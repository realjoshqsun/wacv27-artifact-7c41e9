import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from treebench.cli.validate import _validate_tree
from treebench.data.preprocess import DERIVED_SIZES
from scripts.verify_release import audit_overlay_manifest


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def _run(
    *arguments: str, cwd: Path, env: Optional[dict] = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(arguments),
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _copy_release(destination: Path) -> None:
    shutil.copytree(
        RELEASE_ROOT,
        destination,
        ignore=shutil.ignore_patterns(
            ".git",
            ".pytest_cache",
            "__pycache__",
            "*.pyc",
            "*.egg-info",
            "build",
            "dist",
            "*.zip",
        ),
    )


def _copy_overlay_release(destination: Path) -> tuple[Path, dict]:
    (destination / "baselines").mkdir(parents=True)
    shutil.copytree(
        RELEASE_ROOT / "baselines/overlays", destination / "baselines/overlays"
    )
    shutil.copy2(
        RELEASE_ROOT / "baselines/overlay_manifest.json",
        destination / "baselines/overlay_manifest.json",
    )
    registry = json.loads((RELEASE_ROOT / "baselines/registry.json").read_text())
    return destination, registry


def test_overlay_manifest_gate_rejects_hash_status_and_blocked_payload_drift(
    tmp_path: Path,
) -> None:
    root, registry = _copy_overlay_release(tmp_path / "hash")
    manifest = json.loads((root / "baselines/overlay_manifest.json").read_text())
    payload_value = next(
        entry["payload_path"] for entry in manifest["files"] if entry["materialize"]
    )
    payload = root / payload_value
    payload.write_bytes(payload.read_bytes() + b"\n# drift\n")
    assert any(
        "overlay payload hash mismatch" in failure
        for failure in audit_overlay_manifest(root, registry)
    )

    root, registry = _copy_overlay_release(tmp_path / "status")
    manifest_path = root / "baselines/overlay_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["baseline_status"].pop(next(iter(manifest["baseline_status"])))
    manifest_path.write_text(json.dumps(manifest))
    assert any(
        "overlay status IDs disagree" in failure
        for failure in audit_overlay_manifest(root, registry)
    )

    root, registry = _copy_overlay_release(tmp_path / "blocked")
    manifest = json.loads((root / "baselines/overlay_manifest.json").read_text())
    blocked = next(entry for entry in manifest["files"] if not entry["materialize"])
    blocked_path = root / "baselines/overlays" / blocked["model_ids"][0] / blocked[
        "source_path"
    ]
    blocked_path.parent.mkdir(parents=True, exist_ok=True)
    blocked_path.write_text("forbidden redistributed source\n")
    assert any(
        "LEGAL_BLOCKED payload is present" in failure
        for failure in audit_overlay_manifest(root, registry)
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
def test_package_outside_root_then_initialize_with_forced_anonymous_identity(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    _copy_release(release)

    inside = release / "review.zip"
    rejected = _run(
        sys.executable,
        "scripts/package_review.py",
        "--output",
        str(inside),
        cwd=release,
    )
    assert rejected.returncode != 0
    assert "outside the release root" in (rejected.stdout + rejected.stderr)
    assert not inside.exists()

    packaged = _run(sys.executable, "scripts/package_review.py", cwd=release)
    assert packaged.returncode == 0, packaged.stdout + packaged.stderr
    archive = tmp_path / "treebench_review.zip"
    assert archive.is_file()
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
    assert names
    assert all(not name.endswith(".zip") and "/.git/" not in name for name in names)

    template = tmp_path / "hostile-template"
    template_hook = template / "hooks" / "pre-commit"
    template_hook.parent.mkdir(parents=True)
    template_hook.write_text("#!/usr/bin/env bash\nexit 91\n")
    template_hook.chmod(0o755)
    global_hooks = tmp_path / "hostile-hooks"
    global_hook = global_hooks / "pre-commit"
    global_hooks.mkdir()
    global_hook.write_text("#!/usr/bin/env bash\nexit 92\n")
    global_hook.chmod(0o755)
    global_config = tmp_path / "gitconfig"
    global_excludes = tmp_path / "global-excludes"
    global_excludes.write_text("README.md\n")
    global_attributes = tmp_path / "global-attributes"
    global_attributes.write_text("README.md filter=hostile\n")
    for key, value in (
        ("init.templateDir", str(template)),
        ("init.defaultBranch", "identifying-branch"),
        ("core.hooksPath", str(global_hooks)),
        ("core.excludesFile", str(global_excludes)),
        ("core.attributesFile", str(global_attributes)),
        ("filter.hostile.clean", "/bin/false"),
        ("filter.hostile.required", "true"),
        ("commit.gpgSign", "true"),
        ("gpg.program", "/bin/false"),
    ):
        configured = _run(
            "git", "config", "--file", str(global_config), key, value, cwd=tmp_path
        )
        assert configured.returncode == 0, configured.stderr

    identifying_email = "identifying" + "@" + "example.invalid"
    anonymous_email = "anonymous" + "@" + "users.noreply.github.com"
    env = os.environ.copy()
    env.update(
        {
            "PATH": str(Path(sys.executable).parent)
            + os.pathsep
            + env.get("PATH", ""),
            "GIT_CONFIG_GLOBAL": str(global_config),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "Identifying Person",
            "GIT_AUTHOR_EMAIL": identifying_email,
            "GIT_COMMITTER_NAME": "Identifying Person",
            "GIT_COMMITTER_EMAIL": identifying_email,
        }
    )
    initialized = _run("bash", "scripts/initialize_anonymous_repo.sh", cwd=release, env=env)
    assert initialized.returncode == 0, initialized.stdout + initialized.stderr

    identity = _run(
        "git",
        "log",
        "-1",
        "--format=%an%n%ae%n%cn%n%ce",
        cwd=release,
        env=env,
    )
    assert identity.returncode == 0, identity.stderr
    assert identity.stdout.splitlines() == [
        "Anonymous Authors",
        anonymous_email,
        "Anonymous Authors",
        anonymous_email,
    ]
    commit = _run("git", "cat-file", "commit", "HEAD", cwd=release, env=env)
    assert commit.returncode == 0, commit.stderr
    assert "gpgsig " not in commit.stdout
    hooks_path = _run(
        "git", "config", "--local", "--get", "core.hooksPath", cwd=release, env=env
    )
    assert hooks_path.stdout.strip() == "/dev/null"
    assert not (release / ".git" / "hooks" / "pre-commit").exists()
    branch = _run("git", "branch", "--show-current", cwd=release, env=env)
    assert branch.stdout.strip() == "main"
    tracked = _run("git", "ls-files", "README.md", cwd=release, env=env)
    assert tracked.stdout.strip() == "README.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_valid_upsampling_tree(directory: Path) -> None:
    directory.mkdir(parents=True)
    artifacts = {}
    for size in DERIVED_SIZES:
        points = np.zeros((size, 3), dtype=np.float32)
        for suffix in ("gt", "processed"):
            path = directory / f"{size}_{suffix}.npy"
            np.save(path, points)
            artifacts[path.name] = _sha256(path)
    metadata = {
        "tree_id": directory.name,
        "center": [0.0, 0.0, 0.0],
        "scale": 1.0,
        "artifacts_sha256": artifacts,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata))


def _write_valid_completion_tree(directory: Path) -> None:
    directory.mkdir(parents=True)
    target = directory / "8192_complete.npy"
    np.save(target, np.zeros((8192, 3), dtype=np.float32))
    artifacts = {target.name: _sha256(target)}
    views = []
    for mode in ("uls", "tls"):
        for view_index in range(4):
            path = directory / f"2048_{mode}_{view_index}.npy"
            np.save(path, np.zeros((2048, 3), dtype=np.float32))
            digest = _sha256(path)
            artifacts[path.name] = digest
            views.append(
                {
                    "tree_id": directory.name,
                    "mode": mode,
                    "view_index": view_index,
                    "seed": view_index,
                    "parameters": {},
                    "artifact": path.name,
                    "sha256": digest,
                }
            )
    views_path = directory / "views.json"
    views_path.write_text(json.dumps(views))
    artifacts[views_path.name] = _sha256(views_path)
    metadata = {
        "tree_id": directory.name,
        "center": [0.0, 0.0, 0.0],
        "scale": 1.0,
        "artifacts_sha256": artifacts,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata))


def test_require_provenance_enforces_exact_artifacts_and_view_hashes(
    tmp_path: Path,
) -> None:
    upsampling = tmp_path / "upsampling" / "up"
    _write_valid_upsampling_tree(upsampling)
    _validate_tree(tmp_path / "upsampling", "up", "upsampling", require_provenance=True)
    metadata_path = upsampling / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    unexpected = upsampling / "unexpected.npy"
    unexpected.write_bytes(b"unexpected")
    metadata["artifacts_sha256"][unexpected.name] = _sha256(unexpected)
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="checksum set mismatch"):
        _validate_tree(
            tmp_path / "upsampling", "up", "upsampling", require_provenance=True
        )

    completion = tmp_path / "completion" / "complete"
    _write_valid_completion_tree(completion)
    _validate_tree(
        tmp_path / "completion", "complete", "completion", require_provenance=True
    )
    views_path = completion / "views.json"
    views = json.loads(views_path.read_text())
    views[0]["seed"] = 999
    views_path.write_text(json.dumps(views))
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        _validate_tree(
            tmp_path / "completion", "complete", "completion", require_provenance=True
        )


def test_completion_validator_rejects_hashed_points_outside_target(
    tmp_path: Path,
) -> None:
    completion = tmp_path / "completion" / "complete"
    _write_valid_completion_tree(completion)
    partial_path = completion / "2048_uls_0.npy"
    partial = np.load(partial_path)
    partial[0] = [1.0, 1.0, 1.0]
    np.save(partial_path, partial)
    digest = _sha256(partial_path)

    views_path = completion / "views.json"
    views = json.loads(views_path.read_text())
    next(item for item in views if item["mode"] == "uls" and item["view_index"] == 0)[
        "sha256"
    ] = digest
    views_path.write_text(json.dumps(views))
    metadata_path = completion / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["artifacts_sha256"][partial_path.name] = digest
    metadata["artifacts_sha256"][views_path.name] = _sha256(views_path)
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="not sampled from the completion target"):
        _validate_tree(
            tmp_path / "completion", "complete", "completion", require_provenance=True
        )


def _write_hash_fixture(root: Path, tree_id: str, kind: str) -> None:
    from scripts.hash_derived_dataset import expected_files

    directory = root / tree_id
    directory.mkdir(parents=True)
    for filename in expected_files(kind):
        (directory / filename).write_bytes(f"{tree_id}:{filename}".encode())


def test_completion_hash_manifest_includes_views_and_publishes_atomically(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    _write_hash_fixture(data, "tree", "completion")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["tree"], "val": [], "test": []}))
    output = tmp_path / "completion.sha256.jsonl"
    completed = _run(
        sys.executable,
        str(RELEASE_ROOT / "scripts/hash_derived_dataset.py"),
        "--root",
        str(data),
        "--split",
        str(split),
        "--kind",
        "completion",
        "--output",
        str(output),
        cwd=RELEASE_ROOT,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 11
    assert records[-1]["artifact"] == "tree/views.json"
    summary = output.with_suffix(output.suffix + ".summary.json")
    assert json.loads(summary.read_text())["manifest_sha256"] == _sha256(output)
    assert not list(tmp_path.glob(".*.tmp"))


def test_hash_manifest_failure_leaves_no_partial_outputs(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_hash_fixture(data, "present", "completion")
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps({"train": ["present", "missing"], "val": [], "test": []})
    )
    output = tmp_path / "failed.sha256.jsonl"
    failed = _run(
        sys.executable,
        str(RELEASE_ROOT / "scripts/hash_derived_dataset.py"),
        "--root",
        str(data),
        "--split",
        str(split),
        "--kind",
        "completion",
        "--output",
        str(output),
        cwd=RELEASE_ROOT,
    )
    assert failed.returncode != 0
    assert not output.exists()
    assert not output.with_suffix(output.suffix + ".summary.json").exists()
    assert not list(tmp_path.glob(".*.tmp"))
