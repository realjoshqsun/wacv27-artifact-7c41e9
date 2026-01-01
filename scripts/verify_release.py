#!/usr/bin/env python3
"""Fail-closed structural, anonymity, secret, and protocol release audit."""

import argparse
import hashlib
import json
import py_compile
import re
import sys
from pathlib import Path
from pathlib import PurePosixPath

FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".gz",
    ".ipynb",
    ".las",
    ".laz",
    ".npy",
    ".npz",
    ".o",
    ".pickle",
    ".pkl",
    ".pyc",
    ".pyo",
    ".so",
    ".tar",
    ".pth",
    ".pt",
    ".zip",
}
FORBIDDEN_DIRECTORY_NAMES = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".vscode",
    "__pycache__",
    "build",
    "checkpoints",
    "dist",
    "logs",
    "outputs",
    "runs",
    "tensorboard",
}
FORBIDDEN_FILENAMES = {
    ".DS_Store",
    ".env",
    ".gitmodules",
    ".netrc",
    "credentials",
    "id_dsa",
    "id_ed25519",
    "id_rsa",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def sensitive_patterns() -> list[tuple[str, re.Pattern]]:
    # Build local-only markers in pieces so this scanner does not match itself.
    markers = {
        "private data mount": "/" + "data" + "007/",
        "private home path": "/" + "home/",
        "workspace account": "q84" + "sun",
        "workspace owner": "jo" + "sh",
        "GitHub classic token": "gh" + "p_",
        "GitHub fine-grained token": "github" + "_pat_",
        "Overleaf token helper": "overleaf" + "_git_token",
        "askpass helper": ".overleaf" + "_askpass",
    }
    patterns = [(label, re.compile(re.escape(value), re.IGNORECASE)) for label, value in markers.items()]
    patterns.append(
        (
            "email address",
            re.compile(r"(?<![\w.-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])", re.IGNORECASE),
        )
    )
    patterns.append(
        (
            "generic bearer token",
            re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[A-Za-z0-9._~+/-]{12,}"),
        )
    )
    return patterns


def _safe_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and "." not in path.parts


def audit_overlay_manifest(root: Path, registry: dict) -> list[str]:
    """Verify that the overlay allowlist exactly matches its physical payloads."""
    failures = []
    manifest_path = root / "baselines/overlay_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return [f"cannot read overlay manifest: {error}"]
    if not isinstance(manifest, dict):
        return ["overlay manifest must be a JSON object"]
    if manifest.get("schema_version") != "1.0.0" or manifest.get("hash_algorithm") != "sha256":
        failures.append("overlay manifest schema/hash algorithm is unsupported")
    policy = manifest.get("policy")
    for field in (
        "overlay_allowlist_only",
        "reject_nonempty_materialization_target",
        "reject_hash_mismatch",
        "reject_legal_blocked",
        "bundled_third_party_license_texts",
        "legal_blocked_payload_absent",
    ):
        if not isinstance(policy, dict) or policy.get(field) is not True:
            failures.append(f"overlay policy {field} must be true")

    baselines = registry.get("baselines", [])
    registry_by_id = {
        item.get("id"): item
        for item in baselines
        if isinstance(item, dict) and item.get("id")
    }
    registry_ids = {
        item.get("id") for item in baselines if isinstance(item, dict) and item.get("id")
    }
    statuses = manifest.get("baseline_status")
    if not isinstance(statuses, dict):
        return [*failures, "overlay manifest baseline_status must be an object"]
    status_ids = set(statuses)
    if status_ids != registry_ids:
        failures.append(
            "overlay status IDs disagree with registry: "
            f"missing={sorted(registry_ids - status_ids)}, "
            f"extra={sorted(status_ids - registry_ids)}"
        )
    entries = manifest.get("files")
    if not isinstance(entries, list):
        return [*failures, "overlay manifest files must be a list"]

    materialized_by_model = {model: 0 for model in registry_ids}
    blocked_sources = {model: set() for model in registry_ids}
    declared_payloads = set()
    destinations = set()
    materialized_count = 0
    blocked_count = 0
    for index, entry in enumerate(entries):
        label = f"overlay entry {index}"
        if not isinstance(entry, dict):
            failures.append(f"{label} must be an object")
            continue
        model_ids = entry.get("model_ids")
        if (
            not isinstance(model_ids, list)
            or not model_ids
            or len(model_ids) != len(set(model_ids))
            or any(model not in registry_ids for model in model_ids)
        ):
            failures.append(f"{label} has invalid model_ids")
            continue
        source = entry.get("source_path")
        destination = entry.get("destination_path")
        if not _safe_relative_path(source) or not _safe_relative_path(destination):
            failures.append(f"{label} has an unsafe source/destination path")
            continue
        digest = entry.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            failures.append(f"{label} has an invalid SHA-256")
        materialize = entry.get("materialize")
        if materialize not in (True, False):
            failures.append(f"{label} has a non-boolean materialize field")
            continue
        for model in model_ids:
            key = (model, destination)
            if key in destinations:
                failures.append(f"duplicate overlay destination for {model}: {destination}")
            destinations.add(key)

        payload_value = entry.get("payload_path")
        license_status = entry.get("license_status")
        if materialize:
            materialized_count += 1
            if not _safe_relative_path(payload_value) or not str(payload_value).startswith(
                "baselines/overlays/"
            ):
                failures.append(f"{label} has an invalid materialized payload path")
                continue
            if not isinstance(license_status, dict) or license_status.get("upstream") != "found":
                failures.append(f"{label} lacks a verified upstream license")
            else:
                for model in model_ids:
                    upstream_license = (
                        registry_by_id.get(model, {}).get("upstream", {}).get("license", {})
                    )
                    if (
                        upstream_license.get("status") != "found"
                        or license_status.get("spdx") != upstream_license.get("spdx")
                    ):
                        failures.append(
                            f"{label} license mapping disagrees with registry for {model}"
                        )
            if payload_value in declared_payloads:
                failures.append(f"duplicate overlay payload declaration: {payload_value}")
            declared_payloads.add(payload_value)
            payload = root.joinpath(*PurePosixPath(payload_value).parts)
            if payload.is_symlink() or not payload.is_file():
                failures.append(f"overlay payload is missing or not regular: {payload_value}")
            elif isinstance(digest, str) and sha256(payload) != digest:
                failures.append(f"overlay payload hash mismatch: {payload_value}")
            for model in model_ids:
                materialized_by_model[model] += 1
        else:
            blocked_count += 1
            if payload_value is not None:
                failures.append(f"LEGAL_BLOCKED overlay entry exposes a payload: {source}")
            if entry.get("classification") != "LEGAL_BLOCKED":
                failures.append(f"non-materialized overlay entry is not LEGAL_BLOCKED: {source}")
            for model in model_ids:
                blocked_sources[model].add(source)
                status = statuses.get(model)
                if not isinstance(status, dict) or status.get("status") != "LEGAL_BLOCKED":
                    failures.append(f"non-blocked model has an omitted overlay: {model}/{source}")
                blocked_candidate = root / "baselines/overlays" / model
                blocked_candidate = blocked_candidate.joinpath(*PurePosixPath(source).parts)
                if blocked_candidate.exists() or blocked_candidate.is_symlink():
                    failures.append(f"LEGAL_BLOCKED payload is present: {model}/{source}")

    for model in registry_ids:
        status = statuses.get(model)
        if not isinstance(status, dict):
            continue
        value = status.get("status")
        missing = status.get("missing_required_files")
        if not isinstance(missing, list) or any(
            not _safe_relative_path(item) for item in missing
        ) or len(missing) != len(set(missing)):
            failures.append(f"overlay status has invalid missing_required_files: {model}")
            continue
        if value in {"READY", "ADAPTER_ONLY"}:
            if missing or materialized_by_model[model] == 0 or blocked_sources[model]:
                failures.append(f"runnable overlay status is inconsistent: {model}")
        elif value == "LEGAL_BLOCKED":
            if materialized_by_model[model] or not missing or set(missing) != blocked_sources[model]:
                failures.append(f"LEGAL_BLOCKED overlay status is inconsistent: {model}")
        else:
            failures.append(f"unsupported overlay status for {model}: {value!r}")

    overlay_root = root / "baselines/overlays"
    physical_payloads = {
        path.relative_to(root).as_posix()
        for path in overlay_root.rglob("*")
        if path.is_file()
        and not any(part in FORBIDDEN_DIRECTORY_NAMES for part in path.parts)
    }
    if physical_payloads != declared_payloads:
        failures.append(
            "physical overlay payload set disagrees with manifest: "
            f"missing={sorted(declared_payloads - physical_payloads)[:8]}, "
            f"undeclared={sorted(physical_payloads - declared_payloads)[:8]}"
        )

    license_entries = manifest.get("third_party_licenses")
    declared_license_paths = set()
    licensed_models = set()
    if not isinstance(license_entries, list):
        failures.append("overlay manifest third_party_licenses must be a list")
        license_entries = []
    for index, entry in enumerate(license_entries):
        label = f"third-party license entry {index}"
        if not isinstance(entry, dict):
            failures.append(f"{label} must be an object")
            continue
        model_ids = entry.get("model_ids")
        if (
            not isinstance(model_ids, list)
            or not model_ids
            or len(model_ids) != len(set(model_ids))
            or any(model not in registry_ids for model in model_ids)
        ):
            failures.append(f"{label} has invalid model_ids")
            continue
        overlap = licensed_models.intersection(model_ids)
        if overlap:
            failures.append(f"duplicate third-party license model mapping: {sorted(overlap)}")
        licensed_models.update(model_ids)
        source_path = entry.get("upstream_source_path")
        bundled_path = entry.get("bundled_path")
        if not _safe_relative_path(source_path):
            failures.append(f"{label} has an unsafe upstream source path")
        if (
            not _safe_relative_path(bundled_path)
            or not str(bundled_path).startswith("baselines/third_party_licenses/")
        ):
            failures.append(f"{label} has an invalid bundled path")
            continue
        if bundled_path in declared_license_paths:
            failures.append(f"duplicate bundled third-party license: {bundled_path}")
        declared_license_paths.add(bundled_path)
        digest = entry.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            failures.append(f"{label} has an invalid SHA-256")
        path = root.joinpath(*PurePosixPath(bundled_path).parts)
        if path.is_symlink() or not path.is_file():
            failures.append(f"bundled third-party license is missing: {bundled_path}")
        elif isinstance(digest, str) and sha256(path) != digest:
            failures.append(f"bundled third-party license hash mismatch: {bundled_path}")
        for model in model_ids:
            upstream_license = (
                registry_by_id.get(model, {}).get("upstream", {}).get("license", {})
            )
            if (
                upstream_license.get("status") != "found"
                or entry.get("spdx") != upstream_license.get("spdx")
            ):
                failures.append(f"{label} SPDX disagrees with registry for {model}")
        notice_status = entry.get("notice_status")
        if model_ids == ["pucrn"]:
            if notice_status != "not_present_at_pinned_commit":
                failures.append("PUCRN NOTICE status is not pinned-commit complete")
        elif notice_status != "not_applicable":
            failures.append(f"{label} has an unsupported notice_status")

    physical_license_paths = {
        path.relative_to(root).as_posix()
        for path in (root / "baselines/third_party_licenses").rglob("*")
        if path.is_file()
        and not any(part in FORBIDDEN_DIRECTORY_NAMES for part in path.parts)
    }
    if physical_license_paths != declared_license_paths:
        failures.append(
            "physical third-party license set disagrees with manifest: "
            f"missing={sorted(declared_license_paths - physical_license_paths)}, "
            f"undeclared={sorted(physical_license_paths - declared_license_paths)}"
        )
    ready_models = {
        model
        for model, value in statuses.items()
        if isinstance(value, dict) and value.get("status") in {"READY", "ADAPTER_ONLY"}
    }
    if licensed_models != ready_models:
        failures.append(
            "third-party license coverage disagrees with runnable baselines: "
            f"missing={sorted(ready_models - licensed_models)}, "
            f"extra={sorted(licensed_models - ready_models)}"
        )

    summary = manifest.get("summary")
    expected_summary = {
        "payload_files": materialized_count,
        "legal_blocked_files": blocked_count,
        "materializable_baselines": len(ready_models),
        "evaluation_ready_baselines": len(ready_models),
        "training_launchable_baselines": sum(
            item.get("training_reproduction", {}).get("status") == "VERIFIED"
            for item in baselines
            if isinstance(item, dict)
        ),
        "training_open_baselines": sum(
            item.get("training_reproduction", {}).get("status")
            == "OPEN_COMMAND_PROVENANCE"
            for item in baselines
            if isinstance(item, dict)
        ),
        "legal_blocked_baselines": sum(
            isinstance(value, dict) and value.get("status") == "LEGAL_BLOCKED"
            for value in statuses.values()
        ),
        "bundled_third_party_license_texts": len(license_entries),
    }
    if summary != expected_summary:
        failures.append("overlay manifest summary counts are inconsistent")
    return failures


def audit_registry_publications(registry: dict) -> list[str]:
    failures = []
    venue_pattern = re.compile(r"^[A-Z0-9][A-Z0-9 .&/+\-]{1,31}$")
    baselines = registry.get("baselines")
    if not isinstance(baselines, list):
        return ["registry baselines must be a list before publication audit"]
    for baseline in baselines:
        if not isinstance(baseline, dict):
            failures.append("registry contains a non-object baseline publication entry")
            continue
        identifier = baseline.get("id", "<unknown>")
        publication = baseline.get("publication")
        if not isinstance(publication, dict):
            failures.append(f"baseline publication metadata is missing: {identifier}")
            continue
        venue, year = publication.get("venue"), publication.get("year")
        if not isinstance(venue, str) or not venue_pattern.fullmatch(venue):
            failures.append(f"baseline publication venue is invalid: {identifier}")
        if not isinstance(year, int) or isinstance(year, bool) or not 2000 <= year <= 2026:
            failures.append(f"baseline publication year is invalid: {identifier}")
    return failures


def audit_reference_results(root: Path) -> list[str]:
    try:
        try:
            from scripts.build_paper_tables import collect_reference_results
        except ModuleNotFoundError:
            from build_paper_tables import collect_reference_results

        rows, metrics, _ = collect_reference_results(
            root / "reference_results/full_results.csv",
            manifest_path=root / "reference_results/manifest.json",
            registry_path=root / "baselines/registry.json",
            protocol_path=root / "configs/protocol.json",
        )
    except Exception as error:  # convert the builder's fail-closed result into audit output
        return [f"bundled reference results failed validation: {error}"]
    if len(rows) != 94 or len(metrics) != 32:
        return ["bundled reference results do not contain the frozen 94x32 matrix"]
    return []


def audit(root: Path) -> tuple[list[str], list[str]]:
    failures = []
    warnings = []
    files = []
    excluded_directories = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        forbidden_parts = [
            (index, part)
            for index, part in enumerate(relative.parts)
            if part in FORBIDDEN_DIRECTORY_NAMES
        ]
        if forbidden_parts:
            index, name = forbidden_parts[0]
            excluded = Path(*relative.parts[: index + 1]).as_posix()
            if name == ".git":
                failures.append("repository history is forbidden: .git")
            else:
                excluded_directories.add(excluded)
            continue
        if path.is_symlink():
            failures.append(f"symlink is forbidden: {relative}")
            continue
        if path.is_file():
            files.append(path)
            if path.name in FORBIDDEN_FILENAMES:
                failures.append(f"forbidden file: {relative}")
            if path.suffix.lower() in FORBIDDEN_SUFFIXES:
                failures.append(f"forbidden artifact type: {relative}")
            if path.stat().st_size > 100_000_000:
                failures.append(f"ordinary-Git file exceeds 100 MB: {relative}")

    if excluded_directories:
        warnings.append(
            "generated directories are excluded from Git and the review archive: "
            + ", ".join(sorted(excluded_directories))
        )

    patterns = sensitive_patterns()
    for path in files:
        relative = path.relative_to(root)
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            failures.append(f"binary/non-UTF-8 artifact is forbidden: {relative}")
            continue
        relative_text = relative.as_posix()
        for label, pattern in patterns:
            if pattern.search(text):
                failures.append(f"{label} detected in {relative}")
            if pattern.search(relative_text):
                failures.append(f"{label} detected in path {relative}")
        if path.suffix == ".json":
            try:
                json.loads(text)
            except json.JSONDecodeError as error:
                failures.append(f"invalid JSON {relative}: {error}")
        if path.suffix == ".py":
            try:
                py_compile.compile(str(path), doraise=True, cfile=str(Path("/tmp") / (path.name + ".pyc")))
            except py_compile.PyCompileError as error:
                failures.append(f"Python compile failed for {relative}: {error.msg}")

    required_paths = (
        "README.md",
        "pyproject.toml",
        "configs/protocol.json",
        "manifests/source.json",
        "baselines/registry.json",
        "reference_results/full_results.csv",
        "reference_results/manifest.json",
        "treebench/cli/evaluate.py",
        "treebench/cli/prepare.py",
        "treebench/cli/validate.py",
    )
    for relative in required_paths:
        if not (root / relative).is_file():
            failures.append(f"required release file is missing: {relative}")

    if failures and any("required release file" in item for item in failures):
        return sorted(set(failures)), sorted(set(warnings))

    source = json.loads((root / "manifests/source.json").read_text())
    for split_name, split_spec in source["splits"].items():
        path = root / "manifests" / split_spec["path"]
        if sha256(path) != split_spec["release_compact_sha256"]:
            failures.append(f"split hash mismatch: {split_name}")
        manifest = json.loads(path.read_text())
        actual = {name: len(manifest[name]) for name in ("train", "val", "test")}
        if actual != split_spec["counts"]:
            failures.append(f"split count mismatch: {split_name}")
        sets = [set(manifest[name]) for name in ("train", "val", "test")]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            failures.append(f"split partitions overlap: {split_name}")

    protocol = json.loads((root / "configs/protocol.json").read_text())
    metric_entries = protocol["metrics"]
    if len(metric_entries) != 32:
        failures.append(f"protocol defines {len(metric_entries)} metrics rather than 32")

    registry = json.loads((root / "baselines/registry.json").read_text())
    baselines = registry["baselines"]
    if len(baselines) != 16:
        failures.append(f"registry defines {len(baselines)} baselines rather than 16")
    row_count = sum(len(item["evaluation_scenarios"]) for item in baselines)
    if row_count != registry["expected_method_scenario_rows"] or row_count != 94:
        failures.append(f"baseline scenario count is {row_count}, expected 94")
    for baseline in baselines:
        route = baseline["cross_rate_16x"]
        if route["independent_training_allowed"]:
            failures.append(f"16x training enabled for {baseline['id']}")
        if baseline["entrypoints"]["upsampling_16x"]["train"] is not None:
            failures.append(f"16x train entrypoint present for {baseline['id']}")
    failures.extend(audit_registry_publications(registry))
    failures.extend(audit_overlay_manifest(root, registry))
    failures.extend(audit_reference_results(root))

    checkpoint_manifest = json.loads(
        (root / "manifests/checkpoints.json").read_text()
    )
    if checkpoint_manifest["status"] != "PASS":
        warnings.append("reviewer-accessible checkpoint manifest remains OPEN")
    if not (root / "LICENSE").exists():
        warnings.append("release code license remains OPEN")
    return sorted(set(failures)), sorted(set(warnings))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    failures, warnings = audit(args.root.resolve())
    report = {"failures": failures, "warnings": warnings, "pass": not failures}
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
