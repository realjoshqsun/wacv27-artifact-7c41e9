#!/usr/bin/env python3
"""Clone one pinned upstream baseline and apply its verified TreeBench overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional


RELEASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = RELEASE_ROOT / "baselines" / "registry.json"
DEFAULT_MANIFEST = RELEASE_ROOT / "baselines" / "overlay_manifest.json"


class MaterializationError(RuntimeError):
    """Raised when a baseline cannot be materialized safely."""


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MaterializationError(f"Expected a JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(value: str, field: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise MaterializationError(f"Unsafe {field}: {value!r}")
    return path


def contained_path(root: Path, value: str, field: str) -> Path:
    relative = relative_path(value, field)
    candidate = root.joinpath(*relative.parts)
    resolved_parent = candidate.parent.resolve()
    try:
        resolved_parent.relative_to(root.resolve())
    except ValueError as exc:
        raise MaterializationError(f"{field} escapes its root: {value!r}") from exc
    return candidate


def find_baseline(registry: Dict[str, Any], model: str) -> Dict[str, Any]:
    matches = [entry for entry in registry.get("baselines", []) if entry.get("id") == model]
    if len(matches) != 1:
        known = sorted(entry.get("id", "") for entry in registry.get("baselines", []))
        raise MaterializationError(
            f"Unknown or duplicate baseline {model!r}; known IDs: {', '.join(known)}"
        )
    return matches[0]


def selected_overlay_files(manifest: Dict[str, Any], model: str) -> List[Dict[str, Any]]:
    status = manifest.get("baseline_status", {}).get(model)
    if not isinstance(status, dict):
        raise MaterializationError(f"Overlay status is missing for {model}")
    if status.get("status") == "LEGAL_BLOCKED":
        missing = status.get("missing_required_files", [])
        detail = ", ".join(missing) if missing else "unspecified required files"
        raise MaterializationError(
            f"{model} is LEGAL_BLOCKED: its upstream checkout has no redistribution "
            f"license and required source overrides are not bundled ({detail})."
        )
    if status.get("status") not in {"READY", "ADAPTER_ONLY"}:
        raise MaterializationError(f"Unsupported overlay status for {model}: {status!r}")

    entries = [
        entry
        for entry in manifest.get("files", [])
        if model in entry.get("model_ids", []) and entry.get("materialize") is True
    ]
    if not entries:
        raise MaterializationError(f"No materializable overlay files are registered for {model}")
    return entries


def verify_payloads(entries: Iterable[Dict[str, Any]], release_root: Path) -> None:
    destinations = set()
    for entry in entries:
        payload_value = entry.get("payload_path")
        destination_value = entry.get("destination_path")
        expected = entry.get("sha256")
        if not isinstance(payload_value, str) or not isinstance(destination_value, str):
            raise MaterializationError(f"Malformed overlay entry: {entry!r}")
        if not isinstance(expected, str) or len(expected) != 64:
            raise MaterializationError(f"Malformed SHA-256 for {payload_value}")
        payload = contained_path(release_root, payload_value, "payload_path")
        if payload.is_symlink() or not payload.is_file():
            raise MaterializationError(f"Overlay payload must be a regular file: {payload_value}")
        actual = sha256(payload)
        if actual != expected:
            raise MaterializationError(
                f"Overlay hash mismatch for {payload_value}: expected {expected}, got {actual}"
            )
        destination = str(relative_path(destination_value, "destination_path"))
        if destination in destinations:
            raise MaterializationError(f"Duplicate overlay destination: {destination}")
        destinations.add(destination)


def require_empty_target(target: Path) -> None:
    if target.is_symlink():
        raise MaterializationError("Materialization target cannot be a symbolic link")
    if target.exists():
        if not target.is_dir():
            raise MaterializationError("Materialization target exists and is not a directory")
        try:
            next(target.iterdir())
        except StopIteration:
            return
        raise MaterializationError("Materialization target must be empty")
    if target.parent.exists() and target.parent.is_symlink():
        raise MaterializationError("Materialization target parent cannot be a symbolic link")


def run(command: List[str], *, cwd: Optional[Path] = None) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise MaterializationError(f"Required executable not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or "command failed"
        raise MaterializationError(f"{command[0]} failed: {detail}") from exc
    return completed.stdout.strip()


def clone_pinned(upstream: Dict[str, Any], target: Path, git: str) -> None:
    url = upstream.get("url")
    commit = upstream.get("commit")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise MaterializationError("The upstream URL must be anonymous HTTPS")
    if not isinstance(commit, str) or len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit.lower()
    ):
        raise MaterializationError("The upstream commit must be a full hexadecimal object ID")
    run([git, "clone", "--no-checkout", "--origin", "upstream", url, str(target)])
    run([git, "checkout", "--detach", commit], cwd=target)
    actual = run([git, "rev-parse", "HEAD"], cwd=target)
    if actual != commit:
        raise MaterializationError(
            f"Pinned commit verification failed: expected {commit}, got {actual}"
        )


def safe_destination(target: Path, value: str) -> Path:
    destination = contained_path(target, value, "destination_path")
    relative = relative_path(value, "destination_path")
    cursor = target
    for part in relative.parts[:-1]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise MaterializationError(f"Overlay destination traverses a symlink: {value}")
    if destination.is_symlink():
        raise MaterializationError(f"Overlay destination is a symlink: {value}")
    return destination


def apply_overlay(
    entries: Iterable[Dict[str, Any]], release_root: Path, target: Path
) -> List[Dict[str, str]]:
    applied = []
    for entry in entries:
        payload = contained_path(release_root, entry["payload_path"], "payload_path")
        destination = safe_destination(target, entry["destination_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(payload, destination)
        actual = sha256(destination)
        if actual != entry["sha256"]:
            raise MaterializationError(
                f"Copied overlay hash mismatch for {entry['destination_path']}"
            )
        applied.append(
            {"destination_path": entry["destination_path"], "sha256": actual}
        )
    return applied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="baseline ID from baselines/registry.json")
    parser.add_argument("target", type=Path, help="new or empty checkout directory")
    parser.add_argument("--git", default="git", help="git executable")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        registry = load_json(DEFAULT_REGISTRY)
        manifest = load_json(DEFAULT_MANIFEST)
        baseline = find_baseline(registry, args.model)
        entries = selected_overlay_files(manifest, args.model)
        license_info = baseline.get("upstream", {}).get("license", {})
        if license_info.get("status") != "found" or not license_info.get("spdx"):
            raise MaterializationError(
                f"{args.model} has no verified upstream redistribution license"
            )
        for entry in entries:
            payload_license = entry.get("license_status", {})
            if (
                payload_license.get("upstream") != "found"
                or payload_license.get("spdx") != license_info["spdx"]
            ):
                raise MaterializationError(
                    f"Overlay license mapping disagrees with registry.json: "
                    f"{entry.get('destination_path')}"
                )
        verify_payloads(entries, RELEASE_ROOT)
        target = args.target.expanduser().absolute()
        require_empty_target(target)
        clone_pinned(baseline["upstream"], target, args.git)
        applied = apply_overlay(entries, RELEASE_ROOT, target)
        provenance = {
            "schema_version": "1.0.0",
            "baseline_id": args.model,
            "upstream_url": baseline["upstream"]["url"],
            "upstream_commit": baseline["upstream"]["commit"],
            "registry_sha256": sha256(DEFAULT_REGISTRY),
            "overlay_manifest_sha256": sha256(DEFAULT_MANIFEST),
            "overlay_files": applied,
        }
        provenance_path = target / ".treebench-overlay.json"
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except MaterializationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Materialized {args.model} at pinned commit with {len(applied)} overlay files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
