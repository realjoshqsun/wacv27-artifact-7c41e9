"""Validate split disjointness, cardinalities, metadata, and derived files."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

from treebench.constants import REFERENCE_POINTS
from treebench.data.datasets import load_split
from treebench.data.preprocess import DERIVED_SIZES
from treebench.data.splits import validate_split


def _row_set(points: np.ndarray) -> set[bytes]:
    contiguous = np.ascontiguousarray(points)
    return {row.tobytes() for row in contiguous}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _check_metadata(path: Path) -> dict:
    metadata = json.loads(path.read_text())
    if "center" not in metadata or "scale" not in metadata:
        raise ValueError(f"missing center/scale in {path}")
    center = np.asarray(metadata["center"], dtype=np.float64)
    scale = float(metadata["scale"])
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid center/scale in {path}")
    artifacts = metadata.get("artifacts_sha256", {})
    if not isinstance(artifacts, dict):
        raise ValueError(f"artifacts_sha256 must be an object in {path}")
    for filename, expected in artifacts.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"unsafe artifact path in {path}: {filename!r}")
        if not isinstance(expected, str) or len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected.lower()
        ):
            raise ValueError(f"invalid artifact hash in {path}: {filename!r}")
        artifact = path.parent / filename
        if artifact.is_symlink() or not artifact.is_file() or _sha256(artifact) != expected:
            raise ValueError(f"artifact hash mismatch: {artifact}")
    return metadata


def _expected_artifacts(task: str) -> set[str]:
    if task == "upsampling":
        return {
            filename
            for size in DERIVED_SIZES
            for filename in (f"{size}_gt.npy", f"{size}_processed.npy")
        }
    return {
        "8192_complete.npy",
        *(f"2048_{mode}_{view}.npy" for mode in ("uls", "tls") for view in range(4)),
        "views.json",
    }


def _require_exact_artifacts(metadata: dict, task: str, tree_id: str) -> None:
    artifacts = metadata.get("artifacts_sha256")
    if not isinstance(artifacts, dict):
        raise ValueError(f"generated-data checksums missing for {tree_id}")
    expected = _expected_artifacts(task)
    actual = set(artifacts)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"generated-data checksum set mismatch for {tree_id}; "
            f"missing={missing}, extra={extra}"
        )


def _validate_tree(
    root: Path,
    tree_id: str,
    task: str,
    *,
    require_provenance: bool,
    upsampling_root: Optional[Path] = None,
) -> None:
    directory = root / tree_id
    metadata = _check_metadata(directory / "metadata.json")
    metadata_id = str(metadata.get("tree_id", metadata.get("treeID", tree_id)))
    if metadata_id != tree_id:
        raise ValueError(f"metadata tree ID mismatch for {tree_id}")
    if require_provenance:
        _require_exact_artifacts(metadata, task, tree_id)
    if task == "upsampling":
        arrays, raw_arrays = {}, {}
        for size in DERIVED_SIZES:
            arrays[size] = np.load(directory / f"{size}_processed.npy")
            raw_arrays[size] = np.load(directory / f"{size}_gt.npy")
        for size, points in arrays.items():
            if points.shape != (size, 3) or not np.isfinite(points).all():
                raise ValueError(f"invalid {size}-point array for {tree_id}")
            if raw_arrays[size].shape != (size, 3) or not np.isfinite(raw_arrays[size]).all():
                raise ValueError(f"invalid raw {size}-point array for {tree_id}")
            restored = arrays[size] * float(metadata["scale"]) + np.asarray(metadata["center"])
            if not np.allclose(restored, raw_arrays[size], rtol=1e-6, atol=2e-6):
                raise ValueError(f"normalization mismatch at {size} points for {tree_id}")
        for larger, smaller in zip(DERIVED_SIZES, DERIVED_SIZES[1:]):
            if not _row_set(arrays[smaller]).issubset(_row_set(arrays[larger])):
                raise ValueError(f"{smaller}-point input is not nested for {tree_id}")
    else:
        complete = np.load(directory / "8192_complete.npy")
        if complete.shape != (REFERENCE_POINTS, 3) or not np.isfinite(complete).all():
            raise ValueError(f"invalid completion target for {tree_id}")
        for mode in ("uls", "tls"):
            for view in range(4):
                partial = np.load(directory / f"2048_{mode}_{view}.npy")
                if partial.shape != (2_048, 3) or not np.isfinite(partial).all():
                    raise ValueError(f"invalid {mode} view {view} for {tree_id}")
                if not _row_set(partial).issubset(_row_set(complete)):
                    raise ValueError(
                        f"{mode} view {view} is not sampled from the completion target "
                        f"for {tree_id}"
                    )
        views_path = directory / "views.json"
        if views_path.exists():
            if views_path.is_symlink() or not views_path.is_file():
                raise ValueError(f"views.json must be a regular file for {tree_id}")
            views = json.loads(views_path.read_text())
            if not isinstance(views, list) or len(views) != 8:
                raise ValueError(f"completion needs exactly eight view records for {tree_id}")
            expected_keys = {
                (mode, view) for mode in ("uls", "tls") for view in range(4)
            }
            actual_keys = {(item["mode"], int(item["view_index"])) for item in views}
            if actual_keys != expected_keys:
                raise ValueError(f"completion view provenance mismatch for {tree_id}")
            for item in views:
                mode, view = item["mode"], int(item["view_index"])
                expected_artifact = f"2048_{mode}_{view}.npy"
                if item.get("artifact") != expected_artifact:
                    raise ValueError(f"unexpected view artifact for {tree_id}: {item.get('artifact')!r}")
                if not isinstance(item.get("seed"), int) or not isinstance(
                    item.get("parameters"), dict
                ):
                    raise ValueError(f"incomplete view provenance for {tree_id}")
                artifact = directory / expected_artifact
                if artifact.is_symlink() or _sha256(artifact) != item.get("sha256"):
                    raise ValueError(f"view hash mismatch: {artifact}")
                if metadata.get("artifacts_sha256", {}).get(expected_artifact) != item.get(
                    "sha256"
                ):
                    raise ValueError(f"metadata/view hash disagreement: {artifact}")
        elif require_provenance:
            raise ValueError(f"views.json missing for {tree_id}")
        if upsampling_root is not None:
            up_target = np.load(upsampling_root / tree_id / "8192_processed.npy")
            if not np.allclose(complete, up_target, rtol=0, atol=1e-6):
                raise ValueError(f"completion/reference target mismatch for {tree_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--task", choices=("upsampling", "completion"))
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--require-provenance", action="store_true")
    parser.add_argument(
        "--upsampling-root",
        type=Path,
        help="with completion, verify the target against this upsampling root",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.require_provenance and args.data_root is None:
        raise SystemExit("--require-provenance requires --data-root and --task")
    if args.upsampling_root is not None and args.task != "completion":
        raise SystemExit("--upsampling-root is valid only with --task completion")

    manifest = json.loads(args.split.read_text())
    report = validate_split(manifest, expected_total=args.expected_total)
    if (args.data_root is None) != (args.task is None):
        raise SystemExit("--data-root and --task must be supplied together")
    if args.data_root is not None:
        identifiers = [
            tree_id
            for subset in ("train", "val", "test")
            for tree_id in load_split(args.split, subset)
        ]
        if args.limit is not None:
            identifiers = identifiers[: args.limit]
        for index, tree_id in enumerate(identifiers):
            _validate_tree(
                args.data_root,
                tree_id,
                args.task,
                require_provenance=args.require_provenance,
                upsampling_root=args.upsampling_root,
            )
            print(f"[{index + 1}/{len(identifiers)}] {tree_id}")
        report["files_checked"] = len(identifiers)
        report["task"] = args.task
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
