#!/usr/bin/env python3
"""Create a relative-path SHA-256 manifest for frozen derived arrays."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def expected_files(kind: str) -> list[str]:
    if kind == "upsampling":
        names = ["metadata.json"]
        for size in (8192, 4096, 2048, 1024, 512):
            names.extend((f"{size}_gt.npy", f"{size}_processed.npy"))
        return names
    return [
        "metadata.json",
        "8192_complete.npy",
        *(f"2048_{mode}_{view}.npy" for mode in ("uls", "tls") for view in range(4)),
        "views.json",
    ]


def temporary_path(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def publish_new(temporary: Path, destination: Path) -> None:
    try:
        with destination.open("xb"):
            pass
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite {destination}") from None
    try:
        temporary.replace(destination)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--kind", choices=("upsampling", "completion"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    if args.output.exists() or summary_path.exists():
        existing = args.output if args.output.exists() else summary_path
        raise SystemExit(f"refusing to overwrite {existing}")
    split = json.loads(args.split.read_text())
    identifiers = [
        str(tree_id)
        for subset in ("train", "val", "test")
        for tree_id in split[subset]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    artifact_count = 0
    manifest_temporary = None
    summary_temporary = None
    published = []
    try:
        manifest_temporary = temporary_path(args.output)
        summary_temporary = temporary_path(summary_path)
        with manifest_temporary.open("w") as handle:
            for tree_index, tree_id in enumerate(identifiers):
                for filename in expected_files(args.kind):
                    path = args.root / tree_id / filename
                    if not path.is_file() or path.is_symlink():
                        raise FileNotFoundError(
                            f"missing regular file: {tree_id}/{filename}"
                        )
                    size = path.stat().st_size
                    record = {
                        "tree_id": tree_id,
                        "artifact": f"{tree_id}/{filename}",
                        "bytes": size,
                        "sha256": sha256(path),
                    }
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    total_bytes += size
                    artifact_count += 1
                print(f"[{tree_index + 1}/{len(identifiers)}] {tree_id}")
        summary = {
            "kind": args.kind,
            "tree_count": len(identifiers),
            "artifact_count": artifact_count,
            "total_bytes": total_bytes,
            "manifest": args.output.name,
            "manifest_sha256": sha256(manifest_temporary),
        }
        summary_temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        publish_new(manifest_temporary, args.output)
        published.append(args.output)
        publish_new(summary_temporary, summary_path)
        published.append(summary_path)
    except BaseException:
        if manifest_temporary is not None:
            manifest_temporary.unlink(missing_ok=True)
        if summary_temporary is not None:
            summary_temporary.unlink(missing_ok=True)
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
