#!/usr/bin/env python3
"""Build a deterministic WACV supplementary ZIP after release verification."""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

from verify_release import (
    FORBIDDEN_DIRECTORY_NAMES,
    FORBIDDEN_FILENAMES,
    FORBIDDEN_SUFFIXES,
)

MAX_BYTES = 200_000_000
FIXED_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        help="archive path (default: treebench_review.zip beside the release root)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root.parent / "treebench_review.zip"
    )
    if output == root or root in output.parents:
        raise SystemExit("--output must be outside the release root")
    subprocess.run(
        [sys.executable, str(root / "scripts/verify_release.py"), "--root", str(root)],
        check=True,
    )
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing archive: {output}")
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(
            part in FORBIDDEN_DIRECTORY_NAMES for part in path.relative_to(root).parts
        )
        and path.name not in FORBIDDEN_FILENAMES
        and path.suffix.lower() not in FORBIDDEN_SUFFIXES
    )
    with zipfile.ZipFile(
        output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in files:
            relative = Path("treebench_review") / path.relative_to(root)
            info = zipfile.ZipInfo(str(relative), FIXED_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    size = output.stat().st_size
    if size > MAX_BYTES:
        raise SystemExit(f"archive is {size} bytes; WACV limit is {MAX_BYTES}")
    print(f"created {output} ({size} bytes)")


if __name__ == "__main__":
    main()
