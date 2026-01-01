#!/usr/bin/env python3
"""Download and verify the pinned official FOR-species20K source release."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

FILES = {
    "dev.zip": {
        "url": "https://zenodo.org/api/records/13255198/files/dev.zip/content",
        "bytes": 25_354_980_290,
        "md5": "ecb6196d9d630b095e6f6249b46efdd7",
    },
    "test.zip": {
        "url": "https://zenodo.org/api/records/13255198/files/test.zip/content",
        "bytes": 1_844_174_259,
        "md5": "087eaed01d45bf83643b4d7edcef33f8",
    },
    "tree_metadata_dev.csv": {
        "url": "https://zenodo.org/api/records/13255198/files/tree_metadata_dev.csv/content",
        "bytes": 1_216_814,
        "md5": "603c91ff58c8eab486717ffc82a1b21f",
    },
}


def md5(path: Path) -> str:
    digest = hashlib.md5()  # nosec B324: checksum required by the source record
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, output: Path) -> None:
    spec = FILES[name]
    target = output / name
    if target.exists():
        if target.stat().st_size == spec["bytes"] and md5(target) == spec["md5"]:
            print(f"verified existing {target}")
            return
        raise RuntimeError(f"existing file failed verification: {target}")
    part = output / f"{name}.part"
    if part.exists():
        raise RuntimeError(f"partial download exists; remove or inspect it first: {part}")
    with urlopen(spec["url"]) as response, part.open("xb") as handle:
        while chunk := response.read(8 << 20):
            handle.write(chunk)
    if part.stat().st_size != spec["bytes"] or md5(part) != spec["md5"]:
        raise RuntimeError(f"downloaded file failed verification: {part}")
    part.rename(target)
    print(f"downloaded and verified {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--show-manifest", action="store_true")
    args = parser.parse_args()
    if args.show_manifest:
        print(json.dumps(FILES, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    names = ["dev.zip", "tree_metadata_dev.csv"]
    if args.include_test:
        names.append("test.zip")
    for name in names:
        download(name, args.output)


if __name__ == "__main__":
    main()

