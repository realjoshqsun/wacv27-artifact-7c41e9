"""CLI for reference, completion-view, and split construction."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from treebench.data.completion import derive_seed, prepare_completion_tree
from treebench.data.preprocess import discover_las_files, load_las_xyz, prepare_tree
from treebench.data.splits import (
    make_genus_ood_split,
    make_random_split,
    read_eligible_metadata,
    write_manifest,
)


def _manifest_ids(path: Path) -> set[str]:
    manifest = json.loads(path.read_text())
    return {
        str(tree_id)
        for subset in ("train", "val", "test")
        for tree_id in manifest[subset]
    }


def _references(args: argparse.Namespace) -> None:
    files = list(discover_las_files(args.input))
    if not files:
        raise SystemExit(f"no LAS/LAZ files found under {args.input}")
    selected = _manifest_ids(args.manifest) if args.manifest else None
    files = [path for path in files if selected is None or path.stem in selected]
    if selected is not None:
        found = {path.stem for path in files}
        missing = selected - found
        if missing:
            raise SystemExit(f"{len(missing)} manifest trees are missing from the source root")
    args.output.mkdir(parents=True, exist_ok=True)
    for path in files:
        tree_id = path.stem
        seed = derive_seed(args.seed, tree_id, "reference")
        points = load_las_xyz(path)
        if len(points) < 8_192:
            if selected is not None:
                raise ValueError(f"selected tree {tree_id} has only {len(points)} points")
            continue
        prepare_tree(
            points,
            args.output / tree_id,
            tree_id=tree_id,
            seed=seed,
            fps_backend=args.fps_backend,
            source_path=path,
        )


def _source_metadata(args: argparse.Namespace) -> None:
    files = {path.stem: path for path in discover_las_files(args.input)}
    with args.metadata.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = []
    for row in rows:
        tree_id = str(
            row.get("treeID_str") or row.get("tree_id") or row.get("treeID")
        ).zfill(5)
        path = files.get(tree_id)
        if path is None:
            raise FileNotFoundError(f"source cloud missing for metadata tree {tree_id}")
        try:
            import laspy
        except ImportError as error:
            raise RuntimeError("laspy with a LAZ backend is required") from error
        with laspy.open(path) as reader:
            count = int(reader.header.point_count)
        output.append(
            {
                "tree_id": tree_id,
                "num_points": count,
                "genus": row["genus"],
                "species": row["species"],
                "dataset": row.get("dataset", ""),
                "data_type": row.get("data_type", ""),
                "tree_H": row.get("tree_H", ""),
                "source_filename": path.name,
            }
        )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output[0])
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)


def _completion(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    directories = sorted(path for path in args.upsampling.iterdir() if path.is_dir())
    for directory in directories:
        metadata = json.loads((directory / "metadata.json").read_text())
        reference = np.load(directory / "8192_gt.npy")
        prepare_completion_tree(
            reference,
            args.output / directory.name,
            metadata=metadata,
            tree_id=directory.name,
            base_seed=args.seed,
            tls_distance_fraction=(args.tls_distance, args.tls_distance),
        )


def _splits(args: argparse.Namespace) -> None:
    rows = read_eligible_metadata(args.metadata)
    args.output.mkdir(parents=True, exist_ok=True)
    write_manifest(
        make_random_split(rows, seed=args.seed),
        args.output / "random.json",
    )
    write_manifest(
        make_genus_ood_split(rows, seed=args.seed),
        args.output / "genus_ood.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    references = subparsers.add_parser("references")
    references.add_argument("--input", type=Path, required=True)
    references.add_argument("--output", type=Path, required=True)
    references.add_argument(
        "--manifest",
        type=Path,
        help="optional split manifest whose union fixes the selected source IDs",
    )
    references.add_argument("--seed", type=int, default=42)
    references.add_argument("--fps-backend", choices=("open3d", "numpy"), default="open3d")
    references.set_defaults(function=_references)

    metadata = subparsers.add_parser("source-metadata")
    metadata.add_argument("--input", type=Path, required=True)
    metadata.add_argument("--metadata", type=Path, required=True)
    metadata.add_argument("--output", type=Path, required=True)
    metadata.set_defaults(function=_source_metadata)

    completion = subparsers.add_parser("completion")
    completion.add_argument("--upsampling", type=Path, required=True)
    completion.add_argument("--output", type=Path, required=True)
    completion.add_argument("--seed", type=int, default=42)
    completion.add_argument(
        "--tls-distance",
        type=float,
        default=0.1,
        help="fixed evaluation-view distance as a fraction of tree height",
    )
    completion.set_defaults(function=_completion)

    splits = subparsers.add_parser("splits")
    splits.add_argument("--metadata", type=Path, required=True)
    splits.add_argument("--output", type=Path, required=True)
    splits.add_argument("--seed", type=int, default=42)
    splits.set_defaults(function=_splits)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
