"""Generate a license-clean synthetic fixture for end-to-end smoke tests."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from treebench.data.completion import prepare_completion_tree
from treebench.data.preprocess import prepare_tree


def synthetic_tree(seed: int, count: int = 8_192) -> np.ndarray:
    """Sample a procedural trunk/crown point set; this is not benchmark data."""
    rng = np.random.default_rng(seed)
    trunk_count = count // 5
    crown_count = count - trunk_count
    trunk_z = rng.uniform(0.0, 8.0, trunk_count)
    trunk_angle = rng.uniform(0.0, 2.0 * np.pi, trunk_count)
    trunk_radius = 0.15 + rng.normal(0.0, 0.01, trunk_count)
    trunk = np.column_stack(
        (
            trunk_radius * np.cos(trunk_angle),
            trunk_radius * np.sin(trunk_angle),
            trunk_z,
        )
    )
    direction = rng.normal(size=(crown_count, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radius = 3.0 * np.cbrt(rng.uniform(size=crown_count))
    crown = direction * radius[:, None]
    crown[:, 2] = crown[:, 2] * 1.5 + 8.5
    return np.concatenate((trunk, crown), axis=0).astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty directory: {args.output}")

    upsampling = args.output / "upsampling"
    completion = args.output / "completion"
    identifiers = ("demo_train", "demo_val", "demo_test")
    for index, tree_id in enumerate(identifiers):
        raw = synthetic_tree(args.seed + index)
        metadata = prepare_tree(
            raw,
            upsampling / tree_id,
            tree_id=tree_id,
            seed=args.seed + index,
            fps_backend="numpy",
        )
        reference_raw = np.load(upsampling / tree_id / "8192_gt.npy")
        prepare_completion_tree(
            reference_raw,
            completion / tree_id,
            metadata=metadata,
            tree_id=tree_id,
            base_seed=args.seed,
        )
    split = {
        "train": [identifiers[0]],
        "val": [identifiers[1]],
        "test": [identifiers[2]],
    }
    split_dir = args.output / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "demo.json").write_text(json.dumps(split, indent=2) + "\n")
    predictions = args.output / "predictions"
    up_predictions = predictions / "upsampling-4x"
    up16_predictions = predictions / "upsampling-16x"
    completion_predictions = predictions / "completion"
    up_predictions.mkdir(parents=True, exist_ok=True)
    up16_predictions.mkdir(parents=True, exist_ok=True)
    completion_predictions.mkdir(parents=True, exist_ok=True)
    for prediction_id in identifiers[1:]:
        target = np.load(upsampling / prediction_id / "8192_processed.npy")
        np.save(up_predictions / f"{prediction_id}.npy", target)
        np.save(up16_predictions / f"{prediction_id}.npy", target)
        completion_target = np.load(
            completion / prediction_id / "8192_complete.npy"
        )
        for mode in ("uls", "tls"):
            for view in range(4):
                np.save(
                    completion_predictions / f"{prediction_id}_{mode}_{view}.npy",
                    completion_target,
                )
    with (args.output / "taxonomy.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("tree_id", "genus", "species")
        )
        writer.writeheader()
        for tree_id in identifiers:
            writer.writerow(
                {"tree_id": tree_id, "genus": "Synthetic", "species": "Synthetic_tree"}
            )
    (args.output / "README.md").write_text(
        "This directory is entirely synthetic and is not part of the paper benchmark.\n"
    )
    print(f"generated synthetic fixture at {args.output}")


if __name__ == "__main__":
    main()
