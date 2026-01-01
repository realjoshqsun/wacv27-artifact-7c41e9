"""Evaluate saved normalized predictions with TreeBench's 32 metrics."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from treebench.constants import ALL_METRIC_NAMES, REFERENCE_POINTS
from treebench.data.datasets import CompletionDataset, UpsamplingDataset
from treebench.metrics import compute_all_metrics


def _prediction_path(root: Path, sample: dict) -> Path:
    name = sample["name"]
    if sample["task"] == "completion":
        mode, view = sample["mode"], sample["view_index"]
        candidates = (
            root / f"{name}_{mode}_{view}.npy",
            root / name / f"{mode}_{view}.npy",
            root / name / f"2048_{mode}_{view}.npy",
        )
    else:
        candidates = (root / f"{name}.npy", root / name / "prediction.npy")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "prediction missing; tried " + ", ".join(str(path) for path in candidates)
    )


def _finite_summary(rows: list[dict]) -> dict:
    output = {}
    for name in ALL_METRIC_NAMES:
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        output[name] = {
            "mean": float(finite.mean()) if len(finite) else None,
            "std": float(finite.std()) if len(finite) else None,
            "finite_count": int(len(finite)),
            "total_count": int(len(values)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("upsampling-4x", "upsampling-16x", "completion"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--subset", choices=("val", "test"), default="test")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--point-chunk-size", type=int, default=1_024)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.point_chunk_size < 1:
        raise SystemExit("--point-chunk-size must be positive")
    if not args.predictions.is_dir():
        raise SystemExit(f"prediction directory does not exist: {args.predictions}")

    if args.task == "completion":
        dataset = CompletionDataset(
            args.data_root, args.split, args.subset, frozen=True
        )
    else:
        rate = int(args.task.split("-")[1][:-1])
        dataset = UpsamplingDataset(
            args.data_root, args.split, args.subset, rate=rate
        )

    device = torch.device(args.device)
    rows = []
    total = len(dataset) if args.limit is None else min(len(dataset), args.limit)
    for index in range(total):
        sample = dataset[index]
        path = _prediction_path(args.predictions, sample)
        prediction = np.load(path)
        if prediction.shape != (REFERENCE_POINTS, 3):
            raise ValueError(
                f"{path} has shape {prediction.shape}; expected ({REFERENCE_POINTS}, 3)"
            )
        pred = torch.from_numpy(prediction.astype(np.float32, copy=False)).to(device)
        gt = sample["complete"].to(device)
        metrics = compute_all_metrics(
            pred,
            gt,
            metadata=sample["metadata"],
            point_chunk_size=args.point_chunk_size,
        )
        row = {
            "name": sample["name"],
            "task": sample["task"],
            "mode": sample.get("mode", ""),
            "view_index": sample.get("view_index", ""),
            "prediction": str(path.relative_to(args.predictions)),
        }
        row.update({key: float(value.squeeze().cpu()) for key, value in metrics.items()})
        rows.append(row)
        print(f"[{index + 1}/{total}] {sample['name']}")

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    fields = ["name", "task", "mode", "view_index", "prediction", *ALL_METRIC_NAMES]
    with (args.output / "per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "task": args.task,
        "subset": args.subset,
        "sample_count": len(rows),
        "metrics": _finite_summary(rows),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
