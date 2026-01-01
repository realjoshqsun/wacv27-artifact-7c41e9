"""Join taxonomy and aggregate validation/test metrics into the paper schema."""

import argparse
import csv
from pathlib import Path

from treebench.aggregation import (
    aggregate_val_test,
    attach_taxonomy,
    load_taxonomy,
    read_metric_rows,
)
from treebench.constants import ALL_METRIC_NAMES
from treebench.data.datasets import load_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument(
        "--task",
        choices=("completion", "upsampling-4x", "upsampling-16x"),
        required=True,
    )
    parser.add_argument(
        "--split",
        type=Path,
        required=True,
        help="fixed split manifest used to verify exact val/test sample coverage",
    )
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    taxonomy = load_taxonomy(args.taxonomy)
    val_rows = attach_taxonomy(
        read_metric_rows(
            args.val,
            expected_task=args.task,
            expected_ids=load_split(args.split, "val"),
            subset="val",
        ),
        taxonomy,
    )
    test_rows = attach_taxonomy(
        read_metric_rows(
            args.test,
            expected_task=args.task,
            expected_ids=load_split(args.split, "test"),
            subset="test",
        ),
        taxonomy,
    )
    rows = aggregate_val_test(val_rows, test_rows)
    fields = [
        "group_type",
        "group",
        "n_val",
        *(f"val_{name}" for name in ALL_METRIC_NAMES),
        "n_test",
        *(f"test_{name}" for name in ALL_METRIC_NAMES),
    ]
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} aggregate rows to {args.output}")


if __name__ == "__main__":
    main()
