"""Run the license-clean TreeBench pipeline on procedural point clouds."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(root: Path, *arguments: str) -> None:
    command = [sys.executable, "-m", *arguments]
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=root, check=True)


def _assert_identity_summary(path: Path) -> None:
    summary = json.loads(path.read_text())
    metrics = summary["metrics"]
    for name in ("cd_l2", "cd_l1", "mhd2"):
        if metrics[name]["mean"] != 0.0:
            raise RuntimeError(f"identity smoke check failed for {name}")
    for name in ("f1", "crown_projection_iou", "canopy_occupancy_iou"):
        if metrics[name]["mean"] != 1.0:
            raise RuntimeError(f"identity smoke check failed for {name}")
    if not 0.0 < metrics["dcd"]["mean"] < 1e-5:
        raise RuntimeError("identity DCD is outside the expected epsilon range")


def run_smoke(root: Path, work: Path) -> None:
    demo = work / "demo"
    _run(root, "treebench.cli.generate_demo", "--output", str(demo))
    split = demo / "splits/demo.json"

    _run(
        root,
        "treebench.cli.validate",
        "--split",
        str(split),
        "--data-root",
        str(demo / "upsampling"),
        "--task",
        "upsampling",
        "--expected-total",
        "3",
        "--require-provenance",
    )
    _run(
        root,
        "treebench.cli.validate",
        "--split",
        str(split),
        "--data-root",
        str(demo / "completion"),
        "--task",
        "completion",
        "--expected-total",
        "3",
        "--require-provenance",
        "--upsampling-root",
        str(demo / "upsampling"),
    )

    evaluations = work / "evaluations"
    for task in ("upsampling-4x", "upsampling-16x", "completion"):
        data_kind = "completion" if task == "completion" else "upsampling"
        for subset in ("val", "test"):
            output = evaluations / task / subset
            _run(
                root,
                "treebench.cli.evaluate",
                "--task",
                task,
                "--data-root",
                str(demo / data_kind),
                "--split",
                str(split),
                "--subset",
                subset,
                "--predictions",
                str(demo / "predictions" / task),
                "--output",
                str(output),
                "--device",
                "cpu",
                "--limit",
                "1",
            )
            _assert_identity_summary(output / "summary.json")

    _run(
        root,
        "treebench.cli.aggregate",
        "--val",
        str(evaluations / "upsampling-4x/val/per_sample.csv"),
        "--test",
        str(evaluations / "upsampling-4x/test/per_sample.csv"),
        "--task",
        "upsampling-4x",
        "--split",
        str(split),
        "--taxonomy",
        str(demo / "taxonomy.csv"),
        "--output",
        str(work / "upsampling-4x-aggregate.csv"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        help="optional empty directory to preserve; otherwise use temporary storage",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.workdir is not None:
        if args.workdir.exists() and any(args.workdir.iterdir()):
            raise SystemExit(f"refusing non-empty work directory: {args.workdir}")
        args.workdir.mkdir(parents=True, exist_ok=True)
        run_smoke(root, args.workdir.resolve())
        print(f"TreeBench smoke test passed; artifacts kept at {args.workdir}")
        return
    with tempfile.TemporaryDirectory(prefix="treebench-review-") as temporary:
        run_smoke(root, Path(temporary))
    print("TreeBench smoke test passed")


if __name__ == "__main__":
    main()
