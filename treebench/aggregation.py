"""Taxonomy-aware macro aggregation for TreeBench per-sample metrics."""

import csv
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

from treebench.constants import ALL_METRIC_NAMES, COMPLETION_VIEWS_PER_MODE


TASK_ROW_NAMES = {
    "completion": "completion",
    "upsampling-4x": "upsampling_4x",
    "upsampling-16x": "upsampling_16x",
    "upsampling_4x": "upsampling_4x",
    "upsampling_16x": "upsampling_16x",
}


def load_taxonomy(path: Path) -> Dict[str, dict]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("taxonomy CSV has missing or duplicate headers")
        rows = list(reader)
    output = {}
    for row in rows:
        tree_id = row.get("treeID_str") or row.get("tree_id") or row.get("treeID")
        if tree_id is None:
            raise ValueError("taxonomy CSV needs treeID_str, tree_id, or treeID")
        tree_id = str(tree_id).zfill(5)
        if tree_id in output:
            raise ValueError(f"duplicate taxonomy tree ID: {tree_id}")
        output[tree_id] = {
            "genus": row["genus"],
            "species": row["species"],
        }
    return output


def _canonical_tree_id(value: object) -> str:
    return str(value).zfill(5)


def _expected_sample_keys(
    expected_ids: Sequence[str], expected_task: str
) -> set[tuple[object, ...]]:
    identifiers = [_canonical_tree_id(tree_id) for tree_id in expected_ids]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("expected split identifiers contain duplicates")
    if expected_task == "completion":
        return {
            (tree_id, mode, view)
            for tree_id in identifiers
            for mode in ("uls", "tls")
            for view in range(COMPLETION_VIEWS_PER_MODE)
        }
    return {(tree_id,) for tree_id in identifiers}


def read_metric_rows(
    path: Path,
    *,
    expected_task: str,
    expected_ids: Sequence[str],
    subset: str,
) -> list[dict]:
    """Read one canonical per-sample CSV and verify its exact split coverage."""
    if expected_task not in TASK_ROW_NAMES:
        raise ValueError(f"unsupported expected task: {expected_task!r}")
    if subset not in ("val", "test"):
        raise ValueError("subset must be val or test")
    row_task = TASK_ROW_NAMES[expected_task]
    completion = row_task == "completion"
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if not fieldnames or len(fieldnames) != len(set(fieldnames)):
            raise ValueError(f"{path} has missing or duplicate CSV headers")
        if "mhd2" in fieldnames and "hd" in fieldnames:
            raise ValueError(f"{path} contains both mhd2 and legacy hd")
        rows = list(reader)
    required = {"name", "task", *ALL_METRIC_NAMES}
    if completion:
        required.update(("mode", "view_index"))
    if "mhd2" not in fieldnames and "hd" in fieldnames:
        required.remove("mhd2")
        required.add("hd")
    missing_columns = sorted(required - set(fieldnames))
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {', '.join(missing_columns)}")

    seen: set[tuple[object, ...]] = set()
    for row_index, row in enumerate(rows, start=2):
        if None in row:
            raise ValueError(f"{path}:{row_index} has more values than its header")
        if "mhd2" not in row and "hd" in row:
            row["mhd2"] = row["hd"]
        missing = [name for name in ALL_METRIC_NAMES if name not in row]
        if missing:
            raise ValueError(f"{path} is missing metrics: {', '.join(missing)}")
        if row.get("task") != row_task:
            raise ValueError(
                f"{path}:{row_index} has task {row.get('task')!r}; expected {row_task!r}"
            )
        raw_name = str(row.get("name", "")).strip()
        if not raw_name:
            raise ValueError(f"{path}:{row_index} has an empty sample name")
        tree_id = _canonical_tree_id(raw_name)
        if completion:
            mode = row.get("mode")
            if mode not in ("uls", "tls"):
                raise ValueError(f"{path}:{row_index} has invalid completion mode {mode!r}")
            try:
                view = int(str(row.get("view_index", "")))
            except ValueError as error:
                raise ValueError(
                    f"{path}:{row_index} has a non-integer completion view index"
                ) from error
            if not 0 <= view < COMPLETION_VIEWS_PER_MODE:
                raise ValueError(f"{path}:{row_index} has invalid completion view {view}")
            key: tuple[object, ...] = (tree_id, mode, view)
        else:
            if row.get("mode", "") or row.get("view_index", ""):
                raise ValueError(
                    f"{path}:{row_index} has completion fields for an upsampling task"
                )
            key = (tree_id,)
        if key in seen:
            raise ValueError(f"{path} contains duplicate {subset} sample key {key!r}")
        seen.add(key)
        for metric in ALL_METRIC_NAMES:
            try:
                float(row[metric])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}:{row_index} has a non-numeric value for {metric}"
                ) from error

    expected = _expected_sample_keys(expected_ids, row_task)
    missing_keys = expected - seen
    extra_keys = seen - expected
    if missing_keys or extra_keys:
        details = []
        if missing_keys:
            details.append(f"missing {len(missing_keys)} expected {subset} samples")
        if extra_keys:
            details.append(f"contains {len(extra_keys)} unexpected {subset} samples")
        raise ValueError(f"{path} sample coverage mismatch: {'; '.join(details)}")
    return rows


def attach_taxonomy(rows: Iterable[dict], taxonomy: Dict[str, dict]) -> list[dict]:
    output = []
    for row in rows:
        tree_id = str(row["name"]).zfill(5)
        if tree_id not in taxonomy:
            raise KeyError(f"taxonomy is missing tree {tree_id}")
        output.append({**row, **taxonomy[tree_id]})
    return output


def _metric_mean(rows: list[dict], metric: str) -> float:
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    return float(values.mean()) if len(values) else float("nan")


def aggregate_val_test(val_rows: list[dict], test_rows: list[dict]) -> list[dict]:
    """Return overall, genus, and species rows in the 68-column paper schema."""
    result = []
    for group_type in ("overall", "genus", "species"):
        groups = ["overall"] if group_type == "overall" else sorted(
            {row[group_type] for row in (*val_rows, *test_rows)}
        )
        for group in groups:
            if group_type == "overall":
                val_group, test_group = val_rows, test_rows
            else:
                val_group = [row for row in val_rows if row[group_type] == group]
                test_group = [row for row in test_rows if row[group_type] == group]
            row = {"group_type": group_type, "group": group, "n_val": len(val_group)}
            row.update(
                {f"val_{name}": _metric_mean(val_group, name) for name in ALL_METRIC_NAMES}
            )
            row["n_test"] = len(test_group)
            row.update(
                {f"test_{name}": _metric_mean(test_group, name) for name in ALL_METRIC_NAMES}
            )
            result.append(row)
    return result
