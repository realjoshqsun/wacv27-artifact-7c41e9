"""Creation and validation of TreeBench random and genus-OOD manifests."""

import csv
import json
from pathlib import Path
from typing import Optional

from treebench.constants import HELD_OUT_GENERA, REFERENCE_POINTS


def read_eligible_metadata(path: Path) -> list[dict]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = []
    for row in rows:
        point_count = row.get("num_points")
        if point_count is None:
            raise ValueError("metadata requires a num_points column")
        if int(float(point_count)) < REFERENCE_POINTS:
            continue
        tree_id = row.get("treeID_str") or row.get("tree_id") or row.get("treeID")
        if tree_id is None:
            raise ValueError("metadata requires treeID_str, tree_id, or treeID")
        row = dict(row)
        row["tree_id"] = str(tree_id).zfill(5)
        output.append(row)
    return output


def _split(*values, **kwargs):
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as error:
        raise RuntimeError("scikit-learn is required to regenerate split manifests") from error
    return train_test_split(*values, **kwargs)


def make_random_split(rows: list[dict], *, seed: int = 42) -> dict:
    ids = [row["tree_id"] for row in rows]
    genera = [row["genus"] for row in rows]
    train_ids, remaining_ids, _, remaining_genera = _split(
        ids,
        genera,
        test_size=0.2,
        random_state=seed,
        stratify=genera,
    )
    val_ids, test_ids = _split(
        remaining_ids,
        test_size=0.5,
        random_state=seed,
        stratify=remaining_genera,
    )
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def make_genus_ood_split(rows: list[dict], *, seed: int = 42) -> dict:
    held_out = set(HELD_OUT_GENERA)
    train_ids = [row["tree_id"] for row in rows if row["genus"] not in held_out]
    test_pool = [row["tree_id"] for row in rows if row["genus"] in held_out]
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("pandas is required to regenerate the genus-OOD split") from error
    pool = pd.Series(test_pool)
    validation = pool.sample(frac=0.5, random_state=seed)
    val_ids = validation.tolist()
    test_ids = pool.drop(validation.index).tolist()
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def validate_split(manifest: dict, *, expected_total: Optional[int] = None) -> dict:
    sets = {}
    for subset in ("train", "val", "test"):
        values = manifest.get(subset)
        if not isinstance(values, list):
            raise ValueError(f"manifest field {subset!r} must be a list")
        sets[subset] = set(map(str, values))
        if len(sets[subset]) != len(values):
            raise ValueError(f"duplicate identifier in {subset}")
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise ValueError("train, val, and test identifiers must be disjoint")
    total = sum(map(len, sets.values()))
    if expected_total is not None and total != expected_total:
        raise ValueError(f"manifest has {total} ids; expected {expected_total}")
    return {"total": total, **{key: len(value) for key, value in sets.items()}}


def write_manifest(manifest: dict, path: Path) -> None:
    validate_split(manifest)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
