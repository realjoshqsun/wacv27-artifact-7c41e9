import csv
import math
from pathlib import Path

import pytest

from treebench.aggregation import aggregate_val_test, read_metric_rows
from treebench.constants import ALL_METRIC_NAMES


def _row(name: str, genus: str, species: str, value: float) -> dict:
    return {
        "name": name,
        "genus": genus,
        "species": species,
        **{metric: value for metric in ALL_METRIC_NAMES},
    }


def test_aggregate_has_paper_schema_and_macro_means() -> None:
    val = [_row("a", "G1", "S1", 1.0), _row("b", "G2", "S2", 3.0)]
    test = [_row("c", "G1", "S1", 2.0)]
    rows = aggregate_val_test(val, test)

    overall = rows[0]
    assert overall["group_type"] == "overall"
    assert overall["n_val"] == 2
    assert overall["n_test"] == 1
    assert overall["val_cd_l1"] == 2.0
    assert overall["test_cd_l1"] == 2.0
    assert len(overall) == 68


def test_legacy_hd_column_is_accepted_as_mhd2(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    fields = [
        "name",
        "task",
        "mode",
        "view_index",
        "hd",
        *(name for name in ALL_METRIC_NAMES if name != "mhd2"),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "name": "a",
                "task": "upsampling_4x",
                "mode": "",
                "view_index": "",
                "hd": 0.25,
                **{name: 0.0 for name in ALL_METRIC_NAMES if name != "mhd2"},
            }
        )
    rows = read_metric_rows(
        path,
        expected_task="upsampling-4x",
        expected_ids=["a"],
        subset="test",
    )
    assert rows[0]["mhd2"] == "0.25"


def test_aggregate_propagates_nonfinite_samples() -> None:
    val = [_row("a", "G1", "S1", 1.0), _row("b", "G1", "S1", float("nan"))]
    test = [_row("c", "G1", "S1", 2.0)]
    rows = aggregate_val_test(val, test)
    assert math.isnan(rows[0]["val_cd_l1"])


def test_metric_reader_rejects_duplicate_or_incomplete_sample_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "completion.csv"
    fields = ["name", "task", "mode", "view_index", *ALL_METRIC_NAMES]
    row = {
        "name": "00001",
        "task": "completion",
        "mode": "uls",
        "view_index": 0,
        **{name: 0.0 for name in ALL_METRIC_NAMES},
    }
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([row, row])

    with pytest.raises(ValueError, match="duplicate test sample key"):
        read_metric_rows(
            path,
            expected_task="completion",
            expected_ids=["00001"],
            subset="test",
        )

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="missing 7 expected test samples"):
        read_metric_rows(
            path,
            expected_task="completion",
            expected_ids=["00001"],
            subset="test",
        )
