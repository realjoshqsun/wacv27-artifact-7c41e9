import json
from pathlib import Path

import pytest

from treebench.constants import (
    ALL_METRIC_NAMES,
    COMPLETION_INPUT_POINTS,
    COMPLETION_VIEWS_PER_MODE,
    FORESTRY_VOXEL_SIZE_METRES,
    FUEL_BIN_SIZE_METRES,
    FUEL_LAYER_ACTIVE_FRACTION,
    FUEL_LAYER_MINIMUM_BINS,
    HELD_OUT_GENERA,
    NORMALIZED_F1_TOLERANCE,
    POINT_METRIC_NAMES,
    REFERENCE_POINTS,
    TREE_METRIC_NAMES,
    UPSAMPLING_INPUT_POINTS,
    VERTICAL_PROFILE_BINS,
    VISIBILITY_RESOLUTION,
)
from treebench.data.splits import validate_split


RELEASE_ROOT = Path(__file__).resolve().parents[1]


def _load_json(relative_path: str) -> dict:
    return json.loads((RELEASE_ROOT / relative_path).read_text())


def test_frozen_manifests_are_disjoint_and_have_protocol_counts() -> None:
    protocol = _load_json("configs/protocol.json")
    universes = []
    for split_name in ("random", "genus_ood"):
        manifest = _load_json(f"manifests/splits/{split_name}.json")
        expected = protocol["splits"][split_name]["counts"]
        report = validate_split(manifest, expected_total=expected["total"])
        assert report == {
            "total": expected["total"],
            "train": expected["train"],
            "val": expected["validation"],
            "test": expected["test"],
        }
        universes.append(
            set(manifest["train"]) | set(manifest["val"]) | set(manifest["test"])
        )
    assert universes[0] == universes[1]


def test_split_validator_rejects_duplicates_overlap_and_wrong_total() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate_split({"train": ["a", "a"], "val": [], "test": []})
    with pytest.raises(ValueError, match="disjoint"):
        validate_split({"train": ["a"], "val": ["a"], "test": []})
    with pytest.raises(ValueError, match="expected 4"):
        validate_split(
            {"train": ["a"], "val": ["b"], "test": ["c"]},
            expected_total=4,
        )


def test_protocol_json_matches_package_constants_and_metric_schema() -> None:
    protocol = _load_json("configs/protocol.json")
    metric_names = tuple(metric["name"] for metric in protocol["metrics"])

    assert metric_names == ALL_METRIC_NAMES
    assert len(metric_names) == 32
    assert protocol["metric_counts"] == {
        "point": len(POINT_METRIC_NAMES),
        "structural": 7,
        "canopy_fuel": 20,
        "tree_total": len(TREE_METRIC_NAMES),
        "total": len(ALL_METRIC_NAMES),
    }
    assert sum(metric["direction"] == "higher" for metric in protocol["metrics"]) == 3

    assert protocol["reference"]["points"] == REFERENCE_POINTS
    upsampling = protocol["tasks"]["upsampling"]["settings"]
    assert upsampling["4x"]["input_points"] == UPSAMPLING_INPUT_POINTS[4]
    assert upsampling["16x"]["input_points"] == UPSAMPLING_INPUT_POINTS[16]
    completion = protocol["tasks"]["completion"]
    assert completion["input_points"] == COMPLETION_INPUT_POINTS
    assert completion["visibility_resolution"] == [
        VISIBILITY_RESOLUTION,
        VISIBILITY_RESOLUTION,
    ]
    assert completion["evaluation"]["views_per_mode"] == COMPLETION_VIEWS_PER_MODE
    assert tuple(protocol["splits"]["genus_ood"]["held_out_genera"]) == HELD_OUT_GENERA

    parameters = protocol["metric_parameters"]
    assert parameters["f1_tolerance_normalized_euclidean"] == NORMALIZED_F1_TOLERANCE
    assert parameters["voxel_size_metres"] == FORESTRY_VOXEL_SIZE_METRES
    assert parameters["vertical_profile_bins"] == VERTICAL_PROFILE_BINS
    assert parameters["fuel_bin_size_metres"] == FUEL_BIN_SIZE_METRES
    assert parameters["fuel_layer_active_fraction"] == FUEL_LAYER_ACTIVE_FRACTION
    assert (
        parameters["fuel_layer_minimum_consecutive_bins"]
        == FUEL_LAYER_MINIMUM_BINS
    )
