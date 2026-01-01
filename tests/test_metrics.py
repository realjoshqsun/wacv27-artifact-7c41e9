import math

import numpy as np
import pytest
import torch

from treebench.constants import ALL_METRIC_NAMES, TREE_METRIC_NAMES
from treebench.metrics import (
    compute_all_metrics,
    compute_point_metrics,
    compute_tree_metrics,
    denormalize_points,
)


def _compact_tree() -> torch.Tensor:
    """Return a small non-degenerate cloud with a regular vertical profile."""
    xy = np.array(
        [[0.0, 0.0], [0.2, 0.0], [0.0, 0.2], [0.2, 0.2]],
        dtype=np.float32,
    )
    levels = [
        np.column_stack((xy, np.full(4, z, dtype=np.float32)))
        for z in np.linspace(0.0, 1.0, 5, dtype=np.float32)
    ]
    return torch.from_numpy(np.concatenate(levels, axis=0))


def test_all_32_metrics_have_the_frozen_schema_and_identity_values() -> None:
    cloud = _compact_tree()
    metrics = compute_all_metrics(
        cloud,
        cloud,
        metadata={"center": [10.0, 20.0, 30.0], "scale": 2.0},
        point_chunk_size=8,
    )

    assert tuple(metrics) == ALL_METRIC_NAMES
    assert len(metrics) == 32
    assert all(value.shape == (1,) for value in metrics.values())
    assert all(torch.isfinite(value).all() for value in metrics.values())

    assert metrics["f1"].item() == pytest.approx(1.0)
    assert metrics["crown_projection_iou"].item() == pytest.approx(1.0)
    assert metrics["canopy_occupancy_iou"].item() == pytest.approx(1.0)
    # Assignment-count epsilon makes identity DCD approximately 1e-6.
    assert 0.0 <= metrics["dcd"].item() < 2e-6

    exact_zero = set(ALL_METRIC_NAMES) - {
        "dcd",
        "f1",
        "crown_projection_iou",
        "canopy_occupancy_iou",
    }
    for name in exact_zero:
        assert metrics[name].item() == pytest.approx(0.0, abs=1e-7), name


def test_point_metrics_have_known_translation_values() -> None:
    reference = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    prediction = reference + torch.tensor([[[1.0, 0.0, 0.0]]])

    metrics = compute_point_metrics(prediction, reference, chunk_size=2)

    assert metrics["cd_l2"].item() == pytest.approx(2.0)
    assert metrics["cd_l1"].item() == pytest.approx(1.0)
    assert metrics["mhd2"].item() == pytest.approx(1.0)
    assert metrics["dcd"].item() == pytest.approx(1.0)
    assert metrics["f1"].item() == pytest.approx(0.0)


def test_f1_uses_strict_normalized_tolerance_and_nonfinite_is_visible() -> None:
    reference = torch.tensor([[[0.0, 0.0, 0.0]]])
    boundary = torch.tensor([[[0.01, 0.0, 0.0]]])
    inside = torch.tensor([[[0.009, 0.0, 0.0]]])
    assert compute_point_metrics(boundary, reference, chunk_size=1)["f1"].item() == 0.0
    assert compute_point_metrics(inside, reference, chunk_size=1)["f1"].item() == 1.0

    invalid = reference.clone()
    invalid[0, 0, 0] = float("nan")
    metrics = compute_point_metrics(invalid, reference, chunk_size=1)
    assert all(torch.isnan(value).all() for value in metrics.values())


def test_tree_metrics_have_known_vertical_translation_values() -> None:
    reference = _compact_tree()
    prediction = reference + torch.tensor([0.0, 0.0, 3.0])

    metrics = compute_tree_metrics(prediction, reference)

    translation_invariant = {
        "tree_height_abs_error",
        "tree_height_relative_error",
        "crown_width_abs_error",
        "crown_width_relative_error",
        "crown_projection_area_abs_error",
        "crown_projection_area_relative_error",
        "vertical_profile_l1",
        "crown_length_abs_error",
        "crown_length_relative_error",
        "relative_height_25_abs_error",
        "relative_height_50_abs_error",
        "relative_height_75_abs_error",
        "relative_height_95_abs_error",
        "foliage_height_diversity_abs_error",
        "crown_base_height_relative_error",
        "fuel_stratum_profile_l1",
        "fuel_layer_count_abs_error",
        "dominant_fuel_layer_depth_abs_error",
        "mean_interlayer_distance_abs_error",
        "largest_vertical_gap_abs_error",
        "occupied_volume_abs_error",
        "occupied_volume_relative_error",
    }
    for name in translation_invariant:
        assert metrics[name].item() == pytest.approx(0.0, abs=1e-6), name

    assert metrics["crown_projection_iou"].item() == pytest.approx(1.0)
    assert metrics["canopy_occupancy_iou"].item() == pytest.approx(0.0)
    assert metrics["crown_base_height_abs_error"].item() == pytest.approx(3.0)
    assert metrics["lowest_fuel_layer_base_height_abs_error"].item() == pytest.approx(
        3.0
    )


def test_tree_metric_nonfinite_contract_returns_all_nan() -> None:
    reference = _compact_tree()
    prediction = reference.clone()
    prediction[0, 0] = math.nan

    metrics = compute_tree_metrics(prediction, reference)

    assert tuple(metrics) == TREE_METRIC_NAMES
    assert all(torch.isnan(value).all() for value in metrics.values())


def test_denormalize_points_single_and_collated_metadata() -> None:
    single = torch.tensor([[0.0, 0.5, -1.0], [1.0, -1.0, 0.25]])
    restored = denormalize_points(
        single,
        {"center": [1.0, 2.0, 3.0], "scale": 2.0},
    )
    expected = single * 2.0 + torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(restored, expected)

    batched = torch.stack((single, single + 1.0))
    collated_metadata = {
        "center": [
            torch.tensor([1.0, 10.0]),
            torch.tensor([2.0, 20.0]),
            torch.tensor([3.0, 30.0]),
        ],
        "scale": torch.tensor([2.0, 3.0]),
    }
    restored_batch = denormalize_points(batched, collated_metadata)
    expected_batch = torch.stack(
        (
            batched[0] * 2.0 + torch.tensor([1.0, 2.0, 3.0]),
            batched[1] * 3.0 + torch.tensor([10.0, 20.0, 30.0]),
        )
    )
    torch.testing.assert_close(restored_batch, expected_batch)
