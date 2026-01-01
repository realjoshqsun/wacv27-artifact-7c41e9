import json
from pathlib import Path

import numpy as np
import pytest

from treebench.constants import COMPLETION_INPUT_POINTS
from treebench.data.preprocess import DERIVED_SIZES, nested_subsets, prepare_tree
from treebench.data.visibility import generate_tls_view, generate_uls_view


def _row_set(points: np.ndarray) -> set[tuple[float, float, float]]:
    return {tuple(row) for row in np.asarray(points).tolist()}


def _source_cloud(count: int, seed: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 3))
    points[:, 2] = rng.uniform(0.0, 12.0, size=count)
    return points


def test_nested_subsets_are_exact_nested_and_deterministic() -> None:
    reference = np.arange(96, dtype=np.float64).reshape(32, 3)
    sizes = (32, 16, 8, 4)

    first = nested_subsets(
        reference, sizes=sizes, rng=np.random.default_rng(17)
    )
    second = nested_subsets(
        reference, sizes=sizes, rng=np.random.default_rng(17)
    )

    assert tuple(first) == sizes
    for size in sizes:
        assert first[size].shape == (size, 3)
        np.testing.assert_array_equal(first[size], second[size])
    for larger, smaller in zip(sizes, sizes[1:]):
        assert _row_set(first[smaller]) <= _row_set(first[larger])

    with pytest.raises(ValueError, match="descend"):
        nested_subsets(
            reference,
            sizes=(32, 16, 16),
            rng=np.random.default_rng(17),
        )


def test_prepare_tree_writes_reproducible_nested_artifacts(tmp_path: Path) -> None:
    source = _source_cloud(DERIVED_SIZES[0])
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = prepare_tree(source, first_dir, tree_id="fixture", seed=31)
    second = prepare_tree(source, second_dir, tree_id="fixture", seed=31)

    assert first == second
    assert tuple(first["nested_sizes"]) == DERIVED_SIZES
    for size in DERIVED_SIZES:
        first_raw = np.load(first_dir / f"{size}_gt.npy")
        second_raw = np.load(second_dir / f"{size}_gt.npy")
        first_normalized = np.load(first_dir / f"{size}_processed.npy")
        assert first_raw.shape == (size, 3)
        np.testing.assert_array_equal(first_raw, second_raw)
        np.testing.assert_allclose(
            first_normalized * first["scale"] + np.asarray(first["center"]),
            first_raw,
            rtol=1e-6,
            atol=2e-6,
        )

    for larger, smaller in zip(DERIVED_SIZES, DERIVED_SIZES[1:]):
        assert _row_set(np.load(first_dir / f"{smaller}_gt.npy")) <= _row_set(
            np.load(first_dir / f"{larger}_gt.npy")
        )
    stored_metadata = json.loads((first_dir / "metadata.json").read_text())
    assert stored_metadata == first


@pytest.mark.parametrize(
    ("generator", "kwargs", "expected_mode"),
    [
        (generate_tls_view, {"distance_fraction": (0.1, 0.1)}, "tls"),
        (generate_uls_view, {}, "uls"),
    ],
)
def test_completion_generator_cardinality_and_determinism(
    generator, kwargs: dict, expected_mode: str
) -> None:
    source = _source_cloud(512)

    first, first_parameters = generator(
        source,
        rng=np.random.default_rng(123),
        **kwargs,
    )
    second, second_parameters = generator(
        source,
        rng=np.random.default_rng(123),
        **kwargs,
    )

    assert first.shape == (COMPLETION_INPUT_POINTS, 3)
    np.testing.assert_array_equal(first, second)
    assert first_parameters == second_parameters
    assert first_parameters["mode"] == expected_mode
    assert _row_set(first) <= _row_set(source)
    assert 1 <= len(first_parameters["viewpoints"]) <= (
        1 if expected_mode == "tls" else 10
    )
    if expected_mode == "tls":
        assert first_parameters["distance_fraction"] == pytest.approx(0.1)

