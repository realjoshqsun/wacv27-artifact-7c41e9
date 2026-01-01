import json
from pathlib import Path

import numpy as np
import pytest
import torch

from treebench.constants import (
    COMPLETION_INPUT_POINTS,
    COMPLETION_VIEWS_PER_MODE,
    REFERENCE_POINTS,
)
from treebench.data.datasets import CompletionDataset, UpsamplingDataset, load_split


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value) + "\n")


def _write_fixture_arrays(root: Path, tree_id: str) -> None:
    directory = root / tree_id
    directory.mkdir(parents=True)
    complete = np.arange(REFERENCE_POINTS * 3, dtype=np.float32).reshape(
        REFERENCE_POINTS, 3
    )
    complete /= 10_000.0
    np.save(directory / f"{REFERENCE_POINTS}_processed.npy", complete)
    np.save(directory / "2048_processed.npy", complete[:2048])
    np.save(directory / "512_processed.npy", complete[:512])
    np.save(directory / f"{REFERENCE_POINTS}_complete.npy", complete)
    for mode_index, mode in enumerate(("uls", "tls")):
        for view_index in range(COMPLETION_VIEWS_PER_MODE):
            value = 1 + 10 * mode_index + view_index
            partial = np.full(
                (COMPLETION_INPUT_POINTS, 3), value, dtype=np.float32
            )
            np.save(
                directory / f"{COMPLETION_INPUT_POINTS}_{mode}_{view_index}.npy",
                partial,
            )
    _write_json(
        directory / "metadata.json",
        {"treeID": tree_id, "center": [1.0, 2.0, 3.0], "scale": 2.0},
    )


def test_split_loader_rejects_path_traversal_and_duplicates(tmp_path: Path) -> None:
    split = tmp_path / "unsafe.json"
    _write_json(split, {"train": ["../outside"], "val": [], "test": []})
    with pytest.raises(ValueError, match="unsafe tree identifier"):
        load_split(split, "train")

    _write_json(split, {"train": ["tree_1", "tree_1"], "val": [], "test": []})
    with pytest.raises(ValueError, match="duplicate tree identifier"):
        load_split(split, "train")


def test_upsampling_dataset_loads_4x_and_16x_fixture(tmp_path: Path) -> None:
    root = tmp_path / "data"
    tree_id = "mini_tree"
    _write_fixture_arrays(root, tree_id)
    split = tmp_path / "train.json"
    _write_json(split, {"train": [tree_id], "val": [], "test": []})

    four = UpsamplingDataset(root, split, "train", rate=4)
    sixteen = UpsamplingDataset(root, split, "train", rate=16)

    assert len(four) == len(sixteen) == 1
    assert four[0]["partial"].shape == (2048, 3)
    assert sixteen[0]["partial"].shape == (512, 3)
    assert four[0]["complete"].shape == (REFERENCE_POINTS, 3)
    assert four[0]["task"] == "upsampling_4x"
    assert sixteen[0]["task"] == "upsampling_16x"
    assert four[0]["name"] == tree_id
    assert four[0]["metadata"]["tree_id"] == tree_id


def test_frozen_completion_dataset_uses_uls_then_tls_order(tmp_path: Path) -> None:
    root = tmp_path / "data"
    tree_id = "mini_tree"
    _write_fixture_arrays(root, tree_id)
    split = tmp_path / "validation.json"
    _write_json(split, {"train": [], "val": [tree_id], "test": []})

    dataset = CompletionDataset(root, split, "val", frozen=True)

    assert len(dataset) == 2 * COMPLETION_VIEWS_PER_MODE
    expected = [
        ("uls", 0, 1.0),
        ("uls", 1, 2.0),
        ("uls", 2, 3.0),
        ("uls", 3, 4.0),
        ("tls", 0, 11.0),
        ("tls", 1, 12.0),
        ("tls", 2, 13.0),
        ("tls", 3, 14.0),
    ]
    for index, (mode, view_index, value) in enumerate(expected):
        sample = dataset[index]
        assert sample["mode"] == mode
        assert sample["view_index"] == view_index
        assert sample["name"] == tree_id
        assert sample["partial"].shape == (COMPLETION_INPUT_POINTS, 3)
        assert sample["complete"].shape == (REFERENCE_POINTS, 3)
        assert torch.all(sample["partial"] == value)


def test_training_completion_dataset_is_deterministic_per_epoch(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "data"
    tree_id = "mini_tree"
    _write_fixture_arrays(root, tree_id)
    split = tmp_path / "train.json"
    _write_json(split, {"train": [tree_id], "val": [], "test": []})

    def fake_view(raw, *, rng, **_kwargs):
        indices = rng.integers(0, len(raw), size=COMPLETION_INPUT_POINTS)
        return raw[indices], {"fixture": True}

    monkeypatch.setattr("treebench.data.datasets.generate_tls_view", fake_view)
    monkeypatch.setattr("treebench.data.datasets.generate_uls_view", fake_view)
    dataset = CompletionDataset(root, split, "train", frozen=False, base_seed=73)

    first = dataset[0]
    repeated = dataset[0]
    torch.testing.assert_close(first["partial"], repeated["partial"], rtol=0, atol=0)
    assert first["mode"] == repeated["mode"]
    assert first["view_index"] == -1

    dataset.set_epoch(1)
    next_epoch = dataset[0]
    assert not torch.equal(first["partial"], next_epoch["partial"])
