"""Framework-neutral-on-disk, PyTorch TreeBench dataset loaders."""

import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from treebench.constants import (
    COMPLETION_INPUT_POINTS,
    COMPLETION_VIEWS_PER_MODE,
    REFERENCE_POINTS,
    UPSAMPLING_INPUT_POINTS,
)
from treebench.data.completion import derive_seed
from treebench.data.visibility import generate_tls_view, generate_uls_view


SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def load_split(path: Path, subset: str) -> list[str]:
    manifest = json.loads(Path(path).read_text())
    if subset not in ("train", "val", "test"):
        raise ValueError("subset must be train, val, or test")
    if subset not in manifest or not isinstance(manifest[subset], list):
        raise ValueError(f"split manifest has no list named {subset!r}")
    identifiers = [str(identifier) for identifier in manifest[subset]]
    invalid = [identifier for identifier in identifiers if not SAFE_IDENTIFIER.fullmatch(identifier)]
    if invalid:
        raise ValueError(f"unsafe tree identifier in split: {invalid[0]!r}")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"duplicate tree identifier in split subset {subset!r}")
    return identifiers


def _load_metadata(path: Path) -> dict:
    metadata = json.loads(path.read_text())
    if "treeID" in metadata and "tree_id" not in metadata:
        metadata["tree_id"] = str(metadata.pop("treeID"))
    return metadata


class UpsamplingDataset(Dataset):
    """Load the 4x or evaluation-only 16x upsampling task."""

    def __init__(self, root: Path, split: Path, subset: str, *, rate: int):
        if rate not in UPSAMPLING_INPUT_POINTS:
            raise ValueError("TreeBench upsampling rate must be 4 or 16")
        self.root = Path(root)
        self.ids = load_split(split, subset)
        self.subset = subset
        self.rate = int(rate)
        self.input_points = UPSAMPLING_INPUT_POINTS[rate]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        tree_id = self.ids[index]
        directory = self.root / tree_id
        partial = np.load(directory / f"{self.input_points}_processed.npy")
        complete = np.load(directory / f"{REFERENCE_POINTS}_processed.npy")
        if partial.shape != (self.input_points, 3):
            raise ValueError(f"invalid partial shape for tree {tree_id}: {partial.shape}")
        if complete.shape != (REFERENCE_POINTS, 3):
            raise ValueError(f"invalid target shape for tree {tree_id}: {complete.shape}")
        return {
            "partial": torch.from_numpy(partial.astype(np.float32, copy=False)),
            "complete": torch.from_numpy(complete.astype(np.float32, copy=False)),
            "name": tree_id,
            "metadata": _load_metadata(directory / "metadata.json"),
            "task": f"upsampling_{self.rate}x",
        }


class CompletionDataset(Dataset):
    """Load frozen evaluation views or synthesize one training view per tree."""

    def __init__(
        self,
        root: Path,
        split: Path,
        subset: str,
        *,
        frozen: bool,
        base_seed: int = 21,
        tls_training_distance_fraction: Sequence[float] = (0.5, 1.0),
    ):
        if frozen and subset == "train":
            raise ValueError("frozen views are intended for validation and test")
        self.root = Path(root)
        self.ids = load_split(split, subset)
        self.subset = subset
        self.frozen = bool(frozen)
        self.base_seed = int(base_seed)
        self.epoch = 0
        self.tls_training_distance_fraction = tuple(
            map(float, tls_training_distance_fraction)
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        views_per_tree = 2 * COMPLETION_VIEWS_PER_MODE
        return len(self.ids) * (views_per_tree if self.frozen else 1)

    def __getitem__(self, index: int) -> dict:
        if self.frozen:
            tree_index, within_tree = divmod(
                index, 2 * COMPLETION_VIEWS_PER_MODE
            )
            tree_id = self.ids[tree_index]
            mode = "uls" if within_tree < COMPLETION_VIEWS_PER_MODE else "tls"
            view_index = within_tree % COMPLETION_VIEWS_PER_MODE
            directory = self.root / tree_id
            partial = np.load(
                directory / f"{COMPLETION_INPUT_POINTS}_{mode}_{view_index}.npy"
            )
            complete = np.load(directory / f"{REFERENCE_POINTS}_complete.npy")
            metadata = _load_metadata(directory / "metadata.json")
        else:
            tree_id = self.ids[index]
            directory = self.root / tree_id
            complete = np.load(directory / f"{REFERENCE_POINTS}_complete.npy")
            metadata = _load_metadata(directory / "metadata.json")
            center = np.asarray(metadata["center"])
            scale = float(metadata["scale"])
            raw = complete * scale + center
            seed = derive_seed(self.base_seed, str(self.epoch), tree_id)
            rng = np.random.default_rng(seed)
            if rng.random() < 0.5:
                mode = "tls"
                partial_raw, _ = generate_tls_view(
                    raw,
                    rng=rng,
                    distance_fraction=self.tls_training_distance_fraction,
                )
            else:
                mode = "uls"
                partial_raw, _ = generate_uls_view(raw, rng=rng)
            view_index = -1
            partial = (partial_raw - center) / scale
        if partial.shape != (COMPLETION_INPUT_POINTS, 3):
            raise ValueError(f"invalid partial shape for tree {tree_id}: {partial.shape}")
        if complete.shape != (REFERENCE_POINTS, 3):
            raise ValueError(f"invalid target shape for tree {tree_id}: {complete.shape}")
        return {
            "partial": torch.from_numpy(partial.astype(np.float32, copy=False)),
            "complete": torch.from_numpy(complete.astype(np.float32, copy=False)),
            "name": tree_id,
            "metadata": metadata,
            "task": "completion",
            "mode": mode,
            "view_index": view_index,
        }
