"""Frozen completion-view construction with per-view provenance."""

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from treebench.constants import COMPLETION_VIEWS_PER_MODE
from treebench.data.preprocess import sha256_file
from treebench.data.visibility import generate_tls_view, generate_uls_view


def derive_seed(base_seed: int, *parts: str) -> int:
    payload = ":".join((str(base_seed), *map(str, parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def prepare_completion_tree(
    reference_raw: np.ndarray,
    output_dir: Path,
    *,
    metadata: dict,
    tree_id: str,
    base_seed: int,
    views_per_mode: int = COMPLETION_VIEWS_PER_MODE,
    tls_distance_fraction: Sequence[float] = (0.1, 0.1),
) -> dict:
    """Write one normalized target and four frozen views for each sensor mode.

    ``(0.1, 0.1)`` matches the historical frozen evaluation-view generator.
    Several training adapters instead augment TLS distance in ``[0.5, 1.0]``;
    this distinction is recorded in the baseline registry and data protocol.
    """
    reference_raw = np.asarray(reference_raw, dtype=np.float64)
    center = np.asarray(metadata["center"], dtype=np.float64)
    scale = float(metadata["scale"])
    if reference_raw.shape != (8_192, 3) or not np.isfinite(reference_raw).all():
        raise ValueError("reference_raw must be a finite (8192, 3) array")
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("metadata center must be a finite three-vector")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("metadata scale must be positive")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    target_path = output_dir / "8192_complete.npy"
    np.save(target_path, ((reference_raw - center) / scale).astype(np.float32))
    artifacts = {target_path.name: sha256_file(target_path)}
    views = []
    # Historical generation interleaved ULS then TLS for each view index.
    for view_index in range(views_per_mode):
        for mode in ("uls", "tls"):
            seed = derive_seed(base_seed, str(tree_id), mode, str(view_index))
            rng = np.random.default_rng(seed)
            if mode == "uls":
                partial, parameters = generate_uls_view(reference_raw, rng=rng)
            else:
                partial, parameters = generate_tls_view(
                    reference_raw,
                    rng=rng,
                    distance_fraction=tls_distance_fraction,
                )
            path = output_dir / f"2048_{mode}_{view_index}.npy"
            np.save(path, ((partial - center) / scale).astype(np.float32))
            digest = sha256_file(path)
            artifacts[path.name] = digest
            views.append(
                {
                    "tree_id": str(tree_id),
                    "mode": mode,
                    "view_index": view_index,
                    "seed": seed,
                    "parameters": parameters,
                    "artifact": path.name,
                    "sha256": digest,
                }
            )
    completion_metadata = {
        "tree_id": str(tree_id),
        "center": center.tolist(),
        "scale": scale,
        "base_seed": int(base_seed),
        "views_per_mode": int(views_per_mode),
        "tls_distance_fraction": list(map(float, tls_distance_fraction)),
        "artifacts_sha256": artifacts,
    }
    views_path = output_dir / "views.json"
    views_path.write_text(
        json.dumps(views, indent=2, sort_keys=True) + "\n"
    )
    artifacts[views_path.name] = sha256_file(views_path)
    (output_dir / "metadata.json").write_text(
        json.dumps(completion_metadata, indent=2, sort_keys=True) + "\n"
    )
    return {"metadata": completion_metadata, "views": views}
