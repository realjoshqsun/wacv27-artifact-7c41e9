"""Deterministic TreeBench reference and nested-input construction."""

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from treebench.constants import REFERENCE_POINTS

DERIVED_SIZES = (8_192, 4_096, 2_048, 1_024, 512)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def farthest_point_sample_numpy(
    points: np.ndarray, count: int, *, rng: np.random.Generator
) -> np.ndarray:
    """Reference NumPy FPS with an explicitly seeded initial point."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < count:
        raise ValueError("FPS requires an (N, 3) array with N >= count")
    selected = np.empty(count, dtype=np.int64)
    nearest = np.full(len(points), np.inf)
    selected[0] = int(rng.integers(len(points)))
    for index in range(1, count):
        delta = points - points[selected[index - 1]]
        nearest = np.minimum(nearest, np.einsum("ij,ij->i", delta, delta))
        selected[index] = int(nearest.argmax())
    return points[selected]


def farthest_point_sample_open3d(points: np.ndarray, count: int) -> np.ndarray:
    """Use the Open3D operation used for the paper's 8,192-point references."""
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError(
            "Open3D is required for --fps-backend open3d; install the data extra"
        ) from error
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    sampled = np.asarray(cloud.farthest_point_down_sample(count).points)
    if len(sampled) != count:
        raise RuntimeError(f"Open3D FPS returned {len(sampled)} rather than {count}")
    return sampled


def normalize_tree(points: np.ndarray) -> tuple[np.ndarray, float]:
    center = np.asarray(points, dtype=np.float64).mean(axis=0)
    scale = float(np.linalg.norm(points - center, axis=1).max())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("tree normalization scale must be positive and finite")
    return center, scale


def nested_subsets(
    reference: np.ndarray,
    *,
    sizes: Sequence[int] = DERIVED_SIZES,
    rng: np.random.Generator,
) -> Dict[int, np.ndarray]:
    """Create the exact nested cardinality chain used by the data pipeline."""
    if sizes[0] != len(reference) or any(a <= b for a, b in zip(sizes, sizes[1:])):
        raise ValueError("sizes must descend from the reference cardinality")
    current = np.asarray(reference)
    output = {}
    for size in sizes:
        selected = rng.choice(len(current), size=size, replace=False)
        current = current[selected]
        output[int(size)] = current
    return output


def prepare_tree(
    points_raw: np.ndarray,
    output_dir: Path,
    *,
    tree_id: str,
    seed: int,
    fps_backend: str = "open3d",
    source_path: Optional[Path] = None,
) -> dict:
    """Create references, nested upsampling inputs, metadata, and checksums."""
    points_raw = np.asarray(points_raw, dtype=np.float64)
    if points_raw.ndim != 2 or points_raw.shape[1] != 3:
        raise ValueError("points_raw must have shape (N, 3)")
    if len(points_raw) < REFERENCE_POINTS:
        raise ValueError(f"tree {tree_id} has fewer than {REFERENCE_POINTS} points")
    if not np.isfinite(points_raw).all():
        raise ValueError("points_raw must be finite")
    rng = np.random.default_rng(seed)
    center, scale = normalize_tree(points_raw)
    if len(points_raw) == REFERENCE_POINTS:
        reference = points_raw.copy()
    elif fps_backend == "open3d":
        reference = farthest_point_sample_open3d(points_raw, REFERENCE_POINTS)
    elif fps_backend == "numpy":
        reference = farthest_point_sample_numpy(
            points_raw, REFERENCE_POINTS, rng=rng
        )
    else:
        raise ValueError("fps_backend must be 'open3d' or 'numpy'")
    subsets = nested_subsets(reference, rng=rng)

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for size, raw in subsets.items():
        raw_path = output_dir / f"{size}_gt.npy"
        normalized_path = output_dir / f"{size}_processed.npy"
        np.save(raw_path, raw.astype(np.float32))
        np.save(normalized_path, ((raw - center) / scale).astype(np.float32))
        artifacts[raw_path.name] = sha256_file(raw_path)
        artifacts[normalized_path.name] = sha256_file(normalized_path)

    metadata = {
        "tree_id": str(tree_id),
        "center": center.tolist(),
        "scale": scale,
        "source_point_count": int(len(points_raw)),
        "seed": int(seed),
        "fps_backend": fps_backend,
        "nested_sizes": list(map(int, DERIVED_SIZES)),
        "artifacts_sha256": artifacts,
    }
    if source_path is not None:
        metadata["source_filename"] = Path(source_path).name
        metadata["source_sha256"] = sha256_file(Path(source_path))
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def load_las_xyz(path: Path) -> np.ndarray:
    try:
        import laspy
    except ImportError as error:
        raise RuntimeError("laspy with a LAZ backend is required") from error
    cloud = laspy.read(path)
    return np.column_stack((cloud.x, cloud.y, cloud.z))


def discover_las_files(root: Path) -> Iterable[Path]:
    root = Path(root)
    yield from sorted((*root.rglob("*.laz"), *root.rglob("*.las")))
