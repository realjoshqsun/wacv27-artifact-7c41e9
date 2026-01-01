"""Mesh-free TLS-like and ULS-like partial-observation simulation."""

from typing import Iterable, Literal, Optional, Sequence, Tuple

import numpy as np

from treebench.constants import COMPLETION_INPUT_POINTS, VISIBILITY_RESOLUTION

ProjectionMode = Literal["spherical", "orthographic"]


def tree_bounding_box(
    points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    points = _validate_points(points)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    return minimum, maximum, center, float(maximum[2] - minimum[2])


def _validate_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points must have non-empty shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    return points


def project_and_cull(
    points: np.ndarray,
    viewpoint: np.ndarray,
    *,
    resolution: int = VISIBILITY_RESOLUTION,
    mode: ProjectionMode = "spherical",
) -> np.ndarray:
    """Keep the nearest point in each projected 256 by 256 cell."""
    points = _validate_points(points)
    viewpoint = np.asarray(viewpoint, dtype=np.float64)
    if viewpoint.shape != (3,):
        raise ValueError("viewpoint must have shape (3,)")
    if resolution < 2:
        raise ValueError("resolution must be at least two")

    local = points - viewpoint
    distance = np.linalg.norm(local, axis=1)
    if mode == "spherical":
        u = np.arctan2(local[:, 1], local[:, 0])
        v = np.arccos(np.clip(local[:, 2] / (distance + 1e-8), -1.0, 1.0))
    elif mode == "orthographic":
        u, v = local[:, 0], local[:, 1]
    else:
        raise ValueError(f"unsupported projection mode: {mode}")

    u_min, u_max = u.min(), u.max()
    v_min, v_max = v.min(), v.max()
    if u_max - u_min < 1e-6 or v_max - v_min < 1e-6:
        return np.arange(len(points), dtype=np.int64)
    u_index = (
        (u - u_min) / (u_max - u_min + 1e-8) * (resolution - 1)
    ).astype(np.int32)
    v_index = (
        (v - v_min) / (v_max - v_min + 1e-8) * (resolution - 1)
    ).astype(np.int32)
    cell = u_index.astype(np.int64) * resolution + v_index.astype(np.int64)
    order = np.lexsort((distance, cell))
    first = np.concatenate(([True], cell[order][1:] != cell[order][:-1]))
    return order[first]


def clip_line_to_box(
    start: np.ndarray,
    direction: np.ndarray,
    minimum_xy: np.ndarray,
    maximum_xy: np.ndarray,
) -> Optional[Tuple[float, float]]:
    lower, upper = -1e9, 1e9
    for axis in range(2):
        if abs(direction[axis]) < 1e-6:
            if start[axis] < minimum_xy[axis] or start[axis] > maximum_xy[axis]:
                return None
            continue
        first = (minimum_xy[axis] - start[axis]) / direction[axis]
        second = (maximum_xy[axis] - start[axis]) / direction[axis]
        if first > second:
            first, second = second, first
        lower, upper = max(lower, first), min(upper, second)
    return None if lower > upper else (lower, upper)


def adaptive_visible_sample(
    points: np.ndarray,
    viewpoints: Iterable[np.ndarray],
    target_points: int,
    *,
    rng: np.random.Generator,
    mode: ProjectionMode,
    resolution: int = VISIBILITY_RESOLUTION,
) -> np.ndarray:
    """Union visible returns and sample exactly ``target_points`` points."""
    points = _validate_points(points)
    visible = set()
    for viewpoint in viewpoints:
        visible.update(
            project_and_cull(
                points, viewpoint, resolution=resolution, mode=mode
            ).tolist()
        )
    if not visible:
        raise RuntimeError("visibility projection produced no points")
    # Integer hashes are stable; preserving set iteration matches the original
    # generator more closely than sorting the union before random sampling.
    visible_indices = np.fromiter(visible, dtype=np.int64)
    visible_points = points[visible_indices]
    replace = len(visible_points) < target_points
    selected = rng.choice(len(visible_points), target_points, replace=replace)
    return visible_points[selected]


def generate_tls_view(
    points: np.ndarray,
    *,
    rng: np.random.Generator,
    target_points: int = COMPLETION_INPUT_POINTS,
    distance_fraction: Sequence[float] = (0.5, 1.0),
) -> Tuple[np.ndarray, dict]:
    """Generate one low-lateral terrestrial-laser-scanning-like view."""
    points = _validate_points(points)
    minimum, maximum, _, height = tree_bounding_box(points)
    center_xy = points[:, :2].mean(axis=0)
    low, high = map(float, distance_fraction)
    if low <= 0 or high < low:
        raise ValueError("distance_fraction must be a positive [low, high] pair")
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    distance = float(rng.uniform(low, high) * height)
    scanner_height = float(rng.uniform(1.5, 2.0))
    viewpoint = np.array(
        [
            center_xy[0] + distance * np.cos(angle),
            center_xy[1] + distance * np.sin(angle),
            minimum[2] + scanner_height,
        ]
    )
    partial = adaptive_visible_sample(
        points, [viewpoint], target_points, rng=rng, mode="spherical"
    )
    parameters = {
        "mode": "tls",
        "angle_radians": angle,
        "distance_fraction": distance / max(height, 1e-12),
        "scanner_height_above_tree_min_metres": scanner_height,
        "viewpoints": [viewpoint.tolist()],
        "projection": "spherical",
    }
    return partial, parameters


def generate_uls_view(
    points: np.ndarray,
    *,
    rng: np.random.Generator,
    target_points: int = COMPLETION_INPUT_POINTS,
) -> Tuple[np.ndarray, dict]:
    """Generate one overhead airborne-laser-scanning-like flight-line view."""
    points = _validate_points(points)
    minimum, maximum, center, _ = tree_bounding_box(points)
    flight_height = float(rng.uniform(60.0, 75.0))
    offset = rng.uniform(-0.5, 0.5, size=2)
    pivot = center[:2] + offset
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    direction = np.array([np.cos(angle), np.sin(angle)])
    bounds = clip_line_to_box(pivot, direction, minimum[:2], maximum[:2])
    z = minimum[2] + flight_height
    if bounds is None:
        viewpoints = [np.array([center[0], center[1], z])]
    else:
        positions = np.arange(bounds[0], bounds[1], 0.1)
        if len(positions) > 10:
            positions = positions[
                np.linspace(0, len(positions) - 1, 10).astype(int)
            ]
        viewpoints = [
            np.array(
                [pivot[0] + t * direction[0], pivot[1] + t * direction[1], z]
            )
            for t in positions
        ]
        if not viewpoints:
            viewpoints = [np.array([center[0], center[1], z])]
    partial = adaptive_visible_sample(
        points, viewpoints, target_points, rng=rng, mode="orthographic"
    )
    parameters = {
        "mode": "uls",
        "flight_height_above_tree_min_metres": flight_height,
        "offset_xy_metres": offset.tolist(),
        "flight_angle_radians": angle,
        "viewpoints": [viewpoint.tolist() for viewpoint in viewpoints],
        "projection": "orthographic",
    }
    return partial, parameters

