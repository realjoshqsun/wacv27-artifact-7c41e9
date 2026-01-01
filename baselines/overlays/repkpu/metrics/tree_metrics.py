"""Canonical TreeBench metrics for registered, single-tree XYZ point clouds.

Metric-space attributes require points restored with ``denormalize_points``.
Canopy-fuel outputs describe geometry and occupancy only; XYZ clouds cannot
identify physical fuel mass, moisture, or kg/m3 canopy bulk density.
"""

from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch


TensorLike = Union[torch.Tensor, np.ndarray]

# Evaluation entry points import this tuple so every baseline uses the same
# active metrics and output order.
TREE_METRIC_NAMES = (
    # Tree structural geometry.
    "tree_height_abs_error", "tree_height_relative_error",
    "crown_width_abs_error", "crown_width_relative_error",
    "crown_projection_area_abs_error", "crown_projection_area_relative_error",
    "crown_projection_iou", "occupied_volume_abs_error",
    "occupied_volume_relative_error", "canopy_occupancy_iou",
    "vertical_profile_l1",
    "crown_length_abs_error", "crown_length_relative_error",
    "relative_height_25_abs_error", "relative_height_50_abs_error",
    "relative_height_75_abs_error", "relative_height_95_abs_error",
    "foliage_height_diversity_abs_error",
    # Canopy-fuel structure.
    "crown_base_height_abs_error", "crown_base_height_relative_error",
    "fuel_stratum_profile_l1", "stratum_percentage_cover_l1",
    "fuel_layer_count_abs_error", "lowest_fuel_layer_base_height_abs_error",
    "dominant_fuel_layer_depth_abs_error",
    "mean_interlayer_distance_abs_error",
    "largest_vertical_gap_abs_error",
)


def denormalize_points(points: torch.Tensor, metadata: Optional[Mapping]) -> torch.Tensor:
    """Restore TreeBench coordinates with ``raw = normalized * scale + center``."""
    if metadata is None:
        return points
    if "scale" not in metadata or "center" not in metadata:
        raise KeyError("metadata must contain TreeBench keys 'scale' and 'center'")
    scale = torch.as_tensor(metadata["scale"], dtype=points.dtype, device=points.device)
    center_value = metadata["center"]
    if isinstance(center_value, (list, tuple)) and center_value and torch.is_tensor(center_value[0]):
        center = torch.stack(tuple(center_value), dim=-1).to(
            dtype=points.dtype, device=points.device)
    else:
        center = torch.as_tensor(center_value, dtype=points.dtype, device=points.device)
    if points.ndim == 2:
        points, squeeze = points.unsqueeze(0), True
    else:
        squeeze = False
    restored = points * scale.reshape(-1, 1, 1) + center.reshape(-1, 1, 3)
    return restored.squeeze(0) if squeeze else restored


def _as_bnc(points: TensorLike) -> torch.Tensor:
    points = torch.as_tensor(points, dtype=torch.float32)
    if points.ndim == 2:
        points = points.unsqueeze(0)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape (N, 3) or (B, N, 3)")
    if points.shape[1] < 2:
        raise ValueError("at least two points are required")
    return points


def _safe_relative_error(pred: torch.Tensor, gt: torch.Tensor, eps: float) -> torch.Tensor:
    return (pred - gt).abs() / gt.abs().clamp_min(eps)


def _voxel_keys(points: torch.Tensor, origin: torch.Tensor, voxel_size: float) -> torch.Tensor:
    coords = torch.floor((points - origin) / voxel_size).to(torch.int64)
    return coords[:, 0] * 7_000_001 + coords[:, 1] * 3_001 + coords[:, 2]


def _xy_keys(points: torch.Tensor, origin: torch.Tensor, voxel_size: float) -> torch.Tensor:
    coords = torch.floor((points[:, :2] - origin[:2]) / voxel_size).to(torch.int64)
    return coords[:, 0] * 7_000_001 + coords[:, 1]


def _set_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Exact set semantics, including on older PyTorch versions without isin.
    a, b = torch.unique(a), torch.unique(b)
    intersection = a.numel() + b.numel() - torch.unique(torch.cat((a, b))).numel()
    intersection = torch.as_tensor(intersection, dtype=torch.float32, device=a.device)
    union = a.numel() + b.numel() - intersection
    return intersection / max(float(union), 1.0)


def _crown_base_and_profile(
    points: torch.Tensor,
    z_min: torch.Tensor,
    z_max: torch.Tensor,
    vertical_bins: int,
    threshold: float,
    consecutive_bins: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate an operational crown base and normalized point-height profile."""
    height = (z_max - z_min).clamp_min(1e-8)
    z_bin = torch.floor((points[:, 2] - z_min) / height * vertical_bins).long()
    z_bin = z_bin.clamp(0, vertical_bins - 1)
    profile = torch.bincount(z_bin, minlength=vertical_bins).float()
    profile /= profile.sum().clamp_min(1.0)
    occupied = profile > threshold * profile.max()
    for start in range(vertical_bins - consecutive_bins + 1):
        if bool(occupied[start:start + consecutive_bins].all()):
            return z_min + height * start / vertical_bins, profile
    return z_min, profile


def _vertical_occupied_area_profile(
    points: torch.Tensor,
    z_min: torch.Tensor,
    z_max: torch.Tensor,
    origin: torch.Tensor,
    voxel_size: float,
    vertical_bins: int,
) -> torch.Tensor:
    """Distribution of horizontal occupied cells over relative-height strata."""
    height = (z_max - z_min).clamp_min(1e-8)
    z_bin = torch.floor((points[:, 2] - z_min) / height * vertical_bins).long()
    z_bin = z_bin.clamp(0, vertical_bins - 1)
    xy_keys = _xy_keys(points, origin, voxel_size)
    counts = [
        torch.unique(xy_keys[z_bin == index]).numel()
        for index in range(vertical_bins)
    ]
    profile = torch.as_tensor(counts, dtype=torch.float32, device=points.device)
    return profile / profile.sum().clamp_min(1.0)


def _height_quantile(points: torch.Tensor, z_min: torch.Tensor, q: float) -> torch.Tensor:
    """Linearly interpolated LiDAR relative-height percentile."""
    heights = (points[:, 2] - z_min).sort().values
    position = q * (heights.numel() - 1)
    lower = int(position)
    upper = min(lower + 1, heights.numel() - 1)
    weight = position - lower
    return heights[lower] * (1.0 - weight) + heights[upper] * weight


def _profile_entropy(profile: torch.Tensor, eps: float) -> torch.Tensor:
    """Shannon entropy (FHD proxy) of a point-height profile."""
    positive = profile > 0
    return -(profile[positive] * torch.log(profile[positive].clamp_min(eps))).sum()


def _stratum_cover_profile(
    points: torch.Tensor,
    *,
    xy_origin: torch.Tensor,
    z_origin: torch.Tensor,
    voxel_size: float,
    fuel_bin_size: float,
    number_of_bins: int,
    footprint_size: int,
) -> torch.Tensor:
    """Percentage of a shared XY crown footprint occupied in each metric bin."""
    xy_keys = _xy_keys(points, xy_origin, voxel_size)
    z_bins = torch.floor((points[:, 2] - z_origin) / fuel_bin_size).long()
    z_bins = z_bins.clamp(0, number_of_bins - 1)
    cover = [
        torch.unique(xy_keys[z_bins == index]).numel() / max(footprint_size, 1)
        for index in range(number_of_bins)
    ]
    return torch.as_tensor(cover, dtype=torch.float32, device=points.device)


def _fuel_layers(
    cover: torch.Tensor,
    *,
    active_fraction: float,
    minimum_bins: int,
) -> List[Tuple[int, int]]:
    """Extract consecutive occupancy-derived fuel layers."""
    if cover.numel() == 0 or float(cover.max()) <= 0:
        return []
    active = cover >= active_fraction * cover.max()
    layers: List[Tuple[int, int]] = []
    start = None
    for index in range(active.numel() + 1):
        is_active = index < active.numel() and bool(active[index])
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            if index - start >= minimum_bins:
                layers.append((start, index - 1))
            start = None
    return layers


def _fuel_layer_descriptors(
    cover: torch.Tensor,
    layers: List[Tuple[int, int]],
    *,
    z_origin: torch.Tensor,
    fuel_bin_size: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return count, lowest base, dominant depth, mean gap, and largest gap."""
    zero = cover.new_tensor(0.0)
    if not layers:
        return zero, z_origin, zero, zero, zero
    count = cover.new_tensor(float(len(layers)))
    lowest_base = z_origin + layers[0][0] * fuel_bin_size
    masses = torch.stack([cover[start:end + 1].sum() for start, end in layers])
    dominant_start, dominant_end = layers[int(masses.argmax())]
    dominant_depth = cover.new_tensor(
        (dominant_end - dominant_start + 1) * fuel_bin_size)
    if len(layers) < 2:
        return count, lowest_base, dominant_depth, zero, zero
    gaps = cover.new_tensor([
        max(0, next_start - end - 1) * fuel_bin_size
        for (_, end), (next_start, _) in zip(layers[:-1], layers[1:])
    ])
    return count, lowest_base, dominant_depth, gaps.mean(), gaps.max()


@torch.no_grad()
def compute_tree_metrics(
    pred: TensorLike,
    gt: TensorLike,
    *,
    voxel_size: float,
    vertical_bins: int = 20,
    crown_base_fraction: float = 0.10,
    crown_base_consecutive_bins: int = 2,
    fuel_bin_size: float = 0.25,
    fuel_layer_active_fraction: float = 0.10,
    fuel_layer_minimum_bins: int = 2,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Compute per-tree reconstruction metrics in canonical output order.

    Fuel layers use fixed metric-height bins, a common vertical origin, and a
    shared XY crown footprint.  Active runs must meet the configured fraction
    of peak stratum cover and minimum run length.
    """
    pred, gt = _as_bnc(pred), _as_bnc(gt)
    if pred.shape[0] != gt.shape[0]:
        raise ValueError("pred and gt batch sizes must match")
    if voxel_size <= 0 or fuel_bin_size <= 0:
        raise ValueError("voxel_size and fuel_bin_size must be positive")
    if vertical_bins < 2:
        raise ValueError("vertical_bins must be at least two")
    if not 0 < fuel_layer_active_fraction <= 1:
        raise ValueError("fuel_layer_active_fraction must be in (0, 1]")
    if fuel_layer_minimum_bins < 1:
        raise ValueError("fuel_layer_minimum_bins must be positive")

    output = {name: [] for name in TREE_METRIC_NAMES}
    for p, g in zip(pred, gt):
        origin = torch.cat((p, g), dim=0).min(dim=0).values - voxel_size
        p_min, p_max = p.min(dim=0).values, p.max(dim=0).values
        g_min, g_max = g.min(dim=0).values, g.max(dim=0).values
        p_height, g_height = p_max[2] - p_min[2], g_max[2] - g_min[2]
        p_width = (p_max[:2] - p_min[:2]).mean()
        g_width = (g_max[:2] - g_min[:2]).mean()

        p_xy, g_xy = _xy_keys(p, origin, voxel_size), _xy_keys(g, origin, voxel_size)
        p_area = torch.unique(p_xy).numel() * voxel_size ** 2
        g_area = torch.unique(g_xy).numel() * voxel_size ** 2
        p_xyz, g_xyz = _voxel_keys(p, origin, voxel_size), _voxel_keys(g, origin, voxel_size)
        p_volume = torch.unique(p_xyz).numel() * voxel_size ** 3
        g_volume = torch.unique(g_xyz).numel() * voxel_size ** 3

        p_cbh, p_profile = _crown_base_and_profile(
            p, p_min[2], p_max[2], vertical_bins,
            crown_base_fraction, crown_base_consecutive_bins)
        g_cbh, g_profile = _crown_base_and_profile(
            g, g_min[2], g_max[2], vertical_bins,
            crown_base_fraction, crown_base_consecutive_bins)
        p_fuel_profile = _vertical_occupied_area_profile(
            p, p_min[2], p_max[2], origin, voxel_size, vertical_bins)
        g_fuel_profile = _vertical_occupied_area_profile(
            g, g_min[2], g_max[2], origin, voxel_size, vertical_bins)
        p_crown_length, g_crown_length = p_max[2] - p_cbh, g_max[2] - g_cbh
        p_rh = [_height_quantile(p, p_min[2], q) for q in (0.25, 0.50, 0.75, 0.95)]
        g_rh = [_height_quantile(g, g_min[2], q) for q in (0.25, 0.50, 0.75, 0.95)]
        p_fhd, g_fhd = _profile_entropy(p_profile, eps), _profile_entropy(g_profile, eps)

        shared_z_origin = torch.minimum(p_min[2], g_min[2])
        shared_z_top = torch.maximum(p_max[2], g_max[2])
        # Include the bin containing the upper endpoint.  Without ``+ 1``, a
        # point exactly on the top bin boundary would be clamped downward and
        # could merge or erase the highest detected fuel layer.
        fuel_bins = max(
            1,
            int(torch.floor(
                (shared_z_top - shared_z_origin) / fuel_bin_size).item()) + 1,
        )
        shared_footprint = torch.unique(torch.cat((p_xy, g_xy))).numel()
        p_cover = _stratum_cover_profile(
            p, xy_origin=origin, z_origin=shared_z_origin,
            voxel_size=voxel_size, fuel_bin_size=fuel_bin_size,
            number_of_bins=fuel_bins, footprint_size=shared_footprint)
        g_cover = _stratum_cover_profile(
            g, xy_origin=origin, z_origin=shared_z_origin,
            voxel_size=voxel_size, fuel_bin_size=fuel_bin_size,
            number_of_bins=fuel_bins, footprint_size=shared_footprint)
        p_layers = _fuel_layers(
            p_cover, active_fraction=fuel_layer_active_fraction,
            minimum_bins=fuel_layer_minimum_bins)
        g_layers = _fuel_layers(
            g_cover, active_fraction=fuel_layer_active_fraction,
            minimum_bins=fuel_layer_minimum_bins)
        p_layer_values = _fuel_layer_descriptors(
            p_cover, p_layers, z_origin=shared_z_origin, fuel_bin_size=fuel_bin_size)
        g_layer_values = _fuel_layer_descriptors(
            g_cover, g_layers, z_origin=shared_z_origin, fuel_bin_size=fuel_bin_size)

        values = (
            (p_height - g_height).abs(), _safe_relative_error(p_height, g_height, eps),
            (p_width - g_width).abs(), _safe_relative_error(p_width, g_width, eps),
            p.new_tensor(abs(p_area - g_area)),
            p.new_tensor(abs(p_area - g_area) / max(abs(g_area), eps)),
            _set_iou(p_xy, g_xy),
            p.new_tensor(abs(p_volume - g_volume)),
            p.new_tensor(abs(p_volume - g_volume) / max(abs(g_volume), eps)),
            _set_iou(p_xyz, g_xyz),
            torch.abs(p_profile - g_profile).sum(),
            (p_crown_length - g_crown_length).abs(),
            _safe_relative_error(p_crown_length, g_crown_length, eps),
            *((pv - gv).abs() for pv, gv in zip(p_rh, g_rh)),
            (p_fhd - g_fhd).abs(),
            (p_cbh - g_cbh).abs(),
            torch.abs(
                (p_cbh - p_min[2]) / p_height.clamp_min(eps)
                - (g_cbh - g_min[2]) / g_height.clamp_min(eps)),
            torch.abs(p_fuel_profile - g_fuel_profile).sum(),
            torch.abs(p_cover - g_cover).mean(),
            *((pv - gv).abs() for pv, gv in zip(p_layer_values, g_layer_values)),
        )
        if len(values) != len(TREE_METRIC_NAMES):
            raise RuntimeError("TREE_METRIC_NAMES and metric values are out of sync")
        for key, value in zip(TREE_METRIC_NAMES, values):
            output[key].append(value)
    return {key: torch.stack(values) for key, values in output.items()}


@torch.no_grad()
def compute_tree_descriptors(
    points: TensorLike,
    *,
    voxel_size: float,
    vertical_bins: int = 20,
    crown_base_fraction: float = 0.10,
    crown_base_consecutive_bins: int = 2,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Return per-cloud attributes rather than prediction-reference errors."""
    points = _as_bnc(points)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    names = (
        "tree_height", "crown_width_x", "crown_width_y", "mean_crown_width",
        "crown_projection_area", "occupied_canopy_volume", "crown_base_height",
        "crown_base_height_relative", "crown_length", "crown_ratio",
        "vertical_point_profile", "vertical_occupied_area_profile",
    )
    output = {name: [] for name in names}
    for cloud in points:
        minimum, maximum = cloud.min(dim=0).values, cloud.max(dim=0).values
        height = maximum[2] - minimum[2]
        widths = maximum[:2] - minimum[:2]
        origin = minimum - voxel_size
        xy_keys = _xy_keys(cloud, origin, voxel_size)
        xyz_keys = _voxel_keys(cloud, origin, voxel_size)
        cbh, profile = _crown_base_and_profile(
            cloud, minimum[2], maximum[2], vertical_bins,
            crown_base_fraction, crown_base_consecutive_bins)
        fuel_profile = _vertical_occupied_area_profile(
            cloud, minimum[2], maximum[2], origin, voxel_size, vertical_bins)
        crown_length = maximum[2] - cbh
        values = (
            height, widths[0], widths[1], widths.mean(),
            cloud.new_tensor(torch.unique(xy_keys).numel() * voxel_size ** 2),
            cloud.new_tensor(torch.unique(xyz_keys).numel() * voxel_size ** 3),
            cbh, (cbh - minimum[2]) / height.clamp_min(eps), crown_length,
            # Kept as a descriptor.  Its error equals normalized CBH error.
            crown_length / height.clamp_min(eps),
            profile, fuel_profile,
        )
        for name, value in zip(names, values):
            output[name].append(value)
    return {name: torch.stack(values) for name, values in output.items()}
