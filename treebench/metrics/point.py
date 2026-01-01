"""Portable implementations of TreeBench's five normalized point metrics.

The benchmark runs used compiled Chamfer-distance extensions. This module uses
chunked direct squared differences so evaluation does not depend on a specific
CUDA extension and exact coincident points retain exactly zero distance. It is
slower, but avoids materializing an 8,192 by 8,192 distance matrix.
"""

from typing import Dict, Tuple

import torch

from treebench.constants import NORMALIZED_F1_TOLERANCE, POINT_METRIC_NAMES


def _as_bnc(points: torch.Tensor) -> torch.Tensor:
    points = torch.as_tensor(points)
    if points.ndim == 2:
        points = points.unsqueeze(0)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape (N, 3) or (B, N, 3)")
    if points.shape[1] < 1:
        raise ValueError("point clouds cannot be empty")
    if not points.is_floating_point():
        points = points.float()
    return points


def _directed_nearest_squared(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return squared source-to-target nearest distances and target indices."""
    distances, indices = [], []
    for start in range(0, source.shape[1], chunk_size):
        chunk = source[:, start : start + chunk_size]
        best_distance = torch.full(
            chunk.shape[:2],
            float("inf"),
            dtype=chunk.dtype,
            device=chunk.device,
        )
        best_index = torch.zeros(
            chunk.shape[:2], dtype=torch.long, device=chunk.device
        )
        for target_start in range(0, target.shape[1], chunk_size):
            target_chunk = target[:, target_start : target_start + chunk_size]
            squared = (chunk[:, :, None, :] - target_chunk[:, None, :, :]).square().sum(-1)
            values, nearest = squared.min(dim=2)
            update = values < best_distance
            best_distance = torch.where(update, values, best_distance)
            best_index = torch.where(update, nearest + target_start, best_index)
        distances.append(best_distance)
        indices.append(best_index)
    return torch.cat(distances, dim=1), torch.cat(indices, dim=1)


@torch.no_grad()
def nearest_neighbor_distances(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    chunk_size: int = 1_024,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute both directed squared nearest-neighbor distance fields."""
    pred, gt = _as_bnc(pred), _as_bnc(gt)
    if pred.shape[0] != gt.shape[0]:
        raise ValueError("pred and gt batch sizes must match")
    if pred.device != gt.device:
        raise ValueError("pred and gt must be on the same device")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    pred_to_gt, pred_index = _directed_nearest_squared(
        pred, gt, chunk_size=chunk_size
    )
    gt_to_pred, gt_index = _directed_nearest_squared(
        gt, pred, chunk_size=chunk_size
    )
    return pred_to_gt, gt_to_pred, pred_index, gt_index


def _density_aware_chamfer(
    pred_to_gt: torch.Tensor,
    gt_to_pred: torch.Tensor,
    pred_index: torch.Tensor,
    gt_index: torch.Tensor,
    *,
    alpha: float = 1_000.0,
    density_power: float = 1.0,
) -> torch.Tensor:
    batch, n_pred = pred_to_gt.shape
    n_gt = gt_to_pred.shape[1]

    gt_assignment_count = torch.zeros(
        (batch, n_gt), dtype=pred_to_gt.dtype, device=pred_to_gt.device
    )
    gt_assignment_count.scatter_add_(
        1, pred_index, torch.ones_like(pred_to_gt)
    )
    pred_density = gt_assignment_count.gather(1, pred_index).detach()
    pred_weight = (pred_density.pow(density_power) + 1e-6).reciprocal()
    pred_weight = pred_weight * (n_gt / n_pred)
    loss_pred = (1.0 - torch.exp(-alpha * pred_to_gt) * pred_weight).mean(1)

    pred_assignment_count = torch.zeros(
        (batch, n_pred), dtype=gt_to_pred.dtype, device=gt_to_pred.device
    )
    pred_assignment_count.scatter_add_(
        1, gt_index, torch.ones_like(gt_to_pred)
    )
    gt_density = pred_assignment_count.gather(1, gt_index).detach()
    gt_weight = (gt_density.pow(density_power) + 1e-6).reciprocal()
    gt_weight = gt_weight * (n_pred / n_gt)
    loss_gt = (1.0 - torch.exp(-alpha * gt_to_pred) * gt_weight).mean(1)
    return 0.5 * (loss_pred + loss_gt)


@torch.no_grad()
def compute_point_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    f1_tolerance: float = NORMALIZED_F1_TOLERANCE,
    chunk_size: int = 1_024,
) -> Dict[str, torch.Tensor]:
    """Compute the five per-cloud point metrics in normalized coordinates."""
    pred_bnc, gt_bnc = _as_bnc(pred), _as_bnc(gt)
    invalid = ~torch.isfinite(pred_bnc).all(dim=(1, 2)) | ~torch.isfinite(gt_bnc).all(
        dim=(1, 2)
    )
    safe_pred = torch.where(torch.isfinite(pred_bnc), pred_bnc, torch.zeros_like(pred_bnc))
    safe_gt = torch.where(torch.isfinite(gt_bnc), gt_bnc, torch.zeros_like(gt_bnc))
    pred_to_gt, gt_to_pred, pred_index, gt_index = nearest_neighbor_distances(
        safe_pred, safe_gt, chunk_size=chunk_size
    )
    cd_l2 = pred_to_gt.mean(1) + gt_to_pred.mean(1)
    cd_l1 = 0.5 * (
        pred_to_gt.clamp_min(0).sqrt().mean(1)
        + gt_to_pred.clamp_min(0).sqrt().mean(1)
    )
    mhd2 = 0.5 * (pred_to_gt.amax(1) + gt_to_pred.amax(1))
    precision = (pred_to_gt < f1_tolerance**2).float().mean(1)
    recall = (gt_to_pred < f1_tolerance**2).float().mean(1)
    denominator = precision + recall
    f1 = torch.where(
        denominator > 0,
        2.0 * precision * recall / denominator.clamp_min(1e-12),
        torch.zeros_like(denominator),
    )
    dcd = _density_aware_chamfer(
        pred_to_gt, gt_to_pred, pred_index, gt_index
    )
    output = {
        "cd_l2": cd_l2,
        "cd_l1": cd_l1,
        "mhd2": mhd2,
        "dcd": dcd,
        "f1": f1,
    }
    if bool(invalid.any()):
        for name in output:
            output[name] = output[name].masked_fill(invalid, float("nan"))
    if tuple(output) != POINT_METRIC_NAMES:
        raise RuntimeError("point metric output order is inconsistent")
    return output
