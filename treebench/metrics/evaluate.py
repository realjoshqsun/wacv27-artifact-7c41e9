"""Composition of normalized point metrics and metric-space tree metrics."""

from typing import Dict, Mapping, Optional

import torch

from treebench.constants import ALL_METRIC_NAMES
from treebench.metrics.point import compute_point_metrics
from treebench.metrics.tree import compute_tree_metrics, denormalize_points


@torch.no_grad()
def compute_all_metrics(
    pred_normalized: torch.Tensor,
    gt_normalized: torch.Tensor,
    *,
    metadata: Optional[Mapping] = None,
    pred_metric: Optional[torch.Tensor] = None,
    gt_metric: Optional[torch.Tensor] = None,
    point_chunk_size: int = 1_024,
) -> Dict[str, torch.Tensor]:
    """Compute the canonical 32 metrics.

    Pass either collated ``metadata`` containing center/scale or both clouds in
    metric coordinates.  Point metrics always use the normalized arguments.
    """
    point = compute_point_metrics(
        pred_normalized, gt_normalized, chunk_size=point_chunk_size
    )
    if (pred_metric is None) != (gt_metric is None):
        raise ValueError("pred_metric and gt_metric must be supplied together")
    if pred_metric is None:
        if metadata is None:
            raise ValueError("metadata or metric-coordinate clouds are required")
        pred_metric = denormalize_points(pred_normalized, metadata)
        gt_metric = denormalize_points(gt_normalized, metadata)
    tree = compute_tree_metrics(pred_metric, gt_metric)
    output = {**point, **tree}
    if tuple(output) != ALL_METRIC_NAMES:
        raise RuntimeError("32-metric output order is inconsistent")
    return output

