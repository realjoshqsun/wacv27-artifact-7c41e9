"""TreeBench evaluation shared by PoinTr and AdaPoinTr."""

from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from pointnet2_ops import pointnet2_utils

from metrics.tree_metrics import TREE_METRIC_NAMES, compute_tree_metrics, denormalize_points
from utils.metrics import compute_metrics


FORESTRY_VOXEL_SIZE_METRES = 0.05
POINT_METRICS = ("cd_l2", "cd_l1", "hd", "dcd", "f1")


def _gather_rows(rows):
    if not dist.is_available() or not dist.is_initialized():
        return rows
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, rows)
    return [row for rank_rows in gathered for row in rank_rows]


def _fps(points, count):
    """FPS for the single PoinTr 16x two-stage zero-shot protocol."""
    indices = pointnet2_utils.furthest_point_sample(points.contiguous(), count)
    return pointnet2_utils.gather_operation(
        points.transpose(1, 2).contiguous(), indices).transpose(1, 2).contiguous()


def _predict(model, partial, protocol):
    """Run a TreeBench model under its declared evaluation protocol."""
    if protocol == "two_stage_4x_fps":
        # The 4x PoinTr decoder emits 6144 learned points plus its input.  On
        # 512-point 16x inputs, select 2048 first-stage points and apply the
        # *same* 4x checkpoint again, yielding 6144 + 2048 = 8192 points.
        first = model(partial)[-1]
        if first.shape[1] < 2048:
            raise RuntimeError(
                "PoinTr 16x two-stage protocol needs at least 2048 first-pass "
                f"points, got {first.shape[1]}.")
        return model(_fps(first, 2048))[-1]
    return model(partial)[-1]


@torch.no_grad()
def evaluate(model, loader, device, return_rows=False, treebench_16x_protocol=None,
             max_batches=None):
    """Evaluate each dataset index once and return unrounded sample means."""
    model.eval()
    rows = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        partial = batch["partial"].float().to(device, non_blocking=True)
        gt = batch["complete"].float().to(device, non_blocking=True)
        prediction = _predict(model, partial, treebench_16x_protocol)
        if prediction.shape != gt.shape or prediction.shape[1] != 8192:
            raise RuntimeError(
                "TreeBench requires an exact 8192-point prediction matching GT; got "
                f"{tuple(prediction.shape)} and {tuple(gt.shape)}"
            )

        cd_l1, cd_l2, f1, hd, dcd = compute_metrics(prediction, gt)
        point_values = dict(zip(POINT_METRICS, (cd_l2, cd_l1, hd, dcd, f1)))
        forestry = compute_tree_metrics(
            denormalize_points(prediction, batch["metadata"]),
            denormalize_points(gt, batch["metadata"]),
            voxel_size=FORESTRY_VOXEL_SIZE_METRES,
        )
        names = list(batch["name"])
        modes = list(batch["mode"])
        views = batch["view"].cpu().tolist()
        sample_keys = list(batch["sample_key"])
        for index, name in enumerate(names):
            row = OrderedDict(name=str(name))
            row["mode"] = str(modes[index])
            row["view"] = int(views[index])
            row["sample_key"] = str(sample_keys[index])
            row["input_points"] = int(partial.shape[1])
            row["output_points"] = int(prediction.shape[1])
            for metric, values in point_values.items():
                row[metric] = float(values[index].item())
            for metric, values in forestry.items():
                row[metric] = float(values[index].item())
            rows.append(row)

    rows = _gather_rows(rows)
    # DistributedSampler may pad eval shards. Remove only identical padded
    # records while preserving the eight distinct completion views.
    seen = set()
    unique = []
    for row in rows:
        signature = tuple(row.items())
        if signature not in seen:
            seen.add(signature)
            unique.append(row)
    if not unique:
        raise RuntimeError("TreeBench evaluation produced no samples")
    means = {
        key: float(np.mean([row[key] for row in unique]))
        for key in unique[0] if key != "name"
    }
    return (means, unique) if return_rows else means
