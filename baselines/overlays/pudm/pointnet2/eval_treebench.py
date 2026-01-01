import os
import sys
import time
import json
import re
import argparse
import shutil

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import open3d
from tqdm import tqdm
from shutil import copyfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from util import (
    rescale, find_max_epoch, sampling, sampling_ddim,
    calc_diffusion_hyperparams, AverageMeter, set_seed,
    pc_normalization, numpy_to_pc
)
from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from dataset import get_dataloader
from json_reader import restore_string_to_list_in_a_dict
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist, hausdorff_distance
from metrics.tree_metrics import TREE_METRIC_NAMES, compute_tree_metrics, denormalize_points

FORESTRY_VOXEL_SIZE_METRES = 0.05


# -------------------------
# DDP utils
# -------------------------
def init_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def is_main_process():
    return dist.get_rank() == 0


def gather_list(local_list):
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_list)
    out = []
    for g in gathered:
        out.extend(g)
    return out


# -------------------------
# Metrics
# -------------------------
def calc_dcd(x, gt, alpha=1000, n_lambda=1):
    cd_module = chamfer_3DDist()
    _, n_x, _ = x.shape
    _, n_gt, _ = gt.shape

    frac_12 = n_x / n_gt
    frac_21 = n_gt / n_x

    dist1, dist2, idx1, idx2 = cd_module(x, gt)

    exp_dist1 = torch.exp(-dist1 * alpha)
    exp_dist2 = torch.exp(-dist2 * alpha)

    count1 = torch.zeros_like(idx2)
    count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
    weight1 = count1.gather(1, idx1.long()).float().detach() ** n_lambda
    weight1 = (weight1 + 1e-6) ** (-1) * frac_21
    loss1 = (1 - exp_dist1 * weight1).mean(dim=1)

    count2 = torch.zeros_like(idx1)
    count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
    weight2 = count2.gather(1, idx2.long()).float().detach() ** n_lambda
    weight2 = (weight2 + 1e-6) ** (-1) * frac_12
    loss2 = (1 - exp_dist2 * weight2).mean(dim=1)

    return (loss1 + loss2) / 2


def fscore(dist1, dist2, threshold=1e-4):
    p1 = torch.mean((dist1 < threshold).float(), dim=1)
    p2 = torch.mean((dist2 < threshold).float(), dim=1)
    f = 2 * p1 * p2 / (p1 + p2)
    f[torch.isnan(f)] = 0
    return f, p1, p2


def validate_treebench_cardinality(condition, target, R):
    """Fail before sampling when a batch cannot represent the requested rate."""
    if condition.ndim != 3 or target.ndim != 3:
        raise RuntimeError(
            "PUDM expects [B,N,3] input and target tensors; got {} and {}"
            .format(tuple(condition.shape), tuple(target.shape)))
    if condition.shape[0] != target.shape[0] or condition.shape[2] != 3 or target.shape[2] != 3:
        raise RuntimeError(
            "PUDM input/target batch or coordinate dimensions are invalid: "
            "{} and {}".format(tuple(condition.shape), tuple(target.shape)))
    runtime_rate = int(R)
    expected_points = int(condition.shape[1]) * runtime_rate
    if expected_points != int(target.shape[1]):
        raise RuntimeError(
            "PUDM runtime cardinality mismatch: input={}, target={}, R={}, "
            "input*R={}".format(
                condition.shape[1], target.shape[1], runtime_rate,
                expected_points))
    if runtime_rate == 16 and (int(condition.shape[1]), int(target.shape[1])) != (512, 8192):
        raise RuntimeError(
            "TreeBench 16x protocol requires exactly 512->8192 points; got "
            "{}->{}".format(condition.shape[1], target.shape[1]))
    return expected_points


def validate_treebench_prediction(generated, target, expected_points):
    """Require a dense prediction with the exact requested cardinality."""
    if generated.shape != target.shape or int(generated.shape[1]) != int(expected_points):
        raise RuntimeError(
            "PUDM prediction cardinality mismatch: prediction={}, target={}, "
            "expected_points={}".format(
                tuple(generated.shape), tuple(target.shape), expected_points))


# -------------------------
# Evaluate
# -------------------------
@torch.no_grad()
def evaluate(
    net,
    testloader,
    diffusion_hyperparams,
    scale,
    R,
    gamma,
    T,
    step,
    save_dir,
    save_xyz=False,
    return_names=False,
    max_batches=None,
    conditioning_R=None,
):
    cd_meter = AverageMeter()
    hd_meter = AverageMeter()

    metrics = {
        'mode': [],
        'view': [],
        'sample_key': [],
        'input_points': [],
        'output_points': [],
        'cd_l2': [],
        'cd_l1': [],
        'hd': [],
        'dcd': [],
        'f1': []
    }
    forestry_names = TREE_METRIC_NAMES
    metrics.update({name: [] for name in forestry_names})

    total_name_list = []
    total_meta = []

    cd_module = chamfer_3DDist()

    loader = testloader
    if is_main_process():
        loader = tqdm(testloader, total=len(testloader))

    for batch_index, data in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        label = data['label'].cuda(non_blocking=True)
        condition = data['partial'].cuda(non_blocking=True)
        gt = data['complete'].cuda(non_blocking=True)
        names = data['name']

        batch, num_points, _ = gt.shape
        expected_points = validate_treebench_cardinality(condition, gt, R)
        if conditioning_R is not None:
            conditioning_label = int(conditioning_R) - 1
            if conditioning_label < 0:
                raise RuntimeError(
                    "PUDM checkpoint conditioning rate must be positive; got "
                    f"{conditioning_R}")
            # Cross-rate inference changes the interpolation/output rate only.
            # Keep the learned 4x checkpoint condition (label 3), matching the
            # official PUDM arbitrary-rate evaluation path.
            label = torch.full_like(label, conditioning_label)
        net.module.reset_cond_features()

        if step < T:
            generated, cond_pre, z = sampling_ddim(
                net=net,
                size=(batch, num_points, 3),
                diffusion_hyperparams=diffusion_hyperparams,
                label=label,
                condition=condition,
                R=R,
                gamma=gamma,
                step=step
            )
        else:
            generated, cond_pre, z = sampling(
                net=net,
                size=(batch, num_points, 3),
                diffusion_hyperparams=diffusion_hyperparams,
                label=label,
                condition=condition,
                R=R,
                gamma=gamma
            )

        generated = generated / scale
        gt = gt / scale
        validate_treebench_prediction(generated, gt, expected_points)

        dist1, dist2, _, _ = cd_module(generated, gt)
        cd_l2 = (dist1.mean(1) + dist2.mean(1))
        cd_l1 = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2

        f1, _, _ = fscore(dist1, dist2)
        hd = hausdorff_distance(generated, gt)
        dcd = calc_dcd(generated, gt)

        cd_meter.update(cd_l2.mean().item(), batch)
        hd_meter.update(hd.mean().item(), batch)

        metrics['cd_l2'] += cd_l2.cpu().tolist()
        metrics['cd_l1'] += cd_l1.cpu().tolist()
        metrics['f1'] += f1.cpu().tolist()
        metrics['hd'] += hd.mean(dim=1).cpu().tolist()
        metrics['dcd'] += dcd.cpu().tolist()
        metrics['mode'] += list(data['mode'])
        metrics['view'] += data['view'].cpu().tolist()
        metrics['sample_key'] += list(data['sample_key'])
        metrics['input_points'] += [int(condition.shape[1])] * len(names)
        metrics['output_points'] += [int(generated.shape[1])] * len(names)
        forestry = compute_tree_metrics(
            denormalize_points(generated, data["metadata"]),
            denormalize_points(gt, data["metadata"]),
            voxel_size=FORESTRY_VOXEL_SIZE_METRES,
        )
        for name, values in forestry.items():
            metrics[name] += values.cpu().tolist()

        total_name_list.extend(names)
        total_meta.extend(label.cpu().tolist())

        if save_xyz and is_main_process():
            gen_np = generated.cpu().numpy()
            for i, name in enumerate(names):
                pc = numpy_to_pc(gen_np[i])
                open3d.io.write_point_cloud(
                    os.path.join(save_dir, f"{name}.xyz"), pc
                )

    # reduce meters
    cd_sum = torch.tensor(cd_meter.sum, device='cuda')
    cd_cnt = torch.tensor(cd_meter.count, device='cuda')
    hd_sum = torch.tensor(hd_meter.sum, device='cuda')
    hd_cnt = torch.tensor(hd_meter.count, device='cuda')

    dist.all_reduce(cd_sum)
    dist.all_reduce(cd_cnt)
    dist.all_reduce(hd_sum)
    dist.all_reduce(hd_cnt)

    cd_avg = (cd_sum / cd_cnt).item()
    hd_avg = (hd_sum / hd_cnt).item()

    for k in metrics:
        metrics[k] = gather_list(metrics[k])
    total_name_list = gather_list(total_name_list)
    total_meta = gather_list(total_meta)

    if return_names:
        return cd_avg, hd_avg, metrics, total_name_list
    return cd_avg, hd_avg, metrics


# -------------------------
# Main logic
# -------------------------
def run_eval(
    checkpoint_path,
    phase,
    pointnet_config,
    dataset_config,
    diffusion_config,
    diffusion_hyperparams,
    batch_size,
    gamma,
    R,
    step,
    save_dir
):
    net = PointNet2CloudCondition(pointnet_config).cuda()
    net = DDP(net, device_ids=[torch.cuda.current_device()])

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    net.module.load_state_dict(ckpt['model_state_dict'], strict=False)

    dataset_config['batch_size'] = batch_size
    dataset_config['eval_batch_size'] = batch_size

    loader = get_dataloader(dataset_config, phase=phase, distributed=True)

    return evaluate(
        net,
        loader,
        diffusion_hyperparams,
        scale=dataset_config['scale'],
        R=R,
        gamma=gamma,
        T=diffusion_config['T'],
        step=step,
        save_dir=save_dir,
    )


# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    set_seed(42)
    local_rank = init_distributed()

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default="TreeBench10K")
    parser.add_argument('--step', type=int, default=30)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    config_path = f"./exp_configs/{args.dataset}.json"
    with open(config_path) as f:
        config = restore_string_to_list_in_a_dict(json.load(f))

    pointnet_config = config["pointnet_config"]
    diffusion_config = config["diffusion_config"]
    dataset_config = config["dataset_config"]
    train_config = config["train_config"]

    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)
    for k in diffusion_hyperparams:
        if k != "T":
            diffusion_hyperparams[k] = diffusion_hyperparams[k].cuda()

    ckpt_dir = os.path.join(
        train_config['root_directory'],
        train_config['dataset'],
        train_config['output_directory']
    )

    if is_main_process():
        ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pkl')])
        ckpts.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    else:
        ckpts = None
        
    obj_list = [ckpts]   
    dist.broadcast_object_list(obj_list, src=0)
    ckpts = obj_list[0]

    best_cd = 1e9
    best_ckpt = None
    best_metrics = None

    if is_main_process():
        print(f"Found {len(ckpts)} checkpoints to evaluate.")
    for ckpt in ckpts:
        if is_main_process():
            print('----------------------------------------')
            print(f"Evaluating checkpoint: {ckpt}")
        cd, _, metrics = run_eval(
            os.path.join(ckpt_dir, ckpt),
            'val',
            pointnet_config,
            dataset_config,
            diffusion_config,
            diffusion_hyperparams,
            dataset_config['batch_size'],
            pointnet_config['gamma'],
            dataset_config['R'],
            args.step,
            save_dir=''
        )
        if is_main_process():
            print(f"CD-l2 * 1e3: {cd * 1e3:.4f}, CD-l1 * 1e3: {np.mean(metrics['cd_l1']) * 1e3:.4f}, DCD: {np.mean(metrics['dcd']):.6f}, HD * 1e3: {np.mean(metrics['hd']) * 1e3:.4f}, F1: {np.mean(metrics['f1']):.4f}")

        if cd < best_cd:
            best_cd = cd
            best_ckpt = ckpt
            best_metrics = metrics
        if is_main_process():
            print(f"Current best CD-l2 * 1e3: {best_cd * 1e3:.4f} at {best_ckpt}")

    if is_main_process():
        print('========================================')
        print("Best checkpoint:", best_ckpt)
        print('Val metrics at best checkpoint:')
        print("Avg CD-l2 * 1e3:", np.mean(best_metrics['cd_l2']) * 1e3)
        print("Avg CD-l1 * 1e3:", np.mean(best_metrics['cd_l1']) * 1e3)
        print("Avg DCD:", np.mean(best_metrics['dcd']))
        print("Avg HD * 1e3:", np.mean(best_metrics['hd']) * 1e3)
        print("Avg F1:", np.mean(best_metrics['f1']))
        print('----------------------------------------')

    dist.barrier()

    cd, hd, metrics = run_eval(
        os.path.join(ckpt_dir, best_ckpt),
        'test',
        pointnet_config,
        dataset_config,
        diffusion_config,
        diffusion_hyperparams,
        dataset_config['batch_size'],
        pointnet_config['gamma'],
        dataset_config['R'],
        args.step,
        save_dir=f"./test/{args.dataset.lower()}"
    )
    if is_main_process():
        print('========================================')
        print('Test metrics at best checkpoint:')
        print("Avg CD-l2 * 1e3:", np.mean(metrics['cd_l2']) * 1e3)
        print("Avg CD-l1 * 1e3:", np.mean(metrics['cd_l1']) * 1e3)
        print("Avg DCD:", np.mean(metrics['dcd']))
        print("Avg HD * 1e3:", np.mean(metrics['hd']) * 1e3)
        print("Avg F1:", np.mean(metrics['f1']))
        print('========================================')
