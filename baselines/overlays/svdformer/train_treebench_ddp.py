import os
import torch
import sys
import argparse
import torch.distributed as dist
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import broadcast_object_list
from torch.utils.data.distributed import DistributedSampler
from config_pcn import cfg as svd_cfg
from models.SVDFormer import Model as SVDFormer
from models.model_utils import PCViews
from cfgs.upsampling import parse_TreeBench_args
from cfgs.utils import *
from metrics.CD.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from utils.loss_utils import get_loss
from dataset.TreeBench import TreeBench10K_completion, TreeBench10K_upsampling
from metrics.tree_metrics import TREE_METRIC_NAMES, compute_tree_metrics, denormalize_points
import torch.optim as optim
from einops import rearrange
import time
import psutil
from datetime import datetime
import copy
from tqdm import tqdm

import random
import numpy as np
import csv
from tensorboardX import SummaryWriter

chamfer_dist = chamfer_3DDist()


def get_logger(name, log_dir):
    logger = logging.getLogger("{}:{}".format(name, log_dir))
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(os.path.join(log_dir, "{}.log".format(name)))
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s::%(name)s::%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler())
    return logger


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_distributed():
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

def gather_list(local_list):
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_list)
    out = []
    for g in gathered:
        out.extend(g)
    return out

def save_checkpoint(model, optimizer, scheduler, epoch, filename="checkpoint.pth", best_cd=None):
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_cd': best_cd
    }
    torch.save(state, filename)

# def upsampling(args, model, input_pcd):
#     pcd_pts_num = input_pcd.shape[-1]
#     patch_pts_num = args.num_points
#     sample_num = int(pcd_pts_num / patch_pts_num * args.patch_rate)
#     seed = FPS(input_pcd, sample_num)
#     patches = extract_knn_patch(patch_pts_num, input_pcd, seed)
#     patches, centroid, furthest_distance = normalize_point_cloud(patches)
#     coarse_pts, _= model.forward(patches)
#     coarse_pts = coarse_pts
#     coarse_pts = centroid + coarse_pts * furthest_distance
#     coarse_pts = rearrange(coarse_pts, 'b c n -> c (b n)').contiguous()
#     coarse_pts = FPS(coarse_pts.unsqueeze(0), input_pcd.shape[-1]* args.up_rate)
#     return coarse_pts

def upsampling(args, model, input_pcd):
    configured_input_points = int(args.dataset_config['npoints'])
    actual_input_points = input_pcd.shape[-1]
    if actual_input_points != configured_input_points:
        raise RuntimeError(
            'TreeBench input cardinality does not match the selected configuration: '
            f'got {actual_input_points} points, expected {configured_input_points}.')

    inference_model = model.module if isinstance(model, DDP) else model
    partial = input_pcd.transpose(1, 2).contiguous()
    render = PCViews(TRANS=-svd_cfg.NETWORK.view_distance, RESOLUTION=224)
    depth = render.get_img(partial).unsqueeze(1)
    predictions = inference_model(partial, depth)
    dense = predictions[-1]
    expected_output_points = configured_input_points * int(args.up_rate)
    if dense.ndim != 3 or dense.shape[1:] != (expected_output_points, 3):
        raise RuntimeError(
            'SVDFormer output cardinality does not match the requested cross-rate: '
            f'got {tuple(dense.shape)}, expected '
            f'(batch, {expected_output_points}, 3) for '
            f'{configured_input_points}x{int(args.up_rate)}.')
    return dense.transpose(1, 2).contiguous()


def calc_dcd(x, gt, alpha=1000, n_lambda=1):
    _, n_x, _ = x.shape
    _, n_gt, _ = gt.shape

    frac_12 = n_x / n_gt
    frac_21 = n_gt / n_x

    dist1, dist2, idx1, idx2 = chamfer_dist(x, gt)

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


def hausdorff_distance(X,Y):
    B,N,C = X.shape
    dist1, dist2 ,_ ,_ = chamfer_dist(X, Y)
    h1 = torch.amax(dist1, dim=1).view(B,1)
    h2 = torch.amax(dist2, dim=1).view(B,1)
    hd_loss = torch.cat([h1,h2],dim=-1)
    return hd_loss


def compute_metrics(generated, gt):
    dist1, dist2, _, _ = chamfer_dist(generated, gt)
    cd_l2 = (dist1.mean(1) + dist2.mean(1))
    cd_l1 = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2

    f1, _, _ = fscore(dist1, dist2)
    hd = hausdorff_distance(generated, gt)
    hd = hd.mean(dim=1)
    dcd = calc_dcd(generated, gt)
    
    return cd_l1, cd_l2, f1, hd, dcd


FORESTRY_VOXEL_SIZE_METRES = 0.05


def val(model, args, val_loader, logger=None, local_rank=0, return_names=False, max_batches=None):

    model.eval()
    forestry_metric_names = TREE_METRIC_NAMES
    if return_names: 
        metrics = {
            'name': [],
            'mode': [],
            'view': [],
            'sample_key': [],
            'input_points': [],
            'output_points': [],
            'cd_l2': [],
            'cd_l1': [],
            'hd': [],
            'dcd': [],
            'f1': [],
        }
    else:
        metrics = {
            'cd_l2': [],
            'cd_l1': [],
            'hd': [],
            'dcd': [],
            'f1': [],
        }
    metrics.update({name: [] for name in forestry_metric_names})

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation", disable=(local_rank != 0)):
            if max_batches is not None and idx >= max_batches:
                break
            input_pcd, gt_pcd = batch['partial'], batch['complete']

            input_pcd = rearrange(input_pcd, 'b n c -> b c n').contiguous().float().cuda(local_rank, non_blocking=True)
            gt_pcd = gt_pcd.float().cuda(local_rank, non_blocking=True) 
            pcd_upsampled = upsampling(args, model, input_pcd)
            pcd_upsampled = rearrange(pcd_upsampled, 'b c n -> b n c').contiguous()
            if pcd_upsampled.shape != gt_pcd.shape or pcd_upsampled.shape[1] != 8192:
                raise RuntimeError(
                    'TreeBench evaluation requires an exact 8192-point prediction; '
                    f'got {tuple(pcd_upsampled.shape)} for GT {tuple(gt_pcd.shape)}.')
            
            cd_l1, cd_l2, f1, hd, dcd = compute_metrics(pcd_upsampled, gt_pcd)
            # TreeBench stores normalized clouds.  Restoring each sample's
            # metadata makes height/area/volume errors physically meaningful.
            forestry = compute_tree_metrics(
                denormalize_points(pcd_upsampled, batch['metadata']),
                denormalize_points(gt_pcd, batch['metadata']),
                voxel_size=FORESTRY_VOXEL_SIZE_METRES,
            )
            if return_names:
                metrics['name'] += batch['name']
                metrics['mode'] += batch['mode']
                metrics['view'] += batch['view'].cpu().tolist()
                metrics['sample_key'] += batch['sample_key']
                metrics['input_points'] += [int(input_pcd.shape[-1])] * len(batch['name'])
                metrics['output_points'] += [int(pcd_upsampled.shape[1])] * len(batch['name'])
            metrics['cd_l2'] += cd_l2.cpu().tolist()
            metrics['cd_l1'] += cd_l1.cpu().tolist()
            metrics['f1'] += f1.cpu().tolist()
            metrics['hd'] += hd.cpu().tolist()
            metrics['dcd'] += dcd.cpu().tolist()
            for name, values in forestry.items():
                metrics[name] += values.cpu().tolist()
    if dist.is_initialized():
        for k in metrics:
            metrics[k] = gather_list(metrics[k])

    return metrics



def train(model, args, local_rank, max_train_batches=None, max_val_batches=None):
    set_seed(args.seed)

    before = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    if local_rank == 0:
        print(f"Memory before load: {before:.2f} MB")

    if 'completion' in args.dataset_config['data_dir']:
        train_dataset = TreeBench10K_completion(args.dataset_config, train=True, phase='train')
        val_dataset = TreeBench10K_completion(args.dataset_config, train=False, phase='val')
    else: # upsampling
        assert 'upsampling' in args.dataset_config['data_dir']
        train_dataset = TreeBench10K_upsampling(args.dataset_config, train=True, phase='train')
        val_dataset = TreeBench10K_upsampling(args.dataset_config, train=False, phase='val')
        

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler
    )

    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=int(args.batch_size * 1.5),
        num_workers=8,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False
    )

    if local_rank == 0:
        str_time = datetime.now().isoformat()
    else:
        str_time = None

    obj = [str_time]
    dist.broadcast_object_list(obj, src=0)
    str_time = obj[0]
    output_dir = os.path.join(args.out_path, str_time)
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if local_rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        log_dir = os.path.join(output_dir, 'log')
        os.makedirs(log_dir, exist_ok=True)
    dist.barrier()
    
    if local_rank == 0:
        logger = get_logger('train', log_dir)
        logger.propagate = False
        logger.info('Experiment ID: %s' % (str_time))
        # ========= TensorBoard =========
        input_size = args.dataset_config['npoints']
        exp_name = f"lr{args.lr}_bs{args.batch_size}_up{args.up_rate}_input{input_size}"
        tb_dir = os.path.join(output_dir, 'tensorboard', exp_name)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
    else:
        logger = None
        writer = None

    model = model.cuda(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler_steplr = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.05**(1/150))

    best_cd = 1e4
    best_metrics = None
    best_epoch_num = 0
    metrics_history = []
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        train_batches_seen = 0
        if local_rank == 0:
            print('========================================')
            print(f"Epoch {epoch+1} started on {torch.cuda.device_count()} GPUs")

        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}", disable=(local_rank != 0)):
            if max_train_batches is not None and i >= max_train_batches:
                break
            input_pts, gt_pts = batch['partial'], batch['complete']
            input_pts = rearrange(input_pts, 'b n c -> b c n').contiguous().float().cuda(local_rank, non_blocking=True)
            gt_pts = rearrange(gt_pts, 'b n c -> b c n').contiguous().float().cuda(local_rank, non_blocking=True)

            partial = input_pts.transpose(1, 2).contiguous()
            render = PCViews(TRANS=-svd_cfg.NETWORK.view_distance, RESOLUTION=224)
            depth = render.get_img(partial).unsqueeze(1)
            predictions = model(partial, depth)
            loss_all, _ = get_loss(
                predictions, gt_pts.transpose(1, 2).contiguous(), sqrt=True)

            optimizer.zero_grad(set_to_none=True)
            loss_all.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss_all.item()
            train_batches_seen += 1

        scheduler_steplr.step()
        train_loss_average = epoch_loss / max(train_batches_seen, 1)
        if local_rank == 0:
            logger.info("Epoch %d/%d train loss: %.6f (%d batches)",
                        epoch + 1, args.epochs, train_loss_average, train_batches_seen)

        if (epoch+1) % args.save_rate == 0:
            metrics = val(model.module, args, val_loader, logger, local_rank=local_rank,
                          max_batches=max_val_batches)
            if local_rank == 0:
                metrics_avg = {k: np.mean(v) for k, v in metrics.items()}
                # print(f"Epoch {epoch+1}/{args.epochs} validation completed. CD L2 * 1e3: {metrics_avg['cd_l2']*1e3:.3f}, HD * 1e3: {metrics_avg['hd']*1e3:.3f}")
                cd_l2 = metrics_avg['cd_l2']
                if cd_l2 < best_cd:
                    best_cd = cd_l2
                    best_metrics = metrics_avg
                    best_epoch_num = epoch
                    save_checkpoint(model, optimizer, scheduler_steplr, epoch,
                                    os.path.join(ckpt_dir, 'ckpt-best.pth'),
                                    best_cd)
                    print("Save best checkpoint to "+ os.path.join(ckpt_dir, 'ckpt-best.pth'))
                save_checkpoint(model, optimizer, scheduler_steplr, epoch,
                                os.path.join(ckpt_dir, 'ckpt-last.pth'),
                                best_cd)
                save_checkpoint(model, optimizer, scheduler_steplr, epoch,
                                os.path.join(ckpt_dir, 'ckpt-epoch-{:02d}.pth'.format(epoch + 1)),
                                best_cd)
                
                logger.info(f"Epoch {epoch+1}/{args.epochs} validation completed. CD L2 * 1e3: {cd_l2*1e3:.3f}, CD L1 * 1e3: {metrics_avg['cd_l1']*1e3:.3f}, HD * 1e3: {metrics_avg['hd']*1e3:.3f}, DCD: {metrics_avg['dcd']:.6f}, F1: {metrics_avg['f1']:.4f}")
                logger.info(
                    "Forestry metrics: height MAE (m): %.4f, crown width MAE (m): %.4f, "
                    "footprint IoU: %.4f, occupancy IoU: %.4f, CBH MAE (m): %.4f, "
                    "fuel-stratum L1: %.4f, largest vertical gap error (m): %.4f",
                    metrics_avg['tree_height_abs_error'], metrics_avg['crown_width_abs_error'],
                    metrics_avg['crown_projection_iou'], metrics_avg['canopy_occupancy_iou'],
                    metrics_avg['crown_base_height_abs_error'], metrics_avg['fuel_stratum_profile_l1'],
                    metrics_avg['largest_vertical_gap_abs_error'])
                logger.info("Forestry metrics (all): %s", "; ".join(
                    "{}={:.6f}".format(name, value)
                    for name, value in metrics_avg.items()
                    if name not in ('cd_l1', 'cd_l2', 'dcd', 'hd', 'f1')
                ))
                metrics_history.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss_average,
                    **metrics_avg,
                })
                history_path = os.path.join(output_dir, 'validation_metrics_by_epoch.csv')
                with open(history_path, 'w', newline='') as handle:
                    writer_csv = csv.DictWriter(handle, fieldnames=metrics_history[0].keys())
                    writer_csv.writeheader()
                    writer_csv.writerows(metrics_history)
                logger.info(f"Current Best CD L2 * 1e3: {best_cd*1e3:.3f} at Epoch {best_epoch_num+1}")
                # === TensorBoard logging ===
                writer.add_scalar('loss/train', train_loss_average, epoch)
                writer.add_scalar('cd/val_cd_times1e3', cd_l2*1e3, epoch)
                for name, value in metrics_avg.items():
                    if name not in ('cd_l1', 'cd_l2', 'dcd', 'hd', 'f1'):
                        writer.add_scalar('forestry/{}'.format(name), value, epoch)
    
    if local_rank == 0:
        print('========================================')
        print("Best checkpoint epoch:", best_epoch_num+1)
        print('Val metrics at best checkpoint:')
        print("Avg CD-l2 * 1e3:", np.mean(best_metrics['cd_l2']) * 1e3)
        print("Avg CD-l1 * 1e3:", np.mean(best_metrics['cd_l1']) * 1e3)
        print("Avg DCD:", np.mean(best_metrics['dcd']))
        print("Avg HD * 1e3:", np.mean(best_metrics['hd']) * 1e3)
        print("Avg F1:", np.mean(best_metrics['f1']))
        print('----------------------------------------')
    
    if 'completion' in args.dataset_config['data_dir']:
        test_dataset = TreeBench10K_completion(args.dataset_config, train=False, phase='test')
    else:
        assert 'upsampling' in args.dataset_config['data_dir']
        test_dataset = TreeBench10K_upsampling(args.dataset_config, train=False, phase='test')

    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=int(args.batch_size * 1.5),
        num_workers=8,
        pin_memory=True,
        sampler=test_sampler,
        drop_last=False
    )
    # load best ckpt
    # Rank 0 is the only writer.  All ranks must wait for the file to be
    # fully closed before loading it for distributed test evaluation.
    dist.barrier()
    if local_rank == 0:
        print("Evaluating on test set using best checkpoint...")
    checkpoint = torch.load(os.path.join(ckpt_dir, 'ckpt-best.pth'), map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    # test on test_loader
    test_metrics = val(model, args, test_loader, logger, local_rank=local_rank,
                       max_batches=max_val_batches)
    
    if local_rank == 0:
        test_metrics_avg = {k: np.mean(v) for k, v in test_metrics.items()}
        test_table_path = os.path.join(output_dir, 'test_metrics.csv')
        test_row = {
            'checkpoint': 'ckpt-best.pth',
            'best_epoch': best_epoch_num + 1,
            **test_metrics_avg,
        }
        with open(test_table_path, 'w', newline='') as handle:
            writer_csv = csv.DictWriter(handle, fieldnames=test_row.keys())
            writer_csv.writeheader()
            writer_csv.writerow(test_row)
        print('========================================')
        print('Test set metrics at best checkpoint:')
        print("Avg CD-l2 * 1e3:", np.mean(test_metrics_avg['cd_l2']) * 1e3)
        print("Avg CD-l1 * 1e3:", np.mean(test_metrics_avg['cd_l1']) * 1e3)
        print("Avg DCD:", np.mean(test_metrics_avg['dcd']))
        print("Avg HD * 1e3:", np.mean(test_metrics_avg['hd']) * 1e3)
        print("Avg F1:", np.mean(test_metrics_avg['f1']))
        print("Avg height MAE (m):", np.mean(test_metrics_avg['tree_height_abs_error']))
        print("Avg crown-width MAE (m):", np.mean(test_metrics_avg['crown_width_abs_error']))
        print("Avg footprint IoU:", np.mean(test_metrics_avg['crown_projection_iou']))
        print("Avg occupancy IoU:", np.mean(test_metrics_avg['canopy_occupancy_iou']))
        print("Avg CBH MAE (m):", np.mean(test_metrics_avg['crown_base_height_abs_error']))
        print("Avg fuel-stratum L1:", np.mean(test_metrics_avg['fuel_stratum_profile_l1']))
        print("Avg largest vertical gap error (m):", np.mean(test_metrics_avg['largest_vertical_gap_abs_error']))
        print("All test metrics written to:", test_table_path)
        print('========================================')
    

    cleanup_distributed()
    if local_rank == 0:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ``torch.distributed.launch`` (used by the documented multi-GPU command)
    # injects this legacy argument in addition to LOCAL_RANK in the environment.
    # Consume it here; setup_distributed remains the single source of truth.
    parser.add_argument('--local_rank', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--smoke-epochs', type=int, default=None,
                        help='short-run override; does not alter normal training defaults')
    parser.add_argument('--smoke-train-batches', type=int, default=None,
                        help='limit train batches per epoch for an integration test')
    parser.add_argument('--smoke-val-batches', type=int, default=None,
                        help='limit validation batches for an integration test')
    parser.add_argument('--smoke-batch-size', type=int, default=None,
                        help='per-GPU batch-size override for an integration test')
    # ``parse_TreeBench_args`` owns the normal model arguments and parses
    # ``sys.argv`` itself. Consume smoke-only flags first so that parser does
    # not reject them.
    args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv
    smoke_epochs = args.smoke_epochs
    smoke_train_batches = args.smoke_train_batches
    smoke_val_batches = args.smoke_val_batches
    smoke_batch_size = args.smoke_batch_size
    reset_model_args(parse_TreeBench_args(), args)
    if getattr(args, 'treebench_evaluation_only', False):
        raise RuntimeError(
            'TreeBench 16x configurations are evaluation-only. Reuse the '
            'matching 4x checkpoint through analyze_treebench_taxonomy.py.')
    config_stem = os.path.splitext(os.path.basename(args.config))[0]
    args.out_path = os.environ.get(
        'TREEBENCH_OUTPUT_DIR', os.path.join('./treebench_runs', config_stem))
    if smoke_epochs is not None:
        args.epochs = smoke_epochs
        args.save_rate = 1
    if smoke_batch_size is not None:
        args.batch_size = smoke_batch_size
    local_rank = setup_distributed()
    svd_cfg.NETWORK.step2 = 4
    model = SVDFormer(svd_cfg)
    train(model, args, local_rank, max_train_batches=smoke_train_batches,
          max_val_batches=smoke_val_batches)
