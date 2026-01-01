import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import compute_metrics
import numpy as np
import csv
from utils.treebench_eval import evaluate

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    if getattr(config, 'treebench_evaluation_only', False):
        raise RuntimeError(
            'This TreeBench 16x configuration is evaluation-only. Train its '
            'declared 4x source configuration, then run 16x taxonomy/test '
            'evaluation with that 4x checkpoint.')
    # build dataset
    (train_sampler, train_dataloader), (_, val_dataloader), (_, test_dataloader) = (
        builder.dataset_builder(args, config.dataset.train),
        builder.dataset_builder(args, config.dataset.val),
        builder.dataset_builder(args, config.dataset.test),
    )
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    best_cd = float('inf')
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        if isinstance(best_metrics, dict) and "cd_l2" in best_metrics:
            best_cd = float(best_metrics["cd_l2"])
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # print model info
    print_log('Trainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)
    
    print_log('Untrainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer = builder.build_optimizer(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)
    scheduler = builder.build_scheduler(base_model, optimizer, config, last_epoch=start_epoch-1)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, data in enumerate(train_dataloader):
            
            # start = time.time()  # 开始计时
            # data = next(iter(train_dataloader))  # 模拟加载数据
            # print(f"Data loading time: {time.time() - start:.4f}s")
            
            data_time.update(time.time() - batch_start_time)
            partial, gt = data['partial'].cuda(), data['complete'].cuda()

            num_iter += 1
           
            ret = base_model(partial)
            
            sparse_loss, dense_loss = base_model.module.get_loss(ret, gt, epoch)
         
            _loss = sparse_loss + dense_loss 
            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
            else:
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])


            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)

            if config.scheduler.type == 'GradualWarmup':
                if n_itr < config.scheduler.kwargs_2.total_epoch:
                    scheduler.step()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        # All four baselines validate every epoch and select best by raw CD-L2.
        metrics = validate(base_model, val_dataloader, epoch, val_writer, args, config, logger=logger)
        if args.local_rank == 0:
            cd = metrics['cd_l2']
            if cd < best_cd:
                best_metrics = metrics
                best_cd = cd
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
                print_log(f'[Validation] New best model found at epoch {epoch} with metrics: {metrics}', logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
                                f'ckpt-epoch-{epoch + 1:03d}', args, logger=logger)
    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()

    if args.distributed:
        torch.distributed.barrier()
    best_path = os.path.join(args.experiment_path, 'ckpt-best.pth')
    builder.load_model(base_model.module if hasattr(base_model, "module") else base_model,
                       best_path, logger=logger)
    test_metrics = validate(base_model, test_dataloader, "best", None, args, config, logger=logger)
    if args.local_rank == 0:
        output_path = os.path.join(args.experiment_path, "test_metrics.csv")
        with open(output_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["checkpoint"] + list(test_metrics))
            writer.writeheader()
            writer.writerow({"checkpoint": "ckpt-best.pth", **test_metrics})
        print_log(f"[Test] ckpt-best.pth Metrics = {test_metrics}", logger=logger)


from tqdm import tqdm

def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    metrics = evaluate(
        base_model, test_dataloader, torch.device("cuda", args.local_rank),
        treebench_16x_protocol=getattr(config, 'treebench_16x_protocol', None))
    if args.local_rank == 0:
        print_log(f'[Validation] EPOCH: {epoch} Metrics = {metrics}', logger=logger)
        if val_writer is not None and isinstance(epoch, int):
            for name, value in metrics.items():
                val_writer.add_scalar(f"Metric/{name}", value, epoch)
    return metrics

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
 
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger = logger)
    base_model = base_model.cuda()
    
    metrics = evaluate(
        base_model, test_dataloader, torch.device("cuda", args.local_rank),
        treebench_16x_protocol=getattr(config, 'treebench_16x_protocol', None))
    print_log('[Test] Metrics = %s' % metrics, logger=logger)
