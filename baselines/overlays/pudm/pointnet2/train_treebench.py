import os
import time
import argparse
import json
import copy
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from termcolor import cprint
from shutil import copyfile

from dataset import get_dataloader
from util import (
    training_loss,
    calc_diffusion_hyperparams,
    set_seed,
)
from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from eval_treebench import evaluate
from json_reader import (
    replace_list_with_string_in_a_dict,
    restore_string_to_list_in_a_dict,
)


# -------------------------
# DDP setup
# -------------------------
def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


# -------------------------
# data split
# -------------------------
def split_data(data, device):
    label = data["label"].to(device, non_blocking=True)
    X = data["complete"].to(device, non_blocking=True)
    condition = data["partial"].to(device, non_blocking=True)
    return X, condition, label


# -------------------------
# train
# -------------------------
def train(
    config_file,
    model_path,
    dataset,
    root_directory,
    output_directory,
    tensorboard_directory,
    n_epochs,
    epochs_per_ckpt,
    iters_per_logging,
    learning_rate,
    max_iters,
    min_learning_rate,
    eval_step=30,
):

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    local_path = dataset
    output_directory = os.path.join(root_directory, local_path, output_directory)

    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        try:
            copyfile(
                config_file,
                os.path.join(output_directory, os.path.split(config_file)[1]),
            )
        except Exception:
            print("Config already exists, skip copy")

        print("output directory is", output_directory, flush=True)

    # -------------------------
    # diffusion hyperparams
    # -------------------------
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].to(device)

    # -------------------------
    # dataloader (DDP sampler inside)
    # -------------------------
    trainloader = get_dataloader(trainset_config, distributed=True)
    eval_config = copy.deepcopy(trainset_config)
    eval_config["eval_batch_size"] = eval_config.get("eval_batch_size", eval_config["batch_size"])
    valloader = get_dataloader(eval_config, phase="val", distributed=True)

    # -------------------------
    # model
    # -------------------------
    net = PointNet2CloudCondition(pointnet_config).to(device)

    net = torch.nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_iters,
        eta_min=min_learning_rate,
    )
    loss_function = nn.MSELoss()

    # -------------------------
    # resume
    # -------------------------
    time0 = time.time()
    epoch = 0
    best_cd = float("inf")
    best_epoch = 0

    if model_path and os.path.isfile(model_path):
        map_location = {"cuda:0": f"cuda:{local_rank}"}
        checkpoint = torch.load(model_path, map_location=map_location)

        net.module.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = int(checkpoint["epoch"])
        time0 -= checkpoint["training_time_seconds"]
        best_cd = float(checkpoint.get("best_cd", best_cd))
        best_epoch = int(checkpoint.get("best_epoch", epoch))

        if rank == 0:
            print(f"---- Loading checkpoint : {model_path} ----", flush=True)

    loader_len = len(trainloader)
    n_iters = int(loader_len * n_epochs)
    n_iter = epoch * loader_len

    # -------------------------
    # training loop
    # -------------------------
    loss_list = []
    best_path = os.path.join(output_directory, "ckpt-best.pth")
    history = []
    while n_iter < n_iters:
        epoch += 1

        # important for DDP shuffle
        if hasattr(trainloader.sampler, "set_epoch"):
            trainloader.sampler.set_epoch(epoch)

        for data in trainloader:
            X, condition, label = split_data(data, device)

            optimizer.zero_grad()

            loss = training_loss(
                net,
                loss_function,
                X,
                diffusion_hyperparams,
                label=label,
                condition=condition,
                R=trainset_config["R"],
            )

            loss.backward()
            loss_list.append(loss.item())
            optimizer.step()
            scheduler.step()

            if rank == 0 and n_iter % iters_per_logging == 0:
                cprint(
                    "[{}]\tepoch({}/{})\titer({}/{})\tMSE: {:.6f}".format(
                        time.ctime(),
                        epoch,
                        n_epochs,
                        n_iter,
                        n_iters,
                        np.mean(loss_list),
                    ),
                    "blue",
                )
                loss_list = []

            n_iter += 1

        # Match the other baselines: validate every epoch and select raw CD-L2.
        net.eval()
        _, _, metrics = evaluate(
            net, valloader, diffusion_hyperparams, trainset_config["scale"],
            trainset_config["R"], pointnet_config["gamma"],
            diffusion_config["T"], eval_step, save_dir="",
        )
        means = {name: float(np.mean(values)) for name, values in metrics.items()}
        if rank == 0:
            improved = means["cd_l2"] < best_cd
            if improved:
                best_cd, best_epoch = means["cd_l2"], epoch
            checkpoint = {
                "iter": n_iter,
                "model_state_dict": net.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_time_seconds": int(time.time() - time0),
                "epoch": epoch,
                "best_cd": best_cd,
                "best_epoch": best_epoch,
                "val_metrics": means,
            }
            torch.save(checkpoint, os.path.join(output_directory, "ckpt-last.pth"))
            torch.save(checkpoint, os.path.join(output_directory, f"ckpt-epoch-{epoch:03d}.pth"))
            if improved:
                torch.save(checkpoint, best_path)
            history.append({"epoch": epoch, **means})
            with open(os.path.join(output_directory, "validation_metrics_by_epoch.csv"),
                      "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=history[0].keys())
                writer.writeheader()
                writer.writerows(history)
            cprint(f"epoch {epoch}: val CD-L2={means['cd_l2']:.8f}; "
                   f"best={best_cd:.8f} at epoch {best_epoch}", "red")
        net.train()

    dist.barrier()
    checkpoint = torch.load(best_path, map_location=device)
    net.module.load_state_dict(checkpoint["model_state_dict"])
    testloader = get_dataloader(eval_config, phase="test", distributed=True)
    _, _, test_metrics = evaluate(
        net, testloader, diffusion_hyperparams, trainset_config["scale"],
        trainset_config["R"], pointnet_config["gamma"],
        diffusion_config["T"], eval_step, save_dir="",
    )
    if rank == 0:
        means = {name: float(np.mean(values)) for name, values in test_metrics.items()}
        with open(os.path.join(output_directory, "test_metrics.csv"), "w", newline="") as handle:
            row = {"checkpoint": "ckpt-best.pth", "best_epoch": best_epoch, **means}
            writer = csv.DictWriter(handle, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)
        print("Test metrics from ckpt-best.pth:", means, flush=True)


# -------------------------
# main
# -------------------------
if __name__ == "__main__":
    dataset = "PUGAN"
    check_name = ""
    model_path = f"./exp_{dataset.lower()}/{dataset}/logs/checkpoint/{check_name}"

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default=dataset)
    parser.add_argument("-m", "--model_path", type=str, default="")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    args.config = args.config or f"./exp_configs/{args.dataset}.json"

    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    config = restore_string_to_list_in_a_dict(config)

    dataset_config = config["dataset_config"]
    if (dataset_config.get("dataset") == "TreeBench10K_upsampling" and
            int(dataset_config.get("R", 4)) != 4):
        raise ValueError(
            "TreeBench 16x is evaluation-only. Train/select the corresponding "
            "4x PUDM checkpoint and evaluate it with runtime R=16."
        )

    local_rank = setup_ddp()
    set_seed(42)

    if dist.get_rank() == 0:
        print("The configuration is:")
        print(
            json.dumps(
                replace_list_with_string_in_a_dict(copy.deepcopy(config)),
                indent=4,
            )
        )

    # globals (保持你原来的结构)
    global train_config
    global pointnet_config
    global diffusion_config
    global trainset_config
    global diffusion_hyperparams

    train_config = config["train_config"]
    pointnet_config = config["pointnet_config"]
    diffusion_config = config["diffusion_config"]
    trainset_config = dataset_config

    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)

    train(
        args.config,
        args.model_path,
        **train_config,
    )
