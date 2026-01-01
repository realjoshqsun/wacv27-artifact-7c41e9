"""Evaluate a PUDM checkpoint and aggregate TreeBench metrics by taxonomy."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import get_dataloader
from eval_treebench import evaluate, init_distributed
from json_reader import restore_string_to_list_in_a_dict
from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from util import calc_diffusion_hyperparams, set_seed

DEFAULT_TAXONOMY_CSV = None


def canonical_tree_id(value):
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def resolve_conditioning_rate(config_path, config):
    """Return the rate whose learned class embedding the checkpoint uses.

    Official PUDM cross-rate inference changes the interpolation/output rate
    while retaining the source checkpoint's class condition.  TreeBench keeps
    the runtime dataset in a separate 16x config, so recover and validate the
    source rate explicitly instead of using the runtime dataset label (15).
    """
    runtime_dataset = config["dataset_config"]
    runtime_rate = int(runtime_dataset["R"])
    source_name = config.get("treebench_source_training_config")
    if source_name is None:
        return runtime_rate

    source_path = Path(config_path).resolve().parent / source_name
    with source_path.open() as handle:
        source_config = restore_string_to_list_in_a_dict(json.load(handle))
    source_dataset = source_config["dataset_config"]
    source_rate = int(source_dataset["R"])

    if config.get("treebench_16x_protocol") != "same_4x_checkpoint_runtime_R16":
        raise RuntimeError("Unrecognized PUDM TreeBench cross-rate protocol")
    if (int(runtime_dataset["npoints"]), runtime_rate) != (512, 16):
        raise RuntimeError(
            "PUDM TreeBench cross-rate evaluation requires runtime 512->8192; "
            "got npoints={} and R={}".format(
                runtime_dataset["npoints"], runtime_rate))
    if (int(source_dataset["npoints"]), source_rate) != (2048, 4):
        raise RuntimeError(
            "PUDM cross-rate checkpoint source must be the 2048-point 4x "
            "configuration; got npoints={} and R={}".format(
                source_dataset["npoints"], source_rate))
    if source_config["pointnet_config"] != config["pointnet_config"]:
        raise RuntimeError(
            "Runtime and source PUDM model configurations differ; strict "
            "cross-rate checkpoint reuse is not valid")
    if (source_dataset["CATEGORY_FILE_PATH"] !=
            runtime_dataset["CATEGORY_FILE_PATH"]):
        raise RuntimeError("Runtime and source PUDM taxonomy splits differ")
    return source_rate


def taxonomy_fallback(path):
    table = pd.read_csv(path, dtype=str)
    id_column = "treeID_str" if "treeID_str" in table.columns else "treeID"
    result = {}
    for row in table[[id_column, "species", "genus"]].dropna().itertuples(index=False):
        tree_id, species, genus = map(canonical_tree_id, row)
        result[tree_id] = (species, genus)
        result.setdefault(tree_id.lstrip("0") or "0", (species, genus))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--taxonomy-csv", required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    init_distributed()
    rank = torch.distributed.get_rank()
    set_seed(42)
    with open(args.config) as handle:
        config = restore_string_to_list_in_a_dict(json.load(handle))
    dataset_config = config["dataset_config"]
    conditioning_rate = resolve_conditioning_rate(args.config, config)
    dataset_config["batch_size"] = args.batch_size
    dataset_config["eval_batch_size"] = args.batch_size
    dataset_config["num_workers"] = args.num_workers
    diffusion = calc_diffusion_hyperparams(**config["diffusion_config"])
    for key in diffusion:
        if key != "T":
            diffusion[key] = diffusion[key].cuda()
    model = PointNet2CloudCondition(config["pointnet_config"]).cuda()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = DDP(model, device_ids=[torch.cuda.current_device()])

    summaries = []
    per_sample_tables = []
    fallback = taxonomy_fallback(args.taxonomy_csv)
    with open(dataset_config["CATEGORY_FILE_PATH"]) as handle:
        category_split = json.load(handle)
    for split in ("val", "test"):
        loader = get_dataloader(dataset_config, phase=split, distributed=True)
        _, _, metrics, names = evaluate(
            model, loader, diffusion, dataset_config["scale"], dataset_config["R"],
            config["pointnet_config"]["gamma"], config["diffusion_config"]["T"],
            args.step, "", return_names=True, max_batches=args.max_batches,
            conditioning_R=conditioning_rate,
        )
        table = pd.DataFrame(metrics)
        table.insert(0, "name", names)
        # DistributedSampler pads a non-divisible validation/test set by
        # repeating a few samples so every rank receives the same length.
        # Those are transport duplicates, not additional observations: remove
        # them before taxonomy aggregation and require the exact split IDs.
        table["name"] = table["name"].map(canonical_tree_id)
        table["sample_key"] = table["sample_key"].astype(str)
        table = table.drop_duplicates(subset="sample_key", keep="first")
        expected_names = [canonical_tree_id(name) for name in category_split[split]]
        if dataset_config["dataset"] == "TreeBench10K_completion":
            expected_keys = [
                "{}|{}|{}".format(name, mode, view)
                for name in expected_names
                for mode in ("uls", "tls")
                for view in range(4)
            ]
        else:
            expected_keys = expected_names
        observed_keys = table["sample_key"].tolist()
        if len(observed_keys) != len(expected_keys) or set(observed_keys) != set(expected_keys):
            missing = sorted(set(expected_keys) - set(observed_keys))
            unexpected = sorted(set(observed_keys) - set(expected_keys))
            raise RuntimeError(
                "PUDM {} export does not match the TreeBench split: "
                "rows={}/{}, missing={}, unexpected={}".format(
                    split, len(observed_keys), len(expected_keys),
                    missing[:10], unexpected[:10]))
        order = {key: index for index, key in enumerate(expected_keys)}
        table["_treebench_order"] = table["sample_key"].map(order)
        table = table.sort_values("_treebench_order").drop(columns="_treebench_order")
        names = table["name"].astype(str).tolist()
        taxonomy = []
        for name in names:
            with (Path(dataset_config["data_dir"]) / str(name) / "metadata.json").open() as handle:
                metadata = json.load(handle)
            species, genus = metadata.get("species"), metadata.get("genus")
            key = canonical_tree_id(name)
            taxonomy.append(
                (str(species), str(genus)) if species and genus
                else fallback.get(key) or fallback.get(key.lstrip("0") or "0") or ("Unknown", "Unknown")
            )
        table[["species", "genus"]] = pd.DataFrame(taxonomy, index=table.index)
        metric_names = [
            name for name in metrics
            if name not in {"mode", "view", "sample_key",
                            "input_points", "output_points"}
        ]
        rows = []
        for kind, column in (("overall", None), ("genus", "genus"), ("species", "species")):
            groups = [("overall", table)] if column is None else table.groupby(column, sort=True)
            for group, values in groups:
                row = {"group_type": kind, "group": str(group), f"n_{split}": len(values)}
                row.update({f"{split}_{metric}": values[metric].mean() for metric in metric_names})
                rows.append(row)
        summaries.append(pd.DataFrame(rows))
        per_sample_tables.append((split, table))

    if rank == 0:
        output = Path(args.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        summaries[0].merge(summaries[1], on=["group_type", "group"], how="outer").to_csv(output, index=False)
        for split, table in per_sample_tables:
            table.to_csv(output.with_name(f"{output.stem}_{split}_per_sample.csv"), index=False)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
