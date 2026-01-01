"""Evaluate a TreeBench checkpoint and aggregate metrics by genus and species.

This is intentionally a separate, opt-in analysis entry point: normal training
and evaluation do not perform taxonomy aggregation.  The output is one CSV with
an ``overall`` row followed by ``genus`` and ``species`` rows.  Every metric has
both ``val_*`` and ``test_*`` columns, allowing direct split comparison.

Usage example:
  python analyze_treebench_taxonomy.py \
    --config cfgs/upsampling/TreeBench_upsampling_4x_random.json \
    --checkpoint output/.../ckpt/ckpt-best.pth \
    --output-csv analysis_results/taxonomy/upsampling_4x_random.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cfgs.upsampling import parse_TreeBench_args
from dataset.TreeBench import TreeBench10K_completion, TreeBench10K_upsampling
from config_pcn import cfg as svd_cfg
from models.SVDFormer import Model as SVDFormer
from train_treebench_ddp import set_seed, val


DEFAULT_TAXONOMY_CSV = None


def parse_analysis_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to ckpt-best.pth (or another TreeBench checkpoint).")
    parser.add_argument("--output-csv", required=True, help="Destination combined val/test taxonomy table.")
    parser.add_argument("--taxonomy-csv", default=DEFAULT_TAXONOMY_CSV,
                        help="Fallback taxonomy CSV used only when metadata.json has no species/genus.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Evaluation batch size; defaults to floor(training batch_size * 1.5).")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Optional per-split batch limit for an integration smoke test.")
    analysis_args, treebench_argv = parser.parse_known_args()
    # parse_TreeBench_args owns --config and all model/dataset options.  Keep
    # this script's flags out of its argparse namespace.
    sys.argv = [sys.argv[0]] + treebench_argv
    treebench_args = parse_TreeBench_args()
    if analysis_args.num_workers is not None:
        treebench_args.num_workers = analysis_args.num_workers
    return analysis_args, treebench_args


def canonical_tree_id(value):
    """Return a stable identifier while accepting numeric CSV tree IDs."""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def make_taxonomy_resolver(data_dir, taxonomy_csv):
    """Prefer per-tree metadata; use FOR-species20K CSV only as a fallback."""
    fallback = {}
    if taxonomy_csv and os.path.exists(taxonomy_csv):
        table = pd.read_csv(taxonomy_csv, dtype=str)
        required = {"species", "genus"}
        id_column = "treeID_str" if "treeID_str" in table.columns else "treeID"
        if required.issubset(table.columns) and id_column in table.columns:
            for row in table[[id_column, "species", "genus"]].dropna().itertuples(index=False):
                tree_id, species, genus = map(canonical_tree_id, row)
                fallback[tree_id] = (species, genus)
                fallback.setdefault(tree_id.lstrip("0") or "0", (species, genus))
    elif taxonomy_csv:
        print(f"Warning: fallback taxonomy CSV not found: {taxonomy_csv}")

    metadata_cache = {}
    source_counts = {"metadata": 0, "fallback_csv": 0, "unknown": 0}

    def resolve(tree_id):
        key = canonical_tree_id(tree_id)
        if key not in metadata_cache:
            metadata_path = Path(data_dir) / key / "metadata.json"
            taxonomy = None
            if metadata_path.exists():
                with metadata_path.open() as handle:
                    metadata = json.load(handle)
                species, genus = metadata.get("species"), metadata.get("genus")
                if species and genus:
                    taxonomy = (str(species), str(genus), "metadata")
            if taxonomy is None:
                fallback_taxonomy = fallback.get(key) or fallback.get(key.lstrip("0") or "0")
                if fallback_taxonomy:
                    taxonomy = (*fallback_taxonomy, "fallback_csv")
                else:
                    taxonomy = ("Unknown", "Unknown", "unknown")
            metadata_cache[key] = taxonomy
        species, genus, source = metadata_cache[key]
        source_counts[source] += 1
        return species, genus

    return resolve, source_counts


def build_dataset(args, phase):
    if "completion" in args.dataset_config["data_dir"]:
        return TreeBench10K_completion(args.dataset_config, train=False, phase=phase)
    return TreeBench10K_upsampling(args.dataset_config, train=False, phase=phase)


def evaluate_split(model, args, phase, batch_size, max_batches=None):
    dataset = build_dataset(args, phase)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=args.num_workers,
                        pin_memory=True, shuffle=False)
    values = val(model, args, loader, local_rank=0, return_names=True,
                 max_batches=max_batches)
    return pd.DataFrame(values)


def aggregate_split(per_sample, split, resolve_taxonomy):
    per_sample = per_sample.copy()
    taxonomy = per_sample["name"].map(resolve_taxonomy)
    per_sample[["species", "genus"]] = pd.DataFrame(taxonomy.tolist(), index=per_sample.index)
    metric_columns = [column for column in per_sample.columns
                      if column not in {"name", "mode", "view", "sample_key",
                                        "input_points", "output_points",
                                        "species", "genus"}]

    rows = []
    for group_type, group_column in (("overall", None), ("genus", "genus"), ("species", "species")):
        if group_column is None:
            groups = [("overall", per_sample)]
        else:
            groups = per_sample.groupby(group_column, sort=True, dropna=False)
        for group, values in groups:
            row = {"group_type": group_type, "group": str(group), f"n_{split}": len(values)}
            row.update({f"{split}_{metric}": values[metric].mean() for metric in metric_columns})
            rows.append(row)
    return pd.DataFrame(rows), per_sample


def main():
    analysis_args, args = parse_analysis_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by SVDFormer.")
    set_seed(args.seed)
    batch_size = analysis_args.batch_size or int(args.batch_size * 1.5)

    checkpoint = torch.load(analysis_args.checkpoint, map_location="cpu")
    svd_cfg.NETWORK.step2 = 4
    model = SVDFormer(svd_cfg).cuda(0).eval()
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    # Training checkpoints are saved from DDP.  The analyzer intentionally
    # runs on one GPU, so remove the DDP module prefix when necessary.
    if next(iter(state)).startswith("module."):
        state = {key[len("module."):]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)

    resolve_taxonomy, source_counts = make_taxonomy_resolver(
        args.dataset_config["data_dir"], analysis_args.taxonomy_csv)
    val_per_sample = evaluate_split(
        model, args, "val", batch_size, analysis_args.max_batches)
    test_per_sample = evaluate_split(
        model, args, "test", batch_size, analysis_args.max_batches)
    val_table, val_per_sample = aggregate_split(val_per_sample, "val", resolve_taxonomy)
    test_table, test_per_sample = aggregate_split(test_per_sample, "test", resolve_taxonomy)

    combined = val_table.merge(test_table, on=["group_type", "group"], how="outer")
    order = {"overall": 0, "genus": 1, "species": 2}
    combined["_order"] = combined["group_type"].map(order)
    combined = combined.sort_values(["_order", "group"]).drop(columns="_order")
    output_path = Path(analysis_args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    # Per-sample files make aggregation auditable without mixing them with
    # normal training/evaluation outputs.
    stem = output_path.with_suffix("")
    val_per_sample.to_csv(f"{stem}_val_per_sample.csv", index=False)
    test_per_sample.to_csv(f"{stem}_test_per_sample.csv", index=False)

    print(f"Wrote taxonomy summary: {output_path}")
    print(f"Rows: {len(combined)} (overall + genus + species); taxonomy sources: {source_counts}")


if __name__ == "__main__":
    main()
