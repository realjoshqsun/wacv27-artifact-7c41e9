"""Evaluate a PoinTr/AdaPoinTr checkpoint by TreeBench genus and species."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
from tools import builder
from utils.config import cfg_from_yaml_file
from utils.treebench_eval import evaluate

DEFAULT_TAXONOMY_CSV = None


def canonical_tree_id(value):
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def make_taxonomy_resolver(data_dir, taxonomy_csv):
    fallback = {}
    table = pd.read_csv(taxonomy_csv, dtype=str)
    id_column = "treeID_str" if "treeID_str" in table.columns else "treeID"
    for row in table[[id_column, "species", "genus"]].dropna().itertuples(index=False):
        tree_id, species, genus = map(canonical_tree_id, row)
        fallback[tree_id] = (species, genus)
        fallback.setdefault(tree_id.lstrip("0") or "0", (species, genus))

    def resolve(name):
        key = canonical_tree_id(name)
        metadata_path = Path(data_dir) / key / "metadata.json"
        with metadata_path.open() as handle:
            metadata = json.load(handle)
        species, genus = metadata.get("species"), metadata.get("genus")
        if species and genus:
            return str(species), str(genus)
        return fallback.get(key) or fallback.get(key.lstrip("0") or "0") or ("Unknown", "Unknown")

    return resolve


def resolve_taxonomy(data_dir, name):
    with (Path(data_dir) / str(name) / "metadata.json").open() as handle:
        metadata = json.load(handle)
    return str(metadata.get("species", "Unknown")), str(metadata.get("genus", "Unknown"))


def aggregate(rows, split, resolve):
    table = pd.DataFrame(rows)
    taxonomy = table["name"].map(resolve)
    table[["species", "genus"]] = pd.DataFrame(taxonomy.tolist(), index=table.index)
    metrics = [
        column for column in table
        if column not in {"name", "mode", "view", "sample_key",
                          "input_points", "output_points", "species", "genus"}
    ]
    output = []
    for kind, column in (("overall", None), ("genus", "genus"), ("species", "species")):
        groups = [("overall", table)] if column is None else table.groupby(column, sort=True)
        for group, values in groups:
            row = {"group_type": kind, "group": str(group), f"n_{split}": len(values)}
            row.update({f"{split}_{metric}": values[metric].mean() for metric in metrics})
            output.append(row)
    return pd.DataFrame(output), table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--taxonomy-csv", required=True)
    args = parser.parse_args()

    config = cfg_from_yaml_file(args.config)
    model = build_model_from_cfg(config.model).cuda()
    builder.load_model(model, args.checkpoint)
    tables = []
    samples = []
    data_config = config.dataset.val._base_
    data_dir = data_config.data_dir if hasattr(data_config, "data_dir") else str(
        Path(data_config.CATEGORY_FILE_PATH).parent / "upsampling")
    resolve = make_taxonomy_resolver(data_dir, args.taxonomy_csv)
    for split in ("val", "test"):
        split_config = config.dataset[split]
        dataset = build_dataset_from_cfg(split_config._base_, split_config.others)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        _, rows = evaluate(
            model, loader, torch.device("cuda"), return_rows=True,
            treebench_16x_protocol=getattr(config, 'treebench_16x_protocol', None),
            max_batches=args.max_batches)
        summary, per_sample = aggregate(rows, split, resolve)
        tables.append(summary)
        samples.append((split, per_sample))

    combined = tables[0].merge(tables[1], on=["group_type", "group"], how="outer")
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    for split, table in samples:
        table.to_csv(output.with_name(f"{output.stem}_{split}_per_sample.csv"), index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
