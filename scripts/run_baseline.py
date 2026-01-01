#!/usr/bin/env python3
"""Validate and launch one registered TreeBench baseline command."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple


RELEASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = RELEASE_ROOT / "baselines" / "registry.json"
DEFAULT_MANIFEST = RELEASE_ROOT / "baselines" / "overlay_manifest.json"
SOURCE_MANIFEST = RELEASE_ROOT / "manifests" / "source.json"
SPLIT_MANIFESTS = {
    "random": RELEASE_ROOT / "manifests" / "splits" / "random.json",
    "genus_ood": RELEASE_ROOT / "manifests" / "splits" / "genus_ood.json",
}
POINT_METRIC_NAMES = ("cd_l2", "cd_l1", "hd", "dcd", "f1")
TREE_METRIC_NAMES = (
    "tree_height_abs_error", "tree_height_relative_error",
    "crown_width_abs_error", "crown_width_relative_error",
    "crown_projection_area_abs_error", "crown_projection_area_relative_error",
    "crown_projection_iou", "occupied_volume_abs_error",
    "occupied_volume_relative_error", "canopy_occupancy_iou",
    "vertical_profile_l1", "crown_length_abs_error",
    "crown_length_relative_error", "relative_height_25_abs_error",
    "relative_height_50_abs_error", "relative_height_75_abs_error",
    "relative_height_95_abs_error", "foliage_height_diversity_abs_error",
    "crown_base_height_abs_error", "crown_base_height_relative_error",
    "fuel_stratum_profile_l1", "stratum_percentage_cover_l1",
    "fuel_layer_count_abs_error", "lowest_fuel_layer_base_height_abs_error",
    "dominant_fuel_layer_depth_abs_error",
    "mean_interlayer_distance_abs_error", "largest_vertical_gap_abs_error",
)
METRIC_NAMES = POINT_METRIC_NAMES + TREE_METRIC_NAMES
PER_SAMPLE_IDENTITY_COLUMNS = (
    "name", "mode", "view", "sample_key", "input_points", "output_points",
)


class LaunchError(RuntimeError):
    """Raised when a requested run is outside the registered protocol."""


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LaunchError(f"Expected a JSON object in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: str, field: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise LaunchError(f"Unsafe {field}: {value!r}")
    return path


def repo_path(repo: Path, value: str, field: str) -> Path:
    relative = safe_relative(value, field)
    candidate = repo.joinpath(*relative.parts)
    try:
        candidate.resolve().relative_to(repo.resolve())
    except ValueError as exc:
        raise LaunchError(f"{field} escapes the repository: {value!r}") from exc
    return candidate


def baseline_entry(registry: Dict[str, Any], model: str) -> Dict[str, Any]:
    entries = [entry for entry in registry.get("baselines", []) if entry.get("id") == model]
    if len(entries) != 1:
        known = sorted(entry.get("id", "") for entry in registry.get("baselines", []))
        raise LaunchError(f"Unknown baseline {model!r}; known IDs: {', '.join(known)}")
    return entries[0]


def parse_scenario(scenario: str) -> Tuple[str, str]:
    for split in ("genus_ood", "random"):
        suffix = "_" + split
        if scenario.endswith(suffix):
            return scenario[: -len(suffix)], split
    raise LaunchError(f"Unrecognized scenario: {scenario}")


def gpu_list(value: str) -> List[str]:
    values = value.split(",")
    if not values or any(not item.isdigit() for item in values):
        raise LaunchError("--gpus must be a comma-separated list of non-negative IDs")
    if len(values) != len(set(values)):
        raise LaunchError("--gpus cannot contain duplicate IDs")
    return values


def git_output(repo: Path, arguments: List[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo)] + arguments,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise LaunchError("git is required to verify a materialized baseline") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or "git command failed"
        raise LaunchError(f"Cannot verify materialized Git checkout: {detail}") from exc
    return completed.stdout.strip()


def git_paths(repo: Path, arguments: List[str]) -> set[str]:
    output = git_output(repo, arguments)
    return {value for value in output.split("\0") if value}


def verify_materialized_repo(
    repo: Path,
    baseline: Dict[str, Any],
    model: str,
    manifest: Dict[str, Any],
) -> None:
    if not repo.is_dir() or repo.is_symlink():
        raise LaunchError("--repo must be a materialized baseline directory")
    status = manifest.get("baseline_status", {}).get(model, {})
    if status.get("status") == "LEGAL_BLOCKED":
        missing = ", ".join(status.get("missing_required_files", []))
        raise LaunchError(f"{model} is LEGAL_BLOCKED; missing required files: {missing}")
    if status.get("status") not in {"READY", "ADAPTER_ONLY"}:
        raise LaunchError(f"Unsupported materialization status for {model}")

    provenance_path = repo / ".treebench-overlay.json"
    if provenance_path.is_symlink() or not provenance_path.is_file():
        raise LaunchError("Materialized overlay provenance must be a regular file")
    provenance = load_json(provenance_path)
    provenance_checks = {
        "schema_version": "1.0.0",
        "registry_sha256": sha256(DEFAULT_REGISTRY),
        "overlay_manifest_sha256": sha256(DEFAULT_MANIFEST),
    }
    for field, expected in provenance_checks.items():
        if provenance.get(field) != expected:
            raise LaunchError(f"Materialized provenance mismatch for {field}")
    if provenance.get("baseline_id") != model:
        raise LaunchError(
            f"Repository was materialized for {provenance.get('baseline_id')!r}, not {model!r}"
        )
    expected_commit = baseline.get("upstream", {}).get("commit")
    if provenance.get("upstream_commit") != expected_commit:
        raise LaunchError("Materialized upstream commit does not match registry.json")
    if provenance.get("upstream_url") != baseline.get("upstream", {}).get("url"):
        raise LaunchError("Materialized upstream URL does not match registry.json")

    actual_commit = git_output(repo, ["rev-parse", "HEAD"])
    if actual_commit != expected_commit:
        raise LaunchError(
            f"Git HEAD differs from the pinned upstream commit: {actual_commit}"
        )
    remote_url = git_output(repo, ["remote", "get-url", "upstream"])
    if remote_url != baseline.get("upstream", {}).get("url"):
        raise LaunchError("The upstream Git remote differs from registry.json")

    registered = {
        item["destination_path"]: item["sha256"]
        for item in manifest.get("files", [])
        if model in item.get("model_ids", []) and item.get("materialize") is True
    }
    recorded = {
        item.get("destination_path"): item.get("sha256")
        for item in provenance.get("overlay_files", [])
    }
    if recorded != registered:
        raise LaunchError("Materialized overlay inventory does not match overlay_manifest.json")
    for relative, expected in registered.items():
        path = repo_path(repo, relative, "overlay destination")
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise LaunchError(f"Materialized overlay verification failed: {relative}")

    allowed_changes = set(registered) | {".treebench-overlay.json"}
    changed = set()
    changed.update(git_paths(repo, ["diff", "--name-only", "--relative", "-z", "HEAD", "--"]))
    changed.update(git_paths(repo, ["diff", "--cached", "--name-only", "--relative", "-z", "HEAD", "--"]))
    changed.update(git_paths(repo, ["ls-files", "-z", "--others", "--exclude-standard"]))
    unexpected = sorted(changed - allowed_changes)
    if unexpected:
        preview = ", ".join(unexpected[:8])
        suffix = " ..." if len(unexpected) > 8 else ""
        raise LaunchError(
            "Materialized checkout contains unregistered changes: " + preview + suffix
        )


def config_group(task: str) -> str:
    if task == "completion":
        return "completion"
    if task == "upsampling_4x":
        return "upsampling_4x"
    if task == "upsampling_16x":
        return "upsampling_16x"
    raise LaunchError(f"Unsupported task: {task}")


def resolve_protocol(
    baseline: Dict[str, Any], scenario: str, action: str
) -> Tuple[str, str, Dict[str, Any]]:
    if scenario not in baseline.get("evaluation_scenarios", []):
        raise LaunchError(f"{baseline['id']} does not support {scenario}")
    task, split = parse_scenario(scenario)
    group = config_group(task)
    entrypoint = baseline.get("entrypoints", {}).get(group)
    if not isinstance(entrypoint, dict):
        raise LaunchError(f"Entrypoint is not registered for {scenario}")
    if action == "train":
        if task == "upsampling_16x":
            raise LaunchError("16x is evaluation-only; train the split-matched 4x task")
        if scenario not in baseline.get("trainable_scenarios", []):
            raise LaunchError(f"{scenario} is not a trainable scenario for {baseline['id']}")
        if entrypoint.get("train") is None:
            raise LaunchError(f"No training entrypoint is registered for {scenario}")
        reproduction = baseline.get("training_reproduction", {})
        if reproduction.get("status") != "VERIFIED":
            raise LaunchError(
                f"Training is fail-closed for {baseline['id']}: "
                f"training_reproduction is {reproduction.get('status', 'missing')}"
            )
        launcher = reproduction.get("distributed_launcher", {})
        if launcher.get("status") != "VERIFIED" or not launcher.get("module"):
            raise LaunchError(
                f"Training launcher provenance is OPEN for {baseline['id']}"
            )
    else:
        if entrypoint.get("evaluate") is None:
            raise LaunchError(f"No evaluation entrypoint is registered for {scenario}")
        evaluation = baseline.get("evaluation_reproduction", {})
        if evaluation.get("status") != "VERIFIED":
            raise LaunchError(
                f"Evaluation is fail-closed for {baseline['id']}: "
                f"evaluation_reproduction is {evaluation.get('status', 'missing')}"
            )
    return task, split, entrypoint


def interpolate_config(value: str, split: str) -> str:
    rendered = value.replace("{split}", split)
    if "{" in rendered or "}" in rendered:
        raise LaunchError(f"Unresolved configuration template: {value}")
    return rendered


def data_locations(data_root: Path, task: str, split: str) -> Tuple[Path, Path]:
    task_directory = data_root / ("completion" if task == "completion" else "upsampling")
    category = SPLIT_MANIFESTS[split]
    if not task_directory.is_dir():
        raise LaunchError(f"TreeBench task directory not found: {task_directory}")
    if not category.is_file():
        raise LaunchError(f"Fixed release split manifest not found: {category}")
    source = load_json(SOURCE_MANIFEST)
    expected = source.get("splits", {}).get(split, {}).get("release_compact_sha256")
    if not isinstance(expected, str) or sha256(category) != expected:
        raise LaunchError(f"Fixed release split manifest failed SHA-256 verification: {split}")
    return task_directory.resolve(), category.resolve()


def runtime_directory(output: Path, action: str) -> Path:
    base = output if action == "train" else output.parent
    return base / ".treebench_runtime"


def rewrite_json_config(
    source: Path,
    destination: Path,
    task_directory: Path,
    category: Path,
    output: Path,
    baseline: Dict[str, Any],
    action: str,
) -> Dict[str, Any]:
    config = load_json(source)
    dataset = config.get("dataset_config")
    if not isinstance(dataset, dict):
        raise LaunchError(f"dataset_config is missing from {source}")
    dataset["data_dir"] = str(task_directory)
    dataset["CATEGORY_FILE_PATH"] = str(category)
    model = baseline["id"]
    if action == "train":
        reproduction = baseline["training_reproduction"]
        if model == "pudm":
            train = config.get("train_config")
            if not isinstance(train, dict):
                raise LaunchError("PUDM train_config is missing")
            train["root_directory"] = str(output.resolve().parent)
            train["dataset"] = output.resolve().name
            train["output_directory"] = "."
            train["tensorboard_directory"] = "tensorboard"
            train["n_epochs"] = reproduction["epochs"]
            train["epochs_per_ckpt"] = reproduction["save_every_epochs"]
            train["learning_rate"] = reproduction["learning_rate"]
            train["min_learning_rate"] = reproduction["minimum_learning_rate"]
            dataset["batch_size"] = reproduction["batch_size"]["value"]
            dataset["eval_batch_size"] = reproduction["batch_size"]["value"]
            dataset["num_workers"] = reproduction["dataloader_workers"]
        else:
            config["epochs"] = reproduction["epochs"]
            config["batch_size"] = reproduction["batch_size"]["value"]
            config["lr"] = reproduction["learning_rate"]
            config["save_rate"] = reproduction["save_every_epochs"]
            config["num_workers"] = reproduction["dataloader_workers"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def rewrite_yaml_configs(
    repo: Path,
    source: Path,
    destination: Path,
    scenario: str,
    task_directory: Path,
    category: Path,
    baseline: Dict[str, Any],
    action: str,
) -> None:
    dataset_source = repo / "cfgs" / "dataset_configs" / f"TreeBench_{scenario}.yaml"
    if not dataset_source.is_file() or dataset_source.is_symlink():
        raise LaunchError(f"Dataset configuration is missing: {dataset_source}")
    dataset_destination = destination.with_name(destination.stem + "_dataset.yaml")
    dataset_text = dataset_source.read_text(encoding="utf-8")
    dataset_text, data_count = re.subn(
        r"(?m)^data_dir:\s*.*$", f"data_dir: {json.dumps(str(task_directory))}", dataset_text
    )
    dataset_text, category_count = re.subn(
        r"(?m)^CATEGORY_FILE_PATH:\s*.*$",
        f"CATEGORY_FILE_PATH: {json.dumps(str(category))}",
        dataset_text,
    )
    if data_count != 1 or category_count != 1:
        raise LaunchError("Could not safely rewrite the YAML dataset configuration")
    dataset_destination.parent.mkdir(parents=True, exist_ok=True)
    dataset_destination.write_text(dataset_text, encoding="utf-8")

    model_text = source.read_text(encoding="utf-8")
    pattern = r"cfgs/dataset_configs/TreeBench_[A-Za-z0-9_]+\.yaml"
    replacements = len(re.findall(pattern, model_text))
    if replacements != 3:
        raise LaunchError(
            f"Expected three dataset references in {source}, found {replacements}"
        )
    model_text = re.sub(pattern, json.dumps(str(dataset_destination)), model_text)
    if action == "train":
        reproduction = baseline["training_reproduction"]
        model_text, epoch_count = re.subn(
            r"(?m)^max_epoch\s*:\s*\d+\s*$",
            f"max_epoch : {reproduction['epoch_encoding']['value']}",
            model_text,
        )
        model_text, batch_count = re.subn(
            r"(?m)^total_bs\s*:\s*\d+\s*$",
            f"total_bs : {reproduction['batch_size']['value']}",
            model_text,
        )
        lr_pattern = re.compile(r"(?m)^(\s*lr\s*:\s*)[0-9.eE+-]+(\s*,.*)$")
        model_text, lr_count = lr_pattern.subn(
            lambda match: (
                match.group(1)
                + str(reproduction["learning_rate"])
                + match.group(2)
            ),
            model_text,
        )
        if (epoch_count, batch_count, lr_count) != (1, 1, 1):
            raise LaunchError(
                "Could not safely enforce PoinTr-family training hyperparameters"
            )
    destination.write_text(model_text, encoding="utf-8")


def prepare_runtime_config(
    repo: Path,
    source: Path,
    baseline: Dict[str, Any],
    scenario: str,
    task: str,
    split: str,
    data_root: Path,
    output: Path,
    action: str,
    dry_run: bool,
) -> Path:
    task_directory, category = data_locations(data_root, task, split)
    directory = runtime_directory(output, action)
    destination = directory / baseline["id"] / source.name
    if dry_run:
        return destination
    if source.suffix.lower() == ".json":
        config = rewrite_json_config(
            source, destination, task_directory, category, output, baseline, action
        )
        if baseline["id"] == "pudm" and task == "upsampling_16x":
            source_name = config.get("treebench_source_training_config")
            if not isinstance(source_name, str):
                raise LaunchError("PUDM 16x configuration lacks its 4x source config")
            source_relative = safe_relative(source_name, "treebench_source_training_config")
            source_4x = source.parent.joinpath(*source_relative.parts)
            destination_4x = destination.parent / source_relative.name
            rewrite_json_config(
                source_4x,
                destination_4x,
                task_directory,
                category,
                output,
                baseline,
                "evaluate",
            )
        return destination
    if source.suffix.lower() in {".yaml", ".yml"}:
        rewrite_yaml_configs(
            repo,
            source,
            destination,
            scenario,
            task_directory,
            category,
            baseline,
            action,
        )
        return destination
    raise LaunchError(f"Unsupported configuration format: {source.suffix}")


def pythonpath_entries(model: str, repo: Path) -> List[Path]:
    relative = {
        "snowflakenet": ["pointnet2_ops_lib", "metrics/CD/chamfer3D", "."],
        "pucrn": ["PointUpsampling/PUCRN/.deps", "PointUpsampling/PUCRN/pointnet2_ops_lib", "PointUpsampling/PUCRN/losses", "PointUpsampling/PUCRN"],
        "anchorformer": ["extensions/chamfer_dist", "extensions/expansion_penalty", "."],
        "pu_flow": [".deps", "metric/emd", "."],
        "svdformer": ["metrics/CD/chamfer3D", "."],
        "fsc": ["metrics/CD/chamfer3D", "."],
        "geoformer": ["src", "."],
        "pudm": ["pointnet2", "."],
    }.get(model, ["."])
    return [(repo / value).resolve() for value in relative]


def command_for(
    baseline: Dict[str, Any],
    entrypoint: Dict[str, Any],
    task: str,
    action: str,
    repo: Path,
    runtime_config: Path,
    checkpoint: Optional[Path],
    taxonomy_csv: Optional[Path],
    output: Path,
    processes: int,
    python: str,
) -> List[str]:
    model = baseline["id"]
    if action == "train":
        reproduction = baseline["training_reproduction"]
        launcher = reproduction["distributed_launcher"]["module"]
        entry = repo_path(repo, entrypoint["train"], "training entrypoint")
        command = [
            python,
            "-m",
            launcher,
            "--nproc_per_node",
            str(processes),
            str(entry),
        ]
        if model in {"pointr", "adapointr", "odgnet"}:
            command.extend(
                [
                    "--launcher",
                    "pytorch",
                    "--config",
                    str(runtime_config),
                    "--exp_name",
                    "treebench",
                    "--num_workers",
                    str(reproduction["dataloader_workers"]),
                ]
            )
        elif model == "pudm":
            command.extend(["--config", str(runtime_config)])
        else:
            command.extend(
                [
                    "--config",
                    str(runtime_config),
                    "--epochs",
                    str(reproduction["epochs"]),
                    "--save_rate",
                    str(reproduction["save_every_epochs"]),
                    "--batch_size",
                    str(reproduction["batch_size"]["value"]),
                    "--lr",
                    str(reproduction["learning_rate"]),
                ]
            )
        return command

    if checkpoint is None or taxonomy_csv is None:
        raise LaunchError("Evaluation requires --checkpoint and --taxonomy-csv")
    entry = repo_path(repo, entrypoint["evaluate"], "evaluation entrypoint")
    command = [python]
    if model == "pudm":
        command.extend(
            ["-m", "torch.distributed.launch", "--nproc_per_node", str(processes)]
        )
    command.extend(
        [
            str(entry),
            "--config",
            str(runtime_config),
            "--checkpoint",
            str(checkpoint),
            "--taxonomy-csv",
            str(taxonomy_csv),
            "--output-csv",
            str(output),
            "--batch-size",
            str(baseline["evaluation_reproduction"]["batch_size"][task]),
            "--num-workers",
            str(baseline["evaluation_reproduction"]["dataloader_workers"]),
        ]
    )
    return command


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def validate_output_target(output: Path, action: str, repo: Path) -> None:
    if is_within(output, repo):
        raise LaunchError("--output must be outside the materialized baseline checkout")
    if output.is_symlink():
        raise LaunchError("--output cannot be a symbolic link")
    if action == "train":
        if output.exists():
            if not output.is_dir():
                raise LaunchError("Training --output exists and is not a directory")
            try:
                next(output.iterdir())
            except StopIteration:
                return
            raise LaunchError("Training --output must be new or empty")
    else:
        related = [
            output,
            output.with_name(output.stem + "_val_per_sample.csv"),
            output.with_name(output.stem + "_test_per_sample.csv"),
            output.with_suffix(output.suffix + ".run.json"),
        ]
        existing = [str(path) for path in related if path.exists() or path.is_symlink()]
        if existing:
            raise LaunchError(
                "Evaluation output artifact already exists; refusing to overwrite: "
                + ", ".join(existing)
            )


def checkpoint_metadata_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(checkpoint.name + ".treebench.json")


def verify_cross_rate_checkpoint(
    checkpoint: Path,
    baseline: Dict[str, Any],
    split: str,
) -> Dict[str, Any]:
    metadata_path = checkpoint_metadata_path(checkpoint)
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise LaunchError(
            f"16x evaluation requires regular split-matched checkpoint metadata: {metadata_path}"
        )
    metadata = load_json(metadata_path)
    if metadata.get("schema_version") != "1.0.0":
        raise LaunchError("16x checkpoint metadata has an unsupported schema_version")
    expected_scenario = baseline["cross_rate_16x"]["source_checkpoint"][split]
    checks = {
        "baseline_id": baseline["id"],
        "training_scenario": expected_scenario,
        "upstream_commit": baseline["upstream"]["commit"],
        "checkpoint_sha256": sha256(checkpoint),
    }
    reproduction = baseline.get("training_reproduction", {})
    checks.update(
        {
            "epochs": reproduction.get("epochs"),
            "batch_size": reproduction.get("batch_size"),
            "learning_rate": reproduction.get("learning_rate"),
            "compute_controlled_across_models": False,
        }
    )
    for field, expected in checks.items():
        if metadata.get(field) != expected:
            raise LaunchError(
                f"16x checkpoint metadata mismatch for {field}: "
                f"expected {expected!r}, got {metadata.get(field)!r}"
            )
    runtime_digest = metadata.get("runtime_config_sha256")
    if not isinstance(runtime_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", runtime_digest):
        raise LaunchError("16x checkpoint metadata lacks a valid runtime_config_sha256")
    return metadata


def record_training_checkpoint(
    output: Path,
    baseline: Dict[str, Any],
    registry: Dict[str, Any],
    scenario: str,
    runtime_config: Path,
) -> Path:
    protocol = baseline.get("checkpoint_protocol")
    if not isinstance(protocol, str):
        raise LaunchError("Checkpoint protocol is missing from registry.json")
    best = registry.get("checkpoint_protocols", {}).get(protocol, {}).get("best_filename")
    if not isinstance(best, str):
        raise LaunchError(f"Unknown checkpoint protocol: {protocol}")
    filename = PurePosixPath(best).name
    candidates = sorted(
        path for path in output.rglob(filename) if path.is_file() and not path.is_symlink()
    )
    if len(candidates) != 1:
        raise LaunchError(
            f"Training succeeded but found {len(candidates)} {filename} checkpoints under --output"
        )
    checkpoint = candidates[0]
    reproduction = baseline["training_reproduction"]
    metadata = {
        "schema_version": "1.0.0",
        "baseline_id": baseline["id"],
        "training_scenario": scenario,
        "upstream_commit": baseline["upstream"]["commit"],
        "checkpoint_sha256": sha256(checkpoint),
        "runtime_config_sha256": sha256(runtime_config),
        "epochs": reproduction["epochs"],
        "batch_size": reproduction["batch_size"],
        "learning_rate": reproduction["learning_rate"],
        "compute_controlled_across_models": False,
    }
    metadata_path = checkpoint_metadata_path(checkpoint)
    if metadata_path.exists() or metadata_path.is_symlink():
        raise LaunchError(
            f"Training process created an unexpected checkpoint metadata path: {metadata_path}"
        )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return checkpoint


def read_csv_artifact(path: Path) -> Tuple[Dict[str, Any], List[str], List[Dict[str, str]]]:
    if path.is_symlink() or not path.is_file():
        raise LaunchError(f"Evaluator did not create a required regular CSV: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise LaunchError(f"Evaluator CSV has no header: {path}")
            header = list(reader.fieldnames)
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise LaunchError(f"Cannot validate evaluator CSV {path}: {exc}") from exc
    if not header or any(not value for value in header) or len(header) != len(set(header)):
        raise LaunchError(f"Evaluator CSV has an invalid or duplicate header: {path}")
    artifact = {
        "path": path.name,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
    }
    return artifact, header, rows


def finite_number(value: str, field: str, path: Path) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LaunchError(f"Non-numeric {field} in evaluator CSV {path}: {value!r}") from exc
    if not (number == number and abs(number) != float("inf")):
        raise LaunchError(f"Non-finite {field} in evaluator CSV {path}: {value!r}")
    return number


def exact_integer(value: str, field: str, path: Path) -> int:
    number = finite_number(value, field, path)
    integer = int(number)
    if number != integer:
        raise LaunchError(f"Non-integral {field} in evaluator CSV {path}: {value!r}")
    return integer


def expected_sample_keys(task: str, names: List[str]) -> set[str]:
    if task == "completion":
        return {
            f"{name}|{mode}|{view}"
            for name in names
            for mode in ("uls", "tls")
            for view in range(4)
        }
    return set(names)


def validate_per_sample_csv(
    path: Path,
    task: str,
    split: str,
    split_names: List[str],
    input_points: int,
    output_points: int,
) -> Dict[str, Any]:
    artifact, header, rows = read_csv_artifact(path)
    required = set(PER_SAMPLE_IDENTITY_COLUMNS) | set(METRIC_NAMES) | {"species", "genus"}
    missing_columns = sorted(required - set(header))
    if missing_columns:
        raise LaunchError(
            f"Per-sample CSV {path} lacks required columns: {', '.join(missing_columns)}"
        )
    unexpected_columns = sorted(set(header) - required)
    if unexpected_columns:
        raise LaunchError(
            f"Per-sample CSV {path} has unregistered columns: {', '.join(unexpected_columns)}"
        )

    expected_names = set(split_names)
    expected_keys = expected_sample_keys(task, split_names)
    observed_keys: List[str] = []
    for row_number, row in enumerate(rows, start=2):
        name = row["name"]
        mode = row["mode"]
        view = exact_integer(row["view"], f"view at row {row_number}", path)
        sample_key = row["sample_key"]
        if name not in expected_names:
            raise LaunchError(
                f"Unexpected tree ID in {path} at row {row_number}: {name!r}"
            )
        if task == "completion":
            if mode not in {"uls", "tls"} or view not in range(4):
                raise LaunchError(
                    f"Invalid completion mode/view in {path} at row {row_number}: "
                    f"{mode!r}/{view}"
                )
            derived_key = f"{name}|{mode}|{view}"
        else:
            if mode != "upsampling" or view != -1:
                raise LaunchError(
                    f"Invalid upsampling mode/view in {path} at row {row_number}: "
                    f"{mode!r}/{view}"
                )
            derived_key = name
        if sample_key != derived_key:
            raise LaunchError(
                f"sample_key disagrees with identity fields in {path} at row {row_number}"
            )
        if exact_integer(row["input_points"], "input_points", path) != input_points:
            raise LaunchError(f"Wrong input cardinality in {path} at row {row_number}")
        if exact_integer(row["output_points"], "output_points", path) != output_points:
            raise LaunchError(f"Wrong output cardinality in {path} at row {row_number}")
        for metric in METRIC_NAMES:
            finite_number(row[metric], f"{metric} at row {row_number}", path)
        if not row["species"].strip() or not row["genus"].strip():
            raise LaunchError(f"Blank taxonomy field in {path} at row {row_number}")
        observed_keys.append(sample_key)

    observed_set = set(observed_keys)
    if len(observed_keys) != len(observed_set):
        raise LaunchError(f"Duplicate sample_key values in {path}")
    if observed_set != expected_keys:
        missing = sorted(expected_keys - observed_set)
        unexpected = sorted(observed_set - expected_keys)
        raise LaunchError(
            f"{split} sample-key set does not match the fixed split manifest in {path}: "
            f"rows={len(observed_keys)}/{len(expected_keys)}, missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )

    key_digest = hashlib.sha256()
    for value in sorted(observed_set):
        key_digest.update(value.encode("utf-8"))
        key_digest.update(b"\n")
    artifact.update(
        {
            "unique_sample_keys": len(observed_set),
            "sample_key_set_sha256": key_digest.hexdigest(),
            "finite_metric_values_verified": True,
            "input_points_verified": input_points,
            "output_points_verified": output_points,
        }
    )
    return artifact


def validate_aggregate_csv(
    path: Path,
    expected_counts: Dict[str, int],
) -> Dict[str, Any]:
    artifact, header, rows = read_csv_artifact(path)
    required = {"group_type", "group", "n_val", "n_test"}
    required.update(
        f"{split}_{metric}" for split in ("val", "test") for metric in METRIC_NAMES
    )
    missing = sorted(required - set(header))
    if missing:
        raise LaunchError(f"Aggregate CSV {path} lacks required columns: {', '.join(missing)}")
    unexpected = sorted(set(header) - required)
    if unexpected:
        raise LaunchError(
            f"Aggregate CSV {path} has unregistered columns: {', '.join(unexpected)}"
        )
    identities = [(row["group_type"], row["group"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise LaunchError(f"Aggregate CSV contains duplicate group rows: {path}")
    overall = [
        row for row in rows
        if (row["group_type"], row["group"]) == ("overall", "overall")
    ]
    if len(overall) != 1:
        raise LaunchError(f"Aggregate CSV must contain exactly one overall/overall row: {path}")
    for split in ("val", "test"):
        if exact_integer(overall[0][f"n_{split}"], f"n_{split}", path) != expected_counts[split]:
            raise LaunchError(f"Aggregate {split} count disagrees with per-sample CSV: {path}")
        for metric in METRIC_NAMES:
            finite_number(overall[0][f"{split}_{metric}"], f"{split}_{metric}", path)
    for row_number, row in enumerate(rows, start=2):
        if row["group_type"] not in {"overall", "genus", "species"}:
            raise LaunchError(f"Unknown group_type in {path} at row {row_number}")
        for field in required - {"group_type", "group"}:
            value = row[field]
            if value != "":
                finite_number(value, f"{field} at row {row_number}", path)
    artifact["overall_finite_metrics_verified"] = len(METRIC_NAMES)
    return artifact


def validate_evaluation_artifacts(
    output: Path,
    task: str,
    split: str,
    registry: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, int], str]:
    split_path = SPLIT_MANIFESTS[split]
    split_manifest = load_json(split_path)
    val_names = split_manifest.get("val")
    test_names = split_manifest.get("test")
    if not isinstance(val_names, list) or not isinstance(test_names, list):
        raise LaunchError(f"Fixed split manifest has invalid val/test lists: {split_path}")
    if any(not isinstance(value, str) for value in val_names + test_names):
        raise LaunchError(f"Fixed split manifest contains a non-string tree ID: {split_path}")
    if len(val_names) != len(set(val_names)) or len(test_names) != len(set(test_names)):
        raise LaunchError(f"Fixed split manifest contains duplicate tree IDs: {split_path}")

    protocol = registry["task_protocol"][task]
    input_points = int(protocol["input_points"])
    output_points = int(protocol["output_points"])
    val_path = output.with_name(output.stem + "_val_per_sample.csv")
    test_path = output.with_name(output.stem + "_test_per_sample.csv")
    per_sample = {
        "val": validate_per_sample_csv(
            val_path, task, "val", val_names, input_points, output_points
        ),
        "test": validate_per_sample_csv(
            test_path, task, "test", test_names, input_points, output_points
        ),
    }
    counts = {key: value["rows"] for key, value in per_sample.items()}
    aggregate = validate_aggregate_csv(output, counts)
    return aggregate, per_sample, counts, sha256(split_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("scenario")
    parser.add_argument("action", choices=("train", "evaluate"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--taxonomy-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable from the baseline's registered environment profile",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        registry = load_json(DEFAULT_REGISTRY)
        manifest = load_json(DEFAULT_MANIFEST)
        baseline = baseline_entry(registry, args.model)
        task, split, entrypoint = resolve_protocol(baseline, args.scenario, args.action)
        if args.action == "train" and args.checkpoint is not None:
            raise LaunchError("--checkpoint is not accepted for training; use a registered fresh run")
        if args.action == "train" and args.taxonomy_csv is not None:
            raise LaunchError("--taxonomy-csv is an evaluation argument")
        if args.action == "evaluate" and (args.checkpoint is None or args.taxonomy_csv is None):
            raise LaunchError("Evaluation requires --checkpoint and --taxonomy-csv")

        repo = args.repo.expanduser().resolve()
        verify_materialized_repo(repo, baseline, args.model, manifest)
        gpus = gpu_list(args.gpus)
        if args.action == "train":
            expected_gpus = baseline["training_reproduction"]["gpu_count"]
            if len(gpus) != expected_gpus:
                raise LaunchError(
                    f"The recorded {args.model} training protocol requires "
                    f"{expected_gpus} GPUs, got {len(gpus)}"
                )
        config_value = interpolate_config(entrypoint["config"], split)
        source_config = repo_path(repo, config_value, "configuration path")
        if source_config.is_symlink() or not source_config.is_file():
            raise LaunchError(f"Registered configuration is missing: {config_value}")

        data_root = args.data_root.expanduser().resolve()
        output = args.output.expanduser().absolute()
        validate_output_target(output, args.action, repo)
        checkpoint_arg = args.checkpoint.expanduser().absolute() if args.checkpoint else None
        taxonomy_arg = args.taxonomy_csv.expanduser().absolute() if args.taxonomy_csv else None
        if checkpoint_arg is not None and (
            checkpoint_arg.is_symlink() or not checkpoint_arg.is_file()
        ):
            raise LaunchError(f"Checkpoint not found or not a regular file: {checkpoint_arg}")
        if taxonomy_arg is not None and (
            taxonomy_arg.is_symlink() or not taxonomy_arg.is_file()
        ):
            raise LaunchError(f"Taxonomy CSV not found or not a regular file: {taxonomy_arg}")
        checkpoint = checkpoint_arg.resolve() if checkpoint_arg else None
        taxonomy = taxonomy_arg.resolve() if taxonomy_arg else None
        checkpoint_metadata = None
        if args.action == "evaluate" and task == "upsampling_16x":
            checkpoint_metadata = verify_cross_rate_checkpoint(
                checkpoint, baseline, split
            )
        if not args.dry_run:
            if args.action == "train":
                output.mkdir(parents=True, exist_ok=True)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)

        runtime_config = prepare_runtime_config(
            repo,
            source_config,
            baseline,
            args.scenario,
            task,
            split,
            data_root,
            output,
            args.action,
            args.dry_run,
        )
        command = command_for(
            baseline,
            entrypoint,
            task,
            args.action,
            repo,
            runtime_config,
            checkpoint,
            taxonomy,
            output,
            len(gpus),
            args.python,
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        if args.action == "train":
            env["TREEBENCH_OUTPUT_DIR"] = str(output)
        additions = [str(path) for path in pythonpath_entries(args.model, repo)]
        env["PYTHONPATH"] = os.pathsep.join(additions)

        plan = {
            "baseline_id": args.model,
            "scenario": args.scenario,
            "action": args.action,
            "cwd": str(repo),
            "argv": command,
            "environment": {
                "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
                "PYTHONPATH": env["PYTHONPATH"],
                "PYTHONDONTWRITEBYTECODE": env["PYTHONDONTWRITEBYTECODE"],
                "PYTHONNOUSERSITE": env["PYTHONNOUSERSITE"],
                **(
                    {"TREEBENCH_OUTPUT_DIR": env["TREEBENCH_OUTPUT_DIR"]}
                    if args.action == "train"
                    else {}
                ),
            },
            "runtime_config": str(runtime_config),
        }
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0

        completed = subprocess.run(command, cwd=str(repo), env=env, check=False)
        if completed.returncode != 0:
            raise LaunchError(f"Baseline process exited with code {completed.returncode}")
        verify_materialized_repo(repo, baseline, args.model, manifest)
        if args.action == "train":
            checkpoint = record_training_checkpoint(
                output, baseline, registry, args.scenario, runtime_config
            )
            print(f"Recorded checkpoint provenance: {checkpoint_metadata_path(checkpoint)}")
        else:
            aggregate_artifact, per_sample_artifacts, sample_counts, split_sha256 = (
                validate_evaluation_artifacts(output, task, split, registry)
            )
            source_scenario = (
                baseline["cross_rate_16x"]["source_checkpoint"][split]
                if task == "upsampling_16x"
                else args.scenario
            )
            run_manifest = {
                "schema_version": "1.0.0",
                "baseline_id": args.model,
                "scenario": args.scenario,
                "source_training_scenario": source_scenario,
                "upstream_commit": baseline["upstream"]["commit"],
                "checkpoint_sha256": sha256(checkpoint),
                "config_sha256": sha256(runtime_config),
                "taxonomy_csv_sha256": sha256(taxonomy),
                "split_manifest_sha256": split_sha256,
                "cross_rate_route": (
                    baseline["cross_rate_16x"]["route_id"]
                    if task == "upsampling_16x"
                    else None
                ),
                "input_points": registry["task_protocol"][task]["input_points"],
                "output_points": registry["task_protocol"][task]["output_points"],
                "aggregate_csv": aggregate_artifact,
                "per_sample": per_sample_artifacts,
                "sample_counts": sample_counts,
                "metric_contract": {
                    "count": len(METRIC_NAMES),
                    "names": list(METRIC_NAMES),
                    "all_per_sample_values_finite": True,
                },
                "cardinality_contract": {
                    "per_sample_fields_verified": True,
                    "input_points": registry["task_protocol"][task]["input_points"],
                    "output_points": registry["task_protocol"][task]["output_points"],
                },
                "checkpoint_metadata_sha256": (
                    sha256(checkpoint_metadata_path(checkpoint))
                    if checkpoint_metadata is not None
                    else None
                ),
            }
            output.with_suffix(output.suffix + ".run.json").write_text(
                json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
            )
    except (LaunchError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
