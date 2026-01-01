#!/usr/bin/env python3
"""Dependency-free regression tests for the anonymous baseline release layer."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = RELEASE_ROOT / "scripts" / "run_baseline.py"
SPEC = importlib.util.spec_from_file_location("treebench_run_baseline", RUNNER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load run_baseline.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_values(prefix: str = ""):
    return {prefix + name: "0.125" for name in RUNNER.METRIC_NAMES}


class BaselineReleaseContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def sample_row(self, name: str, mode: str, view: int, key: str, input_points: int):
        row = {
            "name": name,
            "mode": mode,
            "view": str(view),
            "sample_key": key,
            "input_points": str(input_points),
            "output_points": "8192",
            "species": "Synthetic species",
            "genus": "Synthetic",
        }
        row.update(metric_values())
        return row

    def test_synthetic_per_sample_and_aggregate_contracts(self):
        fieldnames = (
            list(RUNNER.PER_SAMPLE_IDENTITY_COLUMNS)
            + list(RUNNER.METRIC_NAMES)
            + ["species", "genus"]
        )
        upsampling = self.root / "upsampling.csv"
        write_csv(
            upsampling,
            fieldnames,
            [self.sample_row("00001", "upsampling", -1, "00001", 2048)],
        )
        artifact = RUNNER.validate_per_sample_csv(
            upsampling, "upsampling_4x", "val", ["00001"], 2048, 8192
        )
        self.assertEqual(artifact["rows"], 1)
        self.assertEqual(artifact["unique_sample_keys"], 1)

        completion = self.root / "completion.csv"
        completion_rows = [
            self.sample_row(
                "00001", mode, view, f"00001|{mode}|{view}", 2048
            )
            for mode in ("uls", "tls")
            for view in range(4)
        ]
        write_csv(completion, fieldnames, completion_rows)
        artifact = RUNNER.validate_per_sample_csv(
            completion, "completion", "test", ["00001"], 2048, 8192
        )
        self.assertEqual(artifact["rows"], 8)
        self.assertEqual(artifact["unique_sample_keys"], 8)

        aggregate = self.root / "aggregate.csv"
        aggregate_fields = ["group_type", "group", "n_val", "n_test"] + [
            f"{split}_{metric}"
            for split in ("val", "test")
            for metric in RUNNER.METRIC_NAMES
        ]
        overall = {
            "group_type": "overall",
            "group": "overall",
            "n_val": "8",
            "n_test": "8",
        }
        overall.update(metric_values("val_"))
        overall.update(metric_values("test_"))
        write_csv(aggregate, aggregate_fields, [overall])
        aggregate_artifact = RUNNER.validate_aggregate_csv(
            aggregate, {"val": 8, "test": 8}
        )
        self.assertEqual(aggregate_artifact["overall_finite_metrics_verified"], 32)

    def test_duplicate_and_nonfinite_rows_fail_closed(self):
        fieldnames = (
            list(RUNNER.PER_SAMPLE_IDENTITY_COLUMNS)
            + list(RUNNER.METRIC_NAMES)
            + ["species", "genus"]
        )
        duplicate = self.root / "duplicate.csv"
        row = self.sample_row("00001", "upsampling", -1, "00001", 2048)
        write_csv(duplicate, fieldnames, [row, row])
        with self.assertRaises(RUNNER.LaunchError):
            RUNNER.validate_per_sample_csv(
                duplicate, "upsampling_4x", "val", ["00001"], 2048, 8192
            )

        nonfinite = self.root / "nonfinite.csv"
        row = dict(row)
        row[RUNNER.METRIC_NAMES[0]] = "nan"
        write_csv(nonfinite, fieldnames, [row])
        with self.assertRaises(RUNNER.LaunchError):
            RUNNER.validate_per_sample_csv(
                nonfinite, "upsampling_4x", "val", ["00001"], 2048, 8192
            )

    def test_all_ready_adapters_have_static_schema_closure(self):
        overlay = RELEASE_ROOT / "baselines" / "overlays"
        adapters = {
            "pointr": (
                ["pointr/datasets/TreeBench.py", "pointr/datasets/TreeBench_completion.py"],
                "pointr/utils/treebench_eval.py",
                "pointr/analyze_treebench_taxonomy.py",
            ),
            "adapointr": (
                ["pointr/datasets/TreeBench.py", "pointr/datasets/TreeBench_completion.py"],
                "pointr/utils/treebench_eval.py",
                "pointr/analyze_treebench_taxonomy.py",
            ),
            "snowflakenet": (["snowflakenet/dataset/TreeBench.py"], "snowflakenet/train_treebench_ddp.py", "snowflakenet/analyze_treebench_taxonomy.py"),
            "pucrn": (["pucrn/PointUpsampling/PUCRN/dataset/TreeBench.py"], "pucrn/PointUpsampling/PUCRN/train_treebench_ddp.py", "pucrn/PointUpsampling/PUCRN/analyze_treebench_taxonomy.py"),
            "pu_flow": (["pu_flow/dataset/TreeBench.py"], "pu_flow/train_treebench_ddp.py", "pu_flow/analyze_treebench_taxonomy.py"),
            "svdformer": (["svdformer/dataset/TreeBench.py"], "svdformer/train_treebench_ddp.py", "svdformer/analyze_treebench_taxonomy.py"),
            "repkpu": (["repkpu/dataset/TreeBench.py"], "repkpu/train_TreeBench_ddp.py", "repkpu/analyze_treebench_taxonomy.py"),
            "pudm": (["pudm/pointnet2/dataloader/dataset_loader.py"], "pudm/pointnet2/eval_treebench.py", "pudm/pointnet2/analyze_treebench_taxonomy.py"),
            "fsc": (["fsc/dataset/TreeBench.py"], "fsc/train_treebench_ddp.py", "fsc/analyze_treebench_taxonomy.py"),
            "geoformer": (["geoformer/dataset/TreeBench.py"], "geoformer/src/train_treebench_ddp.py", "geoformer/src/analyze_treebench_taxonomy.py"),
        }
        identity = set(RUNNER.PER_SAMPLE_IDENTITY_COLUMNS)
        for model, (datasets, evaluator, analyzer) in adapters.items():
            dataset_text = "\n".join((overlay / path).read_text(encoding="utf-8") for path in datasets)
            evaluator_text = (overlay / evaluator).read_text(encoding="utf-8")
            analyzer_text = (overlay / analyzer).read_text(encoding="utf-8")
            for column in {"name", "mode", "view", "sample_key"}:
                self.assertIn(repr(column), dataset_text, f"{model} dataset lacks {column}")
            for column in identity:
                self.assertIn(column, evaluator_text, f"{model} evaluator lacks {column}")
                self.assertIn(column, analyzer_text, f"{model} analyzer includes {column} as a metric")
            for metric in RUNNER.METRIC_NAMES:
                self.assertTrue(
                    metric in evaluator_text or "TREE_METRIC_NAMES" in evaluator_text,
                    f"{model} evaluator lacks metric contract {metric}",
                )

    def test_manifest_hashes_statuses_licenses_and_registry_budgets(self):
        manifest = json.loads(
            (RELEASE_ROOT / "baselines" / "overlay_manifest.json").read_text(encoding="utf-8")
        )
        physical = {
            path.relative_to(RELEASE_ROOT).as_posix()
            for path in (RELEASE_ROOT / "baselines" / "overlays").rglob("*")
            if path.is_file()
        }
        registered = {
            item["payload_path"] for item in manifest["files"] if item["materialize"] is True
        }
        self.assertEqual(physical, registered)
        for item in manifest["files"]:
            if item["materialize"] is True:
                path = RELEASE_ROOT / item["payload_path"]
                self.assertFalse(path.is_symlink())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])
            else:
                self.assertIsNone(item["payload_path"])
        self.assertEqual(len(manifest["third_party_licenses"]), 9)
        for item in manifest["third_party_licenses"]:
            path = RELEASE_ROOT / item["bundled_path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])

        registry = json.loads(
            (RELEASE_ROOT / "baselines" / "registry.json").read_text(encoding="utf-8")
        )
        verified = [
            item for item in registry["baselines"]
            if item["training_reproduction"]["status"] == "VERIFIED"
        ]
        opened = [
            item for item in registry["baselines"]
            if item["training_reproduction"]["status"] == "OPEN_COMMAND_PROVENANCE"
        ]
        self.assertEqual(len(verified), 9)
        self.assertTrue(all(item["training_reproduction"]["epochs"] == 10 for item in verified))
        self.assertEqual([item["id"] for item in opened], ["repkpu"])
        self.assertEqual(opened[0]["training_reproduction"]["epochs"], 50)
        self.assertFalse(registry["training_comparison_policy"]["compute_controlled_across_models"])
        self.assertTrue(all("publication" in item for item in registry["baselines"]))

    def test_materialized_checkout_rejects_unregistered_changes(self):
        repo = self.root / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        upstream_url = "https://example.invalid/upstream.git"
        subprocess.run(
            ["git", "-C", str(repo), "remote", "add", "upstream", upstream_url],
            check=True,
        )
        overlay = repo / "adapter.py"
        overlay.write_text("UPSTREAM = True\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "adapter.py"], check=True)
        environment = os.environ.copy()
        identity = "release" + chr(64) + "invalid"
        environment.update(
            {
                "GIT_AUTHOR_NAME": "Release Test",
                "GIT_AUTHOR_EMAIL": identity,
                "GIT_COMMITTER_NAME": "Release Test",
                "GIT_COMMITTER_EMAIL": identity,
            }
        )
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-q", "-m", "base"],
            check=True,
            env=environment,
        )
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        overlay.write_text("TREEBENCH = True\n", encoding="utf-8")
        digest = hashlib.sha256(overlay.read_bytes()).hexdigest()
        provenance = {
            "schema_version": "1.0.0",
            "baseline_id": "synthetic",
            "upstream_url": upstream_url,
            "upstream_commit": commit,
            "registry_sha256": RUNNER.sha256(RUNNER.DEFAULT_REGISTRY),
            "overlay_manifest_sha256": RUNNER.sha256(RUNNER.DEFAULT_MANIFEST),
            "overlay_files": [
                {"destination_path": "adapter.py", "sha256": digest}
            ],
        }
        (repo / ".treebench-overlay.json").write_text(
            json.dumps(provenance), encoding="utf-8"
        )
        baseline = {
            "id": "synthetic",
            "upstream": {"url": upstream_url, "commit": commit},
        }
        manifest = {
            "baseline_status": {
                "synthetic": {"status": "READY", "missing_required_files": []}
            },
            "files": [
                {
                    "model_ids": ["synthetic"],
                    "destination_path": "adapter.py",
                    "sha256": digest,
                    "materialize": True,
                }
            ],
        }
        RUNNER.verify_materialized_repo(repo, baseline, "synthetic", manifest)
        (repo / "unregistered.txt").write_text("not allowlisted\n", encoding="utf-8")
        with self.assertRaises(RUNNER.LaunchError):
            RUNNER.verify_materialized_repo(repo, baseline, "synthetic", manifest)

    def test_cross_rate_checkpoint_requires_split_matched_provenance(self):
        checkpoint = self.root / "ckpt-best.pth"
        checkpoint.write_bytes(b"synthetic checkpoint")
        baseline = {
            "id": "synthetic",
            "upstream": {"commit": "a" * 40},
            "cross_rate_16x": {
                "source_checkpoint": {
                    "random": "upsampling_4x_random",
                    "genus_ood": "upsampling_4x_genus_ood",
                }
            },
            "training_reproduction": {
                "epochs": 10,
                "batch_size": {"value": 1, "scope": "per_gpu"},
                "learning_rate": 0.001,
            },
        }
        metadata = {
            "schema_version": "1.0.0",
            "baseline_id": "synthetic",
            "training_scenario": "upsampling_4x_random",
            "upstream_commit": "a" * 40,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "runtime_config_sha256": "b" * 64,
            "epochs": 10,
            "batch_size": {"value": 1, "scope": "per_gpu"},
            "learning_rate": 0.001,
            "compute_controlled_across_models": False,
        }
        RUNNER.checkpoint_metadata_path(checkpoint).write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        RUNNER.verify_cross_rate_checkpoint(checkpoint, baseline, "random")
        with self.assertRaises(RUNNER.LaunchError):
            RUNNER.verify_cross_rate_checkpoint(checkpoint, baseline, "genus_ood")

    def test_launcher_consumes_registered_training_values(self):
        registry = json.loads(
            (RELEASE_ROOT / "baselines" / "registry.json").read_text(encoding="utf-8")
        )
        entries = {item["id"]: item for item in registry["baselines"]}
        overlay = RELEASE_ROOT / "baselines" / "overlays"
        roots = {
            "snowflakenet": overlay / "snowflakenet",
            "pucrn": overlay / "pucrn",
            "pu_flow": overlay / "pu_flow",
            "svdformer": overlay / "svdformer",
            "fsc": overlay / "fsc",
            "geoformer": overlay / "geoformer",
        }
        for model, repo in roots.items():
            baseline = entries[model]
            reproduction = baseline["training_reproduction"]
            entrypoint = baseline["entrypoints"]["upsampling_4x"]
            command = RUNNER.command_for(
                baseline,
                entrypoint,
                "upsampling_4x",
                "train",
                repo,
                self.root / "runtime.json",
                None,
                None,
                self.root / "output",
                reproduction["gpu_count"],
                "python",
            )
            self.assertEqual(command[2], reproduction["distributed_launcher"]["module"])
            for option, expected in (
                ("--epochs", reproduction["epochs"]),
                ("--batch_size", reproduction["batch_size"]["value"]),
                ("--lr", reproduction["learning_rate"]),
            ):
                index = command.index(option)
                self.assertEqual(command[index + 1], str(expected), f"{model} {option}")

        pointr_repo = overlay / "pointr"
        for model in ("pointr", "adapointr"):
            baseline = entries[model]
            entrypoint = baseline["entrypoints"]["upsampling_4x"]
            source = pointr_repo / entrypoint["config"].replace("{split}", "random")
            destination = self.root / model / "runtime.yaml"
            RUNNER.rewrite_yaml_configs(
                pointr_repo,
                source,
                destination,
                "upsampling_4x_random",
                self.root / "data",
                self.root / "split.json",
                baseline,
                "train",
            )
            text = destination.read_text(encoding="utf-8")
            reproduction = baseline["training_reproduction"]
            self.assertIn(f"max_epoch : {reproduction['epoch_encoding']['value']}", text)
            self.assertIn(f"total_bs : {reproduction['batch_size']['value']}", text)
            self.assertIn(f"lr : {reproduction['learning_rate']},", text)

        pudm = entries["pudm"]
        source = overlay / "pudm" / "pointnet2" / "exp_configs" / "TreeBench_upsampling_4x_random.json"
        destination = self.root / "pudm-runtime.json"
        config = RUNNER.rewrite_json_config(
            source,
            destination,
            self.root / "data",
            self.root / "split.json",
            self.root / "pudm-output",
            pudm,
            "train",
        )
        reproduction = pudm["training_reproduction"]
        self.assertEqual(config["train_config"]["n_epochs"], reproduction["epochs"])
        self.assertEqual(config["dataset_config"]["batch_size"], reproduction["batch_size"]["value"])
        self.assertEqual(config["train_config"]["learning_rate"], reproduction["learning_rate"])

    def test_ready_entrypoints_configs_and_evaluation_arguments_are_closed(self):
        registry = json.loads(
            (RELEASE_ROOT / "baselines" / "registry.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (RELEASE_ROOT / "baselines" / "overlay_manifest.json").read_text(encoding="utf-8")
        )
        roots = {
            "pointr": "pointr",
            "adapointr": "pointr",
            "snowflakenet": "snowflakenet",
            "pucrn": "pucrn",
            "pu_flow": "pu_flow",
            "svdformer": "svdformer",
            "repkpu": "repkpu",
            "pudm": "pudm",
            "fsc": "fsc",
            "geoformer": "geoformer",
        }
        entries = {item["id"]: item for item in registry["baselines"]}
        for model, overlay_id in roots.items():
            baseline = entries[model]
            registered_destinations = {
                item["destination_path"]
                for item in manifest["files"]
                if item["materialize"] is True and model in item["model_ids"]
            }
            repo = RELEASE_ROOT / "baselines" / "overlays" / overlay_id
            for scenario in baseline["evaluation_scenarios"]:
                task, split = RUNNER.parse_scenario(scenario)
                entrypoint = baseline["entrypoints"][task]
                config = entrypoint["config"].replace("{split}", split)
                self.assertIn(entrypoint["evaluate"], registered_destinations)
                self.assertIn(config, registered_destinations)
                self.assertTrue((repo / entrypoint["evaluate"]).is_file())
                self.assertTrue((repo / config).is_file())
                command = RUNNER.command_for(
                    baseline,
                    entrypoint,
                    task,
                    "evaluate",
                    repo,
                    self.root / "runtime-config",
                    self.root / "checkpoint",
                    self.root / "taxonomy.csv",
                    self.root / "aggregate.csv",
                    1,
                    "python",
                )
                batch_index = command.index("--batch-size")
                worker_index = command.index("--num-workers")
                self.assertEqual(
                    command[batch_index + 1],
                    str(baseline["evaluation_reproduction"]["batch_size"][task]),
                )
                self.assertEqual(
                    command[worker_index + 1],
                    str(baseline["evaluation_reproduction"]["dataloader_workers"]),
                )
            for scenario in baseline["trainable_scenarios"]:
                task, _ = RUNNER.parse_scenario(scenario)
                train_path = baseline["entrypoints"][task]["train"]
                self.assertIn(train_path, registered_destinations)
                self.assertTrue((repo / train_path).is_file())


if __name__ == "__main__":
    unittest.main()
