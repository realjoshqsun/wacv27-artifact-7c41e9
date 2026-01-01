#!/usr/bin/env python3
"""Build audited TreeBench result tables from the canonical CSV layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence


RELEASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = RELEASE_ROOT / "baselines" / "registry.json"
DEFAULT_PROTOCOL = RELEASE_ROOT / "configs" / "protocol.json"
DEFAULT_REFERENCE_RESULTS = RELEASE_ROOT / "reference_results" / "full_results.csv"
DEFAULT_REFERENCE_MANIFEST = RELEASE_ROOT / "reference_results" / "manifest.json"

EXPECTED_BASELINES = 16
EXPECTED_METHOD_SCENARIOS = 94
SCENARIO_ORDER = (
    "completion_random",
    "completion_genus_ood",
    "upsampling_4x_random",
    "upsampling_4x_genus_ood",
    "upsampling_16x_random",
    "upsampling_16x_genus_ood",
)
TASK_SCENARIOS = {
    "completion": ("completion_random", "completion_genus_ood"),
    "upsampling_4x": ("upsampling_4x_random", "upsampling_4x_genus_ood"),
    "upsampling_16x": ("upsampling_16x_random", "upsampling_16x_genus_ood"),
}
MAIN_METRICS = (
    "cd_l1",
    "f1",
    "crown_width_abs_error",
    "crown_width_relative_error",
    "relative_height_95_abs_error",
    "foliage_height_diversity_abs_error",
    "tree_height_abs_error",
    "tree_height_relative_error",
    "stratum_percentage_cover_l1",
    "fuel_layer_count_abs_error",
    "largest_vertical_gap_abs_error",
)
MAIN_LABELS = {
    "cd_l1": "CD-L1",
    "f1": "F1",
    "crown_width_abs_error": "W-E",
    "crown_width_relative_error": "W-RE",
    "relative_height_95_abs_error": "RH95-E",
    "foliage_height_diversity_abs_error": "FHD-E",
    "tree_height_abs_error": "CH-E",
    "tree_height_relative_error": "CH-RE",
    "stratum_percentage_cover_l1": "SPC-E",
    "fuel_layer_count_abs_error": "NFL-E",
    "largest_vertical_gap_abs_error": "LVG-E",
}
TASK_CAPTIONS = {
    "completion": "TreeBench completion test results.",
    "upsampling_4x": "TreeBench 4$\\times$ upsampling test results.",
    "upsampling_16x": "TreeBench 16$\\times$ upsampling test results.",
}
SPLIT_LABELS = {"random": "Random", "genus_ood": "Genus-OOD"}


class TableBuildError(ValueError):
    """Raised when canonical result inputs fail closed validation."""


def _load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise TableBuildError(f"cannot read {label} JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise TableBuildError(f"{label} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_registry(registry: dict) -> list[dict]:
    baselines = registry.get("baselines")
    if not isinstance(baselines, list):
        raise TableBuildError("registry baselines must be a list")
    declared_count = registry.get("baseline_count")
    if len(baselines) != EXPECTED_BASELINES or declared_count != EXPECTED_BASELINES:
        raise TableBuildError(
            f"registry must declare exactly {EXPECTED_BASELINES} baselines"
        )

    identifiers = []
    combinations = 0
    allowed_scenarios = set(SCENARIO_ORDER)
    for baseline in baselines:
        if not isinstance(baseline, dict):
            raise TableBuildError("each registry baseline must be an object")
        identifier = baseline.get("id")
        display_name = baseline.get("display_name")
        scenarios = baseline.get("evaluation_scenarios")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in (".", "..")
            or "/" in identifier
            or "\\" in identifier
            or Path(identifier).name != identifier
        ):
            raise TableBuildError("every baseline id must be a safe path component")
        if not isinstance(display_name, str) or not display_name:
            raise TableBuildError(f"baseline {identifier} has no display_name")
        if not isinstance(scenarios, list) or not all(
            isinstance(value, str) for value in scenarios
        ):
            raise TableBuildError(
                f"baseline {identifier} evaluation_scenarios must be a string list"
            )
        if len(scenarios) != len(set(scenarios)):
            raise TableBuildError(
                f"baseline {identifier} has duplicate evaluation scenarios"
            )
        unknown = set(scenarios) - allowed_scenarios
        if unknown:
            raise TableBuildError(
                f"baseline {identifier} has unsupported scenarios: {sorted(unknown)}"
            )
        identifiers.append(identifier)
        combinations += len(scenarios)

    if len(identifiers) != len(set(identifiers)):
        raise TableBuildError("registry contains duplicate baseline ids")
    declared_rows = registry.get("expected_method_scenario_rows")
    if (
        combinations != EXPECTED_METHOD_SCENARIOS
        or declared_rows != EXPECTED_METHOD_SCENARIOS
    ):
        raise TableBuildError(
            f"registry must define exactly {EXPECTED_METHOD_SCENARIOS} "
            "method-scenario combinations"
        )
    return baselines


def _validate_protocol(protocol: dict) -> tuple[tuple[str, ...], dict[str, str]]:
    metrics = protocol.get("metrics")
    if not isinstance(metrics, list) or len(metrics) != 32:
        raise TableBuildError("protocol must define exactly 32 metrics")
    names = []
    directions = {}
    for metric in metrics:
        if not isinstance(metric, dict):
            raise TableBuildError("every protocol metric must be an object")
        name, direction = metric.get("name"), metric.get("direction")
        if not isinstance(name, str) or direction not in ("lower", "higher"):
            raise TableBuildError("protocol metrics require name and direction")
        names.append(name)
        directions[name] = direction
    if len(names) != len(set(names)):
        raise TableBuildError("protocol contains duplicate metric names")
    missing_main = set(MAIN_METRICS) - set(names)
    if missing_main:
        raise TableBuildError(f"protocol omits main metrics: {sorted(missing_main)}")
    if directions["f1"] != "higher" or any(
        directions[name] != "lower" for name in MAIN_METRICS if name != "f1"
    ):
        raise TableBuildError("protocol main-metric directions are inconsistent")
    return tuple(names), directions


def _expected_pairs(baselines: Sequence[dict]) -> list[tuple[dict, str]]:
    ordered = []
    for scenario in SCENARIO_ORDER:
        for baseline in baselines:
            if scenario in baseline["evaluation_scenarios"]:
                ordered.append((baseline, scenario))
    if len(ordered) != EXPECTED_METHOD_SCENARIOS:
        raise TableBuildError("ordered registry expansion did not produce 94 rows")
    return ordered


def _validate_file_layout(
    results_root: Path, expected_pairs: Sequence[tuple[dict, str]]
) -> None:
    if not results_root.is_dir():
        raise TableBuildError(f"results directory does not exist: {results_root}")
    symlinks = sorted(
        path.relative_to(results_root).as_posix()
        for path in results_root.rglob("*")
        if path.is_symlink()
    )
    if symlinks:
        raise TableBuildError(f"result symlinks are forbidden: {symlinks[:5]}")
    expected = set()
    for baseline, scenario in expected_pairs:
        aggregate = Path(baseline["id"]) / f"{scenario}.csv"
        expected.update(
            {
                aggregate.as_posix(),
                aggregate.with_name(f"{aggregate.stem}_val_per_sample.csv").as_posix(),
                aggregate.with_name(f"{aggregate.stem}_test_per_sample.csv").as_posix(),
                aggregate.with_suffix(aggregate.suffix + ".run.json").as_posix(),
            }
        )
    actual = {
        path.relative_to(results_root).as_posix()
        for path in results_root.rglob("*")
        if path.is_file()
    }
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    messages = []
    if missing:
        messages.append(f"missing {len(missing)} result files: {missing[:5]}")
    if unexpected:
        messages.append(
            f"unexpected {len(unexpected)} result files: {unexpected[:5]}"
        )
    if messages:
        raise TableBuildError("; ".join(messages))


def _normalized_headers(
    fieldnames: Sequence[Optional[str]], path: Path
) -> dict[str, str]:
    if not fieldnames or any(name is None for name in fieldnames):
        raise TableBuildError(f"{path} has no valid CSV header")
    normalized = [str(name).strip() for name in fieldnames]
    if any(not name for name in normalized):
        raise TableBuildError(f"{path} contains an empty column name")
    if len(normalized) != len(set(normalized)):
        raise TableBuildError(f"{path} contains duplicate column names")
    return dict(zip(normalized, (str(name) for name in fieldnames)))


def _select_overall_row(
    rows: Sequence[dict[str, str]], headers: Mapping[str, str], path: Path
) -> dict[str, str]:
    if not rows:
        raise TableBuildError(f"{path} contains no result rows")
    if "group_type" in headers:
        key = headers["group_type"]
        matches = [
            row for row in rows if str(row.get(key, "")).strip().lower() == "overall"
        ]
        if len(matches) != 1:
            raise TableBuildError(
                f"{path} must contain exactly one group_type=overall row"
            )
        return matches[0]
    if len(rows) == 1:
        return rows[0]
    if "group" in headers:
        key = headers["group"]
        matches = [
            row for row in rows if str(row.get(key, "")).strip().lower() == "overall"
        ]
        if len(matches) == 1:
            return matches[0]
    raise TableBuildError(
        f"{path} must be a single-row file or contain exactly one overall row"
    )


def _metric_candidates(metric: str) -> tuple[str, ...]:
    names = (metric, f"test_{metric}")
    if metric == "mhd2":
        names += ("hd", "test_hd")
    return names


def _positive_integer(
    overall: Mapping[str, str], headers: Mapping[str, str], name: str, path: Path
) -> int:
    if name not in headers:
        raise TableBuildError(f"{path} is missing required sample count {name}")
    raw = str(overall.get(headers[name], "")).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise TableBuildError(f"{path} has a non-integer {name}: {raw!r}") from error
    if value <= 0:
        raise TableBuildError(f"{path} requires a positive {name}")
    return value


def _read_metric_row(
    path: Path,
    metric_names: Sequence[str],
    *,
    expected_val: int,
    expected_test: int,
) -> dict[str, float]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            headers = _normalized_headers(reader.fieldnames or (), path)
            rows = list(reader)
    except (OSError, csv.Error) as error:
        raise TableBuildError(f"cannot read result CSV {path}: {error}") from error
    if any(None in row for row in rows):
        raise TableBuildError(f"{path} has rows wider than its header")
    overall = _select_overall_row(rows, headers, path)
    actual_val = _positive_integer(overall, headers, "n_val", path)
    actual_test = _positive_integer(overall, headers, "n_test", path)
    if (actual_val, actual_test) != (expected_val, expected_test):
        raise TableBuildError(
            f"{path} reports val/test counts {(actual_val, actual_test)}; "
            f"expected {(expected_val, expected_test)}"
        )

    values = {}
    for metric in metric_names:
        candidates = [name for name in _metric_candidates(metric) if name in headers]
        if not candidates:
            raise TableBuildError(f"{path} is missing metric {metric}")
        if len(candidates) != 1:
            raise TableBuildError(
                f"{path} has duplicate aliases for metric {metric}: {candidates}"
            )
        column = headers[candidates[0]]
        raw = str(overall.get(column, "")).strip()
        if not raw:
            raise TableBuildError(f"{path} has an empty value for metric {metric}")
        try:
            value = float(raw)
        except ValueError as error:
            raise TableBuildError(
                f"{path} has a non-numeric value for metric {metric}: {raw!r}"
            ) from error
        if not math.isfinite(value):
            raise TableBuildError(
                f"{path} has a non-finite value for metric {metric}"
            )
        values[metric] = value
    return values


def _count_csv_rows(path: Path) -> int:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                raise TableBuildError(f"{path} has no CSV header")
            return sum(1 for _ in reader)
    except (OSError, csv.Error) as error:
        raise TableBuildError(f"cannot inspect per-sample CSV {path}: {error}") from error


def _artifact_record(
    record: object,
    path: Path,
    *,
    expected_rows: Optional[int] = None,
) -> None:
    if not isinstance(record, dict):
        raise TableBuildError(f"run manifest has no artifact record for {path.name}")
    if record.get("path") != path.name:
        raise TableBuildError(f"run manifest artifact identity mismatch for {path.name}")
    if path.is_symlink() or not path.is_file():
        raise TableBuildError(f"run artifact is missing or not regular: {path}")
    if record.get("bytes") != path.stat().st_size:
        raise TableBuildError(f"run artifact byte count mismatch: {path}")
    if record.get("sha256") != _sha256(path):
        raise TableBuildError(f"run artifact SHA-256 mismatch: {path}")
    if expected_rows is not None:
        if record.get("rows") != expected_rows:
            raise TableBuildError(f"run artifact row count mismatch: {path}")
        if _count_csv_rows(path) != expected_rows:
            raise TableBuildError(f"per-sample CSV row count mismatch: {path}")


def _validate_run_manifest(
    aggregate_path: Path,
    *,
    baseline: Mapping[str, object],
    scenario: str,
    task: str,
    split: str,
    expected_val: int,
    expected_test: int,
    task_protocol: Mapping[str, object],
) -> None:
    run_path = aggregate_path.with_suffix(aggregate_path.suffix + ".run.json")
    manifest = _load_json(run_path, "run manifest")
    if manifest.get("schema_version") != "1.0.0":
        raise TableBuildError(f"unsupported run-manifest schema in {run_path}")
    if manifest.get("baseline_id") != baseline["id"]:
        raise TableBuildError(f"baseline identity mismatch in {run_path}")
    if manifest.get("scenario") != scenario:
        raise TableBuildError(f"scenario identity mismatch in {run_path}")
    if manifest.get("input_points") != task_protocol.get("input_points") or manifest.get(
        "output_points"
    ) != task_protocol.get("output_points"):
        raise TableBuildError(f"point-cardinality provenance mismatch in {run_path}")

    expected_source = scenario
    expected_route = None
    if task == "upsampling_16x":
        route = baseline.get("cross_rate_16x")
        if not isinstance(route, dict):
            raise TableBuildError(f"16x route is missing for {baseline['id']}")
        expected_source = route.get("source_checkpoint", {}).get(split)
        expected_route = route.get("route_id")
    if manifest.get("source_training_scenario") != expected_source:
        raise TableBuildError(f"source-checkpoint identity mismatch in {run_path}")
    if manifest.get("cross_rate_route") != expected_route:
        raise TableBuildError(f"cross-rate route mismatch in {run_path}")
    for field in ("checkpoint_sha256", "config_sha256"):
        value = manifest.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise TableBuildError(f"invalid {field} in {run_path}")

    counts = manifest.get("sample_counts")
    if counts != {"val": expected_val, "test": expected_test}:
        raise TableBuildError(f"sample-count provenance mismatch in {run_path}")
    _artifact_record(manifest.get("aggregate_csv"), aggregate_path)
    per_sample = manifest.get("per_sample")
    if not isinstance(per_sample, dict):
        raise TableBuildError(f"per-sample provenance is missing in {run_path}")
    _artifact_record(
        per_sample.get("val"),
        aggregate_path.with_name(f"{aggregate_path.stem}_val_per_sample.csv"),
        expected_rows=expected_val,
    )
    _artifact_record(
        per_sample.get("test"),
        aggregate_path.with_name(f"{aggregate_path.stem}_test_per_sample.csv"),
        expected_rows=expected_test,
    )


def _scenario_parts(scenario: str) -> tuple[str, str]:
    for task, scenarios in TASK_SCENARIOS.items():
        if scenario in scenarios:
            split = "genus_ood" if scenario.endswith("genus_ood") else "random"
            return task, split
    raise TableBuildError(f"unsupported scenario: {scenario}")


def collect_results(
    results_root: Path,
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> tuple[list[dict], tuple[str, ...], dict[str, str]]:
    """Validate and load all 94 canonical method-scenario result files."""
    registry = _load_json(registry_path, "registry")
    protocol = _load_json(protocol_path, "protocol")
    baselines = _validate_registry(registry)
    metric_names, directions = _validate_protocol(protocol)
    pairs = _expected_pairs(baselines)
    results_root = Path(results_root)
    _validate_file_layout(results_root, pairs)

    output = []
    for baseline, scenario in pairs:
        task, split = _scenario_parts(scenario)
        relative_path = Path(baseline["id"]) / f"{scenario}.csv"
        counts = protocol.get("splits", {}).get(split, {}).get("counts", {})
        factor = 2 * 4 if task == "completion" else 1
        try:
            expected_val = int(counts["validation"]) * factor
            expected_test = int(counts["test"]) * factor
        except (KeyError, TypeError, ValueError) as error:
            raise TableBuildError(
                f"protocol has no valid val/test counts for split {split}"
            ) from error
        task_protocol = registry.get("task_protocol", {}).get(task)
        if not isinstance(task_protocol, dict):
            raise TableBuildError(f"registry has no task protocol for {task}")
        aggregate_path = results_root / relative_path
        values = _read_metric_row(
            aggregate_path,
            metric_names,
            expected_val=expected_val,
            expected_test=expected_test,
        )
        _validate_run_manifest(
            aggregate_path,
            baseline=baseline,
            scenario=scenario,
            task=task,
            split=split,
            expected_val=expected_val,
            expected_test=expected_test,
            task_protocol=task_protocol,
        )
        output.append(
            {
                "model_id": baseline["id"],
                "model": baseline["display_name"],
                "scenario": scenario,
                "task": task,
                "split": split,
                "metrics": values,
            }
        )
    return output, metric_names, directions


def collect_reference_results(
    results_path: Path = DEFAULT_REFERENCE_RESULTS,
    *,
    manifest_path: Path = DEFAULT_REFERENCE_MANIFEST,
    registry_path: Path = DEFAULT_REGISTRY,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> tuple[list[dict], tuple[str, ...], dict[str, str]]:
    """Load the frozen test-only paper means after verifying their release hash.

    This path rebuilds tables from the numerical aggregates shipped for review;
    it is deliberately separate from :func:`collect_results`, which requires
    complete per-sample outputs and run provenance from a fresh model run.
    """
    results_path = Path(results_path)
    manifest_path = Path(manifest_path)
    if results_path.is_symlink() or not results_path.is_file():
        raise TableBuildError(
            f"reference results are missing or not a regular file: {results_path}"
        )
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise TableBuildError(
            f"reference manifest is missing or not a regular file: {manifest_path}"
        )

    manifest = _load_json(manifest_path, "reference-results manifest")
    artifact = manifest.get("artifact")
    if (
        manifest.get("schema_version") != "1.0.0"
        or manifest.get("status") != "FROZEN_PAPER_TEST_AGGREGATES"
        or not isinstance(artifact, dict)
    ):
        raise TableBuildError(f"unsupported reference-results manifest: {manifest_path}")
    expected_artifact = {
        "path": results_path.name,
        "bytes": results_path.stat().st_size,
        "sha256": _sha256(results_path),
        "rows": EXPECTED_METHOD_SCENARIOS,
    }
    if artifact != expected_artifact:
        raise TableBuildError(
            f"reference-results artifact does not match {manifest_path}"
        )

    registry = _load_json(registry_path, "registry")
    protocol = _load_json(protocol_path, "protocol")
    baselines = _validate_registry(registry)
    metric_names, directions = _validate_protocol(protocol)
    pairs = _expected_pairs(baselines)
    expected_header = [
        "model_id",
        "model",
        "scenario",
        "task",
        "split",
        "n_test",
        *metric_names,
    ]
    try:
        with results_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            headers = list(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, csv.Error) as error:
        raise TableBuildError(
            f"cannot read frozen reference results {results_path}: {error}"
        ) from error
    if headers != expected_header or len(headers) != len(set(headers)):
        raise TableBuildError(
            f"reference-results header does not match the frozen schema: {results_path}"
        )
    if len(rows) != EXPECTED_METHOD_SCENARIOS or any(None in row for row in rows):
        raise TableBuildError(
            f"reference results must contain exactly {EXPECTED_METHOD_SCENARIOS} rows"
        )

    output = []
    for row_index, (row, (baseline, scenario)) in enumerate(zip(rows, pairs), start=2):
        task, split = _scenario_parts(scenario)
        identities = {
            "model_id": baseline["id"],
            "model": baseline["display_name"],
            "scenario": scenario,
            "task": task,
            "split": split,
        }
        for name, expected in identities.items():
            if row.get(name) != expected:
                raise TableBuildError(
                    f"reference identity mismatch at {results_path}:{row_index}: {name}"
                )
        factor = 8 if task == "completion" else 1
        try:
            expected_test = int(protocol["splits"][split]["counts"]["test"]) * factor
            actual_test = int(row["n_test"])
        except (KeyError, TypeError, ValueError) as error:
            raise TableBuildError(
                f"invalid n_test at {results_path}:{row_index}"
            ) from error
        if actual_test != expected_test:
            raise TableBuildError(
                f"reference n_test mismatch at {results_path}:{row_index}: "
                f"{actual_test} != {expected_test}"
            )
        values = {}
        for metric in metric_names:
            raw = row.get(metric, "")
            try:
                value = float(raw)
            except (TypeError, ValueError) as error:
                raise TableBuildError(
                    f"non-numeric {metric} at {results_path}:{row_index}"
                ) from error
            if not math.isfinite(value):
                raise TableBuildError(
                    f"non-finite {metric} at {results_path}:{row_index}"
                )
            values[metric] = value
        output.append({**identities, "metrics": values})
    return output, metric_names, directions


def _write_csv(path: Path, rows: Sequence[dict], metric_names: Sequence[str]) -> None:
    fieldnames = ["model_id", "model", "scenario", "task", "split", *metric_names]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output = {name: row[name] for name in fieldnames[:5]}
            output.update(
                {name: repr(float(row["metrics"][name])) for name in metric_names}
            )
            writer.writerow(output)


def _latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _format_main_value(metric: str, value: float) -> str:
    if metric == "cd_l1":
        return f"{value:.4f}"
    if metric == "f1":
        return f"{value:.3f}"
    return f"{value:.4f}" if abs(value) < 0.1 else f"{value:.3f}"


def _ranked_values(
    rows: Sequence[dict], metric: str, direction: str
) -> tuple[float, Optional[float]]:
    values = sorted(
        {float(row["metrics"][metric]) for row in rows},
        reverse=direction == "higher",
    )
    return values[0], values[1] if len(values) > 1 else None


def _latex_table(task: str, rows: Sequence[dict], directions: Mapping[str, str]) -> str:
    lines = [
        "% Generated by scripts/build_paper_tables.py; do not edit by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\tiny",
        r"\renewcommand{\arraystretch}{0.90}",
        r"\setlength{\tabcolsep}{1.35pt}",
        rf"\caption{{{TASK_CAPTIONS[task]}}}",
        rf"\label{{tab:treebench_generated_{task}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrr|rrrr|rrrrr}",
        r"\toprule",
        r"& & \multicolumn{2}{c|}{Point} & \multicolumn{4}{c|}{Structural} & \multicolumn{5}{c}{Canopy-fuel}\\",
    ]
    header = ["Split", "Method"] + [
        f"{MAIN_LABELS[name]}$\\{'uparrow' if directions[name] == 'higher' else 'downarrow'}$"
        for name in MAIN_METRICS
    ]
    lines.append(" & ".join(header) + r"\\")
    lines.append(r"\midrule")

    task_rows = [row for row in rows if row["task"] == task]
    for split_index, split in enumerate(("random", "genus_ood")):
        split_rows = [row for row in task_rows if row["split"] == split]
        if not split_rows:
            raise TableBuildError(f"no rows available for {task} {split}")
        ranks = {
            metric: _ranked_values(split_rows, metric, directions[metric])
            for metric in MAIN_METRICS
        }
        for row_index, row in enumerate(split_rows):
            cells = [SPLIT_LABELS[split] if row_index == 0 else ""]
            cells.append(_latex_escape(row["model"]))
            for metric in MAIN_METRICS:
                value = float(row["metrics"][metric])
                rendered = _format_main_value(metric, value)
                best, second = ranks[metric]
                if value == best:
                    rendered = rf"\textbf{{{rendered}}}"
                elif second is not None and value == second:
                    rendered = rf"\underline{{{rendered}}}"
                cells.append(rendered)
            lines.append(" & ".join(cells) + r"\\")
        if split_index == 0:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _write_tables(
    rows: Sequence[dict],
    metric_names: Sequence[str],
    directions: Mapping[str, str],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise TableBuildError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "full_results": output_dir / "full_results.csv",
        "main_metrics": output_dir / "main_metrics.csv",
    }
    _write_csv(paths["full_results"], rows, metric_names)
    _write_csv(paths["main_metrics"], rows, MAIN_METRICS)
    for task in TASK_SCENARIOS:
        path = output_dir / f"{task}.tex"
        path.write_text(_latex_table(task, rows, directions), encoding="utf-8")
        paths[task] = path
    return paths


def build_tables(
    results_root: Path,
    output_dir: Path,
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> dict[str, Path]:
    """Validate a fresh run bundle before creating any output."""
    rows, metric_names, directions = collect_results(
        results_root,
        registry_path=registry_path,
        protocol_path=protocol_path,
    )
    return _write_tables(rows, metric_names, directions, output_dir)


def build_reference_tables(
    results_path: Path,
    output_dir: Path,
    *,
    manifest_path: Path = DEFAULT_REFERENCE_MANIFEST,
    registry_path: Path = DEFAULT_REGISTRY,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> dict[str, Path]:
    """Rebuild tables from the hashed test-only aggregates shipped for review."""
    rows, metric_names, directions = collect_reference_results(
        results_path,
        manifest_path=manifest_path,
        registry_path=registry_path,
        protocol_path=protocol_path,
    )
    return _write_tables(rows, metric_names, directions, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--results",
        type=Path,
        help="fresh results/<model>/<scenario> run-bundle directory",
    )
    inputs.add_argument(
        "--reference-results",
        type=Path,
        help="frozen paper test-aggregate CSV shipped in reference_results",
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=DEFAULT_REFERENCE_MANIFEST,
        help="hash manifest for --reference-results",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.reference_results is not None:
            paths = build_reference_tables(
                args.reference_results,
                args.output,
                manifest_path=args.reference_manifest,
                registry_path=args.registry,
                protocol_path=args.protocol,
            )
        else:
            paths = build_tables(
                args.results,
                args.output,
                registry_path=args.registry,
                protocol_path=args.protocol,
            )
    except TableBuildError as error:
        parser.error(str(error))
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
