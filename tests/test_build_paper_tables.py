import csv
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest


RELEASE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = RELEASE_ROOT / "scripts" / "build_paper_tables.py"
SPEC = importlib.util.spec_from_file_location("build_paper_tables", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
TABLES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TABLES)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_values(
    metric_names: list[str], model_index: int, scenario_index: int
) -> dict[str, float]:
    values = {}
    for metric_index, metric in enumerate(metric_names):
        base = (
            0.01 * (model_index + 1)
            + 0.001 * (scenario_index + 1)
            + 0.00001 * (metric_index + 1)
        )
        values[metric] = 1.0 - base if metric == "f1" else base
    return values


def _write_complete_fixture(
    results_root: Path,
) -> tuple[dict[tuple[str, str], Path], Path]:
    registry = _load_json(RELEASE_ROOT / "baselines" / "registry.json")
    protocol = _load_json(RELEASE_ROOT / "configs" / "protocol.json")
    for split in ("random", "genus_ood"):
        protocol["splits"][split]["counts"]["validation"] = 1
        protocol["splits"][split]["counts"]["test"] = 1
    protocol_path = results_root.parent / "fixture_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    metric_names = [metric["name"] for metric in protocol["metrics"]]
    paths = {}
    for model_index, baseline in enumerate(registry["baselines"]):
        for scenario_index, scenario in enumerate(baseline["evaluation_scenarios"]):
            path = results_root / baseline["id"] / f"{scenario}.csv"
            values = _fixture_values(metric_names, model_index, scenario_index)
            if baseline["id"] == "pu_gcn" and scenario == "completion_random":
                # Exercise the native taxonomy aggregate format and legacy HD alias.
                columns = [
                    "group_type",
                    "group",
                    "n_val",
                    "n_test",
                    *[
                        "test_hd" if metric == "mhd2" else f"test_{metric}"
                        for metric in metric_names
                    ],
                ]
                overall = {
                    "group_type": "overall",
                    "group": "overall",
                    "n_val": "8",
                    "n_test": "8",
                }
                genus = {
                    "group_type": "genus",
                    "group": "FixtureGenus",
                    "n_val": "8",
                    "n_test": "1",
                }
                for metric, value in values.items():
                    column = "test_hd" if metric == "mhd2" else f"test_{metric}"
                    overall[column] = repr(value)
                    genus[column] = repr(value + 0.5)
                _write_csv(path, columns, [genus, overall])
            else:
                _write_csv(
                    path,
                    ["n_val", "n_test", *metric_names],
                    [
                        {
                            "n_val": "8" if scenario.startswith("completion") else "1",
                            "n_test": "8" if scenario.startswith("completion") else "1",
                            **{metric: repr(value) for metric, value in values.items()},
                        }
                    ],
                )
            task, split = TABLES._scenario_parts(scenario)
            expected_rows = 8 if task == "completion" else 1
            val_path = path.with_name(f"{path.stem}_val_per_sample.csv")
            test_path = path.with_name(f"{path.stem}_test_per_sample.csv")
            _write_csv(
                val_path,
                ["name"],
                [{"name": f"val-{index}"} for index in range(expected_rows)],
            )
            _write_csv(
                test_path,
                ["name"],
                [{"name": f"test-{index}"} for index in range(expected_rows)],
            )
            task_protocol = registry["task_protocol"][task]
            cross_rate = baseline["cross_rate_16x"] if task == "upsampling_16x" else None
            run_manifest = {
                "schema_version": "1.0.0",
                "baseline_id": baseline["id"],
                "scenario": scenario,
                "source_training_scenario": (
                    cross_rate["source_checkpoint"][split] if cross_rate else scenario
                ),
                "checkpoint_sha256": "a" * 64,
                "config_sha256": "b" * 64,
                "cross_rate_route": cross_rate["route_id"] if cross_rate else None,
                "input_points": task_protocol["input_points"],
                "output_points": task_protocol["output_points"],
                "sample_counts": {"val": expected_rows, "test": expected_rows},
                "aggregate_csv": {
                    "path": path.name,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                },
                "per_sample": {
                    "val": {
                        "path": val_path.name,
                        "sha256": _sha256(val_path),
                        "bytes": val_path.stat().st_size,
                        "rows": expected_rows,
                    },
                    "test": {
                        "path": test_path.name,
                        "sha256": _sha256(test_path),
                        "bytes": test_path.stat().st_size,
                        "rows": expected_rows,
                    },
                },
            }
            path.with_suffix(path.suffix + ".run.json").write_text(
                json.dumps(run_manifest, indent=2) + "\n"
            )
            paths[(baseline["id"], scenario)] = path
    assert len(paths) == 94
    return paths, protocol_path


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def test_builds_complete_machine_and_latex_tables_deterministically(
    tmp_path: Path,
) -> None:
    results = tmp_path / "results"
    _, protocol_path = _write_complete_fixture(results)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first_paths = TABLES.build_tables(
        results, first_output, protocol_path=protocol_path
    )
    second_paths = TABLES.build_tables(
        results, second_output, protocol_path=protocol_path
    )

    assert set(first_paths) == {
        "full_results",
        "main_metrics",
        "completion",
        "upsampling_4x",
        "upsampling_16x",
    }
    full_header, full_rows = _read_rows(first_paths["full_results"])
    main_header, main_rows = _read_rows(first_paths["main_metrics"])
    protocol = _load_json(RELEASE_ROOT / "configs" / "protocol.json")
    metric_names = [metric["name"] for metric in protocol["metrics"]]

    assert len(full_rows) == len(main_rows) == 94
    assert full_header == [
        "model_id",
        "model",
        "scenario",
        "task",
        "split",
        *metric_names,
    ]
    assert main_header == [
        "model_id",
        "model",
        "scenario",
        "task",
        "split",
        *TABLES.MAIN_METRICS,
    ]
    assert full_rows[0]["model_id"] == "pu_gcn"
    assert full_rows[0]["scenario"] == "completion_random"
    assert float(full_rows[0]["mhd2"]) > 0
    assert "hd" not in full_header

    completion = first_paths["completion"].read_text()
    upsampling_4x = first_paths["upsampling_4x"].read_text()
    assert "SPU-PMD" not in completion
    assert "SPU-PMD" in upsampling_4x
    assert r"CD-L1$\downarrow$" in completion
    assert r"F1$\uparrow$" in completion
    assert r"\textbf{" in completion
    assert r"\underline{" in completion

    for key, first_path in first_paths.items():
        assert first_path.read_bytes() == second_paths[key].read_bytes()


@pytest.mark.parametrize(
    "failure",
    [
        "missing_file",
        "unexpected_file",
        "missing_metric",
        "non_finite",
        "duplicate_alias",
        "duplicate_overall",
        "nonpositive_count",
        "wrong_count",
        "missing_run_manifest",
        "wrong_run_identity",
        "per_sample_tamper",
    ],
)
def test_invalid_result_sets_fail_before_writing(
    tmp_path: Path, failure: str
) -> None:
    results = tmp_path / "results"
    paths, protocol_path = _write_complete_fixture(results)
    target = paths[("pointr", "completion_random")]

    if failure == "missing_file":
        target.unlink()
    elif failure == "unexpected_file":
        _write_csv(results / "unknown" / "extra.csv", ["value"], [{"value": 1}])
    elif failure in {
        "missing_metric",
        "non_finite",
        "duplicate_alias",
        "nonpositive_count",
        "wrong_count",
    }:
        header, rows = _read_rows(target)
        if failure == "missing_metric":
            header.remove("cd_l2")
            rows[0].pop("cd_l2")
        elif failure == "non_finite":
            rows[0]["cd_l2"] = "nan"
        else:
            if failure == "duplicate_alias":
                header.append("hd")
                rows[0]["hd"] = rows[0]["mhd2"]
            elif failure == "nonpositive_count":
                rows[0]["n_test"] = "0"
            else:
                rows[0]["n_test"] = "7"
        _write_csv(target, header, rows)
    elif failure == "duplicate_overall":
        aggregate = paths[("pu_gcn", "completion_random")]
        header, rows = _read_rows(aggregate)
        overall = next(row for row in rows if row["group_type"] == "overall")
        _write_csv(aggregate, header, [*rows, overall])
    elif failure == "missing_run_manifest":
        target.with_suffix(target.suffix + ".run.json").unlink()
    elif failure == "wrong_run_identity":
        run_path = target.with_suffix(target.suffix + ".run.json")
        manifest = _load_json(run_path)
        manifest["scenario"] = "completion_genus_ood"
        run_path.write_text(json.dumps(manifest) + "\n")
    else:
        per_sample = target.with_name(f"{target.stem}_test_per_sample.csv")
        per_sample.write_text(per_sample.read_text() + "extra\n")

    output = tmp_path / "output"
    with pytest.raises(TABLES.TableBuildError):
        TABLES.build_tables(results, output, protocol_path=protocol_path)
    assert not output.exists()


def test_duplicate_csv_headers_fail_closed(tmp_path: Path) -> None:
    results = tmp_path / "results"
    paths, protocol_path = _write_complete_fixture(results)
    target = paths[("pointr", "completion_random")]
    header, rows = _read_rows(target)
    values = [rows[0][column] for column in header]
    target.write_text(
        ",".join([*header, "cd_l2"]) + "\n" + ",".join([*values, rows[0]["cd_l2"]]) + "\n"
    )

    output = tmp_path / "output"
    with pytest.raises(TABLES.TableBuildError, match="duplicate column"):
        TABLES.build_tables(results, output, protocol_path=protocol_path)
    assert not output.exists()


def test_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _, protocol_path = _write_complete_fixture(results)
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep\n")

    with pytest.raises(TABLES.TableBuildError, match="refusing to overwrite"):
        TABLES.build_tables(results, output, protocol_path=protocol_path)
    assert sentinel.read_text() == "keep\n"


def test_bundled_reference_results_rebuild_all_94_rows(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_paths = TABLES.build_reference_tables(
        TABLES.DEFAULT_REFERENCE_RESULTS, first
    )
    second_paths = TABLES.build_reference_tables(
        TABLES.DEFAULT_REFERENCE_RESULTS, second
    )
    _, rows = _read_rows(first_paths["full_results"])
    assert len(rows) == 94
    for name in first_paths:
        assert first_paths[name].read_bytes() == second_paths[name].read_bytes()


def test_reference_results_hash_tamper_fails_before_writing(tmp_path: Path) -> None:
    reference = tmp_path / "full_results.csv"
    manifest = tmp_path / "manifest.json"
    shutil.copyfile(TABLES.DEFAULT_REFERENCE_RESULTS, reference)
    shutil.copyfile(TABLES.DEFAULT_REFERENCE_MANIFEST, manifest)
    reference.write_bytes(reference.read_bytes() + b"\n")
    output = tmp_path / "output"
    with pytest.raises(TABLES.TableBuildError, match="does not match"):
        TABLES.build_reference_tables(
            reference, output, manifest_path=manifest
        )
    assert not output.exists()
