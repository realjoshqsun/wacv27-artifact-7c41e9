# Building Paper Tables

`scripts/build_paper_tables.py` converts the complete set of registered
baseline results into deterministic CSV and LaTeX tables.  It uses
`baselines/registry.json` as the authoritative list of 16 models and 94
method-scenario combinations, and `configs/protocol.json` as the authoritative
32-metric schema.

## Included paper aggregates

`reference_results/full_results.csv` contains the 94 test-only overall means
reported in the paper, with its byte count and SHA-256 fixed by
`reference_results/manifest.json`. To rebuild their numerical tables, run:

```bash
python scripts/build_paper_tables.py \
  --reference-results reference_results/full_results.csv \
  --output generated_tables
```

This mode checks the exact baseline/scenario/task/split order, fixed test
counts, all 32 finite metrics, and the artifact hash. It rebuilds the paper
tables from frozen aggregates and is not evidence of a fresh inference run.

## Fresh-run input layout

Place aggregate test results in this layout:

```text
results/
  pu_gcn/
    completion_random.csv
    completion_random.csv.run.json
    completion_random_val_per_sample.csv
    completion_random_test_per_sample.csv
    completion_genus_ood.csv
    upsampling_4x_random.csv
    upsampling_4x_genus_ood.csv
    upsampling_16x_random.csv
    upsampling_16x_genus_ood.csv
  pointr/
    ...
```

The registry defines the required model directories and scenarios.  SPU-PMD
has four registered upsampling scenarios; each of the other 15 models has all
six scenarios.

Each aggregate CSV must provide positive `n_val` and `n_test` counts matching
the fixed split (completion counts include all eight frozen views per tree),
plus all 32 finite metrics for the test split in one of two forms:

1. one row with count columns and canonical columns such as `cd_l2`, `mhd2`, and
   `tree_height_abs_error`; or
2. an aggregate taxonomy table containing exactly one
   `group_type=overall` row and `test_`-prefixed metric columns such as
   `test_cd_l2`, `test_mhd2`, and `test_tree_height_abs_error`.

The legacy names `hd` and `test_hd` are accepted as aliases for `mhd2`.  Supply
exactly one accepted column for each metric; for example, a file containing both
`mhd2` and `hd` is rejected as ambiguous.

Every aggregate must retain the two per-sample CSVs and the adjacent
`<aggregate>.run.json` written by `scripts/run_baseline.py`. The builder verifies
the baseline/scenario identity, checkpoint and configuration hashes, 16x source
checkpoint route, sample counts, file sizes, row counts, and SHA-256 of all three
CSV artifacts before accepting a result.

## Fresh-run command

Run from the release root:

```bash
python scripts/build_paper_tables.py \
  --results results \
  --output generated_tables
```

The output directory is kept separate from the input `results/` directory.  The
command writes:

- `full_results.csv`: all 32 metrics for all 94 registered combinations;
- `main_metrics.csv`: the 11 metrics displayed in the main paper;
- `completion.tex`;
- `upsampling_4x.tex`;
- `upsampling_16x.tex`.

The LaTeX tables use unrounded values to identify the best and second-best
method in each metric and scenario.  They require the standard `booktabs` and
`graphicx` packages.

## Closed validation

The command validates all inputs before creating the output directory.  It
fails when it finds:

- a missing or unexpected result CSV;
- a registry other than the frozen 16-model, 94-combination protocol;
- duplicate model IDs, scenarios, headers, overall rows, or metric aliases;
- a missing, empty, non-numeric, `NaN`, or infinite metric value;
- missing, zero, or split-inconsistent validation/test sample counts;
- a missing or mismatched run manifest or per-sample artifact;
- a protocol metric count, name, or main-metric direction mismatch.

The included aggregate artifact contains no source point clouds, predictions,
per-sample values, or checkpoints. Fresh results are accepted only through the
stricter run-bundle path above.
