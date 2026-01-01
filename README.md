# TreeBench: Anonymous Review Release

TreeBench evaluates point-cloud upsampling and completion on dense observed
individual trees. It fixes the data splits, constructs sparse and
view-conditioned inputs from one 8,192-point observed reference per tree, and
evaluates every reconstruction with 32 point, tree-structural, and canopy-fuel
geometry metrics.

This repository is an anonymous review artifact. It contains the benchmark
protocol, deterministic data-generation code, the two frozen split manifests,
a shared evaluator, a synthetic smoke fixture, tests, and integration metadata
for the 16 models reported in the paper. It does not contain source point
clouds, derived benchmark arrays, pretrained weights, experiment logs, or Git
history.

## Pipeline

```text
official FOR-species20K development data
        |
        +-- filter clouds with at least 8,192 observed points
        +-- FPS reference + shared centre/radial-scale normalization
        |
        +-- nested random subsets: 2,048 -> 8,192 (4x)
        |                          512 -> 8,192 (16x evaluation)
        |
        +-- mesh-free visibility: 2,048 partial -> 8,192 (completion)
        |
16 pinned baseline integrations
        |
        +-- normalized point metrics (5)
        +-- denormalized tree/canopy-fuel geometry metrics (27)
        |
per-sample results -> overall/genus/species aggregates
```

The benchmark contains 14,884 trees from 33 species and 19 genera. The random
split is genus-stratified 80/10/10. The genus-OOD split trains on nine genera
and validates/tests on ten disjoint held-out genera.

## Repository map

- `treebench/data`: reference construction, nested subsets, TLS-/ULS-like
  visibility simulation, split generation, and PyTorch loaders.
- `treebench/metrics`: the canonical five point metrics and 27 metric-space
  tree/canopy-fuel metrics.
- `treebench/aggregation.py`: taxonomy joins and the 68-column validation/test
  aggregate schema used by the paper pipeline.
- `configs/protocol.json`: machine-readable data and metric definitions.
- `manifests/splits`: exact random and genus-OOD membership.
- `baselines/registry.json`: authoritative 16-model protocol, pinned upstream
  commits, scenario support, 16x routes, environments, and output schemas.
- `baselines/overlays`: allowlisted TreeBench adapters for the ten integrations
  whose pinned upstream source permits redistribution.
- `environments`: the current core environment and three executed legacy
  baseline profiles.
- `reference_results`: hash-pinned overall test means for all 94 reported
  model-scenario combinations.
- `tests`: CPU tests for data construction, loaders, splits, and all metrics.
- `scripts/build_paper_tables.py`: strict reconstruction of the 94-row paper
  tables from registered aggregate outputs.
- `docs`: the benchmark card, complete data protocol, environment notes,
  third-party provenance, paper-table schema, anonymous-repository handoff,
  and WACV compliance audit.

## Quick start without external data

The synthetic fixture is generated procedurally and is not a TreeBench sample
or a paper result.

```bash
conda env create -f environments/core.yml
conda activate treebench-core
python -m pip install -e .
pytest
treebench-smoke
```

`treebench-smoke` generates procedural point clouds in temporary storage and
runs data validation, 4x evaluation, evaluation-only 16x, completion, all 32
metrics, and taxonomy aggregation. To inspect the generated files, pass an
empty directory with `--workdir`.

The individual smoke commands are:

```bash
treebench-generate-demo --output /tmp/treebench-demo
treebench-validate \
  --split /tmp/treebench-demo/splits/demo.json \
  --data-root /tmp/treebench-demo/upsampling \
  --task upsampling

treebench-evaluate \
  --task upsampling-4x \
  --data-root /tmp/treebench-demo/upsampling \
  --split /tmp/treebench-demo/splits/demo.json \
  --subset test \
  --predictions /tmp/treebench-demo/predictions/upsampling-4x \
  --output /tmp/treebench-demo/evaluation/upsampling-4x
```

The bundled identity prediction gives zero geometric error, unit F1 and IoU
values, and the small positive identity value implied by the published DCD
assignment epsilon. Run completion in the same way with `--task completion`,
the completion data root, and its prediction directory.

## Obtain and prepare the source data

FOR-species20K remains third-party data. The pinned source version is Zenodo
record `13255198` / DOI `10.5281/zenodo.13255198`. Inspect the official file
manifest before downloading approximately 25 GB:

```bash
python scripts/download_forspecies20k.py \
  --output /path/to/source-download \
  --show-manifest

python scripts/download_forspecies20k.py \
  --output /path/to/source-download
```

The downloader retrieves from Zenodo and verifies the official byte counts and
MD5 values before renaming a completed file. Extract `dev.zip`, then build the
point-count/taxonomy join and verify the fixed manifests:

```bash
treebench-prepare source-metadata \
  --input /path/to/extracted/dev \
  --metadata /path/to/source-download/tree_metadata_dev.csv \
  --output /path/to/work/source_metadata.csv

treebench-validate \
  --split manifests/splits/random.json \
  --expected-total 14884

treebench-validate \
  --split manifests/splits/genus_ood.json \
  --expected-total 14884
```

Build the reference/subset tree and frozen completion observations:

```bash
treebench-prepare references \
  --input /path/to/extracted/dev \
  --manifest manifests/splits/random.json \
  --output /path/to/treebench/upsampling \
  --seed 42 \
  --fps-backend open3d

treebench-prepare completion \
  --upsampling /path/to/treebench/upsampling \
  --output /path/to/treebench/completion \
  --seed 42 \
  --tls-distance 0.1

treebench-validate \
  --split manifests/splits/random.json \
  --data-root /path/to/treebench/upsampling \
  --task upsampling \
  --expected-total 14884 \
  --require-provenance

treebench-validate \
  --split manifests/splits/random.json \
  --data-root /path/to/treebench/completion \
  --task completion \
  --expected-total 14884 \
  --require-provenance \
  --upsampling-root /path/to/treebench/upsampling
```

Every newly generated tree records normalization, seeds, view parameters, and
SHA-256 values. The historical paper arrays predate that provenance scheme:
their FPS/subset/view RNG state and scanner poses were not saved. The generator
therefore reconstructs the documented protocol deterministically but cannot
recreate the historical arrays byte for byte. Exact reproduction of reported
numbers requires the frozen arrays used by those runs. The data-rights conflict
and permitted release route are documented in
[`docs/THIRD_PARTY_DATA.md`](docs/THIRD_PARTY_DATA.md).

## Tasks and checkpoint policy

There are six evaluation settings: completion, 4x upsampling, and 16x
upsampling, each under random and genus-OOD splits.

Completion and 4x upsampling are trainable. For each split, validation CD-L2
selects the checkpoint. The corresponding 16x setting is evaluation-only: it
uses a 512-point input and the same split-matched 4x checkpoint. Depending on
the architecture, the route composes two 4x passes, requests a runtime 16x
ratio, or feeds 512 points to a fixed-output 8,192-point decoder. A separate
16x checkpoint is outside the benchmark protocol.

Fifteen models cover all six settings. SPU-PMD covers the four upsampling
settings only, giving 94 reported method-scenario rows. The exact routes and
entrypoints live in [`baselines/registry.json`](baselines/registry.json); that
registry overrides historical scripts or upstream defaults.

## Materialize and run a baseline

The release keeps integrations as overlays on pinned public upstream commits.
This avoids copying complete third-party repositories and preserves upstream
attribution. See [`baselines/README.md`](baselines/README.md) for the current
coverage and any license-blocked model patches.

```bash
python scripts/materialize_baseline.py \
  pointr /path/to/materialized/pointr

python scripts/run_baseline.py \
  pointr upsampling_4x_random train \
  --repo /path/to/materialized/pointr \
  --data-root /path/to/treebench \
  --output /path/to/results/pointr/up4-random \
  --gpus 0,1,2,3 \
  --dry-run
```

Remove `--dry-run` only after building that upstream model's CUDA extensions in
the environment assigned by the registry. Evaluation requires `--checkpoint`.
The runner refuses 16x training and validates scenario support before starting
a subprocess.

The baseline implementations retain their published objectives and model
families. The common constraints are the data pairs, split membership,
8,192-point output, validation selection rule, cross-rate checkpoint reuse,
and evaluator.

## Evaluate saved predictions

Predictions are normalized `(8192, 3)` NumPy arrays. Upsampling filenames are
`<tree_id>.npy`; completion filenames are
`<tree_id>_<mode>_<view_index>.npy`, with `mode` equal to `tls` or `uls` and
view indices 0--3.

```bash
treebench-evaluate \
  --task completion \
  --data-root /path/to/treebench/completion \
  --split manifests/splits/genus_ood.json \
  --subset test \
  --predictions /path/to/predictions \
  --output /path/to/evaluation
```

Run validation and test separately, then attach taxonomy and reproduce the
paper aggregation schema:

```bash
treebench-aggregate \
  --val /path/to/val/per_sample.csv \
  --test /path/to/test/per_sample.csv \
  --task completion \
  --split manifests/splits/genus_ood.json \
  --taxonomy /path/to/work/source_metadata.csv \
  --output /path/to/aggregate.csv
```

Point metrics operate in normalized coordinates. Tree-structural and
canopy-fuel metrics restore metres with each tree's saved centre and scale.
[`docs/DATA_PROTOCOL.md`](docs/DATA_PROTOCOL.md) gives every formula,
parameter, unit, and direction.

## Rebuild the paper tables

The release includes the 94 frozen overall test means reported in the paper.
Rebuild their numerical CSV and LaTeX tables without checkpoints or source data:

```bash
python scripts/build_paper_tables.py \
  --reference-results reference_results/full_results.csv \
  --output generated_tables
```

This path verifies the pinned reference-results SHA-256, exact registry
expansion, scenario identities, test counts, metric schema, and finite values.
It reproduces table construction from reported means; it does not rerun model
inference.

For a fresh end-to-end result set, place every registered aggregate, its
adjacent `.run.json`, and its validation and test per-sample CSVs under
`results/<model_id>/`, then run:

```bash
python scripts/build_paper_tables.py \
  --results results \
  --output generated_tables
```

The fresh-run path validates all 16 models and 94 registered method-scenario
combinations before writing the full 32-metric CSV, the 11-metric paper CSV,
and completion, 4x, and 16x LaTeX tables. It rejects missing, extra, duplicate,
ambiguous, non-finite, count-incomplete, or provenance-mismatched results.
[`docs/PAPER_TABLES.md`](docs/PAPER_TABLES.md) defines the accepted overall-row
schemas and required sidecars.

## Reproduction records

Each run manifest should record:

- baseline ID, scenario, seed, environment, GPU, driver, compiler, and runtime;
- relative config and checkpoint paths with SHA-256 values;
- source 4x task/checkpoint and route for every 16x evaluation;
- input/output cardinality and unique completion `(tree, mode, view)` keys;
- aggregate and per-sample output hashes.

Pretrained checkpoints are not stored in ordinary Git because individual files
can exceed GitHub's 100 MB limit and the model set is several gigabytes. If
weights are anonymously hosted for review, populate
`manifests/checkpoints.json` with URLs, sizes, hashes, epochs, and source-config
hashes before packaging.

## Anonymous WACV delivery

Run the release gate and create a deterministic supplementary archive:

```bash
python scripts/verify_release.py
python scripts/package_review.py --output ../treebench_review.zip
```

The verifier rejects repository history, symlinks, binaries, checkpoints,
notebooks, private paths, account markers, common token forms, malformed JSON,
split drift, baseline-count drift, and any 16x training entrypoint. The packer
excludes `.git`, fixes archive timestamps, and enforces the 200 MB WACV limit.

For the double-blind submission, deliver the ZIP through OpenReview. Do not put
a repository link in the manuscript, supplement, or rebuttal. An anonymous
GitHub repository may be used as a staging copy only after its owner, commit
identity, issues, actions, analytics, releases, and Git history pass the same
audit. `scripts/initialize_anonymous_repo.sh` creates a clean local history with
an anonymous commit identity after verification.
[`docs/ANONYMOUS_GITHUB.md`](docs/ANONYMOUS_GITHUB.md) gives the remote-side
inspection checklist and optional read-only mirror procedure.

## Licensing and third-party attribution

Upstream model and dataset citations identify third-party work and remain in an
anonymous submission. Their licenses do not identify the submitting authors.
The baseline registry records where an upstream license was found and blocks
vendoring when it was not found.

No license is assigned here to FOR-species20K or its derived point clouds. A
license for the newly written TreeBench code must be selected before public
distribution; that legal choice remains outside the technical review artifact.
See `docs/THIRD_PARTY_DATA.md` and the baseline overlay manifest before posting
the repository publicly.
