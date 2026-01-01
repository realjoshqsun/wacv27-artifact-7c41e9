# Baseline protocol registry

[`registry.json`](registry.json) is the authoritative machine-readable protocol for the 16 baselines reported in TreeBench. The launcher consumes this registry; upstream defaults and historical shell scripts do not override it. The registry pins public HTTPS repositories to full commits and records each model's environment, scenarios, entrypoints, model-specific training budget, evaluation batches, checkpoint format, and evaluation-only 16x route.

[`overlay_manifest.json`](overlay_manifest.json) is the exhaustive overlay allowlist. Every materialized payload is a regular file with an upstream-relative destination and SHA-256 digest. Data, checkpoints, logs, repository history, notebooks, compiled objects, dependency build trees, and symbolic links are excluded.

## Release status

Ten baselines can be materialized and evaluated: PoinTr, AdaPoinTr, SnowflakeNet, PUCRN, PU-Flow, SVDFormer, RepKPU, PUDM, FSC, and GeoFormer. Nine have a verified training launcher. RepKPU's executed scientific budget is known, but its exact historical launcher command was not preserved; training therefore fails closed as `OPEN_COMMAND_PROVENANCE`, while materialization and evaluation remain available.

PU-GCN, Neural-Points, AnchorFormer, ODGNet, SPU-PMD, and PUFM are `LEGAL_BLOCKED`: no redistribution license was found in their pinned upstream checkout, and their TreeBench integrations require source-code overrides. This release carries no payload from those six integrations. The materializer rejects them before any network access.

The reported training comparison was **not compute-controlled across architectures**. It used the executed, model-specific budgets below.

| Baseline | Launcher status | Epochs | Batch size | LR | Workers | GPUs | Distributed module |
|---|---:|---:|---:|---:|---:|---:|---|
| PoinTr | verified | 10 (`max_epoch=9`) | 512 global | 5e-4 | 8 | 4 | `torch.distributed.launch` |
| AdaPoinTr | verified | 10 (`max_epoch=9`) | 256 global | 1e-4 | 8 | 4 | `torch.distributed.launch` |
| SnowflakeNet | verified | 10 | 64/GPU | 1e-3 | 8 | 4 | `torch.distributed.launch` |
| PUCRN | verified | 10 | 2/GPU | 1e-3 | 8 | 4 | `torch.distributed.run` |
| PU-Flow | verified | 10 | 1/GPU | 1e-3 | 8 | 4 | `torch.distributed.run` |
| SVDFormer | verified | 10 | 1/GPU | 1e-3 | 8 | 4 | `torch.distributed.launch` |
| RepKPU | **open; training disabled** | 50 | 24/GPU | 1e-3 | 8 | 4 | not reconstructed |
| PUDM | verified | 10 | 24/GPU | 1e-4 (minimum 1e-6) | 8 | 4 | `torch.distributed.launch` |
| FSC | verified | 10 | 1/GPU | 1e-3 | 8 | 4 | `torch.distributed.launch` |
| GeoFormer | verified | 10 | 1/GPU | 1e-3 | 8 | 4 | `torch.distributed.launch` |

The launcher writes these values into runtime configurations and passes the applicable command-line values explicitly. It accepts no free-form passthrough arguments, so epochs, batch sizes, and learning rates cannot be silently overridden.

## Materialize a baseline

From the release root, use a new or empty target directory:

```bash
python scripts/materialize_baseline.py snowflakenet ../baseline-work/snowflakenet
```

The materializer verifies payload hashes before cloning, clones the anonymous HTTPS URL, checks out the exact pinned commit, applies only allowlisted files, verifies their copied hashes, and writes `.treebench-overlay.json`. It rejects non-empty targets, symbolic links, unsafe paths, hash mismatches, missing inventory entries, unknown models, and legal-blocked models.

## Data and fixed splits

The data root supplied to the launcher contains only the task directories:

```text
treebench-data/
├── completion/
└── upsampling/
```

The launcher always injects the release-owned split files [`../manifests/splits/random.json`](../manifests/splits/random.json) and [`../manifests/splits/genus_ood.json`](../manifests/splits/genus_ood.json), after checking their SHA-256 values against [`../manifests/source.json`](../manifests/source.json). Legacy category files under the data root are neither required nor accepted as split overrides.

## Train

```bash
python scripts/run_baseline.py snowflakenet upsampling_4x_random train \
  --repo ../baseline-work/snowflakenet \
  --python python \
  --gpus 0,1,2,3 \
  --data-root ../treebench-data \
  --output ../runs/snowflakenet-4x-random
```

Training accepts only completion and 4x scenarios, requires exactly the recorded GPU count, and requires a new or empty output directory outside the upstream checkout. The checkout must have the pinned `HEAD`, exact `upstream` remote, verified overlay hashes, and no change outside registered overlay destinations. External `PYTHONPATH` and user-site packages are not inherited. A successful run must produce exactly one registered best-checkpoint filename; the launcher writes an adjacent `<checkpoint>.treebench.json` with the checkpoint/config hashes and training provenance.

RepKPU training and all six legal-blocked integrations stop before process launch. A 16x training request always stops before process launch.

## Evaluate

```bash
python scripts/run_baseline.py snowflakenet upsampling_4x_random evaluate \
  --repo ../baseline-work/snowflakenet \
  --python python \
  --gpus 0 \
  --data-root ../treebench-data \
  --taxonomy-csv ../treebench-data/taxonomy.csv \
  --checkpoint ../checkpoints/snowflakenet-4x-random.pth \
  --output ../results/snowflakenet-4x-random.csv \
  --dry-run
```

Remove `--dry-run` after inspecting the emitted argument array and environment. The checkpoint and taxonomy file must be regular, non-symbolic-link files. The aggregate CSV, both per-sample CSVs, and the run sidecar must not already exist.

Completion and 4x upsampling map 2,048 input points to 8,192 output points. The 16x setting is evaluation-only: it maps 512 points to 8,192 using the split-matched 4x checkpoint, without 16x training or fine-tuning. Before a 16x launch, the runner requires and verifies the adjacent checkpoint provenance written by a matching 4x training run, including baseline, split-specific source scenario, pinned upstream commit, and checkpoint SHA-256.

After evaluation, the runner fails closed unless all of the following hold:

1. the aggregate CSV and both `_val_per_sample.csv` and `_test_per_sample.csv` files exist as regular files;
2. completion contains exactly one unique `name|mode|view` key for each fixed split tree, four ULS views and four TLS views, while upsampling contains exactly one key per tree;
3. every per-sample row reports the registered input cardinality and an actual 8,192-point output;
4. all 32 registered per-sample metrics are finite;
5. the aggregate `overall` counts match the per-sample files and all 32 validation/test means are finite.

Only then does it write `<aggregate.csv>.run.json`. The sidecar records the baseline/scenario identity, source training scenario, pinned upstream commit, checkpoint/config/taxonomy/split hashes, cross-rate route, cardinality and metric contracts, sample counts, and SHA-256/byte/row metadata for all three CSV artifacts. Per-sample artifacts also carry a canonical sample-key-set hash.

## Third-party licenses

Pinned upstream license texts are retained under [`third_party_licenses/`](third_party_licenses/) and mapped in `overlay_manifest.json`:

| Baseline IDs | SPDX | Bundled upstream text |
|---|---|---|
| PoinTr, AdaPoinTr | MIT | `third_party_licenses/pointr/LICENSE` |
| SnowflakeNet | MIT | `third_party_licenses/snowflakenet/LICENSE` |
| PUCRN | Apache-2.0 | `third_party_licenses/pucrn/LICENSE` |
| PU-Flow | MIT | `third_party_licenses/pu_flow/LICENSE` |
| SVDFormer | MIT | `third_party_licenses/svdformer/LICENSE` |
| RepKPU | MIT | `third_party_licenses/repkpu/LICENSE` |
| PUDM | MIT | `third_party_licenses/pudm/LICENSE` |
| FSC | MIT | `third_party_licenses/fsc/LICENSE` |
| GeoFormer | MIT | `third_party_licenses/geoformer/LICENSE` |

The pinned PUCRN checkout contains the Apache-2.0 `LICENSE` under `PointUpsampling/PUCRN/` and no `NOTICE` file; the manifest records that absence. A repository marked `not_found_in_upstream_checkout` must remain code-free until redistribution permission or a valid upstream license is confirmed.

## Environment profiles

The registry records compatibility environments for legacy CUDA operators. In particular, PUDM uses Python 3.7, PyTorch 1.9.1+cu111, PyTorch3D 0.6.1, and pytorch-lightning 0.7.1. These records describe execution dependencies; they do not grant permission to redistribute third-party source.

The complete benchmark contains 94 method-scenario rows: 15 models define six scenarios, while SPU-PMD defines only the four upsampling scenarios. Availability in the paper registry does not override the legal release gate above.
