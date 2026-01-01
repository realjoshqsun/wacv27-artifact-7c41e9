# Frozen paper aggregates

`full_results.csv` contains the finite overall test-set means used by the
paper: 16 registered baselines, 94 supported method--scenario rows, and all 32
TreeBench metrics. Completion counts include four TLS-like and four ULS-like
observations for every test tree. Repeated rows introduced only by distributed
sampler padding were removed before the affected means were computed.

The byte count and SHA-256 are pinned in `manifest.json`. Rebuild the paper
tables with:

```bash
python scripts/build_paper_tables.py \
  --reference-results reference_results/full_results.csv \
  --output generated_tables
```

This command checks the manifest hash, the exact 94-row registry expansion,
model/scenario/task/split identities, fixed test counts, the 32-metric schema,
and finite values before writing any output. It reproduces table construction
from the paper's frozen numerical results. It does not rerun inference; fresh
model outputs use `--results` and must include per-sample CSVs and run
provenance as described in `docs/PAPER_TABLES.md`.
