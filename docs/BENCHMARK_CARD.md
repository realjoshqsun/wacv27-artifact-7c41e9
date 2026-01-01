# TreeBench Benchmark Card

## Evaluative role

TreeBench tests whether point-cloud reconstruction methods preserve tree geometry
when they densify a sparse observation or complete a view-conditioned partial
observation. It evaluates the same prediction in two coordinate systems:

- five point-set metrics in the normalized model space; and
- 27 tree-structural and canopy-fuel geometry metrics after restoring metres.

The benchmark supports claims about reconstruction fidelity under its fixed
8,192-point protocol. It does not validate downstream forestry products or the
physical accuracy of a scanner.

## Composition

TreeBench derives its cohort from the labelled development partition of
FOR-species20K. A tree is eligible when its observed cloud contains at least
8,192 points. The fixed cohort contains 14,884 individual trees, 33 species,
and 19 genera.

Each tree contributes one 8,192-point observed reference. The upsampling tasks
use nested 2,048-point and 512-point subsets for 4x and 16x evaluation. The
completion task uses four TLS-like and four ULS-like frozen 2,048-point views.
These views use mesh-free visibility on the observed reference; they are not
renderings of a complete geometric surface.

The random split is genus-stratified 80/10/10. The genus-OOD split trains on
nine genera and validates/tests on ten disjoint held-out genera. Exact sample
membership is stored in `manifests/splits`.

## Tasks and comparison unit

There are six evaluation settings: completion, 4x upsampling, and 16x
upsampling under both random and genus-OOD splits. Completion and 4x are
trainable. The 16x setting reuses the split-matched 4x checkpoint and is
evaluation-only.

The paper compares 16 baselines. Fifteen cover all six settings; SPU-PMD covers
the four upsampling settings. The complete comparison therefore has 94
method-scenario rows. A sample is one tree for upsampling and one
`(tree, sensor mode, view index)` tuple for completion.

## Measurements

Point metrics are CD-L2, CD-L1, MHD2, DCD, and F1. Tree-space metrics include
seven structural descriptors and 20 canopy-fuel descriptors. The canonical
names, formulas, units, directions, thresholds, bin sizes, and coordinate
systems are defined in `configs/protocol.json` and `docs/DATA_PROTOCOL.md`.

Metrics are first computed per sample, then averaged without sample weights.
Taxonomic reports use separate overall, genus, and species rows. Non-finite
predictions produce non-finite per-sample metrics and are counted explicitly;
they must not be silently replaced with favourable values.

## Intended uses

- compare point-cloud upsampling and completion methods under fixed inputs,
  outputs, splits, and metrics;
- examine whether rankings from common point metrics agree with rankings from
  tree geometry;
- measure the change from random to genus-disjoint evaluation;
- test cross-rate behaviour by applying a 4x-trained model at 16x; and
- develop additional reconstruction metrics or baselines against the same
  protocol.

## Uses outside the benchmark claim

TreeBench does not establish that 8,192 points are sufficient for a forestry or
fire-management product. It does not provide ground-truth meshes, branch or
leaf labels, biomass, fuel loads, species recognition targets, scanner
calibration, or geographic generalization guarantees. Its TLS-like and
ULS-like observations model directional sparsity, not all physical effects of a
specific instrument. Results should not be presented as operational forestry
validation without an independent downstream study.

## Known limitations

- The reference is a dense observation of a physical tree, not a complete
  ground-truth surface.
- A single fixed output cardinality makes methods comparable but does not test
  the density needed by every downstream application.
- Genus-OOD measures taxonomic shift within the source collection; it does not
  isolate geography, acquisition system, season, age, or forest structure.
- Mesh-free visibility depends on projection resolution and the sampled
  reference. Thin or unseen structures cannot be recovered as ground truth.
- Unweighted per-sample means give each tree or frozen view equal influence;
  they are not prevalence-weighted population estimates.
- Historical paper arrays did not retain the random state and poses needed for
  byte-exact regeneration. The release generator fixes those values for new
  builds, while the reported numbers remain tied to the frozen historical
  arrays.

## Data governance and ethics

The benchmark uses existing non-human point clouds. Release metadata omits
precise geographic coordinates and fields unrelated to evaluation. The source
dataset and its derivatives are not redistributed because official rights
records conflict. `docs/THIRD_PARTY_DATA.md` records the provenance, checksums,
rights issue, and permitted release route.

## Versioning and maintenance

`configs/protocol.json`, split manifests, source checksums, baseline commit
hashes, overlay hashes, checkpoint hashes, and generated-data hashes define a
release version. Any change to sample membership, task construction, metric
formula, or 16x route creates a new benchmark version and must not be mixed with
the paper's reported results. Public maintainer and support information should
be added only after double-blind review ends.
