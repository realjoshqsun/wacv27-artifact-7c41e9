# TreeBench Data and Evaluation Protocol

This document fixes the dataset construction, split, input-generation, and
evaluation conventions used by TreeBench.  The machine-readable counterpart is
[`configs/protocol.json`](../configs/protocol.json).  Implementations should use
that file for constants and this document for the meaning of each operation.

## 1. Source cohort

TreeBench is derived from the labelled/public training partition of
FOR-species20K.  The construction metadata contains 17,707 candidate trees.  A
tree is retained when its source point cloud contains at least 8,192 XYZ points,
which yields 14,884 trees spanning 19 genera and 33 species.

The source `.laz` files and their labels remain governed by the FOR-species20K
terms described in `THIRD_PARTY_DATA.md`.  The release repository distributes
construction code and manifests; it does not relicense the source data.

## 2. Canonical 8,192-point references

For each retained source cloud \(X\):

1. Compute its centroid \(c = |X|^{-1}\sum_{x\in X}x\).
2. Compute the isotropic scale \(s = \max_{x\in X}\lVert x-c\rVert_2\).
3. Select 8,192 source points by farthest-point sampling (FPS).
4. Store raw metric coordinates and normalized coordinates
   \(x'=(x-c)/s\).

The per-tree metadata required for evaluation is `treeID`, the three-component
`center`, and scalar `scale`.  Denormalization is
`raw = normalized * scale + center`.

The historical construction used Open3D FPS.  Its NumPy fallback selected a
random initial point.  Input ordering and library versions can therefore affect
a regenerated reference; the frozen 8,192-point arrays are the benchmark
references.

## 3. Upsampling inputs

Nested subsets are sampled without replacement in this order:

`8192 -> 4096 -> 2048 -> 1024 -> 512`.

The benchmark evaluates:

| Setting | Input | Reference |
|---|---:|---:|
| 4x upsampling | 2,048 | 8,192 |
| 16x upsampling | 512 | 8,192 |

The input is consequently a subset of the reference for both settings.  Files
at 16,384 points found in 46 legacy tree directories are remnants of an earlier
experiment and are outside the protocol.

## 4. Dataset splits

Both manifests contain the same 14,884 tree IDs and have no duplicate IDs
within or across train, validation, and test.

### Random split

The random split is stratified by genus.  Scikit-learn `train_test_split` first
selects 20% into a temporary set and then divides that set equally into
validation and test, using `random_state=42` for both calls.

| Partition | Trees |
|---|---:|
| Train | 11,907 |
| Validation | 1,488 |
| Test | 1,489 |

All 19 genera occur in every partition.

### Genus-OOD split

The genus-OOD training partition contains nine genera:

`Acer`, `Betula`, `Carpinus`, `Eucalyptus`, `Fagus`, `Fraxinus`, `Picea`,
`Pinus`, and `Pseudotsuga`.

Validation and test contain only ten held-out genera:

`Prunus`, `Corylus`, `Euonymus`, `Larix`, `Abies`, `Populus`, `Crataegus`,
`Ulmus`, `Tilia`, and `Quercus`.

The held-out pool is split with pandas `sample(frac=0.5,
random_state=42)`; the sampled IDs form validation and the remainder form test.

| Partition | Trees |
|---|---:|
| Train | 12,111 |
| Validation | 1,386 |
| Test | 1,387 |

The frozen manifest SHA-256 values are:

- `8-1-1_random_N>=8192.json`:
  `6f138f406968c2e7fce878038d92d49b85e49e547a6a2145434bdfe15b3dcac7`
- `8-1-1_genus-OOD_N>=8192.json`:
  `2e59045aee309976c7621c3efd8ea9ea7f24f87f13f70204e1e83bf523c64125`

These manifests, rather than a newly repeated random split, define benchmark
membership.

## 5. Completion observations

Completion uses scanner-inspired visibility simulation on the raw metric cloud,
then applies the same per-tree normalization as the complete reference.

### Visibility operator

The simulator projects points onto a 256 by 256 image and retains the closest
point in each occupied pixel:

- TLS-like views use spherical azimuth/elevation projection from a terrestrial
  viewpoint.
- ULS-like views use orthographic XY projection from aerial viewpoints.

Visible indices from all viewpoints in one observation are united.  The union
is sampled without replacement when it contains at least 2,048 points and with
replacement otherwise, producing exactly 2,048 input points.

### TLS-like observation

For the canonical frozen evaluation views, sample an azimuth uniformly from
\([0,2\pi)\).  Place the scanner at radial distance \(0.1H\) from the XY tree
centre and at height `z_min + U(1.5, 2.0)` metres, where \(H\) is tree height.
One spherical view forms each observation.

### ULS-like observation

Place the flight line at `z_min + U(60, 75)` metres.  Draw a pivot by adding an
independent `U(-0.5, 0.5)` metre offset to the tree's XY bounding-box centre,
draw a uniform heading, and clip the infinite line to the XY bounding box.
Sample positions every 0.1 metre along the clipped line, retaining at most ten
evenly spaced viewpoints.  Each viewpoint contributes orthographic visible
points.  If clipping yields no usable segment, use one overhead viewpoint at
the tree centre.

### Frozen validation and test views

Each tree has four TLS-like and four ULS-like inputs:

- `2048_tls_0.npy` through `2048_tls_3.npy`
- `2048_uls_0.npy` through `2048_uls_3.npy`
- `8192_complete.npy`

The evaluation order is ULS indices 0--3 followed by TLS indices 0--3.  A
portable evaluator should emit a unique sample key `(tree_id, mode, view_index)`
and aggregate all eight samples.  A bare tree ID is not a unique completion
sample identifier.

### Historical training-view variants

Training generated a fresh TLS- or ULS-like observation with equal probability
at each access.  The ULS implementation was shared.  Two TLS radial-distance
variants existed in the historical model ports:

- fixed `0.1H`: PoinTr/AdaPoinTr, ODGNet, and PUDM;
- `U(0.5H, 1.0H)`: APU-LDI, AnchorFormer, FBNet, FSC, GeoFormer, Grad-PU,
  Neural-Points, PU-Flow, PU-GCN, PUCRN, PUFM, RepKPU, SeedFormer,
  SnowflakeNet, SPU-PMD, and SVDFormer.

The frozen validation/test observations use the fixed `0.1H` construction.
Reproducing a reported model run requires selecting the training variant used
by that port; the release configuration records the distinction explicitly.

## 6. Evaluation coordinates and aggregation

The five point-distribution metrics operate on normalized coordinates.  The 27
tree metrics operate in metres after denormalization with the reference
`center` and `scale`.  Prediction and reference must have shape `(8192, 3)` for
the reported benchmark.  Non-finite inputs produce `NaN` for all 32 metrics for
that sample and must propagate to the aggregate rather than being silently
discarded. Diagnostic summaries also record finite and total sample counts.

Metrics are computed per sample and macro-averaged over the requested split and
setting.  Distributed evaluators must remove sampler padding by unique sample
key, never by comparing metric rows or cloud contents.

## 7. Five normalized point metrics

Let \(d_{P\to G}(p)=\min_{g\in G}\lVert p-g\rVert_2^2\) and define
\(d_{G\to P}\) symmetrically.

- `cd_l2` (lower):
  \(\operatorname{mean}d_{P\to G}+\operatorname{mean}d_{G\to P}\).
- `cd_l1` (lower): one half of the sum of the two mean Euclidean nearest-neighbour
  distances.
- `mhd2` (lower; called `HD` in legacy code): one half of the sum of the two
  maximum squared nearest-neighbour distances.  This symmetric mean of directed
  maxima is the benchmark's MHD2 convention.
- `dcd` (lower): density-aware Chamfer distance with `alpha=1000`, density power
  `lambda=1`, and assignment-count epsilon `1e-6`.
- `f1` (higher): harmonic mean of precision and recall using the strict test
  `squared_distance < 1e-4`, equivalent to normalized Euclidean tolerance
  `0.01`.

No factor of one half is applied to `cd_l2`.  These scaling choices are part of
the protocol.

## 8. Twenty-seven tree metrics

The shared spatial grid uses voxel/cell size 0.05 m and origin
`min(prediction union reference) - 0.05 m`.  Relative errors divide by the
absolute reference value clamped below at `1e-8`.

### Structural metrics (7)

- `crown_width_abs_error`, `crown_width_relative_error`: crown width is the mean
  of the X and Y axis-aligned spans.
- `relative_height_{25,50,75,95}_abs_error`: absolute error in the corresponding
  linearly interpolated quantile of height above the cloud's own minimum Z.
- `foliage_height_diversity_abs_error`: absolute error in Shannon entropy of the
  normalized 20-bin relative-height point-count profile.

### Canopy-fuel geometry metrics (20)

- `tree_height_abs_error`, `tree_height_relative_error`: errors in the Z span.
- `crown_projection_area_abs_error`,
  `crown_projection_area_relative_error`, `crown_projection_iou`: area and set
  IoU of occupied 0.05 m XY cells.
- `occupied_volume_abs_error`, `occupied_volume_relative_error`,
  `canopy_occupancy_iou`: volume and set IoU of occupied 0.05 m XYZ voxels.
- `vertical_profile_l1`: L1 distance between normalized 20-bin relative-height
  point-count profiles.
- `crown_length_abs_error`, `crown_length_relative_error`: crown length is
  `z_max - crown_base_height`.
- `crown_base_height_abs_error`, `crown_base_height_relative_error`: crown base
  is the lower edge of the first of at least two consecutive relative-height
  bins whose normalized point count is strictly greater than 10% of the maximum
  bin count.  The relative metric compares `(CBH - z_min) / tree_height`.
- `fuel_stratum_profile_l1`: L1 distance between normalized 20-bin profiles of
  occupied XY-cell counts.
- `stratum_percentage_cover_l1`: mean absolute difference between 0.25 m
  vertical-bin cover profiles.  Cover is occupied XY cells divided by the joint
  prediction/reference footprint size.
- `fuel_layer_count_abs_error`,
  `lowest_fuel_layer_base_height_abs_error`,
  `dominant_fuel_layer_depth_abs_error`,
  `mean_interlayer_distance_abs_error`, and
  `largest_vertical_gap_abs_error`: absolute errors in layer descriptors derived
  from the 0.25 m cover profile.  Active bins have cover greater than or equal
  to 10% of the profile maximum; a layer contains at least two consecutive
  active bins.  The dominant layer has the largest summed cover.

The two IoUs and F1 are higher-is-better.  The other 29 metrics are
lower-is-better.  These canopy-fuel metrics quantify geometry and occupancy
from XYZ; they do not estimate mass, moisture, or live/dead fuel state.

## 9. Historical provenance and exact reproducibility

The frozen arrays and manifests preserve the evaluated benchmark instances.
The historical generators did not set a global or per-sample random seed for
FPS fallback, nested subset selection, scanner poses, or final visible-point
sampling.  Multiprocessing could also inherit NumPy RNG state.  Regenerating
from source is therefore statistically reproducible under this protocol but is
not guaranteed to reproduce the frozen arrays bit for bit.

Legacy per-tree metadata records only `treeID`, `center`, and `scale`.  It does
not record completion mode/view, scanner pose, flight trajectory, generator
parameters, or per-array checksum.  Mode and view index can be recovered from
filenames; the sampled poses cannot be reconstructed after the fact.  A new
release-generated corpus should add:

- a stable per-sample seed;
- `(tree_id, mode, view_index)`;
- all sampled scanner/flight parameters;
- generator and dependency versions;
- SHA-256 for every array.

The source statistics CSV used to select eligible trees is preserved by its
manifest outputs, but its original producer script is unavailable.  Its audited
SHA-256 is
`ab3d92916a716b69426c57975caf7774bf6f7cc845569440d0ca11714b8e3650`.
The joined taxonomy/height metadata CSV has SHA-256
`29548180da80f94f6028b7efdcfbc839d0ede801fcb003a0b244e549cabb8ae0`.

The completion `8192_complete.npy` files were normalized in a separate pass.
They match the upsampling 8,192-point references geometrically, with possible
float32 round-trip differences (observed maximum absolute difference
approximately `5.96e-08`).

## 10. Required release checks

Before publishing data or reporting a benchmark result, validate all of the
following:

1. Manifest hashes and split counts match Section 4.
2. Tree IDs are unique and partitions are disjoint.
3. Every upsampling input has the required point count and is a subset of its
   8,192-point reference.
4. Every completion tree has exactly four TLS and four ULS inputs of 2,048
   points plus one 8,192-point reference.
5. All arrays have shape `(N, 3)`, finite numeric values, and consistent
   normalization metadata.
6. Point metrics run in normalized coordinates and tree metrics in metres.
7. Evaluation emits unique completion sample keys and includes all eight fixed
   views per tree.
8. Metric names, parameter values, direction, and count match the 32 entries in
   `configs/protocol.json`.
9. Checksums and complete generator provenance accompany newly generated
   arrays.
