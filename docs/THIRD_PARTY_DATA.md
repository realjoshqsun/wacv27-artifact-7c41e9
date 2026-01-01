# Third-Party Data and Code

## Release decision

The anonymous review release distributes no FOR-species20K point clouds and no
pre-generated derivatives of those point clouds. It should distribute:

1. official versioned download instructions;
2. a machine-readable source and split manifest;
3. checksum verification;
4. deterministic benchmark-generation code; and
5. a small, independently generated synthetic fixture for smoke testing.

This design preserves reproducibility without mirroring a large upstream
dataset whose official rights metadata is inconsistent. Download instructions
alone are insufficient because they do not fix sample selection, splits,
generation parameters, or random seeds.

## Official sources

| Resource | Official reference |
|---|---|
| Dataset paper | [Puliti et al., 2025](https://doi.org/10.1111/2041-210X.14503) |
| Versioned dataset record | [Zenodo 13255198](https://doi.org/10.5281/zenodo.13255198) |
| Dataset concept DOI | [Zenodo 13255197](https://doi.org/10.5281/zenodo.13255197) |
| Permanent original-code snapshot | [Zenodo 14650218](https://doi.org/10.5281/zenodo.14650218) |
| Updated upstream repository | [FOR-species20K](https://github.com/stefp/FOR-species20K) |
| Original classification benchmark | [Codabench competition 3667](https://www.codabench.org/competitions/3667/) |

Use the version DOI `10.5281/zenodo.13255198` in manifests and reproducibility
instructions. The concept DOI may resolve to a later record with different
files or metadata.

## Official dataset files

The current Zenodo record exposes the following files through its [official
API](https://zenodo.org/api/records/13255198):

| File | Size in bytes | MD5 | Official content URL |
|---|---:|---|---|
| `dev.zip` | 25,354,980,290 | `ecb6196d9d630b095e6f6249b46efdd7` | [download](https://zenodo.org/api/records/13255198/files/dev.zip/content) |
| `test.zip` | 1,844,174,259 | `087eaed01d45bf83643b4d7edcef33f8` | [download](https://zenodo.org/api/records/13255198/files/test.zip/content) |
| `tree_metadata_dev.csv` | 1,216,814 | `603c91ff58c8eab486717ffc82a1b21f` | [download](https://zenodo.org/api/records/13255198/files/tree_metadata_dev.csv/content) |

The upstream description permits development data to be split for model
development. Species labels for the original test split are withheld for the
Codabench classification benchmark. The release must not reconstruct, infer,
or publish those withheld labels.

## Rights metadata

### Dataset

The current [Zenodo record
metadata](https://zenodo.org/api/records/13255198) identifies the dataset
license as `GPL-3.0-or-later`. The [DataCite record for the same
DOI](https://api.datacite.org/dois/10.5281/zenodo.13255198) lists both
`GPL-3.0-or-later` and `AGPL-3.0-or-later`. An [institutional record from a
dataset contributor](https://biblio.ugent.be/publication/01J9RJN98GQ9EZ2HBJE3A9GPC8)
instead reports `CC-BY-4.0`.

These records do not establish one consistent redistribution license. The
article's publication license is separate from the dataset license and does
not resolve this conflict. The dataset also combines 25 source collections,
including public datasets and in-kind contributions, as described in the
paper.

GPL can cover non-software works when the preferred form for modification is
defined. Distribution of a modified covered work generally requires preserved
notices, a copy of the license, clear modification notices and dates, and
continued GPL coverage. See the [GNU GPLv3
text](https://www.gnu.org/licenses/gpl.en.html) and [GNU license
FAQ](https://www.gnu.org/licenses/gpl-faq.en.html).

Accordingly, the review release must not assign a new license to copied or
derived FOR-species20K point clouds. Distribution of a pre-generated TreeBench
dataset requires written clarification from the Zenodo record owner identifying
the applicable data license, followed by full attribution, change notices,
license propagation, and release of the preferred editable form.

### Upstream code

The [permanent Zenodo code
snapshot](https://zenodo.org/api/records/14650218) is marked `CC-BY-4.0`.
CC BY 4.0 permits sharing and adaptation with appropriate credit, a license
link, and an indication of changes. See the [official CC BY 4.0
terms](https://creativecommons.org/licenses/by/4.0/).

The updated GitHub repository currently has no detected license. GitHub's
repository metadata reports `license: null`, and its license endpoint returns
no license file. Do not copy later GitHub-only changes unless a separate license
or permission covers them. Code copied from the fixed Zenodo snapshot must keep
its CC BY attribution and change notice.

## Required release artifacts

### Source manifest

Record at least:

- dataset title and citation;
- version DOI and official file checksums;
- required source archives;
- source tree identifier to benchmark sample identifier mapping;
- the `>= 8,192` observed-point eligibility decision;
- fixed train, validation, in-distribution test, and out-of-distribution test
  membership;
- all task variants, view identifiers, random seeds, and output checksums; and
- generator version and configuration hash.

Include only metadata needed to reproduce the benchmark. Omit species labels,
precise geographic coordinates, and other source fields that the benchmark
does not use.

### Download and generation code

The data preparation command should:

1. accept an existing local Zenodo download or fetch the official files;
2. verify size and MD5 before extraction;
3. select exactly the samples listed in the manifest;
4. generate the fixed 8,192-point references;
5. generate all upsampling, completion, visibility, and OOD variants with fixed
   seeds;
6. write a generated-data manifest and checksums; and
7. refuse to continue when source or output validation fails.

### Synthetic smoke fixture

The repository may include a small point cloud created independently by the
release code. Label it as a functional test fixture, not as a benchmark sample
or evidence supporting paper results. Its test should exercise loading,
generation, metric computation, and schema validation without downloading the
full upstream dataset.

## Prohibited release actions

- Do not include `dev.zip`, `test.zip`, source `.laz` files, or a mirror of the
  upstream dataset.
- Do not include pre-generated derivative point clouds until the dataset
  license conflict is resolved.
- Do not describe third-party data as covered by the release code license.
- Do not relabel the data as MIT, Apache-2.0, BSD, CC BY, or another license.
- Do not remove the upstream citation or rights notices for anonymity.
- Do not copy unlicensed GitHub-only code.
- Do not publish the original withheld classification labels.

Upstream attribution identifies a third-party resource and must remain in an
anonymous submission.
