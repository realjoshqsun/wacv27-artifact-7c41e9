# WACV 2027 anonymous-release audit

Audit date: 2026-08-28. This report distinguishes conference rules, the
Evaluations & Datasets (E&D) guidance that WACV asks authors to follow, and
technical limitations of this artifact.

Status labels are **PASS**, **PARTIAL**, **OPEN**, **BLOCKED**, and **N/A**.
`OPEN` requires a submission-time choice or an external artifact. `BLOCKED`
cannot be resolved safely inside this source release.

## Controlling guidance

The [WACV 2027 Author
Guidelines](https://wacv.thecvf.com/Conferences/2027/AuthorGuides) require the
paper and additional material to remain double blind. Code in supplementary
material is voluntary but strongly encouraged; it must be anonymized and
include environment and execution instructions. The code should target the
main results and as many secondary experiments as practical. Supplementary
material must be a PDF or ZIP no larger than 200 MB and may not add results on
new datasets, results from an improved method, or a corrected paper PDF.

The [WACV 2027 Call for
Papers](https://wacv.thecvf.com/Conferences/2027/CallForPapers) introduces the
E&D track and says it will be evaluated similarly to the NeurIPS E&D track.
WACV encourages, but does not state that it wholesale incorporates, the
NeurIPS 2026 rules. The active OpenReview fields therefore control whether an
anonymous `code_URL` is mandatory for this submission.

Under the [NeurIPS 2026 E&D reviewer
guidelines](https://neurips.cc/Conferences/2026/EvaluationsDatasetsReviewerGuidelines),
code is mandatory when the primary contribution is a reusable executable
artifact such as a benchmark platform, evaluation toolkit, or data generator.
It must be accessible, executable, anonymous, and independently documented.
The [E&D
FAQ](https://neurips.cc/Conferences/2026/EvaluationsDatasetsFAQ) says that an
existing public dataset need not be rehosted, but the code that transforms or
uses it must be accessible and executable. A new dataset contribution has
additional hosting, Croissant/RAI metadata, and reviewer-access requirements;
large datasets over 4 GB should also provide a smaller representative sample.

## WACV anonymity and supplementary-material audit

| Check | Status | Evidence or remaining action |
|---|---|---|
| Separate review tree | **PASS** | This directory was assembled independently and contains no inherited `.git` history. |
| Author/affiliation removal | **PASS** | The release gate scans paths and UTF-8 contents for local accounts, private mounts, email addresses, token forms, and known workspace markers. |
| Secret and binary exclusion | **PASS** | Tokens, credential files, notebooks, archives, checkpoints, object files, shared libraries, source clouds, predictions, and logs are rejected. |
| Third-party attribution | **PASS** | Dataset and upstream-model citations, pinned commits, and found upstream licenses are retained. These identify third parties, not the submitting authors. |
| Anonymous Git history creation | **PASS** | The initializer ignores system/global Git configuration and attributes, disables templates, hooks, and signing, fixes all four author/committer identity fields and dates, and verifies the resulting commit. |
| Environment and run instructions | **PASS** | The README, exact core environment, three executed legacy profiles, CLI help, data commands, model runner, and smoke path are present. |
| Supplement format and size | **PASS** | The deterministic packer writes outside the source tree, excludes forbidden artifacts, and enforces the 200,000,000-byte limit. Re-run it on the exact submission snapshot. |
| No unreported/improved experiments | **PASS** | The bundle contains the reported protocol and frozen reported test means, not a revised paper or new-dataset experiment. |
| Manuscript/rebuttal links | **OPEN** | Confirm in the submitted files that no author-owned public repository is cited and that any rebuttal contains no external link. |
| Remote repository metadata | **OPEN** | The user must create the remote, push the single anonymous commit, inspect it logged out, and provide only the URL allowed by the active OpenReview form. |

WACV warns that identity leakage in source code can cause summary rejection.
The source-tree scan is therefore necessary but not sufficient: the final
remote owner/profile, commit page, contributor list, issues, actions, releases,
LFS metadata, analytics, and downloaded remote archive must also be inspected.

## E&D artifact audit

| Area | Status | Evidence or limitation |
|---|---|---|
| Evaluative purpose and claims | **PASS** | `BENCHMARK_CARD.md` states the tasks, supported claims, assumptions, intended/non-intended uses, and limitations. |
| Public source-data access | **PASS** | The exact FOR-species20K Zenodo version, official files, byte counts, and checksums are pinned; the downloader verifies before publishing a file. |
| Deterministic benchmark generator | **PASS** | Filtering, FPS reference construction, normalization, nested subsets, TLS-/ULS-like visibility, seeds, manifests, per-artifact hashes, and validation are executable. |
| Exact split availability | **PASS** | Both 14,884-tree split manifests are bundled and hash-pinned: random 11,907/1,488/1,489 and genus-OOD 12,111/1,386/1,387. |
| Metric implementation | **PASS** | Five point metrics and 27 tree/canopy-fuel metrics are present; all 32 names and parameters are machine-readable. The tree implementation was regression-checked against the production evaluator with zero numerical difference on a finite random batch. |
| CPU usability | **PASS** | Unit tests and the procedural smoke path exercise generation, validation, 4x, evaluation-only 16x, completion, 32 metrics, and taxonomy aggregation without downloading research data. |
| Paper-table reconstruction | **PASS** | `reference_results/full_results.csv` supplies the hash-pinned 94 reported test rows. The reference mode checks identities, counts, all 32 finite metrics, and regenerates the numerical CSV/LaTeX tables. |
| Fresh-result integrity | **PASS** | Fresh table inputs require aggregate, validation/test per-sample files, exact sample counts, run provenance, hashes, 8,192-point outputs, finite metrics, unique sample keys, and the registered 16x source route. |
| Baseline execution coverage | **PARTIAL** | All 16 protocols and 94 scenarios are registered. Ten licensed integrations are materializable; six required source overrides are withheld because no upstream redistribution license was found. See `baselines/README.md`. |
| Full GPU rerun | **OPEN** | CUDA extensions and all training/inference jobs were not rerun from this assembled folder. Static checks and dry-run construction do not replace a clean multi-GPU reproduction. |
| Reviewer checkpoints | **OPEN** | `manifests/checkpoints.json` is intentionally empty until anonymous URLs, sizes, hashes, epochs, and config hashes are available. |
| Byte-identical historical arrays | **BLOCKED** | Historical FPS/subset/view RNG state, poses, and generated-array hashes were not retained. The deterministic generator implements the documented protocol but cannot recreate those frozen arrays byte for byte. |
| Frozen derived-data distribution | **BLOCKED** | Official records disagree on the applicable FOR-species20K rights. The release cannot redistribute the historical derived arrays until the record owner clarifies the license in writing. |
| New-dataset Croissant metadata | **N/A / CONDITIONAL** | This artifact uses an existing public source plus executable transformation code and does not host a new static dataset. If the active WACV form treats the processed TreeBench arrays as a submitted dataset contribution, hosting, valid Croissant/RAI metadata, and reviewer access become required before submission. |
| New TreeBench code license | **OPEN** | No license is assigned to newly written code. Select one only after confirming ownership; no license is needed merely for confidential WACV review, but public reuse remains unlicensed. |

## Interpretation

The folder is a runnable, anonymous benchmark-code artifact and can rebuild the
reported numerical tables from frozen aggregate means. It is not yet a
complete independent reproduction of every paper number from raw data and
weights. The three material gaps are reviewer-accessible checkpoints, the six
license-blocked baseline overrides, and reviewer access to the exact frozen
derived arrays.

For a WACV supplementary-code submission, those gaps should be disclosed
exactly as above. For a strict NeurIPS-style E&D executable-artifact review,
create the anonymous code URL before submission and resolve as many of the
three gaps as possible. Do not describe procedural synthetic smoke data as a
representative sample of the real benchmark.

## Final release gate

Before sharing a URL or uploading the ZIP:

1. run the full test suite, smoke command, release verifier, and deterministic
   packer on the final tree;
2. record the ZIP SHA-256 and size;
3. create one unsigned anonymous commit and push it to a new empty remote;
4. inspect the remote and its downloaded archive while logged out;
5. populate the active OpenReview `code_URL` only if requested, and keep the
   author-owned repository URL out of the manuscript and rebuttal; and
6. recheck the active WACV/OpenReview deadline, because conference pages should
   not be assumed to override the form shown to the authors.
