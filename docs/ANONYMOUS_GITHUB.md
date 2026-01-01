# Anonymous Repository Handoff

## Reviewer delivery

The WACV review artifact should be the verified ZIP uploaded through
OpenReview. WACV permits code in the supplementary ZIP, limits the upload to
200 MB, and requires the code and all additional material to remain anonymous.
The conference discourages external links, and rebuttals cannot contain links,
including anonymous ones.

An anonymous repository is useful as a staging copy or when the active
submission form explicitly requests a code URL. It does not replace the exact
ZIP submitted by the supplementary deadline.

## Create a clean source repository

Do not fork, mirror, or import the research repository. Those operations retain
owner or history links. From the verified release directory, run:

```bash
python scripts/verify_release.py
bash scripts/initialize_anonymous_repo.sh
git log --format=fuller --show-signature
git config --local --list
git remote -v
```

The initialization script creates one root commit with the local identity
`Anonymous Authors`. It refuses an existing `.git` directory. Add a remote only
after inspecting the commit and repository contents.

Create the remote with an account and repository name that do not identify the
submission. Do not reuse an organization, avatar, profile text, SSH key label,
verified signature, personal access-token name, or public activity that links
the account to an author. Never store a token, credential helper, or remote URL
containing credentials in this directory.

## GitHub-side checklist

- Start from an empty repository; do not initialize it with a README, license,
  or `.gitignore`.
- Push only the single clean release history.
- Keep Actions disabled unless a workflow and every log line have passed the
  anonymity scan.
- Keep Issues, Discussions, Projects, sponsorship, Pages, releases, and package
  publishing disabled during review unless they are required and anonymous.
- Do not add identifiable collaborators, CODEOWNERS, signed commits, badges,
  acknowledgements, funding metadata, or personal domains.
- Inspect the public file browser, commit page, contributor list, archive name,
  release metadata, and raw-file URLs from a logged-out browser.
- Download the remote archive and rerun `scripts/verify_release.py` on the
  extracted tree before sharing it.

## Optional read-only anonymized mirror

[Anonymous GitHub](https://anonymous.4open.science/) can expose a read-only,
identity-stripped mirror of a GitHub repository. Configure explicit redaction
terms for every author name, username, affiliation, email, project codename,
and private path fragment. Disable images, PDFs, notebooks, comments, and links
unless each item has been inspected. Use a random slug and a review-period
expiration date. Check the rendered mirror from a logged-out browser and
download its archive for a second scan.

Do not put the resulting URL in the WACV manuscript, supplement, or rebuttal
unless the current conference instructions or submission form explicitly
permit it. The OpenReview supplementary ZIP remains the conservative delivery
route.

## Final command sequence

After the code-license choice and checkpoint decision are recorded:

```bash
pytest
python scripts/verify_release.py
python scripts/package_review.py --output ../treebench_review.zip
sha256sum ../treebench_review.zip
```

Keep the ZIP hash with the submission record. A later repository update does
not change the artifact that reviewers received through OpenReview.
