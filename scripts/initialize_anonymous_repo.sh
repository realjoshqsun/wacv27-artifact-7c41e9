#!/usr/bin/env bash
set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$release_root"

python scripts/verify_release.py
if [[ -e .git ]]; then
  echo "Refusing to replace an existing .git directory." >&2
  exit 1
fi

empty_template="$(mktemp -d)"
cleanup() {
  rmdir -- "$empty_template" 2>/dev/null || true
}
trap cleanup EXIT
git_executable="$(command -v git)"
git_clean() {
  env -i \
    PATH="$PATH" \
    LANG=C \
    LC_ALL=C \
    GIT_CONFIG_NOSYSTEM=1 \
    GIT_CONFIG_GLOBAL=/dev/null \
    GIT_ATTR_NOSYSTEM=1 \
    "$git_executable" "$@"
}

git_clean init -b main --template="$empty_template"
rmdir -- "$empty_template"
trap - EXIT

anonymous_name="Anonymous Authors"
anonymous_email="anonymous"@"users.noreply.github.com"
git_clean config user.name "$anonymous_name"
git_clean config user.email "$anonymous_email"
git_clean config commit.gpgSign false
git_clean config tag.gpgSign false
git_clean config core.hooksPath /dev/null
git_clean add --all -- .
env -i \
  PATH="$PATH" \
  LANG=C \
  LC_ALL=C \
  GIT_CONFIG_NOSYSTEM=1 \
  GIT_CONFIG_GLOBAL=/dev/null \
  GIT_ATTR_NOSYSTEM=1 \
  GIT_AUTHOR_NAME="$anonymous_name" \
  GIT_AUTHOR_EMAIL="$anonymous_email" \
  GIT_COMMITTER_NAME="$anonymous_name" \
  GIT_COMMITTER_EMAIL="$anonymous_email" \
  GIT_AUTHOR_DATE="2026-01-01T00:00:00Z" \
  GIT_COMMITTER_DATE="2026-01-01T00:00:00Z" \
  "$git_executable" -c core.hooksPath=/dev/null -c commit.gpgSign=false \
    commit --no-verify --no-gpg-sign -m "Release anonymous TreeBench review code"

identity="$(git_clean log -1 --format='%an <%ae>|%cn <%ce>')"
expected_identity="$anonymous_name <$anonymous_email>|$anonymous_name <$anonymous_email>"
if [[ "$identity" != "$expected_identity" ]]; then
  echo "Anonymous commit identity verification failed." >&2
  exit 1
fi
commit_object="$(git_clean cat-file commit HEAD)"
if [[ "$commit_object" == gpgsig\ * || "$commit_object" == *$'\ngpgsig '* ]]; then
  echo "Anonymous commit unexpectedly contains a signature." >&2
  exit 1
fi
if [[ -n "$(git_clean status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "Anonymous repository is not clean after initialization." >&2
  exit 1
fi

echo "Created a clean local repository. Inspect git log/show before adding a remote."
