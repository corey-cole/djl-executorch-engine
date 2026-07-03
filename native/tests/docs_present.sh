#!/usr/bin/env bash
# Host-fast presence checks for the documentation deliverables.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

test -f docs/executorch-build-notes.md || fail "build-notes doc missing"
grep -q 'executorch-runtime-dist' docs/executorch-build-notes.md || fail "build-notes must point at Repo A"
grep -qi 'glibc' docs/executorch-build-notes.md || fail "build-notes must cover the glibc floor"
grep -qi 'vnni' docs/executorch-build-notes.md || fail "build-notes must cover the VNNI check rationale"

grep -q 'gh attestation verify' README.md || fail "README must document local attestation verify"
grep -qi 'FetchContent\|downloads\|prebuilt' README.md || fail "README must describe the fetch model"

grep -q 'et-install/' .gitignore && fail "obsolete et-install/ still ignored (no longer produced)"
echo "PASS: docs present"
