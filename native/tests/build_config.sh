#!/usr/bin/env bash
# Host-fast test of build.sh's config seam via PRINT_BUILD_CONFIG, plus a guard that Stage-A is gone.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

out="$(PRINT_BUILD_CONFIG=1 bash native/build.sh)"
grep -q 'ET_RUNTIME_VARIANT=logging' <<<"${out}" || fail "default variant not logging"
grep -q 'STAGE_SO=1'                  <<<"${out}" || fail "default STAGE_SO not 1"
grep -q 'NATIVE_BUILD_DIR=native/build\b' <<<"${out}" || fail "default NATIVE_BUILD_DIR changed"

out="$(ET_RUNTIME_VARIANT=bare STAGE_SO=0 NATIVE_BUILD_DIR=/tmp/nb ET_INSTALL=/tmp/xi \
       PRINT_BUILD_CONFIG=1 bash native/build.sh)"
grep -q 'ET_RUNTIME_VARIANT=bare'  <<<"${out}" || fail "variant override ignored"
grep -q 'STAGE_SO=0'               <<<"${out}" || fail "STAGE_SO override ignored"
grep -q 'NATIVE_BUILD_DIR=/tmp/nb\b' <<<"${out}" || fail "NATIVE_BUILD_DIR override ignored"
grep -q 'ET_INSTALL=/tmp/xi\b'     <<<"${out}" || fail "ET_INSTALL passthrough missing"

# Stage-A must be fully gone from build.sh.
grep -qE 'SKIP_ET_BUILD|EXECUTORCH_ENABLE_LOGGING|torch==2\.12|avx512vnni' native/build.sh \
  && fail "Stage-A remnants still present in build.sh"

echo "PASS: build.sh config"
