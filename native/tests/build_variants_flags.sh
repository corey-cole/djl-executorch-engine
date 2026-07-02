#!/usr/bin/env bash
# Fast host test for build.sh's ET_VARIANT flag map via the PRINT_ET_FLAGS early-exit path.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

out="$(PRINT_ET_FLAGS=1 bash native/build.sh)"
grep -q 'ET_VARIANT=logging' <<<"$out" || fail "default variant not logging"
grep -q 'EXECUTORCH_ENABLE_LOGGING=ON' <<<"$out" || fail "default flags missing LOGGING=ON"
grep -q 'ET_INSTALL=/workspace/et-install\b' <<<"$out" || fail "default ET_INSTALL changed"
grep -q 'ET_BUILD=/workspace/et-cmake-out\b' <<<"$out" || fail "default ET_BUILD changed"
grep -q 'STAGE_SO=1' <<<"$out" || fail "default STAGE_SO not 1"

out="$(ET_VARIANT=bare PRINT_ET_FLAGS=1 bash native/build.sh)"
grep -q 'EXECUTORCH_ENABLE_LOGGING=OFF' <<<"$out" || fail "bare missing LOGGING=OFF"
grep -q 'DEVTOOLS' <<<"$out" && fail "bare should not enable devtools"

out="$(ET_VARIANT=devtools PRINT_ET_FLAGS=1 bash native/build.sh)"
grep -q 'EXECUTORCH_BUILD_DEVTOOLS=ON' <<<"$out" || fail "devtools missing DEVTOOLS=ON"
grep -q 'EXECUTORCH_ENABLE_EVENT_TRACER=ON' <<<"$out" || fail "devtools missing EVENT_TRACER=ON"
grep -q 'EXECUTORCH_ENABLE_LOGGING=OFF' <<<"$out" || fail "devtools should hold logging OFF"

out="$(ET_VARIANT=bare ET_INSTALL=/tmp/xi ET_BUILD=/tmp/xb STAGE_SO=0 PRINT_ET_FLAGS=1 bash native/build.sh)"
grep -q 'ET_INSTALL=/tmp/xi\b' <<<"$out" || fail "ET_INSTALL override ignored"
grep -q 'ET_BUILD=/tmp/xb\b' <<<"$out" || fail "ET_BUILD override ignored"
grep -q 'STAGE_SO=0' <<<"$out" || fail "STAGE_SO override ignored"

if ET_VARIANT=bogus PRINT_ET_FLAGS=1 bash native/build.sh >/dev/null 2>&1; then
  fail "unknown ET_VARIANT should exit non-zero"
fi

echo "PASS: build.sh flag map"
