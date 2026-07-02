#!/usr/bin/env bash
# Build + run the Release timing harness (no sanitizers) against a prebuilt ExecuTorch runtime
# ($ET_INSTALL). A cheap gross-regression screen for the logging/devtools ship-or-not decision;
# see docs/benchmarking.md. Times ONE install prefix per run — native/build_variants.sh drives it
# across the bare/logging/devtools configs.
#
# CI env: ET_INSTALL (required; same contract as build_qa.sh). ITERS (default 1000) and WARMUP
# (default 100) tune the timed loop. Run in the SAME manylinux_2_28 container as native/build.sh so
# the toolchain matches the runtime. No JDK needed: the ET_BUILD_BENCH configure skips the JNI shim.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ET_INSTALL="${ET_INSTALL:-${REPO_ROOT}/et-install}"
ITERS="${ITERS:-1000}"
WARMUP="${WARMUP:-100}"

test -f "${ET_INSTALL}/lib/cmake/ExecuTorch/executorch-config.cmake" \
  || { echo "ET_INSTALL=${ET_INSTALL} has no executorch-config.cmake; build the runtime first (native/build.sh)"; exit 1; }

# Release, no sanitizer, own build tree (distinct from native/asan QA and native/build shim).
cmake -B native/bench -S native -G "Unix Makefiles" -DET_INSTALL="${ET_INSTALL}" \
  -DET_BUILD_BENCH=ON -DCMAKE_BUILD_TYPE=Release
cmake --build native/bench --target et_timing_harness

echo "--- Release timing harness (iters=${ITERS} warmup=${WARMUP}) ---"
./native/bench/et_timing_harness native/spike/add.pte "${ITERS}" "${WARMUP}"
