#!/usr/bin/env bash
# Build + run the native QA targets (Catch2 units + ASan/LSan leak harness) against a
# prebuilt ExecuTorch runtime ($ET_INSTALL). Not part of the shipping build — the QA targets
# are gated behind -DET_BUILD_QA=ON and built with AddressSanitizer/LeakSanitizer, so they are
# a distinct build tree (native/asan) from the Release .so (native/build via build_desktop.sh).
#
# Prerequisites: an ExecuTorch runtime built per native/build.sh (ET_INSTALL), a clang/gcc with
# -fsanitize=address, cmake + ninja, and network access (Catch2 is fetched at configure time).
set -euo pipefail

# Run from the repo root regardless of where this script is invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# This default is specific to my local directory structure; override with ET_INSTALL to point to a different runtime.
ET_INSTALL="${ET_INSTALL:-$HOME/workspace/executorch/cmake-out}"
ITERS="${ITERS:-1000}"

test -f "${ET_INSTALL}/lib/cmake/ExecuTorch/executorch-config.cmake" \
  || { echo "ET_INSTALL=${ET_INSTALL} has no executorch-config.cmake; build the runtime first (native/build.sh)"; exit 1; }

cmake -B native/asan -S native -DET_INSTALL="${ET_INSTALL}" -DET_BUILD_QA=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"
cmake --build native/asan --target et_runtime_test et_leak_harness

echo "--- Catch2 unit suite ---"
./native/asan/et_runtime_test

echo "--- ASan/LSan leak harness (${ITERS} iterations) ---"
./native/asan/et_leak_harness native/spike/add.pte "${ITERS}"
