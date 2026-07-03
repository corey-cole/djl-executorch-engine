#!/usr/bin/env bash
# Build + run the native QA targets (Catch2 units + ASan/LSan leak harness) against the resolved
# ExecuTorch runtime (default fetches the pinned logging tarball; runtime fetched by CMake, or set
# ET_INSTALL). Not part of the shipping build — the QA targets are gated behind -DET_BUILD_QA=ON
# and built with AddressSanitizer/LeakSanitizer, so they are a distinct build tree (native/asan)
# from the Release .so (native/build via native/build.sh).
#
# Prerequisites: a C++ compiler with -fsanitize=address, cmake + make, and network access (Catch2
# and the ExecuTorch runtime tarball are fetched at configure time).
#
# CI env vars: ET_RUNTIME_VARIANT (default logging), ET_INSTALL (escape hatch). The QA targets are
# JVM-free and the shared native/CMakeLists.txt skips the JNI shim under -DET_BUILD_QA=ON, so NO
# JAVA_HOME/JDK is needed. In GitHub Actions, run this in the SAME manylinux_2_28 container as
# native/build.sh (matching gcc-toolset toolchain).
set -euo pipefail

# Run from the repo root regardless of where this script is invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ITERS="${ITERS:-1000}"

# QA is the only ASan consumer; install the toolset's ASan runtime here (moved out of build.sh).
if command -v dnf >/dev/null 2>&1; then
  TOOLSET_VER="$(gcc -dumpversion | cut -d. -f1)"
  dnf install -y -q "gcc-toolset-${TOOLSET_VER}-libasan-devel" || true
fi

# Runtime comes from CMake resolution: default fetches the pinned logging tarball; set ET_INSTALL
# to point at an existing install (escape hatch). No precondition to check here.
ET_ARGS=(-DET_RUNTIME_VARIANT="${ET_RUNTIME_VARIANT:-logging}")
[ -n "${ET_INSTALL:-}" ] && ET_ARGS+=(-DET_INSTALL="${ET_INSTALL}")

cmake -B native/asan -S native -G "Unix Makefiles" "${ET_ARGS[@]}" -DET_BUILD_QA=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"
cmake --build native/asan --target et_runtime_test et_leak_harness

echo "--- Catch2 unit suite ---"
./native/asan/et_runtime_test

echo "--- ASan/LSan leak harness (${ITERS} iterations) ---"
./native/asan/et_leak_harness native/spike/add.pte "${ITERS}"
