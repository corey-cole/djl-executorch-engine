#!/usr/bin/env bash
# Build + run the native QA targets (Catch2 units + ASan/LSan leak harness) against a
# prebuilt ExecuTorch runtime ($ET_INSTALL). Not part of the shipping build — the QA targets
# are gated behind -DET_BUILD_QA=ON and built with AddressSanitizer/LeakSanitizer, so they are
# a distinct build tree (native/asan) from the Release .so (native/build via native/build.sh).
#
# Prerequisites: an ExecuTorch runtime built per native/build.sh (ET_INSTALL), a C++ compiler with
# -fsanitize=address, cmake + make, and network access (Catch2 is fetched at configure time).
#
# CI env vars: ET_INSTALL is the ONLY required variable. The QA targets are JVM-free and the shared
# native/CMakeLists.txt skips the JNI shim under -DET_BUILD_QA=ON, so NO JAVA_HOME/JDK is needed.
# In GitHub Actions, run this in the SAME manylinux_2_28 container as native/build.sh (matching
# gcc-toolset toolchain); build.sh publishes ET_INSTALL to $GITHUB_ENV, so a later `run:` step that
# calls this script inherits it automatically.
set -euo pipefail

# Run from the repo root regardless of where this script is invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Default to the runtime produced by the container build (native/build.sh installs to
# ${REPO_ROOT}/et-install, which lives under the /workspace mount so it persists on the host).
# Override ET_INSTALL to point at a different runtime. Note: that runtime is built in the
# manylinux_2_28 container (gcc-toolset-14); if you hit cross-toolchain link/ASan noise running QA
# on the host, either point ET_INSTALL at a host-built runtime or run this script in the container.
ET_INSTALL="${ET_INSTALL:-${REPO_ROOT}/et-install}"
ITERS="${ITERS:-1000}"

test -f "${ET_INSTALL}/lib/cmake/ExecuTorch/executorch-config.cmake" \
  || { echo "ET_INSTALL=${ET_INSTALL} has no executorch-config.cmake; build the runtime first (native/build.sh)"; exit 1; }

# Pin the Make generator so the build doesn't depend on the container's CMAKE_GENERATOR (the image
# may default to Ninja, which isn't guaranteed on PATH in a fresh QA step).
cmake -B native/asan -S native -G "Unix Makefiles" -DET_INSTALL="${ET_INSTALL}" -DET_BUILD_QA=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"
cmake --build native/asan --target et_runtime_test et_leak_harness

echo "--- Catch2 unit suite ---"
./native/asan/et_runtime_test

echo "--- ASan/LSan leak harness (${ITERS} iterations) ---"
./native/asan/et_leak_harness native/spike/add.pte "${ITERS}"
