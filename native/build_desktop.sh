#!/usr/bin/env bash
# Build the desktop-Linux ExecuTorch JNI shim against a prebuilt runtime ($ET_INSTALL)
# and stage the .so for JAR bundling. The runtime itself is built per the Prerequisite.
# See docs/superpowers/plans/2026-06-28-executorch-desktop-linux-native-build.md
set -euo pipefail

# Run from the repo root regardless of where this script is invoked from
# (the relative paths below — native/, src/main/... — assume the repo root as CWD).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ET_INSTALL="${ET_INSTALL:-$HOME/workspace/executorch/cmake-out}"
JOBS="${JOBS:-$(nproc)}"

test -f "${ET_INSTALL}/lib/cmake/ExecuTorch/executorch-config.cmake" \
  || { echo "ET_INSTALL=${ET_INSTALL} has no executorch-config.cmake; build the runtime first (see Prerequisite)"; exit 1; }

rm -rf native/build
cmake -B native/build -S native -G Ninja -DET_INSTALL="${ET_INSTALL}"
cmake --build native/build -j"${JOBS}"

OUT="src/main/resources/native/linux-x86_64"
mkdir -p "${OUT}"
cp native/build/libexecutorch_djl.so "${OUT}/"
echo "Artifact: ${OUT}/libexecutorch_djl.so"
ls -lh "${OUT}/libexecutorch_djl.so"

# Third-party notices from the prebuilt runtime ($ET_INSTALL) — parity with native/build.sh.
test -f "${ET_INSTALL}/LICENSE" && test -d "${ET_INSTALL}/THIRD-PARTY-NOTICES" \
  || { echo "runtime notices missing under ${ET_INSTALL} (LICENSE + THIRD-PARTY-NOTICES/)"; exit 1; }
LIC_OUT="${OUT}/licenses"
rm -rf "${LIC_OUT}"
mkdir -p "${LIC_OUT}"
cp "${ET_INSTALL}/LICENSE" "${LIC_OUT}/"
cp -r "${ET_INSTALL}/THIRD-PARTY-NOTICES" "${LIC_OUT}/"
echo "Notices: ${LIC_OUT} ($(find "${LIC_OUT}" -type f | wc -l) files)"