#!/usr/bin/env bash
# Build + run the Release timing harness (no sanitizers) against the resolved ExecuTorch runtime
# (default fetches the pinned tarball for $ET_RUNTIME_VARIANT; runtime is fetched by CMake, or set
# ET_INSTALL to point at an existing install). A cheap gross-regression screen for the
# logging/devtools ship-or-not decision; see docs/benchmarking.md. Times ONE variant per run —
# native/build_variants.sh drives it across the bare/logging/devtools configs.
#
# CI env: ET_RUNTIME_VARIANT (default logging), ET_INSTALL (escape hatch). ITERS (default 1000) and
# WARMUP (default 100) tune the timed loop; MODEL picks the .pte (default the mobilenet demo
# artifact, falling back to add.pte). Run in the SAME manylinux_2_28 container as
# native/build.sh so the toolchain matches the runtime. No JDK needed: the ET_BUILD_BENCH configure
# skips the JNI shim.
set -euo pipefail

# shellcheck source=native/container_env.sh
. "${BASH_SOURCE[0]%/*}/container_env.sh"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ET_RUNTIME_VARIANT="${ET_RUNTIME_VARIANT:-logging}"
ITERS="${ITERS:-1000}"
WARMUP="${WARMUP:-100}"

# add.pte is small enough that per-call overhead, not the runtime, dominates -- so a build-flag
# delta measured on it is buried under timer resolution. Prefer the XNNPACK-delegated mobilenet.
if [ -z "${MODEL:-}" ]; then
  if [ -f example/build/models/mobilenet_v2.pte ]; then
    MODEL=example/build/models/mobilenet_v2.pte
  else
    MODEL=native/spike/add.pte
    echo "WARNING: mobilenet_v2.pte not found (./gradlew :example:exportModels); falling back to" >&2
    echo "         ${MODEL}, which is too small to resolve a build-flag delta." >&2
  fi
fi

# Runtime comes from CMake resolution: default fetches the pinned ${ET_RUNTIME_VARIANT} tarball;
# set ET_INSTALL to point at an existing install (escape hatch). No precondition to check here.
ET_ARGS=(-DET_RUNTIME_VARIANT="${ET_RUNTIME_VARIANT}")
[ -n "${ET_INSTALL:-}" ] && ET_ARGS+=(-DET_INSTALL="${ET_INSTALL}")

# Release, no sanitizer, own build tree (distinct from native/asan QA and native/build shim).
# Drop the tree only if it was configured for a different source root (container vs host); a same-root
# re-run keeps it so the FetchContent'd runtime tarball (native/bench/_deps) is reused. CLEAN=1 forces.
et_chown_outputs_on_exit native/bench native/bench-results
bash native/clean_stale_tree.sh native/bench native
cmake -B native/bench -S native -G "Unix Makefiles" "${ET_ARGS[@]}" \
  -DET_BUILD_BENCH=ON -DCMAKE_BUILD_TYPE=Release
cmake --build native/bench --target et_timing_harness

echo "--- Release timing harness (model=${MODEL} iters=${ITERS} warmup=${WARMUP}) ---"
./native/bench/et_timing_harness "${MODEL}" "${ITERS}" "${WARMUP}"
