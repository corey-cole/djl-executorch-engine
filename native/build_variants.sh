#!/usr/bin/env bash
# Build all three ExecuTorch runtime variants (bare/logging/devtools), time each, and print a
# comparison table with pairwise deltas vs. bare. Run INSIDE the manylinux_2_28 container (same as
# native/build.sh) so the torch wheel is installed once by the first variant and reused. Never
# stages the shipped artifact (STAGE_SO=0). See docs/benchmarking.md for how to read the numbers.
#
# Env: ITERS/WARMUP forwarded to bench.sh. BUILD_SH/BENCH_SH/WORKSPACE/NATIVE_BUILD_DIR are
# override seams (tests inject stubs; WORKSPACE defaults to the container's /workspace mount;
# NATIVE_BUILD_DIR defaults to native/build where build.sh leaves the shim .so).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BUILD_SH="${BUILD_SH:-native/build.sh}"
BENCH_SH="${BENCH_SH:-native/bench.sh}"
WORKSPACE="${WORKSPACE:-/workspace}"
NATIVE_BUILD_DIR="${NATIVE_BUILD_DIR:-native/build}"
VARIANTS=(bare logging devtools)

RESULTS_DIR="native/bench-results"
mkdir -p "${RESULTS_DIR}"
RESULTS="${RESULTS_DIR}/variants-$(date -u +%Y%m%dT%H%M%SZ).txt"

declare -A SIZE LOAD COLD WARM

for v in "${VARIANTS[@]}"; do
  install="${WORKSPACE}/et-install-${v}"
  build="${WORKSPACE}/et-cmake-out-${v}"

  skip=0
  [ -f "${install}/lib/cmake/ExecuTorch/executorch-config.cmake" ] && skip=1  # reuse on re-run

  echo "=== variant ${v} (SKIP_ET_BUILD=${skip}) ==="
  ET_VARIANT="${v}" ET_INSTALL="${install}" ET_BUILD="${build}" STAGE_SO=0 \
    SKIP_ET_BUILD="${skip}" NATIVE_BUILD_DIR="${NATIVE_BUILD_DIR}" bash "${BUILD_SH}"

  SIZE[$v]="$(stat -c%s "${NATIVE_BUILD_DIR}/libexecutorch_djl.so")"

  line="$(ET_INSTALL="${install}" ITERS="${ITERS:-1000}" WARMUP="${WARMUP:-100}" \
          bash "${BENCH_SH}" | grep '^et_timing:')"
  echo "${line}" >> "${RESULTS}"
  LOAD[$v]="$(sed -n 's/.*[[:space:]]load_ms=\([0-9.]*\).*/\1/p' <<<"${line}")"
  COLD[$v]="$(sed -n 's/.*[[:space:]]cold_ms=\([0-9.]*\).*/\1/p' <<<"${line}")"
  WARM[$v]="$(sed -n 's/.*[[:space:]]warm_mean_ms=\([0-9.]*\).*/\1/p' <<<"${line}")"
done

table() {
  printf '%-10s %14s %10s %10s %14s\n' variant shim_so_bytes load_ms cold_ms warm_mean_ms
  for v in "${VARIANTS[@]}"; do
    printf '%-10s %14s %10s %10s %14s\n' "$v" "${SIZE[$v]}" "${LOAD[$v]}" "${COLD[$v]}" "${WARM[$v]}"
  done
  echo
  echo "deltas vs bare (each isolates one axis):"
  awk -v lb="${SIZE[logging]}" -v bb="${SIZE[bare]}" -v db="${SIZE[devtools]}" \
      -v lw="${WARM[logging]}" -v bw="${WARM[bare]}" -v dw="${WARM[devtools]}" 'BEGIN{
    printf "  logging - bare : shim %+d bytes (%+.1f%%), warm_mean %+.3f ms\n",
           lb-bb, (bb ? (lb-bb)*100.0/bb : 0), lw-bw
    printf "  devtools - bare: shim %+d bytes (%+.1f%%), warm_mean %+.3f ms\n",
           db-bb, (bb ? (db-bb)*100.0/bb : 0), dw-bw
  }'
}

table | tee -a "${RESULTS}"
echo "Wrote ${RESULTS}"
