#!/usr/bin/env bash
# Benchmark all three ExecuTorch runtime variants (bare/logging/devtools) and print a comparison
# table with warm-mean deltas vs. bare. Each variant's prebuilt tarball is DOWNLOADED by CMake
# (FetchContent) inside bench.sh — no from-source build. Run INSIDE manylinux_2_28 (matches the
# shipped toolchain). See docs/benchmarking.md.
#
# Env: ITERS/WARMUP forwarded to bench.sh. BENCH_SH/WORKSPACE are override seams (tests inject a
# stub bench.sh; WORKSPACE is unused by the real path but kept so the stub test can sandbox).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BENCH_SH="${BENCH_SH:-native/bench.sh}"
VARIANTS=(bare logging devtools)

RESULTS_DIR="native/bench-results"
mkdir -p "${RESULTS_DIR}"
RESULTS="${RESULTS_DIR}/variants-$(date -u +%Y%m%dT%H%M%SZ).txt"

declare -A LOAD COLD WARM
for v in "${VARIANTS[@]}"; do
  echo "=== variant ${v} ==="
  line="$(ET_RUNTIME_VARIANT="${v}" ITERS="${ITERS:-1000}" WARMUP="${WARMUP:-100}" \
          bash "${BENCH_SH}" | grep '^et_timing:')"
  echo "${line}" >> "${RESULTS}"
  LOAD[$v]="$(sed -n 's/.*[[:space:]]load_ms=\([0-9.]*\).*/\1/p' <<<"${line}")"
  COLD[$v]="$(sed -n 's/.*[[:space:]]cold_ms=\([0-9.]*\).*/\1/p' <<<"${line}")"
  WARM[$v]="$(sed -n 's/.*[[:space:]]warm_mean_ms=\([0-9.]*\).*/\1/p' <<<"${line}")"
done

table() {
  printf '%-10s %10s %10s %14s\n' variant load_ms cold_ms warm_mean_ms
  for v in "${VARIANTS[@]}"; do
    printf '%-10s %10s %10s %14s\n' "$v" "${LOAD[$v]}" "${COLD[$v]}" "${WARM[$v]}"
  done
  echo
  echo "deltas vs bare (warm_mean isolates the runtime cost of each build flag):"
  awk -v lw="${WARM[logging]}" -v bw="${WARM[bare]}" -v dw="${WARM[devtools]}" 'BEGIN{
    printf "  logging - bare : warm_mean %+.3f ms\n", lw-bw
    printf "  devtools - bare: warm_mean %+.3f ms\n", dw-bw
  }'
}

table | tee -a "${RESULTS}"
echo "Wrote ${RESULTS}"
