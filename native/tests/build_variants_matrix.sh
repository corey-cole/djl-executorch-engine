#!/usr/bin/env bash
# Fast host test for build_variants.sh orchestration using stub drivers (no real builds).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
mkdir -p "${TMP}/bin" "${TMP}/nativebuild" native/bench-results

# Stub build.sh: fabricates a per-variant-sized shim + install config; logs SKIP_ET_BUILD received.
cat > "${TMP}/bin/build.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "stub-build ${ET_VARIANT} SKIP_ET_BUILD=${SKIP_ET_BUILD}" >> "${STUB_LOG}"
mkdir -p "${ET_INSTALL}/lib/cmake/ExecuTorch"
: > "${ET_INSTALL}/lib/cmake/ExecuTorch/executorch-config.cmake"
mkdir -p "${NATIVE_BUILD_DIR}"
case "${ET_VARIANT}" in
  bare)     head -c 1000 /dev/zero > "${NATIVE_BUILD_DIR}/libexecutorch_djl.so" ;;
  logging)  head -c 1350 /dev/zero > "${NATIVE_BUILD_DIR}/libexecutorch_djl.so" ;;
  devtools) head -c 1200 /dev/zero > "${NATIVE_BUILD_DIR}/libexecutorch_djl.so" ;;
esac
EOF
chmod +x "${TMP}/bin/build.sh"

# Stub bench.sh: emits an et_timing line whose warm_mean varies by install prefix.
cat > "${TMP}/bin/bench.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
case "${ET_INSTALL}" in
  *bare)     wm=0.100 ;;
  *logging)  wm=0.110 ;;
  *devtools) wm=0.150 ;;
  *)         wm=0.000 ;;
esac
echo "et_timing: model=add.pte iters=5 warmup=1 load_ms=1.000 cold_ms=0.500 warm_min_ms=0.090 warm_mean_ms=${wm} warm_max_ms=0.200 sink=1"
EOF
chmod +x "${TMP}/bin/bench.sh"

export STUB_LOG="${TMP}/stub.log"
: > "${STUB_LOG}"

out="$(BUILD_SH="${TMP}/bin/build.sh" BENCH_SH="${TMP}/bin/bench.sh" WORKSPACE="${TMP}/ws" \
       NATIVE_BUILD_DIR="${TMP}/nativebuild" \
       bash native/build_variants.sh)"

# Table has a row per variant with the fabricated sizes.
grep -qE '^bare .* 1000 ' <<<"$out" || fail "bare row/size missing"
grep -qE '^logging .* 1350 ' <<<"$out" || fail "logging row/size missing"
grep -qE '^devtools .* 1200 ' <<<"$out" || fail "devtools row/size missing"

# Deltas vs bare: logging shim +350 bytes; devtools warm_mean +0.050 ms.
grep -q 'logging - bare' <<<"$out" || fail "logging delta line missing"
grep -q '+350 bytes' <<<"$out" || fail "logging shim delta wrong"
grep -q 'devtools - bare' <<<"$out" || fail "devtools delta line missing"
grep -qE 'warm_mean \+0\.050' <<<"$out" || fail "devtools warm delta wrong"

# Results file written with the three verbatim et_timing lines.
latest="$(ls -t native/bench-results/variants-*.txt | head -1)"
test "$(grep -c '^et_timing:' "${latest}")" -eq 3 || fail "results file missing et_timing lines"

# First run: no reuse (fresh workspace) -> SKIP_ET_BUILD=0 for every variant.
test "$(grep -c 'SKIP_ET_BUILD=0' "${STUB_LOG}")" -eq 3 || fail "first run should not skip"

# Second run: installs now exist -> SKIP_ET_BUILD=1 for every variant.
: > "${STUB_LOG}"
BUILD_SH="${TMP}/bin/build.sh" BENCH_SH="${TMP}/bin/bench.sh" WORKSPACE="${TMP}/ws" \
  NATIVE_BUILD_DIR="${TMP}/nativebuild" \
  bash native/build_variants.sh >/dev/null
test "$(grep -c 'SKIP_ET_BUILD=1' "${STUB_LOG}")" -eq 3 || fail "second run should reuse"

echo "PASS: build_variants.sh matrix"
