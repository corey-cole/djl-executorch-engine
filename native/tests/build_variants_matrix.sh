#!/usr/bin/env bash
# Host-fast test: build_variants.sh loops the 3 variants through a STUBBED bench.sh (keyed on
# ET_RUNTIME_VARIANT) and produces the table + deltas. No real build/download.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

STUB_DIR="$(mktemp -d)"
BENCH_STUB="${STUB_DIR}/bench.sh"
cat > "${BENCH_STUB}" <<'EOF'
#!/usr/bin/env bash
case "${ET_RUNTIME_VARIANT}" in
  bare)     echo "et_timing: model=add.pte iters=10 warmup=1 load_ms=5.0 cold_ms=2.0 warm_min_ms=1.0 warm_mean_ms=1.100 warm_max_ms=1.2 sink=0";;
  logging)  echo "et_timing: model=add.pte iters=10 warmup=1 load_ms=5.5 cold_ms=2.1 warm_min_ms=1.1 warm_mean_ms=1.250 warm_max_ms=1.3 sink=0";;
  devtools) echo "et_timing: model=add.pte iters=10 warmup=1 load_ms=6.0 cold_ms=2.2 warm_min_ms=1.2 warm_mean_ms=1.400 warm_max_ms=1.5 sink=0";;
  *) echo "bad variant" >&2; exit 1;;
esac
EOF
chmod +x "${BENCH_STUB}"

out="$(BENCH_SH="${BENCH_STUB}" WORKSPACE="${STUB_DIR}" bash native/build_variants.sh)"

grep -qE '^bare\s' <<<"${out}"     || fail "no bare row"
grep -qE '^logging\s' <<<"${out}"  || fail "no logging row"
grep -qE '^devtools\s' <<<"${out}" || fail "no devtools row"
grep -q '1.250' <<<"${out}"        || fail "logging warm_mean missing from table"
# deltas vs bare: logging - bare warm_mean = +0.150; devtools - bare = +0.300
grep -q '+0.150' <<<"${out}"       || fail "logging delta wrong"
grep -q '+0.300' <<<"${out}"       || fail "devtools delta wrong"

RESULTS="$(ls -t native/bench-results/variants-*.txt | head -1)"
[ "$(grep -c '^et_timing:' "${RESULTS}")" -eq 3 ] || fail "results file should hold 3 et_timing lines"

echo "PASS: build_variants matrix"
