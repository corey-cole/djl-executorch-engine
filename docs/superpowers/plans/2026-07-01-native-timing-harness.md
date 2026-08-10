# Native Timing Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Release timing harness plus the build wiring to measure the logging/devtools "ship or not" screen described in `docs/benchmarking.md`.

**Architecture:** A JVM-free `et_timing_harness` links the existing `et_runtime` core behind a new `ET_BUILD_BENCH` CMake gate; `bench.sh` builds and runs it against one ExecuTorch install prefix. `build.sh` gains an `ET_VARIANT` flag map (plus overridable paths and a `STAGE_SO` guard) so one script produces the bare/logging/devtools runtimes. `build_variants.sh` orchestrates all three inside one container, records each config's shim size + timing, and emits a comparison table with pairwise deltas.

**Tech Stack:** C++20, CMake ≥ 3.24, Bash, ExecuTorch 1.3.x, manylinux_2_28 container (gcc-toolset-14, cp312).

## Global Constraints

- **Default `build.sh` invocation must stay byte-identical to today:** no env set ⇒ `ET_VARIANT=logging` (logging ON, no devtools), `ET_INSTALL=/workspace/et-install`, `ET_BUILD=/workspace/et-cmake-out`, and the shim IS staged into `src/main/resources/native/linux-x86_64/`.
- **Never commit native binaries** (`.so`, install trees). The harness/matrix must not stage or commit any artifact; `STAGE_SO=0` for all variant builds.
- **`bench.sh` needs no JDK:** the `ET_BUILD_BENCH` configure must not build the JNI shim (shim gate excludes both QA and bench).
- **C++20** — the core header (`et_runtime.h`) uses `std::span`; the harness compiles under the existing `CMAKE_CXX_STANDARD 20`.
- **Container-only native builds:** compiling/running the harness and any `build.sh`/`bench.sh` runtime build happen inside `manylinux_2_28` against an ExecuTorch install. Fast host-runnable tests exist only for the pure-shell logic (build.sh flag map via `PRINT_ET_FLAGS`; build_variants orchestration via stub drivers).
- **User-managed, do not edit:** `.gitignore`, `local_build_wrapper.sh`, any host docker-run wrapper. Flag needed `.gitignore` entries in the final report; do not add them.
- **Commit trailer** for every commit: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Timing harness + `ET_BUILD_BENCH` gate + `bench.sh`

**Files:**
- Create: `native/harness/dtype_size.h` (shared ScalarType→byte-width helper)
- Create: `native/harness/et_timing_harness.cpp`
- Create: `native/bench.sh`
- Modify: `native/harness/et_leak_harness.cpp` (use the shared helper; drop its local copy)
- Modify: `native/CMakeLists.txt` (add `ET_BUILD_BENCH` option + guarded target; widen shim gate)

**Interfaces:**
- Consumes (from `native/core/et_runtime.h`, unchanged): `EtRuntime(const std::string&)` (throws `std::runtime_error`), `MethodMeta EtRuntime::methodMeta() const`, `ForwardResult EtRuntime::forward(std::span<const InputDesc>)`, `std::span<const OutputView> ForwardResult::outputs() const`, `InputDesc{const void* data; std::vector<int64_t> shape; int8_t scalarType;}`, `MethodMeta{int numInputs; std::vector<int8_t> inputScalarTypes; std::vector<std::vector<int64_t>> inputShapes;}`.
- Produces: `measly::et::dtypeSize(int8_t)` in `native/harness/dtype_size.h`, included by both harnesses (same directory, no extra include dir needed).
- Produces (relied on by Task 3): `bench.sh` reads `ET_INSTALL` (default `${REPO_ROOT}/et-install`), `ITERS` (default 1000), `WARMUP` (default 100); prints exactly one line beginning `et_timing:` with space-separated `key=value` tokens including `load_ms`, `cold_ms`, `warm_mean_ms`. Exit 0 on success; non-zero on load/meta/output failure.

- [ ] **Step 1: Create the shared `dtypeSize` helper**

Create `native/harness/dtype_size.h`:

```cpp
#ifndef MEASLY_ET_DTYPE_SIZE_H
#define MEASLY_ET_DTYPE_SIZE_H

#include <cstddef>
#include <cstdint>

namespace measly::et {

// Byte width of an ExecuTorch ScalarType code, for the subset the JNI-free harnesses build
// 1-filled host buffers for. Shared by et_leak_harness.cpp and et_timing_harness.cpp.
inline size_t dtypeSize(int8_t st) {
  switch (st) {
    case 6:            // FLOAT32
    case 3: return 4;  // INT32
    case 7:            // FLOAT64
    case 4: return 8;  // INT64
    case 0:            // UINT8
    case 1:            // INT8
    case 11: return 1;  // BOOL
    default: return 4;
  }
}

}  // namespace measly::et

#endif  // MEASLY_ET_DTYPE_SIZE_H
```

- [ ] **Step 2: Refactor `et_leak_harness.cpp` to use the shared helper**

In `native/harness/et_leak_harness.cpp`, add `#include "dtype_size.h"` alongside the other includes and **delete** the local `static size_t dtypeSize(int8_t st) { ... }` definition. It already does `using namespace measly::et;`, so unqualified `dtypeSize(...)` calls resolve to the header's. No call-site changes.

- [ ] **Step 3: Write the timing harness (the deliverable tested by the smoke run in Step 6)**

Create `native/harness/et_timing_harness.cpp`:

```cpp
// JNI-free timing harness: load once, then time load/cold/warm forwards over EtRuntime.
// Built Release, no sanitizers (ET_BUILD_BENCH), for the logging/devtools ship-or-not screen
// in docs/benchmarking.md. Model-agnostic: tensor inputs derived from methodMeta(), backed by
// 1-filled host buffers (mirrors et_leak_harness.cpp).
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "dtype_size.h"
#include "et_runtime.h"

using namespace measly::et;
using clock_type = std::chrono::steady_clock;

static double ms_since(clock_type::time_point start) {
  return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

int main(int argc, char** argv) {
  const char* pte = (argc > 1) ? argv[1] : "add.pte";
  const int iters = (argc > 2) ? std::atoi(argv[2]) : 1000;
  const int warmup = (argc > 3) ? std::atoi(argv[3]) : 100;

  try {
    // --- load (single sample; where ET_LOG is concentrated) ---
    auto t_load = clock_type::now();
    EtRuntime rt(pte);
    const double load_ms = ms_since(t_load);

    MethodMeta meta = rt.methodMeta();
    if (meta.numInputs <= 0) {
      std::fprintf(stderr, "et_timing: model %s has no inputs\n", pte);
      return 2;
    }

    // --- 1-filled inputs, kept alive for the whole run (borrowed by InputDesc) ---
    std::vector<std::vector<uint8_t>> buffers(meta.numInputs);
    std::vector<InputDesc> inputs;
    inputs.reserve(meta.numInputs);
    for (int i = 0; i < meta.numInputs; ++i) {
      if (meta.inputScalarTypes[i] < 0) continue;  // non-tensor input
      size_t count = 1;
      for (int64_t d : meta.inputShapes[i]) count *= static_cast<size_t>(d);
      size_t bytes = count * dtypeSize(meta.inputScalarTypes[i]);
      buffers[i].assign(bytes, 0);
      if (meta.inputScalarTypes[i] == 6) {  // float32 -> 1.0f
        float one = 1.0f;
        for (size_t b = 0; b + sizeof(float) <= bytes; b += sizeof(float))
          std::memcpy(buffers[i].data() + b, &one, sizeof(float));
      } else {
        std::memset(buffers[i].data(), 1, bytes);
      }
      inputs.push_back(InputDesc{buffers[i].data(), meta.inputShapes[i],
                                 meta.inputScalarTypes[i]});
    }

    volatile unsigned char sink = 0;  // defeat dead-code elimination of the forward loop

    // --- cold: first forward ---
    auto t_cold = clock_type::now();
    {
      ForwardResult r = rt.forward(inputs);
      auto outs = r.outputs();
      if (outs.empty()) {
        std::fprintf(stderr, "et_timing: forward produced no outputs\n");
        return 3;
      }
      sink ^= *static_cast<const unsigned char*>(outs[0].data);
    }
    const double cold_ms = ms_since(t_cold);

    // --- warmup (discarded) ---
    for (int i = 0; i < warmup; ++i) {
      ForwardResult r = rt.forward(inputs);
      sink ^= *static_cast<const unsigned char*>(r.outputs()[0].data);
    }

    // --- timed warm loop: min / mean / max ---
    double warm_min = 0, warm_max = 0, warm_sum = 0;
    for (int i = 0; i < iters; ++i) {
      auto t = clock_type::now();
      ForwardResult r = rt.forward(inputs);
      sink ^= *static_cast<const unsigned char*>(r.outputs()[0].data);
      double e = ms_since(t);
      warm_sum += e;
      if (i == 0 || e < warm_min) warm_min = e;
      if (i == 0 || e > warm_max) warm_max = e;
    }
    const double warm_mean = iters > 0 ? warm_sum / iters : 0.0;

    std::printf("et_timing: model=%s iters=%d warmup=%d load_ms=%.3f cold_ms=%.3f "
                "warm_min_ms=%.3f warm_mean_ms=%.3f warm_max_ms=%.3f sink=%d\n",
                pte, iters, warmup, load_ms, cold_ms, warm_min, warm_mean, warm_max,
                static_cast<int>(sink));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "et_timing: error: %s\n", e.what());
    return 1;
  }
}
```

- [ ] **Step 4: Add the CMake gate + widen the shim gate**

In `native/CMakeLists.txt`, add next to the existing `ET_BUILD_QA` option (line ~6):

```cmake
option(ET_BUILD_BENCH "Build the Release timing harness (no sanitizers)" OFF)
```

Change the shim gate (currently `if(NOT ET_BUILD_QA)` around the `JAVA_HOME` + `executorch_djl` block) to:

```cmake
if(NOT ET_BUILD_QA AND NOT ET_BUILD_BENCH)
```

After the `if(ET_BUILD_QA) ... endif()` block, add:

```cmake
# --- Release timing harness (no sanitizers). OFF for the shipping build and for QA. ---
if(ET_BUILD_BENCH)
  add_executable(et_timing_harness ${CMAKE_CURRENT_SOURCE_DIR}/harness/et_timing_harness.cpp)
  target_link_libraries(et_timing_harness PRIVATE et_runtime)
endif()
```

- [ ] **Step 5: Write `bench.sh` (the runnable smoke test lives here)**

Create `native/bench.sh` (mode 0755):

```bash
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
```

- [ ] **Step 6: Run the smoke test to verify it passes (in the manylinux_2_28 container, against an existing `et-install`)**

Run: `ITERS=5 WARMUP=1 ./native/bench.sh`
Expected: configures + builds `et_timing_harness`, then a final line like
`et_timing: model=native/spike/add.pte iters=5 warmup=1 load_ms=... cold_ms=... warm_min_ms=... warm_mean_ms=... warm_max_ms=... sink=...`; exit 0.

- [ ] **Step 7: Verify the exit-code teeth (in the same container)**

Run: `./native/bench/et_timing_harness /nonexistent/nope.pte 1 1; echo "exit=$?"`
Expected: prints `et_timing: error: ...` to stderr and `exit=1` (load threw).

- [ ] **Step 8: Verify the shared helper didn't break the QA build (in the same container)**

The leak harness now includes `dtype_size.h`; confirm the QA tree still builds and links it.
Run: `./native/build_qa.sh` (or, faster, just reconfigure+build the leak target).
Expected: `et_runtime_test` + `et_leak_harness` build and run clean, as before.

- [ ] **Step 9: Commit**

```bash
git add native/harness/dtype_size.h native/harness/et_timing_harness.cpp \
        native/harness/et_leak_harness.cpp native/bench.sh native/CMakeLists.txt
git commit -m "feat(native): Release timing harness + ET_BUILD_BENCH gate + bench.sh

Extract shared dtypeSize() into native/harness/dtype_size.h, used by both the leak
and timing harnesses.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `build.sh` variant wiring

**Files:**
- Modify: `native/build.sh`

**Interfaces:**
- Produces (relied on by Task 3): `ET_VARIANT` (`bare`|`logging`|`devtools`, default `logging`) selects ExecuTorch cmake flags; `ET_INSTALL`/`ET_BUILD` honor an incoming value (defaults `/workspace/et-install`, `/workspace/et-cmake-out`); `STAGE_SO` (default `1`) guards the copy into `src/main/resources`; `PRINT_ET_FLAGS` (non-empty) prints the resolved config to stdout and exits 0 before any heavy setup. Unknown `ET_VARIANT` ⇒ exit 2.
- Consumes: the existing `SKIP_ET_BUILD` and `$GITHUB_ENV` behavior (unchanged).

- [ ] **Step 1: Write the failing test (host, fast — this exercises the `PRINT_ET_FLAGS` path)**

Save as `native/tests/build_variants_flags.sh` (mode 0755):

```bash
#!/usr/bin/env bash
# Fast host test for build.sh's ET_VARIANT flag map via the PRINT_ET_FLAGS early-exit path.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

out="$(PRINT_ET_FLAGS=1 bash native/build.sh)"
grep -q 'ET_VARIANT=logging' <<<"$out" || fail "default variant not logging"
grep -q 'EXECUTORCH_ENABLE_LOGGING=ON' <<<"$out" || fail "default flags missing LOGGING=ON"
grep -q 'ET_INSTALL=/workspace/et-install\b' <<<"$out" || fail "default ET_INSTALL changed"
grep -q 'ET_BUILD=/workspace/et-cmake-out\b' <<<"$out" || fail "default ET_BUILD changed"
grep -q 'STAGE_SO=1' <<<"$out" || fail "default STAGE_SO not 1"

out="$(ET_VARIANT=bare PRINT_ET_FLAGS=1 bash native/build.sh)"
grep -q 'EXECUTORCH_ENABLE_LOGGING=OFF' <<<"$out" || fail "bare missing LOGGING=OFF"
grep -q 'DEVTOOLS' <<<"$out" && fail "bare should not enable devtools"

out="$(ET_VARIANT=devtools PRINT_ET_FLAGS=1 bash native/build.sh)"
grep -q 'EXECUTORCH_BUILD_DEVTOOLS=ON' <<<"$out" || fail "devtools missing DEVTOOLS=ON"
grep -q 'EXECUTORCH_ENABLE_EVENT_TRACER=ON' <<<"$out" || fail "devtools missing EVENT_TRACER=ON"
grep -q 'EXECUTORCH_ENABLE_LOGGING=OFF' <<<"$out" || fail "devtools should hold logging OFF"

out="$(ET_VARIANT=bare ET_INSTALL=/tmp/xi ET_BUILD=/tmp/xb STAGE_SO=0 PRINT_ET_FLAGS=1 bash native/build.sh)"
grep -q 'ET_INSTALL=/tmp/xi\b' <<<"$out" || fail "ET_INSTALL override ignored"
grep -q 'ET_BUILD=/tmp/xb\b' <<<"$out" || fail "ET_BUILD override ignored"
grep -q 'STAGE_SO=0' <<<"$out" || fail "STAGE_SO override ignored"

if ET_VARIANT=bogus PRINT_ET_FLAGS=1 bash native/build.sh >/dev/null 2>&1; then
  fail "unknown ET_VARIANT should exit non-zero"
fi

echo "PASS: build.sh flag map"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./native/tests/build_variants_flags.sh`
Expected: FAIL (build.sh has no `PRINT_ET_FLAGS` path yet; it proceeds past the flag check or errors on host setup).

- [ ] **Step 3: Add the variant-resolution block near the top of `build.sh`**

Insert immediately after `set -ex` (before the JDK-extract section), so `PRINT_ET_FLAGS` exits before any container-only work:

```bash
# --- Runtime variant selection. Flags only — paths stay overridable so the default artifact and
#     all downstream defaults (build_qa.sh, bench.sh, README) are unchanged. See the timing-harness
#     spec + docs/benchmarking.md. ---
ET_VARIANT="${ET_VARIANT:-logging}"
case "${ET_VARIANT}" in
  bare)     ET_VARIANT_FLAGS=(-DEXECUTORCH_ENABLE_LOGGING=OFF) ;;
  logging)  ET_VARIANT_FLAGS=(-DEXECUTORCH_ENABLE_LOGGING=ON) ;;
  devtools) ET_VARIANT_FLAGS=(-DEXECUTORCH_ENABLE_LOGGING=OFF
                              -DEXECUTORCH_BUILD_DEVTOOLS=ON
                              -DEXECUTORCH_ENABLE_EVENT_TRACER=ON) ;;
  *) echo "Unknown ET_VARIANT='${ET_VARIANT}' (want bare|logging|devtools)"; exit 2 ;;
esac
ET_BUILD="${ET_BUILD:-/workspace/et-cmake-out}"
ET_INSTALL="${ET_INSTALL:-/workspace/et-install}"
STAGE_SO="${STAGE_SO:-1}"

# Fast diagnostic: print the resolved config and exit before any heavy setup (JDK/pip/build).
if [ -n "${PRINT_ET_FLAGS:-}" ]; then
  echo "ET_VARIANT=${ET_VARIANT} ET_BUILD=${ET_BUILD} ET_INSTALL=${ET_INSTALL} STAGE_SO=${STAGE_SO} FLAGS=${ET_VARIANT_FLAGS[*]}"
  exit 0
fi
```

Then delete the later duplicate assignments (the existing `ET_BUILD="/workspace/et-cmake-out"` and `ET_INSTALL="/workspace/et-install"` lines, ~47-48) since they are now set above. Leave the `$GITHUB_ENV` export block in place (it runs after JDK extraction, where `JAVA_HOME` is available, and uses the already-resolved `ET_INSTALL`).

- [ ] **Step 4: Use the variant flags in the configure and guard the staging copy**

In the `cmake -B ${ET_BUILD} ...` configure (Stage A), replace the hardcoded line `-DEXECUTORCH_ENABLE_LOGGING=ON \` with:

```bash
    "${ET_VARIANT_FLAGS[@]}" \
```

Replace the final staging block (currently unconditional `OUT=...; mkdir; cp; echo; ls`) with:

```bash
if [ "${STAGE_SO}" = "1" ]; then
  OUT="src/main/resources/native/linux-x86_64"
  mkdir -p "${OUT}"
  cp native/build/libexecutorch_djl.so "${OUT}/"
  echo "Artifact: ${OUT}/libexecutorch_djl.so"
  ls -lh "${OUT}/libexecutorch_djl.so"
else
  echo "STAGE_SO=0: built shim but not staging into resources"
  ls -lh native/build/libexecutorch_djl.so
fi
```

- [ ] **Step 5: Run the flag test to verify it passes (host)**

Run: `./native/tests/build_variants_flags.sh`
Expected: `PASS: build.sh flag map`.

- [ ] **Step 6: Integration check (manylinux_2_28 container) — bare variant, no clobber**

Run (in-container; reuses runtime if already built):
```bash
sha_before=$(sha256sum src/main/resources/native/linux-x86_64/libexecutorch_djl.so 2>/dev/null || echo none)
ET_VARIANT=bare ET_INSTALL=/workspace/et-install-bare ET_BUILD=/workspace/et-cmake-out-bare STAGE_SO=0 bash native/build.sh
sha_after=$(sha256sum src/main/resources/native/linux-x86_64/libexecutorch_djl.so 2>/dev/null || echo none)
test -f /workspace/et-install-bare/lib/cmake/ExecuTorch/executorch-config.cmake && echo "OK: bare runtime installed"
test "$sha_before" = "$sha_after" && echo "OK: shipped .so untouched (STAGE_SO=0)"
```
Expected: both `OK:` lines print; the Stage A configure log shows `EXECUTORCH_ENABLE_LOGGING=OFF`.

- [ ] **Step 7: Commit**

```bash
git add native/build.sh native/tests/build_variants_flags.sh
git commit -m "feat(native): build.sh ET_VARIANT flag map + STAGE_SO guard + overridable paths

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `build_variants.sh` matrix

**Files:**
- Create: `native/build_variants.sh`
- Create: `native/tests/build_variants_matrix.sh` (host test with stub drivers)

**Interfaces:**
- Consumes: `build.sh` (`ET_VARIANT`/`ET_INSTALL`/`ET_BUILD`/`STAGE_SO`/`SKIP_ET_BUILD`) from Task 2; `bench.sh` (`ET_INSTALL`, emits an `et_timing:` line) from Task 1.
- Produces: a comparison table to stdout and to `native/bench-results/variants-<UTC>.txt`; overridable `BUILD_SH`/`BENCH_SH`/`WORKSPACE` for testing.

- [ ] **Step 1: Write the failing test (host, fast — stubs stand in for the real drivers)**

Save as `native/tests/build_variants_matrix.sh` (mode 0755):

```bash
#!/usr/bin/env bash
# Fast host test for build_variants.sh orchestration using stub drivers (no real builds).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
mkdir -p "${TMP}/bin" native/build native/bench-results

# Stub build.sh: fabricates a per-variant-sized shim + install config; logs SKIP_ET_BUILD received.
cat > "${TMP}/bin/build.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "stub-build ${ET_VARIANT} SKIP_ET_BUILD=${SKIP_ET_BUILD}" >> "${STUB_LOG}"
mkdir -p "${ET_INSTALL}/lib/cmake/ExecuTorch"
: > "${ET_INSTALL}/lib/cmake/ExecuTorch/executorch-config.cmake"
mkdir -p native/build
case "${ET_VARIANT}" in
  bare)     head -c 1000 /dev/zero > native/build/libexecutorch_djl.so ;;
  logging)  head -c 1350 /dev/zero > native/build/libexecutorch_djl.so ;;
  devtools) head -c 1200 /dev/zero > native/build/libexecutorch_djl.so ;;
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
  bash native/build_variants.sh >/dev/null
test "$(grep -c 'SKIP_ET_BUILD=1' "${STUB_LOG}")" -eq 3 || fail "second run should reuse"

echo "PASS: build_variants.sh matrix"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./native/tests/build_variants_matrix.sh`
Expected: FAIL (`native/build_variants.sh` does not exist).

- [ ] **Step 3: Write `build_variants.sh`**

Create `native/build_variants.sh` (mode 0755):

```bash
#!/usr/bin/env bash
# Build all three ExecuTorch runtime variants (bare/logging/devtools), time each, and print a
# comparison table with pairwise deltas vs. bare. Run INSIDE the manylinux_2_28 container (same as
# native/build.sh) so the torch wheel is installed once by the first variant and reused. Never
# stages the shipped artifact (STAGE_SO=0). See docs/benchmarking.md for how to read the numbers.
#
# Env: ITERS/WARMUP forwarded to bench.sh. BUILD_SH/BENCH_SH/WORKSPACE are override seams (tests
# inject stubs; WORKSPACE defaults to the container's /workspace mount).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BUILD_SH="${BUILD_SH:-native/build.sh}"
BENCH_SH="${BENCH_SH:-native/bench.sh}"
WORKSPACE="${WORKSPACE:-/workspace}"
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
    SKIP_ET_BUILD="${skip}" bash "${BUILD_SH}"

  SIZE[$v]="$(stat -c%s native/build/libexecutorch_djl.so)"

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
```

- [ ] **Step 4: Run the matrix test to verify it passes (host)**

Run: `./native/tests/build_variants_matrix.sh`
Expected: `PASS: build_variants.sh matrix`.

- [ ] **Step 5: Commit**

```bash
git add native/build_variants.sh native/tests/build_variants_matrix.sh
git commit -m "feat(native): build_variants.sh matrix (bare/logging/devtools + deltas)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Post-implementation notes (for the final report, not tasks)

- **`.gitignore` (user-managed):** add `et-install-*/`, `et-cmake-out-*/`, `native/bench-results/`, and (if not already ignored) `native/bench/`.
- **Devtools flag risk:** the `devtools` variant is the one unverified flag combination — `EXECUTORCH_BUILD_DEVTOOLS=ON` + `EXECUTORCH_ENABLE_EVENT_TRACER=ON` may require additional ET 1.3.x options (etdump/flatccrt) to configure cleanly. If the `devtools` runtime build fails, this is expected; resolve the flag set against the ExecuTorch 1.3.x devtools docs and pin it in `build.sh`'s `case` arm.
- **Full end-to-end** (all three real runtime builds + timing) is a container run of `native/build_variants.sh`; the host tests cover only the shell logic.
