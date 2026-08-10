# Native Timing Harness — Design

> **Status:** design/spec for review. Scope is the harness + its build gate + its driver script,
> plus the variant wiring in `build.sh` and a manager script that builds all three runtime configs
> (bare / logging / devtools) and reports the comparison. The *interpretation* of the resulting
> numbers, and the full MobileNetV2/ResNet benchmark, remain in `docs/benchmarking.md`.

## Purpose

A cheap, early **gross-regression screen** for the two "should this ship in the release build?"
questions in `docs/benchmarking.md`:

- **Logging** (`EXECUTORCH_ENABLE_LOGGING`) — runtime cost, complementing the already-measured
  binary-size delta (8.5 → 11.5 MB).
- **Profiling** (`EXECUTORCH_BUILD_DEVTOOLS` + `EXECUTORCH_ENABLE_EVENT_TRACER`, **not tracing**) —
  the one-artifact-vs-two gating cost, which needs **no core changes** to measure.

It times `add.pte` (or any single-tensor model) through the existing `EtRuntime` core and reports
load / cold / warm latency. Read a large delta as "don't ship without a real reason"; read a clean
result as "no gross regression," **not** "confirmed cheap" — a 1-op model under-reports costs that
scale with model complexity (see `benchmarking.md`). The real MobileNetV2/ResNet numbers remain the
final arbiter.

## Why not reuse the leak harness

`et_leak_harness` is built under `-fsanitize=address` by `native/build_qa.sh`, which inflates and
distorts wall-clock non-uniformly. Timing must run against a **Release, no-sanitizer** build. Same
`EtRuntime` core, same model-agnostic input construction, different build config and different
measured quantities — so it is a sibling target, not a flag on the leak harness.

## Architecture

Five pieces, all reusing existing seams:

1. **`native/harness/et_timing_harness.cpp`** — a new JVM-free executable linking only `et_runtime`
   (the same core the `.so`, the units, and the leak harness link). It loads the model once, builds
   1-filled inputs from `methodMeta()` (mirroring the leak harness), and times three phases.
2. **CMake gate `ET_BUILD_BENCH`** (new, distinct from `ET_BUILD_QA`) — builds *only* the timing
   harness. Kept separate from QA so a timing build pulls neither Catch2 (FetchContent) nor the
   ASan/LSan flags QA implies.
3. **`native/bench.sh`** — a driver mirroring `build_qa.sh`, but **Release / no sanitizer**, its own
   build tree (`native/bench`), parameterized by `ET_INSTALL` so it can be pointed at each of the
   3 runtime install prefixes in turn.
4. **`native/build.sh` variant wiring** — `build.sh` gains an `ET_VARIANT` knob (and a couple of
   supporting env vars) so one script produces any of the three runtime configs without editing it.
   The default invocation stays byte-identical to today's shipping build. See *build.sh wiring*.
5. **`native/build_variants.sh`** — the manager: runs inside the container, builds all three runtime
   variants (reusing one torch install), records each variant's shim `.so` size + timing, and emits
   a comparison table with the pairwise deltas. See *Build-variant matrix*.

### Why a separate `ET_BUILD_BENCH` gate

`build_qa.sh` configures `native/asan` with `-DET_BUILD_QA=ON` **and** ASan flags. If the timing
harness lived under `ET_BUILD_QA`, `bench.sh` would have to configure a second tree with
`-DET_BUILD_QA=ON` too — dragging in the Catch2 download and the leak/unit targets it doesn't need,
and risking someone timing an ASan tree. A dedicated gate keeps the bench build minimal and makes
"never time a sanitized build" structural rather than a convention.

## Measured phases

Using `std::chrono::steady_clock`, reported as `double` milliseconds:

- **`load_ms`** — a single `EtRuntime rt(pte)` construction. This is where `ET_LOG` is concentrated
  (init/load, not the steady `forward` hot loop), so it is the phase most likely to reveal logging's
  runtime cost. One sample; report as-is.
- **`cold_ms`** — the first `forward()` after load (captures any first-call/JIT-ish warmup).
- **`warm_*_ms`** — `WARMUP` discarded forwards, then `ITERS` timed forwards over one loaded model.
  Report **min / mean / max**: `min` is the least-noisy estimate of true compute cost, `mean` the
  typical, `max` the tail/jitter. (Median/p95 deliberately omitted — YAGNI for a screen.)

Each timed `forward()` result's first output byte is accumulated into a `volatile` sink and printed,
to defeat dead-code elimination of the loop body.

## Interfaces

Consumes (from `native/core/et_runtime.h`, unchanged):

- `EtRuntime(const std::string& ptePath)` — load; throws `std::runtime_error`.
- `MethodMeta EtRuntime::methodMeta() const` — `{numInputs, inputScalarTypes, inputShapes}`.
- `ForwardResult EtRuntime::forward(std::span<const InputDesc> inputs)`.
- `std::span<const OutputView> ForwardResult::outputs() const`.
- `InputDesc{const void* data, std::vector<int64_t> shape, int8_t scalarType}`.

### CLI

Positional, mirroring `et_leak_harness`'s `<pte> [iters]` style, plus a warmup slot:

```
et_timing_harness [pte] [iters] [warmup]
```

- `pte` — model path (default `add.pte`).
- `iters` — timed warm forwards (default `1000`; env `ITERS` via the driver script).
- `warmup` — discarded forwards before timing (default `100`; env `WARMUP` via the driver script).

Argv takes precedence over env; the driver script passes the env-resolved values positionally.

### Output

One human block plus one machine-parseable summary line (so results across the 3 configs can be
diffed/grepped without re-running):

```
et_timing: model=add.pte iters=1000 warmup=100 load_ms=1.234 cold_ms=0.456 warm_min_ms=0.010 warm_mean_ms=0.012 warm_max_ms=0.031 sink=1
```

### Exit codes

- `0` — completed; summary printed.
- non-zero — load threw, `methodMeta().numInputs <= 0`, or a `forward()` produced zero outputs
  (gives the smoke check teeth; see Testing).

## Input construction

Reuse the leak harness's model-agnostic approach verbatim: a local `dtypeSize(int8_t)` mapping ET
scalar-type codes to byte widths, 1-filled host buffers per tensor input (float32 filled with
`1.0f`, others `memset` to 1), skipping non-tensor inputs (`scalarType < 0`). Buffers are `std::vector<uint8_t>`
kept alive for the whole run (borrowed by `InputDesc`, valid across every `forward()`).

## CMake changes (`native/CMakeLists.txt`)

Add an `option(ET_BUILD_BENCH ...)` near the existing `ET_BUILD_QA` option, and a guarded block
parallel to the QA block:

```cmake
option(ET_BUILD_BENCH "Build the Release timing harness (no sanitizers)" OFF)
...
if(ET_BUILD_BENCH)
  add_executable(et_timing_harness ${CMAKE_CURRENT_SOURCE_DIR}/harness/et_timing_harness.cpp)
  target_link_libraries(et_timing_harness PRIVATE et_runtime)
endif()
```

The shim's gate must also exclude bench builds, or a `-DET_BUILD_BENCH=ON` configure (where
`ET_BUILD_QA` is OFF) would build the JNI shim and demand a JDK — the exact `JAVA_HOME` requirement
`bench.sh` is meant to avoid. Change the existing `if(NOT ET_BUILD_QA)` guarding the shim +
`JAVA_HOME` block to `if(NOT ET_BUILD_QA AND NOT ET_BUILD_BENCH)`. No change to the `ET_BUILD_QA`
block or the core.

## Driver script (`native/bench.sh`)

Mirrors `build_qa.sh`'s structure (repo-root cd, `ET_INSTALL` default + `executorch-config.cmake`
guard, pinned `-G "Unix Makefiles"`), but:

- **Release, no sanitizer:** `-DCMAKE_BUILD_TYPE=Release`, no `-fsanitize` flags.
- **Own tree:** `native/bench` (not `native/asan`).
- **Gate:** `-DET_BUILD_BENCH=ON` (not `-DET_BUILD_QA=ON`).
- **Env:** `ET_INSTALL` (required, same contract as `build_qa.sh`), `ITERS` (default 1000),
  `WARMUP` (default 100).
- Builds target `et_timing_harness`, then runs it against `native/spike/add.pte` with the resolved
  `ITERS`/`WARMUP`.

`bench.sh` times **one** install prefix per run. Driving it across the three variants is the job of
`build_variants.sh` below; running it standalone against a single `ET_INSTALL` remains valid for
one-off timing.

## Testing

The harness is a measurement tool, not correctness-critical, so no Catch2 unit is added (that would
pull it under the QA/Catch2 gate we deliberately kept it out of). Its acceptance check is a **smoke
run** via the driver: `ITERS=5 WARMUP=1 ./native/bench.sh` against `add.pte` must exit `0` and print
a summary line with `iters=5`. The non-zero exit on `numInputs<=0` / zero outputs / load failure is
what makes that smoke run a real check rather than a no-op.

Correctness of `forward()` itself is already covered by `et_runtime_test.cpp` (`add(2,3)==5`); the
timing harness intentionally does not re-assert numeric results (it is model-agnostic and can't know
the expected output for an arbitrary `.pte`).

## build.sh wiring

`build.sh` currently hardcodes the runtime config (logging on, no devtools) and the output paths.
To let one script produce any variant *without changing the default artifact*, it gains three env
knobs, each defaulting to today's behavior so a no-env run is byte-identical:

- **`ET_VARIANT`** (default `logging`) — selects the ExecuTorch cmake flags, replacing the current
  hardcoded `-DEXECUTORCH_ENABLE_LOGGING=ON`:

  | `ET_VARIANT` | logging | devtools + event tracer |
  |---|---|---|
  | `bare`     | `OFF` | `OFF` |
  | `logging`  | `ON`  | `OFF` |  ← default; reproduces today's flags exactly
  | `devtools` | `OFF` | `ON`  |

- **`ET_BUILD` / `ET_INSTALL`** — already internal vars; change the assignments to honor an incoming
  value (`ET_INSTALL="${ET_INSTALL:-/workspace/et-install}"`, likewise `ET_BUILD`). Defaults
  unchanged, so downstream (`build_qa.sh`, `bench.sh`, README) still find `et-install`. The matrix
  passes per-variant prefixes.
- **`STAGE_SO`** (default `1`) — guards the final copy of the shim into
  `src/main/resources/native/linux-x86_64/`. The matrix sets `STAGE_SO=0` so a bench build still
  *produces* `native/build/libexecutorch_djl.so` (whose size is a measurement) but does **not**
  clobber the shipped artifact.

`ET_VARIANT` drives **flags only**, not paths — otherwise the default `logging` variant would
relocate the install to `et-install-logging` and break every downstream default. Path relocation is
the matrix's responsibility, done explicitly via `ET_INSTALL`/`ET_BUILD`. The existing `SKIP_ET_BUILD`
and `$GITHUB_ENV` export blocks are unaffected (the export publishes whatever `ET_INSTALL` resolved
to).

**Devtools flag risk:** `EXECUTORCH_BUILD_DEVTOOLS=ON` is expected to also build the `etdump` /
`flatccrt` libraries that today log as benign "library is not found" notices. ET 1.3.x's devtools
preset may require additional options beyond `DEVTOOLS` + `EVENT_TRACER`; the exact set is to be
confirmed against the ExecuTorch 1.3.x devtools docs during implementation of the `devtools` branch,
and pinned in `build.sh` once it configures and builds cleanly. This is the one genuinely unverified
flag combination in the design.

## Build-variant matrix (`native/build_variants.sh`)

The manager script. It runs **inside the same manylinux_2_28 container** as `build.sh`/`bench.sh`
(launched the same way — via the host wrapper or `docker run`), so the torch wheel is installed once
by the first variant's Stage A and reused by the rest (pip sees it satisfied).

Behavior:

1. **Variants:** `bare`, `logging`, `devtools`. Each maps to `ET_INSTALL=/workspace/et-install-<v>`
   and `ET_BUILD=/workspace/et-cmake-out-<v>`.
2. **Baseline choice:** `bare` is the common baseline for both deltas — `logging − bare` isolates the
   logging cost (devtools held off) and `devtools − bare` isolates the devtools-enabled-not-tracing
   cost (logging held off). Holding one axis fixed per delta is why `devtools` runs with logging
   `OFF` rather than matching the shipped logging-on build. (A 4th `devtools+logging` point can be
   added later if the isolated deltas prove material; three keeps the matrix cheap, per the scope.)
3. **Per variant:**
   - Reuse if present: if `et-install-<v>/lib/cmake/ExecuTorch/executorch-config.cmake` already
     exists, pass `SKIP_ET_BUILD=1` so re-running the matrix skips completed runtime builds.
   - `ET_VARIANT=<v> ET_INSTALL=… ET_BUILD=… STAGE_SO=0 bash native/build.sh` — builds the runtime
     and the shim for that variant.
   - Record the shim size (`stat -c%s native/build/libexecutorch_djl.so`) immediately, before the
     next variant's `rm -rf native/build` overwrites it.
   - `ET_INSTALL=/workspace/et-install-<v> bash native/bench.sh` — capture the single `et_timing:`
     summary line.
4. **Output:** a comparison table to stdout and to a results file under a gitignored dir
   (`native/bench-results/variants-<UTC-timestamp>.txt`), columns:
   `variant | shim_so_bytes | load_ms | cold_ms | warm_mean_ms`, followed by the two pairwise deltas
   (absolute and %). The `et_timing:` machine line for each variant is preserved verbatim in the file
   for traceability.

The matrix owns **build-output management**: it names and creates the per-variant install/build dirs
and the results dir, and never touches the shipped `src/main/resources` artifact (via `STAGE_SO=0`).

**Gitignore (user-managed):** the per-variant trees and results need ignoring —
`et-install-*/`, `et-cmake-out-*/`, `native/bench-results/`. Flagged for the user to add to
`.gitignore`; not edited here.

## Out of scope

- Tracer-**attached** profiling cost — needs `EtRuntime` to construct an `ETDumpGen` and expose the
  dump buffer; that is the future profiling spec, not this screen.
- JVM-side / JMH numbers, MobileNetV2/ResNet artifacts — the full benchmark, `benchmarking.md`.
- The host launch wrapper (`local_build_wrapper.sh` and any docker-run for the matrix) — user-managed;
  this spec assumes `build_variants.sh` is launched inside the container the same way `build.sh` is.
- `.gitignore` edits — user-managed; the needed entries are flagged above.

## Files

- Create: `native/harness/et_timing_harness.cpp`
- Create: `native/bench.sh`
- Create: `native/build_variants.sh`
- Modify: `native/CMakeLists.txt` (add `ET_BUILD_BENCH` option + guarded target)
- Modify: `native/build.sh` (add `ET_VARIANT` flag map; honor incoming `ET_INSTALL`/`ET_BUILD`;
  add `STAGE_SO` guard around the `src/main/resources` copy)
- Reference (unchanged): `native/core/et_runtime.h`, `native/harness/et_leak_harness.cpp`
  (input-construction pattern), `native/build_qa.sh` (driver-script pattern)
