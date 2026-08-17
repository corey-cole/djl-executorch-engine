# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A DJL ([Deep Java Library](https://djl.ai/)) engine plugin that runs ExecuTorch (`.pte`) models. DJL 0.36.0 only supports the deprecated TorchScript export API; this engine adds ExecuTorch as a *separate* DJL engine so PyTorch models exported via the newer ExecuTorch backend can run under DJL, and to allow gradual migration off TorchScript. CPU-only, limited NDArray support. Group/coordinates: `org.measly:djl-executorch-engine`.

Supported platforms: `linux-x86_64`, `linux-aarch64` and `windows-x86_64` (all ship the `logging` runtime variant). `bare`/`devtools` runtime variants are Linux-only benchmarking builds.

## Two-layer architecture

**Java layer** (`src/main/java/org/measly/executorch/`)
- `engine/` — the DJL SPI implementation. `EtEngineProvider` is registered via `META-INF/services/ai.djl.engine.EngineProvider`. `EtEngine` (rank 10) → `EtModel` (loads `.pte`, owns the native handle) → `EtSymbolBlock` (runs `forward()`, marshals `NDList` ↔ `EtTensor[]`) → `EtNDManager`/`EtNDArray` (minimal tensor factory).
- `jni/` — the JNI boundary. `EtNative` holds the `native` method declarations and loads the `.so` on class init; `EtTensor`/`EtMethodMeta` are the marshalling structs.
- `translate/` — DJL `Translator` support types.
- `LibUtils` resolves and loads the native library: `EXECUTORCH_LIBRARY_PATH` env override wins; otherwise the `.so`/`.dll` is extracted from the classpath (`/native/<platform>/`) into a **content-addressed cache** (`~/.cache/executorch-djl/<sha256>/` on Linux, `%LOCALAPPDATA%\executorch-djl\<sha256>\` on Windows) and `System.load`ed. Windows can't delete a loaded DLL, hence the stable per-content dir. Keep `LibUtils.libName` in sync with `nativeLibName` in `build.gradle.kts`.

**Native layer** (`native/`)
- `core/et_runtime.{h,cpp}` — a **JNIEnv-free** C++ core (`measly::et::EtRuntime`) that wraps the
  ExecuTorch `Module`. Deliberately free of any JVM dependency so it can be linked by the shim, the
  Catch2 unit tests, and the leak harness alike. Borrowed input pointers, single-copy out (into a
  heap `byte[]`, so outputs are *not* direct buffers). **The input borrow is not zero-copy end to
  end:** ExecuTorch's `Method::set_input` copies into its own arena for any input where
  `is_memory_planned()` is true — the export default, `MemoryPlanningPass(alloc_graph_input=True)`,
  and so the case for any model a user brings unless they went out of their way — and honors the
  borrow only for models exported with `alloc_graph_input=False`. This repo ships a handful of such
  unplanned fixtures deliberately, to exercise the borrow path: `native/spike/add_unplanned.pte`,
  `clamp5.pte`, and `lin129.pte`. `example/build/models/mobilenet_v2_unplanned.pte` exercises the
  same path but is not shipped — it's a gitignored build output that
  `tools/scripts/export_mobilenet.py` generates on demand. See `docs/native-architecture.md` §3 and
  `docs/executorch-host-buffer-contract-brief.md`.
- `jni/executorch_djl_jni.cpp` + `jni/et_logging.cpp` — the JNI shim (`executorch_djl` shared library). `et_logging.cpp` is a PAL bridge that forwards native `ET_LOG` output to slf4j via `EtNative.nativeLog` (level codes: 0=debug 1=info 2=warn 3=error).
- `harness/` — `et_timing_harness` (Release benchmark) and `et_leak_harness` (ASan/LSan). `test/et_runtime_test.cpp` — Catch2 units. These link only the JNIEnv-free core, so QA/bench configures need no JDK.

### The ExecuTorch runtime is NOT built here

The engine links against the ExecuTorch runtime, but that runtime is **downloaded**, not compiled. CMake `FetchContent`s a hash-pinned, build-attested tarball published by the separate [`executorch-runtime-dist`](https://github.com/measly-java-learning/executorch-runtime-dist) repo. The pin lives in `native/cmake/EtRuntimePin.cmake` (**generated — do not hand-edit**; bump by replacing the whole file with the asset from the next `v<etver>-<pkgrev>` release, then re-applying the comment header). The SHA256 change is the supply-chain review gate. **After a pin bump, re-run `./native/gen_clangd_db.sh`** — the clangd database is refreshed only by that script, so it otherwise keeps resolving against the previous runtime's headers, silently and with no warning.

- **Escape hatch**: set `ET_INSTALL=/path/to/et-install` to link an existing runtime tree; CMake then skips the download.
- ExecuTorch runtime version is currently `1.3.1` (pin `1.3.1-10`); mirrored in `EtEngine.EXECUTORCH_VERSION`.
- The pin file defines `et_runtime_dist_url(<variant> <row> <out_url> <out_sha>)` and
  `native/CMakeLists.txt` resolves rows through it. Do not rebuild `ET_RUNTIME_URL_<variant>_<row>`
  names by hand: an unpublished pair expands to an empty string and surfaces as an opaque
  `FetchContent` error, where the selector fails at configure time naming the pair it could not find.
- The pin also carries `ET_RUNTIME_OPENVINO_*` rows (an OpenVINO CPU runtime bundle, linux-x86_64
  only). This engine does not link the OpenVINO delegate, so those rows are unconsumed.
- Two runtime behaviours changed at pin `1.3.1-10`: `EXECUTORCH_XNNPACK_SHARED_WORKSPACE` is pinned
  **ON**, so one workspace arena is shared across delegate instances; and the `devtools` variant is
  built **with** logging, so it is no longer a logging-free comparison point in
  `native/build_variants.sh` — only `bare` is.
- A post-link CMake guard (`assert_xnnpack_registered.cmake`, Linux only) fails the build if the XNNPACK backend registration got GC'd out of the `.so`. Windows covers the same property at runtime via the Catch2 suite executing an XNNPACK-delegated `add.pte`.
- The runtime's first-party custom op `etnp::lstm` (linux-x86_64 `logging` tarball only) is
  whole-archived into the shim when the tarball ships `lib/cmake/ETNPExtras/ETNPExtras.cmake`
  (auto-detected in `native/CMakeLists.txt`). Exercised end-to-end by `LstmModelIT`.
- **Windows links the `-static` (`/MT`) pin row** so the shipped DLL needs no VC++ redistributable.
  Windows publishes *two* rows for one platform, hence two variables: `ET_PLATFORM` is the platform
  identity (`windows-x86_64`) and `ET_RUNTIME_ROW` is the pin-row key (`windows-x86_64-static`). The
  `/MD` row exists for CPython consumers and is **not** what we link — it stays in the pin file only
  so `cmake_resolution.sh` can prove the row is a real choice. MSVC does **not** reliably diagnose a
  CRT mismatch (no `LNK2038`, not even an `LNK4098`), so `native/tests/check_windows_crt.sh` is the
  real gate; it runs over both the shim tree and the QA tree.
- **`find_package(executorch)` supplies no language standard.** ExecuTorch's headers require C++17 and
  enforce it with a hard `#error` in `runtime/platform/compiler.h`, but the installed CMake package
  exports **no** `INTERFACE_COMPILE_FEATURES` on any target — verified on both Linux and Windows
  builds of v1.3.1. `native/CMakeLists.txt` therefore states the standard itself
  (`CMAKE_CXX_STANDARD 20` + `CMAKE_CXX_STANDARD_REQUIRED ON`); do **not** delete those as redundant.
  Removing them breaks MSVC only — GCC defaults to `gnu++17` and masks it — surfacing as
  `fatal error C1189: #error: "You need C++17 to compile ExecuTorch"`. Measured on MSVC 19.51: the
  compiler defaults to `_MSVC_LANG=201402`, i.e. C++14, so the flag is doing real work.

### glibc floor (important for releases)

ExecuTorch 1.3 pins `torch==2.12.0`, whose wheel needs **glibc ≥ 2.28**. So the shipped `.so` must be built inside a `manylinux_2_28` container to keep that floor (covers RHEL/Rocky 8+, Ubuntu 20.04+, Debian 11+). Building on the host produces a `.so` linked against host glibc that **breaks the floor** — fine for local `./gradlew test`, never for a release.

## Build & test

**The JVM integration tests load the native library, so the native shim must be built and staged first.**

### Native shim (do this first)

```bash
./native/local_build_wrapper.sh          # the blessed Linux path: runs the pinned engine-build image, keeps glibc-2.28 floor
```

`local_build_wrapper.sh` runs the **pinned shared engine-build image** — a `manylinux_2_28`
derivative whose digest lives in `.engine-build-image` — rather than building one. It is the
**blessed** Linux path because it holds the glibc-2.28 floor. `native/build.sh`, what the wrapper
invokes *inside* the container, no longer installs anything: it **asserts** its toolchain (JDK
headers via `JAVA_HOME`, `ninja` on PATH). On a suitably equipped Linux host, `native/build.sh`
directly now **works** — but the artifact links host glibc and **breaks the floor**, so that is
for local `./gradlew test` only, never a release. The wrapper stages the `.so` into
`src/main/resources/native/linux-x86_64/`. The runtime is fetched by CMake during the run — no
ExecuTorch checkout needed (network access required).

#### Windows build

There is no container on Windows (the manylinux image only bakes the glibc floor for Linux), so the shim is built directly on the host by the same `native/build.sh` — it detects Git-Bash (`uname -s` = `MINGW*`/`MSYS*`) and takes the Windows path. Requirements, all generic (no assumptions about VS edition or a specific machine):

- **Visual Studio 2022 with the C++ toolchain (any edition** — Community/Professional/Enterprise). CI discovers it edition-agnostically via `vswhere -latest -products *` and activates it with `Launch-VsDevShell.ps1 -Arch amd64`. `build.sh` does **not** activate VS itself — the caller must already have the MSVC dev shell active (it just asserts `cl` and `ninja` are on PATH).
- **Ninja** and **CMake** on PATH (both ship with the VS C++ workload).
- **Git-Bash** to run `build.sh` (invoke it by explicit path so PATH order can't pick WSL's `bash.exe`; use a non-login shell so the profile doesn't reset the VS env).
- **A JDK for headers only** — set `JAVA_HOME` to any JDK; the build compiles against `include/win32/jni_md.h` and never links `libjvm`. CI binds JDK 8 deliberately (oldest supported `jni.h` = widest runtime compatibility), but any JDK's headers work.

Key ABI constraint: the build passes `-DCMAKE_BUILD_TYPE=Release` on Windows because MSVC encodes the CRT flavour into every object and refuses to mix them. The pinned runtime tarball is built Release (`/MD`), so a non-Release shim fails to link with `LNK2038` `RuntimeLibrary`/`_ITERATOR_DEBUG_LEVEL` mismatches. GCC/ELF has no such ABI tag, so the Linux leg leaves the build type unset. Output is `executorch_djl.dll` (no `lib` prefix), staged into `src/main/resources/native/windows-x86_64/`.

### JVM build/test (Gradle, JDK 17)

```bash
./gradlew test        # unit + native integration tests (excludes @Tag("leak"))
./gradlew leakTest    # JVM memory-leak stress test (constrained heap/direct memory)
./gradlew build       # full build incl. jacoco coverage report
```

Every root-project `Test` task runs with `-Xcheck:jni`, the JVM's JNI-contract checker. The flag is attached to
the test-task umbrella in `build.gradle.kts` (`tasks.withType<Test>().configureEach`) rather than to
`tasks.test`, because `tasks.test` excludes eight tags including `oom` — and `oomTest`, the task
that drives the allocation-failure paths this checker polices, is one of the excluded ones. A JNI
contract violation surfaces as a `WARNING in native method:` line or a VM abort, not a test
failure, so it is easy to miss. `JniCheckFlagTest` and `JniCheckFlagTaggedTest` prove the flag is
attached (the tagged subclass carries the assertion into the eight tag-filtered tasks); deleting
either silently removes the proof.

### Threading / workspace stress (local only, opt-in)

```bash
./gradlew stressGate                     # 8-thread correctness gate, ~30s (add -PstressSeconds=N)
./gradlew stressSweep                    # 9-cell throughput matrix -> build/reports/stress/sweep.tsv
ET_STRESS=1 ./native/local_build_wrapper.sh native/build_qa.sh   # native harness under ASan + UBSan
```

**None of these run in CI, deliberately** — they saturate every core for their whole duration. The
`stress`, `stress-sweep`, and `stress-baseline` tags are excluded from `tasks.test`.

The fixture is `src/test/resources/models/stress/` — a bucket-gather + 5-layer MLP whose `.pte` and
golden digests are **committed together**; regenerating one without the other is a silent
wrong-answer bug, so `tools/scripts/export_stress_model.py` always writes both.

`stressSweep` is two forked JVMs (`stressSweepCore` + `stressSweepBaseline`) because the intra-op
pool is process-global and write-once, so intra-op=1 cells and the intra-op=default cell cannot
share a process.

`src/test/java/org/measly/executorch/stress/PerThreadContext.java` is the reference pattern for
multi-threaded use: **one `ZooModel` per thread**, not a shared model behind a `ThreadLocal`.

Run a single test class/method:
```bash
./gradlew test --tests 'org.measly.executorch.engine.EtModelTest'
./gradlew test --tests 'org.measly.executorch.engine.EtModelTest.loadAndForwardAddModel'
```

Native library JARs are published per-platform with a classifier (`djl-executorch-engine-<platform>.jar`), sourced from `build/native-staging/<platform>/`.

### Native QA / benchmarking (optional)

Run these **through the container wrapper** so the toolchain matches:
```bash
./native/local_build_wrapper.sh native/build_qa.sh        # Catch2 units + leak harness under ASan + UBSan
./native/local_build_wrapper.sh native/bench.sh           # Release timing harness
ITERS=2000 ./native/local_build_wrapper.sh native/build_variants.sh   # times all 3 runtime variants
```

Shell-level tests for the build machinery live in `native/tests/` (e.g. `cmake_resolution.sh` exercises pin resolution for a foreign platform without that hardware, via `-DET_PRINT_RESOLUTION=ON`).

`native/build_qa.sh` builds the QA tree under **both** ASan and UBSan, so the Catch2 units and the
leak harness run under UndefinedBehaviorSanitizer too. UBSan is a gate, not a log: UB **aborts** the
run rather than printing, so a QA failure may be a `runtime error:` line rather than a failed
assertion — treat either as a finding. The check set lives in the `ET_UBSAN_CHECKS` CMake cache
variable (`undefined,float-cast-overflow,float-divide-by-zero`, minus `vptr`) and can be narrowed
for a one-off run. GCC has no ignorelist, so the way to exempt a function is
`__attribute__((no_sanitize("undefined")))` (or a per-TU compile-option override).
`implicit-signed-integer-truncation` is clang-only and therefore uncovered by this gate.

The gate also runs in CI on both Linux arches — `native-build-job.yml` invokes `build_qa.sh` in its
`linux-x86_64` and `linux-aarch64` rows — and UBSan adds compile and run time to both.

### The JNI shim UBSan gate (JVM-driven)

`native/ubsan_gate.sh` is the **only** configuration that instruments the JNI shim:
`native/CMakeLists.txt` skips `jni/` under `ET_BUILD_QA`, and `build_qa.sh` — et_runtime + Catch2 +
harnesses — is JVM-free by design. This gate builds the shim under UBSan and runs the JVM suite
against it, so the marshalling code the JNI tests exercise is checked.

It runs in **two phases** because no single environment has both the toolchain and a usable JDK: the
pinned image carries Corretto 8 (chosen for the oldest supported `jni.h`), which cannot start Gradle
9.6.1's JDK 17 toolchain. `ET_UBSAN_MODE` selects the phase — `build` | `test` | `all`, default
`auto` (build-only inside the pinned image, both phases outside it). The local invocation is the two
commands the script prints:

```bash
./native/local_build_wrapper.sh native/ubsan_gate.sh   # build phase (in-container)
ET_UBSAN_MODE=test ./native/ubsan_gate.sh              # JVM phase (host, JDK 17)
```

A UB hit presents as a **JVM hard crash** mid-test, not a Java exception or assertion failure: the
`runtime error:` line and its stack trace sit **above** the JVM's own crash dump. That is the gate
working, not a flake.

The instrumented library is never staged into `src/main/resources/native/`. It is reached through
`EXECUTORCH_LIBRARY_PATH`, which `LibUtils` honours ahead of the classpath copy and which
`build.gradle.kts` declares as a `Test` task input — so the plain tree is untouched and no rebuild
is needed afterwards. The gate links with `-static-libubsan` so the UBSan runtime travels inside
the `.so` and a stock JVM can `dlopen` it, and it asserts the result carries **no dynamic `libubsan`
dependency**.

CI runs the gate on **`linux-x86_64` only**: `native-build-job.yml` builds the instrumented shim in
that matrix row, and `native-build.yml`'s `ubsan-jvm-gate` job downloads the
`executorch-ubsan-linux-x86_64` artifact and runs the JVM phase. The aarch64 row is deliberately
not gated — a second native build plus a `--rerun-tasks` JVM suite is real CI time, and it would
double for no new defect class.

## Example module

`example/` is a standalone MobileNetV2 image-classification demo (`org.measly.example.MobilenetExample`) that benchmarks this ExecuTorch engine against the LibTorch/PyTorch DJL engine (JMH benchmarks in `src/jmh/`). Model artifacts (`.pte`/`.pt`) are generated on demand by `./gradlew :example:exportModels` (needs `uv` on PATH; runs `tools/scripts/export_mobilenet.py`).

## Conventions worth knowing

- `EtSymbolBlock.forward()` is **not thread-safe** on the same model — one `Model`/`Predictor` per thread, and never `close()` a model with a forward in flight.

  > **Threading, and why more threads is usually wrong _under the default sharing mode_.** The rule
  > above is about *safety*, not throughput. XNNPACK-delegated models already parallelize inside a
  > single `forward()` on ExecuTorch's shared intra-op pool, and under the shipped `global`
  > workspace sharing mode concurrent delegate calls serialize on one process-global workspace
  > mutex — so N `Predictor`s on N threads is typically slower than one, not N× faster. Tune
  > `ai.djl.executorch.num_threads` before adding caller threads. Measured on a 4-core/8-thread
  > host with MobileNetV2: 1 thread 462 forwards/s, 4 threads 305, 8 threads 147 (peak RSS 33 MB →
  > 224 MB). Ratios on larger hosts are unmeasured.
  >
  > **Those figures are conditional on that mutex.** With `workspaceSharingMode=disabled` the model
  > gets a private workspace and caller threads scale — achieved parallelism at 1 intra-op thread
  > was 1.12/1.12/1.12/1.17 at 1/2/4/8 caller threads under `global`, versus 1.12/2.23/4.35/7.13
  > under `disabled`.
- `ai.djl.executorch.num_threads` (JVM flag) or `EtEngine.setIntraOpThreads(n)` sizes ExecuTorch's intra-op (XNNPACK) threadpool. Process-global, write-once: applied and sealed at the first model load; the effective native count is `EtEngine.getIntraOpThreads()`.
- `Criteria.optOption("workspaceSharingMode", "disabled"|"per_model"|"global")` picks the XNNPACK workspace sharing mode **per model**; `ai.djl.executorch.workspace_sharing_mode` (JVM flag) is the default for models that don't specify. These two strings are published as `EtEngine.WORKSPACE_SHARING_MODE_OPTION` and `EtEngine.WORKSPACE_SHARING_MODE_PROPERTY`. Unlike `num_threads` this is neither process-global nor write-once — ExecuTorch resolves it per delegate at load, so modes compose and load order is irrelevant. An unrecognized *option* fails the load; an unrecognized *property* warns and is ignored. Absent both, no spec is sent and the runtime default (`global` for our pin) applies. See `docs/superpowers/specs/2026-08-08-workspace-sharing-mode-design.md`.
- `EtEngineStats.snapshot()` is the production monitoring surface: effective config, process totals, and per-model counters/native footprint. A JMX MXBean (`org.measly.executorch:type=EtEngineStats`) auto-registers at the first model load; `ai.djl.executorch.jmx_enabled=false` (published as `EtEngine.JMX_ENABLED_PROPERTY`) opts out, and a registration failure is a logged warning, never a failed load. Two conventions matter when reading it: byte fields use **`-1` = unavailable** vs **`0` = genuinely zero** (`stagingBytes` is legitimately `0` for memory-planned models, i.e. nearly all of them), and `modelsLive` is an **upper bound** — the registry holds models weakly, so one dropped without `close()` still counts until the GC reclaims it. Forward counters are hot-path `volatile long`s written count→total→max, an order the `max <= total` invariant depends on. DJL's `Predictor.setMetrics(...)` is for profiling only: its `limit` defaults to uncapped, so it retains every sample forever. See `docs/superpowers/specs/2026-08-09-production-observability-design.md`.
- `EtStatsSnapshot.getXnnpackWorkspaceBytes()` reports the XNNPACK delegate's arena size. It is the
  one byte field that is **not** a sum over live models: the pin sets
  `EXECUTORCH_XNNPACK_SHARED_WORKSPACE=ON`, so one arena backs the whole process and the figure is
  already the total — never sum it, and there is deliberately no per-model counterpart. It is a
  high-water mark including alignment padding, so it is the peak and not the live footprint. `0` is
  a real answer that stands until a delegated method **executes** — loading one leaves the arena at
  0, and a delegated model that allocates nothing never grows it (both an elementwise add and a
  Linear measure 0; only a conv allocates, which is why `native/spike/conv.pte` exists). The backing
  option is a **vendored patch in the pinned distribution**, not upstream ExecuTorch, and requires
  runtime `1.3.1-10`+; against a stock runtime the key does not resolve and this reads `-1`.
- `weight_cache_enabled` is deliberately **not** exposed. `XnnpackBackend::execute()` holds a second process-global mutex (`weights_cache_mutex_`) for the whole delegate call whenever a model uses the cache, which would undo everything `workspaceSharingMode=disabled` buys. It is off in our pin (`EXECUTORCH_XNNPACK_ENABLE_WEIGHT_CACHE=OFF`), which is what makes the `disabled` numbers above real — treat a pin bump that flips it as a performance regression. To enable it anyway no rebuild is needed: the macro guards only the *default*, and `XNNWeightsCache` is compiled into the shipped `libxnnpack_backend.a`. Set `weight_cache_enabled` (a **bool**) in the same `LoadBackendOptionsMap` built in `native/core/et_runtime.cpp`, and keep those models off the hot path.
- `EtRuntime`'s constructor calls `Module::load_forward()` unconditionally so the workspace sharing mode (and any XNNPACK delegate) is resolved at construction, not deferred to the first `forward()`. This applies to every XNNPACK-delegated model, not only ones that set the new option: model loading is slower and the first inference is faster, an invalid sharing mode now fails at load rather than at first predict, and in `native/harness/et_timing_harness.cpp` this shifts measured cost from `cold_ms` into `load_ms` (steady-state throughput is unaffected since warmup is discarded there).
- The `native/spike/` directory holds throwaway spike/smoke files (`EtNative.java`, `cpp_smoke.cpp`, `add.pte`), not production code.
- `docs/README.md` is the documentation index. Current reference material (`docs/building.md`,
  `docs/native-architecture.md`, `docs/benchmarking.md`, `docs/ci-native-build.md`,
  `docs/executorch-build-notes.md`, `docs/executorch-host-buffer-contract-brief.md`) lives directly
  under `docs/`; point-in-time records (handovers, work-in-progress notes, superseded research) live
  under `docs/research/`, kept for their reasoning but not current guidance. Design docs live in
  `docs/superpowers/specs/` and `docs/superpowers/plans/`; the top-level
  `djl-executorch-engine-design.md` is the overall design writeup.
