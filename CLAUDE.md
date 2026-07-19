# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A DJL ([Deep Java Library](https://djl.ai/)) engine plugin that runs ExecuTorch (`.pte`) models. DJL 0.36.0 only supports the deprecated TorchScript export API; this engine adds ExecuTorch as a *separate* DJL engine so PyTorch models exported via the newer ExecuTorch backend can run under DJL, and to allow gradual migration off TorchScript. CPU-only, limited NDArray support. Group/coordinates: `org.measly:djl-executorch-engine`.

Supported platforms: `linux-x86_64` and `windows-x86_64` (both ship the `logging` runtime variant). `bare`/`devtools` runtime variants are Linux-only benchmarking builds.

## Two-layer architecture

**Java layer** (`src/main/java/org/measly/executorch/`)
- `engine/` — the DJL SPI implementation. `EtEngineProvider` is registered via `META-INF/services/ai.djl.engine.EngineProvider`. `EtEngine` (rank 10) → `EtModel` (loads `.pte`, owns the native handle) → `EtSymbolBlock` (runs `forward()`, marshals `NDList` ↔ `EtTensor[]`) → `EtNDManager`/`EtNDArray` (minimal tensor factory).
- `jni/` — the JNI boundary. `EtNative` holds the `native` method declarations and loads the `.so` on class init; `EtTensor`/`EtMethodMeta` are the marshalling structs.
- `translate/` — DJL `Translator` support types.
- `LibUtils` resolves and loads the native library: `EXECUTORCH_LIBRARY_PATH` env override wins; otherwise the `.so`/`.dll` is extracted from the classpath (`/native/<platform>/`) into a **content-addressed cache** (`~/.cache/executorch-djl/<sha256>/` on Linux, `%LOCALAPPDATA%\executorch-djl\<sha256>\` on Windows) and `System.load`ed. Windows can't delete a loaded DLL, hence the stable per-content dir. Keep `LibUtils.libName` in sync with `nativeLibName` in `build.gradle.kts`.

**Native layer** (`native/`)
- `core/et_runtime.{h,cpp}` — a **JNIEnv-free** C++ core (`measly::et::EtRuntime`) that wraps the ExecuTorch `Module`. Deliberately free of any JVM dependency so it can be linked by the shim, the Catch2 unit tests, and the leak harness alike. Zero-copy in (borrowed input pointers), single-copy out.
- `jni/executorch_djl_jni.cpp` + `jni/et_logging.cpp` — the JNI shim (`executorch_djl` shared library). `et_logging.cpp` is a PAL bridge that forwards native `ET_LOG` output to slf4j via `EtNative.nativeLog` (level codes: 0=debug 1=info 2=warn 3=error).
- `harness/` — `et_timing_harness` (Release benchmark) and `et_leak_harness` (ASan/LSan). `test/et_runtime_test.cpp` — Catch2 units. These link only the JNIEnv-free core, so QA/bench configures need no JDK.

### The ExecuTorch runtime is NOT built here

The engine links against the ExecuTorch runtime, but that runtime is **downloaded**, not compiled. CMake `FetchContent`s a hash-pinned, build-attested tarball published by the separate [`executorch-runtime-dist`](https://github.com/measly-java-learning/executorch-runtime-dist) repo. The pin lives in `native/cmake/EtRuntimePin.cmake` (**generated — do not hand-edit**; bump by replacing the whole file with the asset from the next `v<etver>-<pkgrev>` release, then re-applying the comment header). The SHA256 change is the supply-chain review gate.

- **Escape hatch**: set `ET_INSTALL=/path/to/et-install` to link an existing runtime tree; CMake then skips the download.
- ExecuTorch runtime version is currently `1.3.1` (pin `1.3.1-8`); mirrored in `EtEngine.EXECUTORCH_VERSION`.
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
./native/local_build_wrapper.sh          # BLESSED: builds inside manylinux_2_28, keeps glibc-2.28 floor
./native/build.sh                          # LOCAL FAST PATH — host build, breaks the floor, do NOT ship
```

The wrapper stages the `.so` into `src/main/resources/native/linux-x86_64/`. The runtime is fetched by CMake during the run — no ExecuTorch checkout needed (network access required).

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

Run a single test class/method:
```bash
./gradlew test --tests 'org.measly.executorch.engine.EtModelTest'
./gradlew test --tests 'org.measly.executorch.engine.EtModelTest.loadAndForwardAddModel'
```

Native library JARs are published per-platform with a classifier (`djl-executorch-engine-<platform>.jar`), sourced from `build/native-staging/<platform>/`.

### Native QA / benchmarking (optional)

Run these **through the container wrapper** so the toolchain matches:
```bash
./native/local_build_wrapper.sh native/build_qa.sh        # Catch2 units + ASan/LSan leak harness
./native/local_build_wrapper.sh native/bench.sh           # Release timing harness
ITERS=2000 ./native/local_build_wrapper.sh native/build_variants.sh   # times all 3 runtime variants
```

Shell-level tests for the build machinery live in `native/tests/` (e.g. `cmake_resolution.sh` exercises pin resolution for a foreign platform without that hardware, via `-DET_PRINT_RESOLUTION=ON`).

**Container file-ownership gap**: container builds run as root. `native/build.sh` chowns *its own* outputs back via an EXIT trap when passed `HOST_UID`/`HOST_GID`, but `bench.sh`/`build_variants.sh`/`build_qa.sh` do not — they leave root-owned `native/bench/`, `native/bench-results/`, `native/asan/`. Fix by hand with `sudo chown -R "$(id -u):$(id -g)" ...` after running them.

## Example module

`example/` is a standalone MobileNetV2 image-classification demo (`org.measly.example.MobilenetExample`) that benchmarks this ExecuTorch engine against the LibTorch/PyTorch DJL engine (JMH benchmarks in `src/jmh/`). Model artifacts (`.pte`/`.pt`) are generated on demand by `./gradlew :example:exportModels` (needs `uv` on PATH; runs `tools/scripts/export_mobilenet.py`).

## Conventions worth knowing

- `EtSymbolBlock.forward()` is **not thread-safe** on the same model — one `Model`/`Predictor` per thread, and never `close()` a model with a forward in flight.
- The `native/spike/` directory holds throwaway spike/smoke files (`EtNative.java`, `cpp_smoke.cpp`, `add.pte`), not production code.
- Design docs live in `docs/superpowers/specs/` and `docs/superpowers/plans/`; the top-level `djl-executorch-engine-design.md` is the overall design writeup.
