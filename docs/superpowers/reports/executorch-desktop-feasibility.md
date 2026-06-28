# ExecuTorch Desktop-Linux Native Build — Feasibility Report

> **Verdict: Desktop-Linux native build is practical — GO.**
>
> A standard (non-Android) desktop JVM can load a `.pte` and run ExecuTorch inference via a thin
> custom JNI shim with no `fbjni` and no Android dependencies. All three gates in the
> [feasibility plan](../plans/2026-06-28-executorch-desktop-linux-native-build.md) passed end to end.

Date: 2026-06-28 · ExecuTorch **v1.3.1** · Path **B** (thin custom JNI shim).

## Environment

| | |
|---|---|
| OS / glibc | Ubuntu 24.04, glibc 2.39 |
| Compiler | GCC 13.3.0 (C++17) |
| CMake | 3.28.3 |
| JDK | Zulu OpenJDK 17.0.19 (headers only) |
| Python | 3.12.3 (ExecuTorch AOT export) |
| ExecuTorch | v1.3.1, built from source with submodules |

## Gate results (evidence)

**Prerequisite — runtime builds & is linkable.** `cmake --preset linux` with PIC + extensions +
XNNPACK, then `cmake --install`. Produced `libexecutorch.a`, `extension_module*`,
`extension_tensor`, `libxnnpack_backend.a`, and `lib/cmake/ExecuTorch/executorch-config.cmake`.

**Task 1 — pure-C++ inference (no JVM).** `cpp_smoke add.pte` (forward(2.0, 3.0)):
```
RESULT=5.0      # exit 0
```
Proves the runtime runs `.pte` inference on desktop Linux independent of any JVM.

**Task 2 — JVM bridge.** `libexecutorch_djl.so` built; `ldd -r` shows **no undefined
ExecuTorch/XNNPACK symbols** (only `Java_*`, resolved by the JVM at load). End-to-end Java smoke:
```
RESULT=5.0
JVM<->ExecuTorch desktop bridge OK
```
Clears the show-stopper: a desktop JVM loads a `.pte` and runs inference.

**Task 3 — reproducible packaging.** `native/build_desktop.sh` builds the shim against
`$ET_INSTALL` and stages the artifact; the bundled `.so` re-runs the Java smoke green.

## The artifact

`src/main/resources/native/linux-x86_64/libexecutorch_djl.so` — **8,158,784 bytes (7.8 MB)**.

Self-contained (ExecuTorch runtime, XNNPACK, op kernels statically linked); only universal
dynamic deps:
```
$ ldd libexecutorch_djl.so
    linux-vdso, libstdc++.so.6, libm.so.6, libgcc_s.so.1, libc.so.6, ld-linux-x86-64.so.2
$ ldd … | grep -E 'libexecutorch|libxnnpack|libtokenizer'   →  none (static)
```
Confirms the design's footprint claim: single-digit MB vs. hundreds of MB for libtorch, with no
transitive native dependency chain.

## Chosen approach and why

**Path B — thin custom JNI shim** (`native/jni/executorch_djl_jni.cpp`, raw `<jni.h>` over
`executorch::extension::Module`). The design doc's original "Option A" (recompile ExecuTorch's
stock Android JNI) was rejected after inspection: `extension/android/CMakeLists.txt` hard-fails
with `if(NOT ANDROID) → FATAL_ERROR`, downloads a prebuilt **Android** `fbjni` AAR, and relies on
the NDK for `jni.h`. Path B avoids `fbjni`, Android, and that whole entanglement; cost is that we
do not reuse the upstream `org.pytorch.executorch.*` Java classes (the DJL `EtModel`/`EtNDArray`
layer calls our own small `org.measly.executorch.jni.EtNative` surface instead).

## Non-obvious build facts (carry forward into the engine)

Against ExecuTorch v1.3.1's installed CMake package, several assumptions in the original design
were wrong and are now encoded in `native/CMakeLists.txt`:
- Config dir is `lib/cmake/**ExecuTorch**` (capital E), config file `executorch-config.cmake`.
- `executorch-config.cmake` calls `find_package(tokenizers CONFIG)` with no hint → must set
  `tokenizers_DIR` / prefix path or imported targets go missing (`tokenizers::tokenizers`).
- Exported targets are **bare** — no `executorch::` namespace.
- Backend/ops targets (`xnnpack_backend`, `portable_ops_lib`) **self-embed `--whole-archive`** in
  `INTERFACE_LINK_OPTIONS`; wrapping again causes "multiple definition".
- The shim links **no** JDK libs — `jni.h` headers only (`find_package(JNI)` fails seeking
  AWT/JVM); derive include dir from `JAVA_HOME`.

## Known limitation → handed to the release pipeline

The dev-box artifact requires **GLIBC_2.38 / GLIBCXX_3.4.29** (built on glibc 2.39), which is too
high to distribute (excludes RHEL 8, Ubuntu 20.04–22.04, Amazon Linux 2). This is **not** a
feasibility blocker — it is a build-environment choice. Production binaries must be built in an
old-glibc (manylinux) container. Full analysis, two-stage pipeline, build-container options, and
the portability gate: [`docs/ci-native-build.md`](../../ci-native-build.md).

## Recommendation

Proceed to the DJL engine implementation (`EtEngine`, `EtEngineProvider`, `EtModel`,
`EtNDManager`/`EtNDArray`, `MapTranslator`, SPI registration, hybrid mode) per the design doc,
loading this `.so` via `System.load()`. Defer multi-platform natives and the manylinux release
pipeline to their own plans.
