# Windows x86_64 builds — design

**Date:** 2026-07-15
**Status:** Approved for planning
**Goal:** Make `windows-x86_64` a first-class published platform alongside `linux-x86_64`: CI builds and QAs `executorch_djl.dll`, it ships in the published jar, and `LibUtils` loads it on Windows.

---

## 1. Context and the corrected premise

An external handoff note (`windows-jni-handoff.md`, deliberately not in git — it contains host
credentials) states the Windows runtime artifact is **core-only: no XNNPACK, no extras, "only rely on
`executorch`"**.

**That claim is stale and does not describe the artifact we consume.** Verified directly against
`executorch-runtime-1.3.1-logging-windows-x86_64.tar.gz` from release `v1.3.1-6`:

- `BUILDINFO` records `-DEXECUTORCH_BUILD_XNNPACK=ON`, plus the `EXTENSION_MODULE` / `EXTENSION_TENSOR`
  / `EXTENSION_DATA_LOADER` flags.
- `lib/` contains `xnnpack_backend.lib`, `extension_module_static.lib`, `extension_tensor.lib`,
  `portable_ops_lib.lib` — all four targets `native/CMakeLists.txt` links.
- All four are exported as CMake targets by `lib/cmake/ExecuTorch/ExecuTorchTargets.cmake`.

The Windows artifact is therefore **feature-equivalent to Linux for this project's needs**. The design
below assumes that. If a future artifact regresses to genuinely core-only, this design is invalid and
the engine would need a degraded, XNNPACK-less Windows story — a materially different project.

**Corollary:** the handoff's "warm-host blind spots" (MAX_PATH/`core.longpaths`, `core.symlinks`,
multi-Python `find_package(Python3)` mismatch) bite when *building ExecuTorch from source*. This repo
only *consumes* a prebuilt tarball and never checks out ExecuTorch. **None of them apply here.**

## 2. Runtime pin

Replace `native/cmake/EtRuntimePin.cmake` wholesale with the `EtRuntimePin.cmake` asset from release
`v1.3.1-6`, per the procedure in the file's own header ("replace this whole file... The SHA256 change
is the supply-chain review gate").

Consequences, all intended:

- Adds `ET_RUNTIME_URL_logging_windows-x86_64` + SHA — the row this project needs.
- **Moves every Linux SHA from `v1.3.1-2` to `v1.3.1-6`.** This PR is *not* a Windows-only diff. Linux
  CI revalidates on the same PR, but review must treat the Linux SHA changes as a supply-chain gate.
- Drags in unused `linux-aarch64` rows (see §9, out of scope).
- Loses the repo's hand-written commentary in the current pin header. Acceptable: the file is marked
  "Do not edit by hand", and the commentary's content is preserved here in §9.

Windows ships **`logging` only** — there is no `bare` or `devtools` Windows row. `native/build_variants.sh`
benchmarking remains Linux-only.

## 3. CMake (`native/CMakeLists.txt`)

Four narrow changes.

### 3.1 Platform becomes a cache variable (test seam)

Replace the hardcoded `set(_ET_PLATFORM "linux-x86_64")` (line 17) with a cache variable defaulting to
the detected platform:

```cmake
if(WIN32)
  set(_ET_PLATFORM_DEFAULT "windows-x86_64")
else()
  set(_ET_PLATFORM_DEFAULT "linux-x86_64")
endif()
set(ET_PLATFORM "${_ET_PLATFORM_DEFAULT}" CACHE STRING "Runtime platform row to resolve from the pin")
```

**Cache variable, not a plain `set`, is deliberate.** It composes with the existing `ET_PRINT_RESOLUTION`
seam so Windows pin resolution is testable **from a Linux host** with no Windows machine:

```
cmake -S native -B <tmp> -DET_PLATFORM=windows-x86_64 -DET_PRINT_RESOLUTION=ON
```

All downstream uses of `_ET_PLATFORM` (the `_ET_STEM`, `_ET_RESOLVED_URL`, `_ET_RESOLVED_SHA` lookups at
lines 25-27) switch to `ET_PLATFORM`. The `ET_RESOLUTION` status line gains `platform=${ET_PLATFORM}`.

### 3.2 Named failure for unavailable variants on Windows

Windows has only a `logging` row. Today `-DET_RUNTIME_VARIANT=bare` on Windows falls through to the
generic `No pin row for variant=... platform=...` error (line 29). Add an explicit check that names the
real constraint:

```cmake
if(ET_PLATFORM STREQUAL "windows-x86_64" AND NOT ET_RUNTIME_VARIANT STREQUAL "logging")
  message(FATAL_ERROR "windows-x86_64 ships the 'logging' variant only; got '${ET_RUNTIME_VARIANT}'. "
                      "bare/devtools are Linux-only (benchmarking; see native/build_variants.sh).")
endif()
```

This must sit **before** the `ET_PRINT_RESOLUTION` early-return so the seam exercises it.

### 3.3 JNI header directory

Line 81 hardcodes `"${JAVA_HOME}/include/linux"`. The `jni_md.h` subdirectory is platform-named:

```cmake
if(WIN32)
  set(_JNI_MD_DIR "win32")
else()
  set(_JNI_MD_DIR "linux")
endif()
target_include_directories(executorch_djl PRIVATE
  "${JAVA_HOME}/include" "${JAVA_HOME}/include/${_JNI_MD_DIR}")
```

### 3.4 Gate the `nm` XNNPACK guard off on Windows

Wrap the POST_BUILD command (lines 87-96) in `if(NOT WIN32)`.

`native/cmake/assert_xnnpack_registered.cmake` is not portable in a deep way, not merely a tooling gap:

- `_GLOBAL__sub_I_XNNPACKBackend` is **Itanium C++ ABI** static-init mangling. MSVC registers static
  initializers via `.CRT$XCU` pointers with no equivalent stable symbol name.
- The `[Tt]` symbol-class regex is **ELF/`nm`** output format.
- MSVC's `/OPT:REF` is the analogous stripper, and `/WHOLEARCHIVE:` the analogous fix — the *risk* is
  real on Windows, but this implementation can only be re-invented, not ported.

Windows coverage for the same property comes from the Catch2 suite instead (see §6). Porting the guard
to MSVC link maps is out of scope (§9).

No `-fPIC` handling is needed — position-independent code is a non-concept on Windows. The existing
`POSITION_INDEPENDENT_CODE ON` on `et_runtime` (line 55) is harmless there.

## 4. Artifact naming: `executorch_djl.dll`

MSVC sets `CMAKE_SHARED_LIBRARY_PREFIX` to empty, so the Windows output is **`executorch_djl.dll`**, not
`libexecutorch_djl.dll`. (A `.lib` import library is also produced; it is a build byproduct and is not
shipped.) JNI entry points are already correctly exported: `native/jni/executorch_djl_jni.cpp` uses
`JNIEXPORT` on all five entry points, which `jni_md.h` defines as `__declspec(dllexport)` on win32. No
`.def` file is needed.

The filename `libexecutorch_djl.so` is hardcoded in three places, each needing a per-platform mapping:

| Site | Current |
|---|---|
| `build.gradle.kts:67` | `.file("libexecutorch_djl.so")` in the release `require()` check |
| `src/main/java/.../LibUtils.java:12` | `private static final String LIB = "libexecutorch_djl.so"` |
| `.github/workflows/native-build-job.yml:75` | upload path glob `**/*.so` |

Packaging otherwise needs no structural change: `build.gradle.kts:56` already emits one
classifier-per-platform jar via `nativePlatforms`, so Windows is an append to that list (plus the
per-platform filename mapping above feeding the `require()` check).

## 5. Build scripts (`native/build.sh`, `native/build_qa.sh`)

One script serves both platforms, branching on `uname -s` (`MINGW64_NT*` / `MSYS_NT*` under Git-Bash).
The caller supplies an already-activated MSVC environment; the scripts do not activate VS themselves.

`native/build.sh` on Windows:
- Skip Corretto RPM extraction — take `JAVA_HOME` from the runner. Assert `include/win32/jni_md.h` exists
  (mirroring the existing Linux `include/linux/jni_md.h` assertion at line 42).
- Skip the `chown` cleanup trap (bind-mount root-ownership is a container concern; there is no container).
- Skip `pip install ninja` — Ninja ships with VS and is on PATH after dev-shell activation.
- `JOBS` from `NUMBER_OF_PROCESSORS` rather than `nproc`.
- Stage to `src/main/resources/native/windows-x86_64/executorch_djl.dll`.

`native/build_qa.sh` on Windows:
- Skip the `dnf` ASan-runtime install and the `-fsanitize=address` flags entirely.
- Build and run **`et_runtime_test` only**. Never `et_leak_harness`.
- Use `-G Ninja` (the Linux QA path's `Unix Makefiles` generator does not apply).

## 6. QA posture

| | linux-x86_64 | windows-x86_64 |
|---|---|---|
| Catch2 unit suite | yes (ASan) | yes (no sanitizers) |
| ASan/LSan leak harness | yes | **no — structurally impossible** |
| XNNPACK survived link | `nm` post-link guard | Catch2 executing `add.pte` |

**The leak harness cannot run on Windows.** MSVC provides ASan (`/fsanitize=address`) but has no
LeakSanitizer implementation. `et_leak_harness` is therefore Linux-only. This is acceptable: the C++
under test (`native/core/et_runtime.cpp`) is platform-neutral, so leak defects it could find are found
on Linux.

**Windows XNNPACK coverage needs no new test.** `native/spike/export_add.py:17` lowers `add.pte` through
`XnnpackPartitioner`, so it is XNNPACK-delegated; `native/test/et_runtime_test.cpp:35` asserts
`add(2,3) == 5` by executing it. If the XNNPACK backend failed to register, that test fails. Running the
existing suite on Windows *is* the runtime check the `nm` guard only proxies for.

Trade-off accepted: on Windows a dropped backend surfaces as a test failure rather than the `nm` guard's
named diagnostic. Judged acceptable given the guard exists for a regression in the *upstream tarball*,
which Linux CI catches first on the same PR.

## 7. CI (`.github/workflows/native-build-job.yml`)

**Split as a sibling job in the same file**, following the precedent set by
`measly-java-learning/executorch-runtime-dist`'s `release.yml`, which pairs a containerized `build` job
with a native `build-windows` job feeding a common `pin` (its lines 99-104 enshrine this as the pattern
for future platforms).

Rationale for same-file over a second workflow file: `native-build-job.yml` is a **reusable** workflow
called by both `native-build.yml` and `publish.yml`, exposing one `artifact-pattern` output. A second
file would force `publish.yml` to call two workflows and merge two outputs. A sibling job keeps
`publish.yml` **entirely untouched** — the existing `executorch-libs-*` pattern already matches both
platforms, and `download-artifact`'s `merge-multiple: true` lands them as sibling directories
`native-staging/linux-x86_64/` and `native-staging/windows-x86_64/`, exactly what `build.gradle.kts`
expects.

New job `build-executorch-shim-windows`:
- `runs-on: windows-2022`. **This runner is the acceptance gate, not winbox** — winbox is VS 18
  Community, the runner is VS 2022/17 Enterprise.
- No `container:`.
- Provenance step (`gh attestation verify`) unchanged; it runs natively on the Windows runner, matching
  on the `logging-windows-x86_64` URL.
- Build step in `shell: pwsh`: discover VS via `vswhere -latest -products *` (**edition-agnostic** — a
  runner-image edition change must not silently break the build), `Launch-VsDevShell.ps1 -Arch amd64
  -SkipAutomaticLocation`, then invoke `native/build.sh` through **Git-Bash by explicit path**
  (`${env:ProgramFiles}\Git\bin\bash.exe`).
- Upload as `executorch-libs-windows-x86_64`, path `src/main/resources/native/**/*.dll`.

Two failure modes to design against, both borrowed from the reference implementation:

- **WSL bash hijack.** Invoking bare `bash` can resolve to WSL's `System32\bash.exe` via PATH order,
  running the build in a Linux environment with no MSVC toolchain. Always use the explicit Git-Bash path.
- **Login shell drops the VS env.** `Launch-VsDevShell` mutates the *current* PowerShell process env;
  a bash child inherits it — but only a **non-login** (`bash -c`, no `-l`) shell. A login shell
  re-sources the profile and resets PATH.

Not needed here (unlike the reference): `git config --system core.longpaths/core.symlinks`. Those exist
in the runtime project for its recursive ExecuTorch checkout. This repo checks out only its own small
tree.

## 8. `LibUtils` — content-addressed cache

`LibUtils.loadLibrary()` currently extracts the shim to a fresh temp file per JVM and calls
`deleteOnExit`. The shim is **11.5 MB**. On Windows the OS holds a loaded DLL open, so that delete
silently fails and **every JVM run permanently leaks 11.5 MB into `%TEMP%`** — unacceptable for an
embedded library.

Replace with a cache directory, on **both** platforms:

```
Windows: %LOCALAPPDATA%\executorch-djl\<sha256>\executorch_djl.dll
Linux:   $XDG_CACHE_HOME (or ~/.cache)/executorch-djl/<sha256>/libexecutorch_djl.so
```

**Cache key is the SHA-256 of the resource bytes**, not a jar-manifest version. A manifest version is
null in dev and test builds, and a stale cache silently serving yesterday's `.so` after a local rebuild
is a nasty debugging session. A content hash is never stale, is self-versioning, and handles dev
rebuilds correctly. Cost is ~40 ms on a cache hit — negligible against loading an 11.5 MB shim that then
runs ML models.

Algorithm:
1. `EXECUTORCH_LIBRARY_PATH` override is honored first, unchanged (existing behavior, `LibUtils.java:23`).
2. Stream the classpath resource, computing SHA-256. On a **hit**, `System.load` the cached file and
   return without writing.
3. On a **miss**, hash *while copying* in a single pass to a `.tmp` sibling, then `ATOMIC_MOVE` into
   place. Atomic rename makes concurrent JVMs racing on the same path safe: last writer wins, and both
   observe identical bytes since the path is content-addressed.
4. If `ATOMIC_MOVE` fails because another JVM won the race, fall back to loading the existing file.

`platform()` gains `windows-x86_64` (`os.name` contains `windows`, `os.arch` in {`amd64`, `x86_64`}),
and the library filename becomes a per-platform mapping rather than the `LIB` constant.

## 9. Out of scope

- **`linux-aarch64`** — rows arrive with the `v1.3.1-6` pin and would unblock the commented-out matrix
  entry at `native-build-job.yml:20-24`, but it is a separate platform with its own runner and QA story.
  Folding it in would double this project's surface. Deserves its own follow-up.
- **`bare` / `devtools` on Windows** — not built upstream. `native/build_variants.sh` stays Linux-only.
- **Windows leak coverage** — no LSan on MSVC (§6).
- **Porting the `nm` guard to MSVC link maps** (`/MAP` + parse) — rejected in favor of the runtime check;
  it would add a second brittle symbol-format dependency.
- **macOS** — the runtime project has no macOS artifact yet.

## 10. Testing

Host-fast (run on Linux, no Windows machine needed):
- `native/tests/cmake_resolution.sh` — assert `-DET_PLATFORM=windows-x86_64 -DET_PRINT_RESOLUTION=ON`
  resolves the `logging_windows-x86_64` URL and stem; assert the §3.2 named failure fires for
  `-DET_PLATFORM=windows-x86_64 -DET_RUNTIME_VARIANT=bare`.
- `native/tests/ci_workflow.sh` — assert the `build-executorch-shim-windows` job exists, runs on
  `windows-2022`, uses edition-agnostic `vswhere`, and uploads `executorch-libs-windows-x86_64`.
- JUnit — `LibUtils.platform()` returns `windows-x86_64` for `os.name=Windows 11`/`os.arch=amd64`.
  (`platform()` is already the only unit-tested method in that class.)

Real gate:
- `windows-2022` runner green on build + Catch2.
- winbox is for iteration only; it is not an acceptance gate (VS edition differs from the runner).

## 11. Risks

| Risk | Mitigation |
|---|---|
| Pin bump changes Linux artifacts, not just adds Windows | Linux CI revalidates on the same PR; SHA diff is an explicit review gate |
| A future upstream Windows artifact regresses to genuinely core-only | §1 premise is stated explicitly; link failure would be loud and immediate at `find_package` |
| winbox (VS 18 Community) diverges from runner (VS 17 Enterprise) | Edition-agnostic `vswhere`; runner is the sole acceptance gate |
| WSL `bash.exe` hijacks the build | Explicit Git-Bash path |
| Concurrent JVMs race on the cache file | Content-addressed path + `ATOMIC_MOVE`; identical bytes either way |

## 12. Security note

`windows-jni-handoff.md` contains the winbox hostname, user, and key path. It lives **outside this repo**
and must stay out of git. Nothing in this design requires committing host or credential details: CI uses
GitHub-hosted runners exclusively, and winbox access is developer-local.
