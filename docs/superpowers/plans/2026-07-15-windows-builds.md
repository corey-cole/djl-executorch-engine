# Windows x86_64 Builds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `windows-x86_64` a first-class published platform: CI builds and QAs `executorch_djl.dll` on a GitHub `windows-2022` runner, it ships in a classifier jar, and `LibUtils` loads it on Windows.

**Architecture:** The ExecuTorch runtime is **downloaded, never built here** — `native/CMakeLists.txt` `FetchContent`s a hash-pinned tarball and links its static archives into one JNI shared library. Windows support is therefore mostly *parameterization*: turn the hardcoded `linux-x86_64` platform into a variable, fork the shell scripts and CI on OS, map the artifact filename per platform, and teach `LibUtils` a second platform. The verified premise (design §1) is that the Windows tarball is **feature-equivalent to Linux** — it ships `xnnpack_backend`, `extension_module_static`, `extension_tensor`, `portable_ops_lib`, all exported by `lib/cmake/ExecuTorch`.

**Tech Stack:** CMake 3.24+ / Ninja, MSVC 2022 (C++20), JNI (JDK 8 headers), Java 17 / Gradle Kotlin DSL, JUnit 5, Catch2 v3, GitHub Actions, bash (Git-Bash on Windows), PowerShell.

**Spec:** `docs/superpowers/specs/2026-07-15-windows-builds-design.md`

## Global Constraints

- **PREREQUISITE: PR 1 must be merged and green.** This plan assumes `native/cmake/EtRuntimePin.cmake` is at `v1.3.1-6` and `ET_RUNTIME_URL_logging_windows-x86_64` exists. **Do not bump the pin in this PR.**
- Windows ships the **`logging` variant only**. There is no `bare`/`devtools` Windows row; `native/build_variants.sh` benchmarking stays Linux-only.
- Windows artifact filename is **`executorch_djl.dll`** — MSVC emits **no `lib` prefix**. Linux stays `libexecutorch_djl.so`.
- **Acceptance gate is the GitHub `windows-2022` runner**, never winbox. winbox is VS 18 Community; the runner is VS 2022/17 Enterprise. winbox is for iteration only.
- **`et_leak_harness` never runs on Windows** — MSVC has ASan but no LeakSanitizer. Leak coverage is Linux-only, permanently.
- The `nm`-based XNNPACK guard is **Linux-only** and must be gated off under `WIN32`. Windows coverage comes from the Catch2 suite executing the XNNPACK-delegated `add.pte`.
- **`linux-aarch64` is out of scope.** Its pin rows exist but stay inert; leave `native-build-job.yml:20-24` commented out.
- Always invoke Git-Bash **by explicit path** (`${env:ProgramFiles}\Git\bin\bash.exe`) and **non-login** (`bash -c`, never `bash -lc`).
- `docs_present.sh` is failing on `main` for unrelated reasons (`et-install/` drift). **Out of scope — do not fix it here.**

---

### Task 1: Make the runtime platform a variable

**Files:**
- Modify: `native/CMakeLists.txt:17` (and the `_ET_PLATFORM` uses at lines 25-27, 29, 34)
- Test: `native/tests/cmake_resolution.sh`

**Interfaces:**
- Consumes: `ET_RUNTIME_URL_logging_windows-x86_64` / `ET_RUNTIME_SHA256_logging_windows-x86_64` from `EtRuntimePin.cmake` (PR 1).
- Produces: CMake cache variable **`ET_PLATFORM`** (`STRING`), defaulting to `windows-x86_64` under `WIN32` else `linux-x86_64`. The `ET_RESOLUTION` diagnostic line gains a **`platform=<value>`** field. Tasks 2 and 9 rely on both.

Why a cache variable rather than a plain `set`: it composes with the existing `ET_PRINT_RESOLUTION`
early-return so Windows pin resolution is testable **from a Linux host with no Windows machine**.

- [ ] **Step 1: Write the failing test**

In `native/tests/cmake_resolution.sh`, after the `devtools` block (after line 28) and before the escape-hatch block, add:

```bash
# Windows resolution, asserted from a Linux host: ET_PLATFORM is a cache var, so the ET_PRINT_RESOLUTION
# seam can resolve a foreign platform's pin row without that platform being present.
out="$(probe -DET_PLATFORM=windows-x86_64)"
grep -q 'platform=windows-x86_64'                                <<<"${out}" || fail "windows platform not echoed"
grep -q 'stem=executorch-runtime-1.3.1-logging-windows-x86_64'   <<<"${out}" || fail "windows stem wrong"
grep -q 'executorch-runtime-1.3.1-logging-windows-x86_64.tar.gz' <<<"${out}" || fail "windows url wrong"

# The default on this (Linux) host must be unaffected by ET_PLATFORM existing.
out="$(probe)"
grep -q 'platform=linux-x86_64'                                  <<<"${out}" || fail "default platform not linux-x86_64"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `bash native/tests/cmake_resolution.sh`

Expected: `FAIL: windows platform not echoed`. `ET_PLATFORM` is not a real variable yet, so `-DET_PLATFORM=...` is ignored and the status line has no `platform=` field.

- [ ] **Step 3: Introduce the ET_PLATFORM cache variable**

In `native/CMakeLists.txt`, replace line 17:

```cmake
set(_ET_PLATFORM "linux-x86_64")
```

with:

```cmake
# Which pin row to resolve. A CACHE var (not a plain set) is deliberate: it lets the
# ET_PRINT_RESOLUTION host-test seam force a foreign platform (-DET_PLATFORM=windows-x86_64) and assert
# the pin resolves, with no machine of that platform present. See native/tests/cmake_resolution.sh.
if(WIN32)
  set(_ET_PLATFORM_DEFAULT "windows-x86_64")
else()
  set(_ET_PLATFORM_DEFAULT "linux-x86_64")
endif()
set(ET_PLATFORM "${_ET_PLATFORM_DEFAULT}" CACHE STRING "Runtime platform row to resolve from EtRuntimePin.cmake")
```

- [ ] **Step 4: Repoint every `_ET_PLATFORM` use**

In the same file, replace the three lookups (lines 25-27) and the error message (line 29):

```cmake
  set(_ET_STEM "executorch-runtime-${ET_RUNTIME_ET_VERSION}-${ET_RUNTIME_VARIANT}-${ET_PLATFORM}")
  set(_ET_RESOLVED_URL "${ET_RUNTIME_URL_${ET_RUNTIME_VARIANT}_${ET_PLATFORM}}")
  set(_ET_RESOLVED_SHA "${ET_RUNTIME_SHA256_${ET_RUNTIME_VARIANT}_${ET_PLATFORM}}")
  if(NOT _ET_RESOLVED_URL)
    message(FATAL_ERROR "No pin row for variant='${ET_RUNTIME_VARIANT}' platform='${ET_PLATFORM}' in EtRuntimePin.cmake")
  endif()
```

And add `platform=` to the diagnostic (line 34):

```cmake
  message(STATUS "ET_RESOLUTION resolution=${_ET_RESOLUTION} variant=${ET_RUNTIME_VARIANT} platform=${ET_PLATFORM} stem=${_ET_STEM} url=${_ET_RESOLVED_URL} et_install=${ET_INSTALL}")
```

Confirm no stale references remain: `grep -n '_ET_PLATFORM' native/CMakeLists.txt` should match only the
`_ET_PLATFORM_DEFAULT` lines from Step 3.

- [ ] **Step 5: Run the test to verify it passes**

Run: `bash native/tests/cmake_resolution.sh`

Expected: `PASS: cmake resolution`

- [ ] **Step 6: Commit**

```bash
git add native/CMakeLists.txt native/tests/cmake_resolution.sh
git commit -m "feat(native): make the runtime platform a cache variable

ET_PLATFORM defaults to windows-x86_64 under WIN32, else linux-x86_64. As a
cache var it composes with the ET_PRINT_RESOLUTION seam, so Windows pin
resolution is asserted from a Linux host with no Windows machine.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Name the logging-only constraint on Windows

**Files:**
- Modify: `native/CMakeLists.txt` (inside the `fetch` branch, before the URL lookup)
- Test: `native/tests/cmake_resolution.sh`

**Interfaces:**
- Consumes: `ET_PLATFORM` (Task 1).
- Produces: a `FATAL_ERROR` containing the exact substring `ships the 'logging' variant only`. Task 9's CI job depends on nothing here; this is purely diagnostic quality.

Windows has no `bare`/`devtools` pin row. Today `-DET_RUNTIME_VARIANT=bare` on Windows dies with the
generic `No pin row for variant=...`, which reads like a broken pin rather than a platform that never
had those artifacts. Name the real reason.

The guard belongs **inside the `fetch` branch**: under the `ET_INSTALL` escape hatch the user supplies
their own prefix and no pin row is consulted, so the variant is irrelevant and must not fail.

- [ ] **Step 1: Write the failing test**

`cmake_resolution.sh`'s `probe()` helper treats a configure failure as a test failure, so a helper that
asserts the *opposite* is needed. Add it directly beneath `probe()` (after line 15):

```bash
probe_fails() {  # extra -D args; asserts configure FAILS and echoes the captured error text
  local b err; b="$(mktemp -d)"; err="${b}/err"
  if cmake -S native -B "${b}" -DET_PRINT_RESOLUTION=ON "$@" >"${err}" 2>&1; then
    cat "${err}"; rm -rf "${b}"; fail "cmake configure unexpectedly succeeded ($*)"
  fi
  cat "${err}"; rm -rf "${b}"
}
```

Then, after the Windows resolution block added in Task 1, add:

```bash
# windows-x86_64 has no bare/devtools pin row upstream. Fail with the real reason, not "no pin row".
out="$(probe_fails -DET_PLATFORM=windows-x86_64 -DET_RUNTIME_VARIANT=bare)"
grep -q "ships the 'logging' variant only" <<<"${out}" || fail "windows+bare should fail with a named error"

# The escape hatch bypasses the pin entirely, so the variant constraint must NOT apply there.
out="$(probe -DET_PLATFORM=windows-x86_64 -DET_RUNTIME_VARIANT=bare -DET_INSTALL=/tmp/my-et)"
grep -q 'resolution=escape-hatch' <<<"${out}" || fail "escape hatch must bypass the windows variant guard"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `bash native/tests/cmake_resolution.sh`

Expected: `FAIL: windows+bare should fail with a named error`. Configure does fail, but with
`No pin row for variant='bare' platform='windows-x86_64'`, which lacks the asserted substring.

- [ ] **Step 3: Add the named guard**

In `native/CMakeLists.txt`, inside the `else()` (fetch) branch, immediately after `set(_ET_RESOLUTION "fetch")` and **before** the `_ET_STEM` line:

```cmake
  # windows-x86_64 ships `logging` only; bare/devtools are Linux-only benchmarking builds that upstream
  # never produced for Windows. Without this, the generic "No pin row" error below reads like a broken
  # pin rather than a platform that never had those artifacts. Inside the fetch branch on purpose: the
  # ET_INSTALL escape hatch consults no pin row, so the variant is irrelevant there.
  if(ET_PLATFORM STREQUAL "windows-x86_64" AND NOT ET_RUNTIME_VARIANT STREQUAL "logging")
    message(FATAL_ERROR
      "windows-x86_64 ships the 'logging' variant only; got '${ET_RUNTIME_VARIANT}'. "
      "bare/devtools are Linux-only (benchmarking; see native/build_variants.sh).")
  endif()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `bash native/tests/cmake_resolution.sh`

Expected: `PASS: cmake resolution`

- [ ] **Step 5: Commit**

```bash
git add native/CMakeLists.txt native/tests/cmake_resolution.sh
git commit -m "feat(native): name the windows logging-only constraint

-DET_RUNTIME_VARIANT=bare on windows-x86_64 died with a generic 'no pin row'
that read like a broken pin. Fail with the real reason instead. Scoped to the
fetch branch so the ET_INSTALL escape hatch is unaffected.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Make the JNI shim compile under MSVC

**Files:**
- Modify: `native/CMakeLists.txt:80-81` (JNI include dirs), `native/CMakeLists.txt:87-96` (post-link guard)
- Test: covered by Task 7's real Windows build; no host test (this is compiler/linker behavior)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: on Windows, the target `executorch_djl` emits **`executorch_djl.dll`** (MSVC sets `CMAKE_SHARED_LIBRARY_PREFIX` to empty). Tasks 6, 7 and 9 depend on that exact filename.

Two changes. No `.def` file and no export macros are needed: `native/jni/executorch_djl_jni.cpp` already
marks all five entry points `JNIEXPORT`, which `jni_md.h` defines as `__declspec(dllexport)` on win32.
No `-fPIC` work either — position-independent code is a non-concept on Windows, and the existing
`POSITION_INDEPENDENT_CODE ON` on `et_runtime` is harmless there.

- [ ] **Step 1: Fix the JNI header directory**

`jni_md.h` lives in a platform-named subdirectory. Replace `native/CMakeLists.txt:80-81`:

```cmake
  target_include_directories(executorch_djl PRIVATE
    "${JAVA_HOME}/include" "${JAVA_HOME}/include/linux")
```

with:

```cmake
  # jni_md.h sits in a platform-named subdir: include/linux vs include/win32.
  if(WIN32)
    set(_JNI_MD_DIR "win32")
  else()
    set(_JNI_MD_DIR "linux")
  endif()
  target_include_directories(executorch_djl PRIVATE
    "${JAVA_HOME}/include" "${JAVA_HOME}/include/${_JNI_MD_DIR}")
```

- [ ] **Step 2: Gate the nm post-link guard to non-Windows**

`native/cmake/assert_xnnpack_registered.cmake` cannot run under MSVC, and this is not a mere tooling gap:
`_GLOBAL__sub_I_XNNPACKBackend` is **Itanium C++ ABI** static-init mangling with no MSVC equivalent (MSVC
uses `.CRT$XCU` pointers), and the `[Tt]` symbol-class regex is **ELF/`nm`** output format. Porting it is
out of scope per design §9; Windows gets equivalent coverage from Catch2 executing the XNNPACK-delegated
`add.pte` (Task 8).

Wrap the block at `native/CMakeLists.txt:87-96`. It currently reads:

```cmake
  if(NOT CMAKE_NM)
    find_program(CMAKE_NM NAMES nm)
  endif()
  add_custom_command(TARGET executorch_djl POST_BUILD
    COMMAND ${CMAKE_COMMAND}
      -DSO=$<TARGET_FILE:executorch_djl>
      -DNM=${CMAKE_NM}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/assert_xnnpack_registered.cmake
    VERBATIM
    COMMENT "Asserting XNNPACK backend survived the link")
```

Replace with:

```cmake
  # Linux-only: the guard greps Itanium-ABI static-init mangling out of `nm` (ELF) output. MSVC has
  # neither (.CRT$XCU pointers, no nm). Windows covers the same property at runtime instead — the
  # Catch2 suite executes the XNNPACK-delegated add.pte, which fails if the backend never registered.
  # See docs/superpowers/specs/2026-07-15-windows-builds-design.md §3.4 and §6.
  if(NOT WIN32)
    if(NOT CMAKE_NM)
      find_program(CMAKE_NM NAMES nm)
    endif()
    add_custom_command(TARGET executorch_djl POST_BUILD
      COMMAND ${CMAKE_COMMAND}
        -DSO=$<TARGET_FILE:executorch_djl>
        -DNM=${CMAKE_NM}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/assert_xnnpack_registered.cmake
      VERBATIM
      COMMENT "Asserting XNNPACK backend survived the link")
  endif()
```

- [ ] **Step 3: Verify the Linux build is unregressed**

These edits touch shared code paths, so prove Linux still builds and the guard still fires.

```bash
docker run --rm -v "$PWD":/workspace -w /workspace \
  quay.io/pypa/manylinux_2_28_x86_64 /bin/bash /workspace/native/build.sh
```

Expected: build completes with `Artifact: src/main/resources/native/linux-x86_64/libexecutorch_djl.so`,
and the log contains `XNNPACK post-link assertion OK: backend registration present, <N> xnn_* text symbols`
with N in the hundreds. If that line is absent, the `if(NOT WIN32)` gate is wrong — it is suppressing the
guard on Linux.

- [ ] **Step 4: Run the host test suite**

Run: `bash native/tests/cmake_resolution.sh && bash native/tests/build_config.sh`

Expected: `PASS: cmake resolution` and `PASS: build config`.

- [ ] **Step 5: Commit**

```bash
git add native/CMakeLists.txt
git commit -m "feat(native): make the JNI shim buildable under MSVC

Select include/win32 vs include/linux for jni_md.h, and gate the nm-based
XNNPACK post-link guard to non-Windows: it greps Itanium-ABI static-init
mangling from ELF nm output, neither of which MSVC has. Windows covers the
same property at runtime via Catch2 executing the XNNPACK-delegated add.pte.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Teach LibUtils the windows-x86_64 platform

**Files:**
- Modify: `src/main/java/org/measly/executorch/engine/LibUtils.java:12` (the `LIB` constant), `:47-55` (`platform()`)
- Test: `src/test/java/org/measly/executorch/engine/LibUtilsTest.java:36-47`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `static String platform()` now returns `"windows-x86_64"`; **new** `static String libName(String platform)` returning `"executorch_djl.dll"` for `windows-*` else `"libexecutorch_djl.so"`. Task 5 calls both.

> **Read this before you start.** `LibUtilsTest.platformRejectsUnsupportedOs` (line 36) currently asserts
> that **Windows 11 / amd64 throws** — and it passes today. This task **inverts that existing test**. When
> it goes red, that is the intended signal, not a regression to revert. Replace it; do not delete the
> unsupported-OS coverage (repoint it at a genuinely unsupported OS).

- [ ] **Step 1: Write the failing test**

In `src/test/java/org/measly/executorch/engine/LibUtilsTest.java`, **replace** the entire
`platformRejectsUnsupportedOs` method (lines 35-47) with:

```java
    @Test
    void platformResolvesWindowsX8664() {
        String os = System.getProperty("os.name");
        String arch = System.getProperty("os.arch");
        try {
            System.setProperty("os.name", "Windows 11");
            System.setProperty("os.arch", "amd64");
            assertEquals("windows-x86_64", LibUtils.platform());
        } finally {
            System.setProperty("os.name", os);
            System.setProperty("os.arch", arch);
        }
    }

    @Test
    void platformRejectsUnsupportedOs() {
        String os = System.getProperty("os.name");
        String arch = System.getProperty("os.arch");
        try {
            System.setProperty("os.name", "Mac OS X");
            System.setProperty("os.arch", "amd64");
            assertThrows(UnsupportedOperationException.class, LibUtils::platform);
        } finally {
            System.setProperty("os.name", os);
            System.setProperty("os.arch", arch);
        }
    }

    @Test
    void libNameIsPlatformSpecific() {
        assertEquals("libexecutorch_djl.so", LibUtils.libName("linux-x86_64"));
        assertEquals("executorch_djl.dll", LibUtils.libName("windows-x86_64"));
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./gradlew test --tests 'org.measly.executorch.engine.LibUtilsTest'`

Expected: FAIL. `platformResolvesWindowsX8664` throws `UnsupportedOperationException`, and
`libNameIsPlatformSpecific` fails to compile (`libName` does not exist).

- [ ] **Step 3: Implement platform() and libName()**

In `src/main/java/org/measly/executorch/engine/LibUtils.java`, **delete** the `LIB` constant at line 12:

```java
    private static final String LIB = "libexecutorch_djl.so";
```

and replace the `platform()` method (lines 47-55) with:

```java
    static String platform() {
        String os = System.getProperty("os.name").toLowerCase();
        String arch = System.getProperty("os.arch").toLowerCase();
        boolean x64 = arch.equals("amd64") || arch.equals("x86_64");
        if (os.contains("linux") && x64) {
            return "linux-x86_64";
        }
        if (os.contains("windows") && x64) {
            return "windows-x86_64";
        }
        throw new UnsupportedOperationException(
                "ExecuTorch engine supports only linux-x86_64 and windows-x86_64, got: " + os + "/" + arch);
    }

    /** MSVC emits no `lib` prefix and a .dll suffix. Keep in sync with nativeLibName in build.gradle.kts. */
    static String libName(String platform) {
        return platform.startsWith("windows-") ? "executorch_djl.dll" : "libexecutorch_djl.so";
    }
```

Then fix the now-broken reference in `loadLibrary()` — replace:

```java
        String platform = platform();
        String resource = "/native/" + platform + "/" + LIB;
```

with:

```java
        String platform = platform();
        String resource = "/native/" + platform + "/" + libName(platform);
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./gradlew test --tests 'org.measly.executorch.engine.LibUtilsTest'`

Expected: BUILD SUCCESSFUL, 5 tests passing (`platformResolvesLinuxX8664`, `platformRejectsUnsupportedArch`, `platformResolvesWindowsX8664`, `platformRejectsUnsupportedOs`, `libNameIsPlatformSpecific`).

- [ ] **Step 5: Commit**

```bash
git add src/main/java/org/measly/executorch/engine/LibUtils.java \
        src/test/java/org/measly/executorch/engine/LibUtilsTest.java
git commit -m "feat(engine): recognize windows-x86_64 in LibUtils

platform() now resolves windows-x86_64, and libName() maps the artifact name
per platform (MSVC emits executorch_djl.dll, no lib prefix). Inverts the old
platformRejectsUnsupportedOs assertion, which asserted Windows throws;
unsupported-OS coverage moves to Mac OS X.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Content-addressed native library cache

**Files:**
- Modify: `src/main/java/org/measly/executorch/engine/LibUtils.java` (`loadLibrary()` and new private helpers)
- Test: `src/test/java/org/measly/executorch/engine/LibUtilsTest.java`

**Interfaces:**
- Consumes: `platform()`, `libName(String)` (Task 4).
- Produces: **new** `static Path cacheRoot()` returning `%LOCALAPPDATA%\executorch-djl` on Windows and `$XDG_CACHE_HOME`-or-`~/.cache` + `/executorch-djl` elsewhere. Behavior change: `loadLibrary()` loads from `cacheRoot()/<sha256>/<libName>` instead of a per-JVM temp file.

The shim is **11.5 MB**. Today `loadLibrary()` extracts it to a fresh temp file per JVM and calls
`deleteOnExit`. On Windows the OS holds a loaded DLL open, so **that delete silently fails and every JVM
run permanently leaks 11.5 MB into `%TEMP%`** — unacceptable for an embedded library.

Cache key is the **SHA-256 of the resource bytes**, not a jar-manifest version: a manifest version is null
in dev and test builds, and a stale cache silently serving yesterday's `.so` after a local rebuild is a
nasty debugging session. A content hash is never stale and self-versions.

Two passes on a miss, one on a hit, is deliberate: the key must be known before the path exists, so a
hit must be able to answer without writing 11.5 MB.

- [ ] **Step 1: Write the failing test**

Append to `src/test/java/org/measly/executorch/engine/LibUtilsTest.java`:

```java
    // Asserts the OS branch is actually taken. Checking only the leaf name would pass on either branch
    // and prove nothing, so compare the two roots against each other: they must differ, and both must be
    // absolute (System.load requires an absolute path).
    @Test
    void cacheRootBranchesOnOs() {
        String os = System.getProperty("os.name");
        try {
            System.setProperty("os.name", "Windows 11");
            Path windows = LibUtils.cacheRoot();
            System.setProperty("os.name", "Linux");
            Path linux = LibUtils.cacheRoot();

            assertEquals("executorch-djl", windows.getFileName().toString());
            assertEquals("executorch-djl", linux.getFileName().toString());
            assertNotEquals(windows, linux, "cache root must differ per OS");
            assertTrue(windows.isAbsolute(), "windows cache root must be absolute: " + windows);
            assertTrue(linux.isAbsolute(), "linux cache root must be absolute: " + linux);
            assertTrue(linux.toString().contains(".cache") || System.getenv("XDG_CACHE_HOME") != null,
                    "linux cache root should honour XDG_CACHE_HOME or fall back to ~/.cache: " + linux);
        } finally {
            System.setProperty("os.name", os);
        }
    }
```

Add the imports at the top of the file:

```java
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./gradlew test --tests 'org.measly.executorch.engine.LibUtilsTest'`

Expected: compilation failure — `cacheRoot()` does not exist.

- [ ] **Step 3: Implement the cache**

Replace the **entire contents** of `src/main/java/org/measly/executorch/engine/LibUtils.java` with the
following. It re-states `platform()` and `libName()` from Task 4 unchanged — this listing is the whole
file, so paste it wholesale rather than merging by hand:

```java
package org.measly.executorch.engine;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/** Resolves and loads the executorch_djl native shim. */
public final class LibUtils {

    private static final int BUF = 64 * 1024;
    private static boolean loaded;

    private LibUtils() {}

    // Not unit-tested: drives System.load, the EXECUTORCH_LIBRARY_PATH env override, and classpath
    // extraction, all of which need the real native library and JVM state. platform(), libName() and
    // cacheRoot() are the unit-tested seams.
    public static synchronized void loadLibrary() {
        if (loaded) {
            return;
        }
        String override = System.getenv("EXECUTORCH_LIBRARY_PATH");
        if (override != null && !override.isEmpty()) {
            System.load(override);
            loaded = true;
            return;
        }
        String platform = platform();
        String lib = libName(platform);
        String resource = "/native/" + platform + "/" + lib;
        try {
            // Hash-only first pass. The cache key must be known before the path exists, and a hit must
            // not rewrite 11.5 MB. A miss pays a second read; that happens once per version per host.
            Path target = cacheRoot().resolve(sha256(resource)).resolve(lib);
            if (!Files.isRegularFile(target)) {
                extract(resource, target);
            }
            System.load(target.toAbsolutePath().toString());
            loaded = true;
        } catch (IOException e) {
            throw new IllegalStateException("Failed to extract native library " + resource, e);
        }
    }

    static String platform() {
        String os = System.getProperty("os.name").toLowerCase();
        String arch = System.getProperty("os.arch").toLowerCase();
        boolean x64 = arch.equals("amd64") || arch.equals("x86_64");
        if (os.contains("linux") && x64) {
            return "linux-x86_64";
        }
        if (os.contains("windows") && x64) {
            return "windows-x86_64";
        }
        throw new UnsupportedOperationException(
                "ExecuTorch engine supports only linux-x86_64 and windows-x86_64, got: " + os + "/" + arch);
    }

    /** MSVC emits no `lib` prefix and a .dll suffix. Keep in sync with nativeLibName in build.gradle.kts. */
    static String libName(String platform) {
        return platform.startsWith("windows-") ? "executorch_djl.dll" : "libexecutorch_djl.so";
    }

    /**
     * Cache location. Windows cannot delete a loaded DLL, so a per-JVM temp file would leak 11.5 MB per
     * run; a stable per-content directory is reused instead.
     */
    static Path cacheRoot() {
        String os = System.getProperty("os.name").toLowerCase();
        if (os.contains("windows")) {
            String localAppData = System.getenv("LOCALAPPDATA");
            if (localAppData != null && !localAppData.isEmpty()) {
                return Paths.get(localAppData, "executorch-djl");
            }
            return Paths.get(System.getProperty("user.home"), "AppData", "Local", "executorch-djl");
        }
        String xdg = System.getenv("XDG_CACHE_HOME");
        if (xdg != null && !xdg.isEmpty()) {
            return Paths.get(xdg, "executorch-djl");
        }
        return Paths.get(System.getProperty("user.home"), ".cache", "executorch-djl");
    }

    private static InputStream open(String resource) {
        InputStream is = LibUtils.class.getResourceAsStream(resource);
        if (is == null) {
            throw new IllegalStateException(
                    "Native library not found on classpath: " + resource
                            + " (set EXECUTORCH_LIBRARY_PATH or run native/build_desktop.sh)");
        }
        return is;
    }

    private static String sha256(String resource) throws IOException {
        MessageDigest md;
        try {
            md = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is required of every JRE but is unavailable", e);
        }
        byte[] buf = new byte[BUF];
        try (InputStream is = new DigestInputStream(open(resource), md)) {
            while (is.read(buf) != -1) {
                // DigestInputStream updates the digest as a side effect of reading.
            }
        }
        StringBuilder sb = new StringBuilder(64);
        for (byte b : md.digest()) {
            sb.append(Character.forDigit((b >> 4) & 0xF, 16)).append(Character.forDigit(b & 0xF, 16));
        }
        return sb.toString();
    }

    private static void extract(String resource, Path target) throws IOException {
        Path dir = target.getParent();
        Files.createDirectories(dir);
        Path tmp = Files.createTempFile(dir, target.getFileName().toString(), ".tmp");
        try {
            try (InputStream is = open(resource); OutputStream os = Files.newOutputStream(tmp)) {
                byte[] buf = new byte[BUF];
                int n;
                while ((n = is.read(buf)) != -1) {
                    os.write(buf, 0, n);
                }
            }
            try {
                Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE);
            } catch (FileAlreadyExistsException | AtomicMoveNotSupportedException e) {
                // A concurrent JVM published first. The path is content-addressed, so the winner's bytes
                // are ours byte-for-byte: adopt it rather than overwrite a file another process may have
                // already mapped (which Windows would refuse anyway).
                if (!Files.isRegularFile(target)) {
                    throw e;
                }
            }
        } finally {
            Files.deleteIfExists(tmp);
        }
    }
}
```

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `./gradlew test --tests 'org.measly.executorch.engine.LibUtilsTest'`

Expected: BUILD SUCCESSFUL, 6 tests passing (the five from Task 4 plus `cacheRootBranchesOnOs`).

- [ ] **Step 5: Verify the cache end-to-end on Linux**

Unit tests don't exercise `System.load`. Drive the real path: run the full suite twice and confirm a
cache directory is created on the first run and *reused* (not rewritten) on the second.

```bash
rm -rf ~/.cache/executorch-djl
./gradlew test
find ~/.cache/executorch-djl -name '*.so' -printf '%p %s bytes\n'
```

Expected: BUILD SUCCESSFUL, and exactly one `libexecutorch_djl.so` of ~11.5 MB under a 64-hex-character
directory. Note its inode, then re-run and confirm it is unchanged:

```bash
ls -i ~/.cache/executorch-djl/*/libexecutorch_djl.so
./gradlew test --rerun-tasks
ls -i ~/.cache/executorch-djl/*/libexecutorch_djl.so
```

Expected: identical inode both times, and no `.tmp` files left behind. A changed inode means the hit path
is rewriting the file.

- [ ] **Step 6: Commit**

```bash
git add src/main/java/org/measly/executorch/engine/LibUtils.java \
        src/test/java/org/measly/executorch/engine/LibUtilsTest.java
git commit -m "feat(engine): content-addressed native library cache

Windows cannot delete a loaded DLL, so the old per-JVM temp file plus
deleteOnExit leaked 11.5 MB into %TEMP% on every run. Extract to
<cache>/<sha256>/<lib> instead, reused across runs on both platforms.

Keyed on the content hash rather than a jar manifest version: the manifest is
null in dev/test builds, and a stale cache serving a pre-rebuild .so is a nasty
debugging session. Concurrent JVMs are safe via atomic rename onto a
content-addressed path.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Publish a windows-x86_64 classifier jar

**Files:**
- Modify: `build.gradle.kts:56-72`
- Test: `./gradlew nativeJar-windows-x86_64`

**Interfaces:**
- Consumes: the artifact filename contract from Task 4 (`executorch_djl.dll`).
- Produces: Gradle task **`nativeJar-windows-x86_64`** emitting a jar with classifier `windows-x86_64` containing `native/windows-x86_64/executorch_djl.dll`. Consumed by `publish.yml` with no change to that file.

`build.gradle.kts` is already correctly shaped — it builds one classifier jar per entry in
`nativePlatforms`. Only the platform list and the hardcoded `.so` filename need to move.

- [ ] **Step 1: Add the platform and a filename mapping**

Replace `build.gradle.kts:56`:

```kotlin
val nativePlatforms = listOf("linux-x86_64")
```

with:

```kotlin
val nativePlatforms = listOf("linux-x86_64", "windows-x86_64")

// MSVC emits no `lib` prefix and a .dll suffix. Keep in sync with LibUtils.libName.
fun nativeLibName(platform: String): String =
    if (platform.startsWith("windows-")) "executorch_djl.dll" else "libexecutorch_djl.so"
```

Then replace the hardcoded filename at line 67:

```kotlin
    val resolvedSo = nativeStaging.get().dir(platform).file("libexecutorch_djl.so").asFile
    doFirst { // Fail a release rather than ship an empty native jar
        require(resolvedSo.exists()) { "Missing native library for ${platform}: ${resolvedSo}" }
    }
```

with:

```kotlin
    val resolvedLib = nativeStaging.get().dir(platform).file(nativeLibName(platform)).asFile
    doFirst { // Fail a release rather than ship an empty native jar
        require(resolvedLib.exists()) { "Missing native library for ${platform}: ${resolvedLib}" }
    }
```

Also update the stale comment at line 57 (`Look for .so files in ...`):

```kotlin
// Look for the platform's native library in build/native-staging/<platform>/<nativeLibName>
```

- [ ] **Step 2: Verify the task exists and gates on a missing library**

The `require()` must fail loudly rather than ship an empty jar:

```bash
./gradlew nativeJar-windows-x86_64
```

Expected: FAIL with `Missing native library for windows-x86_64: .../build/native-staging/windows-x86_64/executorch_djl.dll`.
That is the correct behavior with no staged DLL present — it proves both the task registration and the guard.

- [ ] **Step 3: Verify the jar builds when the library is staged**

Use a stand-in to prove the packaging path, then clean it up:

```bash
mkdir -p build/native-staging/windows-x86_64
printf 'not-a-real-dll' > build/native-staging/windows-x86_64/executorch_djl.dll
./gradlew nativeJar-windows-x86_64
unzip -l build/libs/*-windows-x86_64.jar
```

Expected: BUILD SUCCESSFUL, and the listing contains `native/windows-x86_64/executorch_djl.dll`.

```bash
rm -rf build/native-staging/windows-x86_64
```

- [ ] **Step 4: Verify Linux packaging is unregressed**

```bash
./gradlew nativeJar-linux-x86_64 && unzip -l build/libs/*-linux-x86_64.jar
```

Expected: BUILD SUCCESSFUL, listing contains `native/linux-x86_64/libexecutorch_djl.so`.

- [ ] **Step 5: Commit**

```bash
git add build.gradle.kts
git commit -m "feat(build): publish a windows-x86_64 classifier jar

nativePlatforms gains windows-x86_64 and the hardcoded .so filename becomes a
per-platform mapping. publish.yml needs no change: its executorch-libs-*
pattern already matches both platforms.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Fork native/build.sh for Windows

**Files:**
- Modify: `native/build.sh`
- Test: `native/tests/build_config.sh` (existing `PRINT_BUILD_CONFIG` seam), then a real winbox build

**Interfaces:**
- Consumes: `ET_PLATFORM` default detection (Task 1), the MSVC-buildable target (Task 3).
- Produces: on Windows, `src/main/resources/native/windows-x86_64/executorch_djl.dll`. Task 9's CI job invokes this script; Task 6's Gradle task consumes its output path.

The caller supplies an already-activated MSVC environment (Task 9 does this via `Launch-VsDevShell`); the
script does **not** activate VS itself. Under Git-Bash `uname -s` reports `MINGW64_NT-10.0-*` or `MSYS_NT-*`.

- [ ] **Step 1: Add the host-OS fork**

In `native/build.sh`, immediately after `set -ex` (line 2), insert:

```bash
# Host fork. Under Git-Bash on Windows `uname -s` is MINGW64_NT-* or MSYS_NT-*. The caller must have
# already activated the MSVC dev shell (see .github/workflows/native-build-job.yml); this script does
# not activate VS itself. Everything Linux-only below (Corretto RPM, chown, dnf, nproc) is skipped.
case "$(uname -s)" in
  MINGW*|MSYS*) ET_HOST_OS=windows ;;
  *)            ET_HOST_OS=linux ;;
esac
```

- [ ] **Step 2: Skip the container chown trap on Windows**

The `chown` cleanup exists because container bind-mount outputs are root-owned on the host. There is no
container on Windows. Replace the `trap cleanup EXIT` line (line 12) with:

```bash
[ "${ET_HOST_OS}" = "linux" ] && trap cleanup EXIT
```

- [ ] **Step 3: Fork JDK discovery**

Replace the Corretto RPM extraction block (lines 34-44, from `echo "--- Extracting Corretto JDK headers`
through `echo "JAVA_HOME=${JAVA_HOME}"`) with:

```bash
if [ "${ET_HOST_OS}" = "windows" ]; then
  echo "--- Using the runner's JDK headers (headers-only; we never link libjvm) ---"
  test -n "${JAVA_HOME:-}" || { echo "JAVA_HOME must be set on Windows (see setup-java)"; exit 1; }
  # Git-Bash gives JAVA_HOME as a Windows path; cmake accepts it, but the test below needs a POSIX path.
  JAVA_HOME="$(cygpath -u "${JAVA_HOME}" 2>/dev/null || echo "${JAVA_HOME}")"
  export JAVA_HOME
  test -f "${JAVA_HOME}/include/win32/jni_md.h" \
    || { echo "JDK headers not found under JAVA_HOME=${JAVA_HOME} (expected include/win32/jni_md.h)"; exit 1; }
  echo "JAVA_HOME=${JAVA_HOME}"
else
  echo "--- Extracting Corretto JDK headers (headers-only; we never link libjvm) ---"
  JDK_EXTRACT=/opt/corretto
  mkdir -p "${JDK_EXTRACT}"
  cp /workspace/amazon-corretto-linux-jdk.rpm /tmp/corretto.rpm
  rpm2archive /tmp/corretto.rpm            # -> /tmp/corretto.rpm.tgz (no cpio in this image)
  tar -C "${JDK_EXTRACT}" -xzf /tmp/corretto.rpm.tgz
  JNI_H="$(find "${JDK_EXTRACT}" -path '*/include/jni.h' | head -1)"
  export JAVA_HOME="${JNI_H%/include/jni.h}"
  test -f "${JAVA_HOME}/include/linux/jni_md.h" \
    || { echo "JDK headers not found under JAVA_HOME=${JAVA_HOME}"; exit 1; }
  echo "JAVA_HOME=${JAVA_HOME}"
fi
```

- [ ] **Step 4: Fork toolchain setup and job count**

Ninja ships with Visual Studio and is on PATH once the dev shell is activated; there is no `pip`-installed
Python toolchain to lean on and no `nproc`. Replace the Ninja/toolchain block (lines 46-51):

```bash
if [ "${ET_HOST_OS}" = "windows" ]; then
  echo "--- Toolchain Versions (MSVC dev shell must already be activated by the caller) ---"
  command -v cl >/dev/null 2>&1 || { echo "cl.exe not on PATH: activate the VS dev shell first"; exit 1; }
  command -v ninja >/dev/null 2>&1 || { echo "ninja not on PATH: activate the VS dev shell first"; exit 1; }
  cl 2>&1 | head -1; cmake --version; ninja --version
else
  echo "--- Setting up Ninja (the shim configures with -G Ninja) ---"
  export PATH="/opt/python/cp312-cp312/bin:${PATH}"
  pip install ninja
  echo "--- Toolchain Versions ---"
  gcc --version; g++ --version; cmake --version; ninja --version
fi
```

Replace the `JOBS`/`cd` block (lines 59-60):

```bash
if [ "${ET_HOST_OS}" = "windows" ]; then
  JOBS="${JOBS:-${NUMBER_OF_PROCESSORS:-4}}"
else
  JOBS="${JOBS:-$(nproc)}"
  cd /workspace
fi
```

- [ ] **Step 4b: Match the MSVC CRT to the prebuilt runtime (`CMAKE_BUILD_TYPE`)**

**Discovered during execution — the original plan omitted this and the Windows link failed with 460
`LNK2038` errors.** `build.sh`'s cmake configure never passes `CMAKE_BUILD_TYPE`. On GCC/ELF that is
harmless (no CRT-flavour ABI tag). On MSVC it is fatal: our objects compile against the **Debug** CRT
(`MDd_DynamicDebug`, `_ITERATOR_DEBUG_LEVEL=2`) while the pinned runtime's `.lib`s are **Release**
(`MD_DynamicRelease`) — `BUILDINFO` records `-DCMAKE_BUILD_TYPE=Release`. The link dies with:

```
error LNK2038: mismatch detected for 'RuntimeLibrary': value 'MD_DynamicRelease' doesn't match value 'MDd_DynamicDebug'
fatal error LNK1319: 460 mismatches detected
```

Fork the configure so Windows builds Release (matching the runtime's `/MD`), leaving the Linux
invocation byte-identical so the Linux artifact does not change. Replace:

```bash
cmake -B "${NATIVE_BUILD_DIR}" -S native -G Ninja \
  -DET_RUNTIME_VARIANT="${ET_RUNTIME_VARIANT}" "${ET_INSTALL_ARG[@]}"
```

with:

```bash
# MSVC encodes the CRT flavour into every object and the linker refuses to mix them. The pinned runtime
# tarball is built Release (/MD — see its BUILDINFO cmake_flags), so the shim must be Release too or the
# link dies with LNK2038 'RuntimeLibrary' / '_ITERATOR_DEBUG_LEVEL' mismatches. GCC/ELF has no such ABI
# tag, so the Linux leg stays as-is (unset) and its artifact is unchanged.
BUILD_TYPE_ARG=()
[ "${ET_HOST_OS}" = "windows" ] && BUILD_TYPE_ARG=(-DCMAKE_BUILD_TYPE=Release)
cmake -B "${NATIVE_BUILD_DIR}" -S native -G Ninja \
  -DET_RUNTIME_VARIANT="${ET_RUNTIME_VARIANT}" "${BUILD_TYPE_ARG[@]}" "${ET_INSTALL_ARG[@]}"
```

- [ ] **Step 5: Fork the staging step**

Replace the `STAGE_SO` block (lines 72-81):

```bash
if [ "${ET_HOST_OS}" = "windows" ]; then
  OUT_PLATFORM="windows-x86_64"; OUT_LIB="executorch_djl.dll"
else
  OUT_PLATFORM="linux-x86_64";   OUT_LIB="libexecutorch_djl.so"
fi

if [ "${STAGE_SO}" = "1" ]; then
  OUT="src/main/resources/native/${OUT_PLATFORM}"
  mkdir -p "${OUT}"
  cp "${NATIVE_BUILD_DIR}/${OUT_LIB}" "${OUT}/"
  echo "Artifact: ${OUT}/${OUT_LIB}"
  ls -lh "${OUT}/${OUT_LIB}"
else
  echo "STAGE_SO=0: built shim but not staging into resources"
  ls -lh "${NATIVE_BUILD_DIR}/${OUT_LIB}"
fi
```

- [ ] **Step 6: Verify the config seam and the Linux build still work**

```bash
bash native/tests/build_config.sh
PRINT_BUILD_CONFIG=1 bash native/build.sh
```

Expected: `PASS: build config`, then a config line naming `ET_RUNTIME_VARIANT=logging STAGE_SO=1 ...`.
The `PRINT_BUILD_CONFIG` early-exit precedes all of the forked logic, so it must be unaffected.

```bash
docker run --rm -v "$PWD":/workspace -w /workspace \
  quay.io/pypa/manylinux_2_28_x86_64 /bin/bash /workspace/native/build.sh
```

Expected: `Artifact: src/main/resources/native/linux-x86_64/libexecutorch_djl.so`.

- [ ] **Step 7: Verify the real build on winbox**

This is the first step that proves MSVC compilation. **winbox is for iteration, not acceptance** (VS 18
Community vs the runner's VS 17 Enterprise) — Task 9's runner is the gate.

Sync the tree and build, using the activation → Git-Bash handoff (non-login shell, so the dev-shell PATH
survives):

```bash
ssh winbox "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force -Path C:\\Users\\cored\\djl-et\""
rsync -az --delete --exclude build --exclude .git ./ 'winbox:C:/Users/cored/djl-et/' 2>/dev/null \
  || scp -qr native src build.gradle.kts 'winbox:C:/Users/cored/djl-et/'
```

Then run the build via a base64-encoded PowerShell payload (the remote default shell is `cmd`, so this
avoids quoting hell):

```bash
cat > /tmp/winbuild.ps1 <<'PS'
$ErrorActionPreference = "Stop"
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath  = & $vswhere -latest -products * -property installationPath
if (-not $vsPath) { throw "vswhere found no Visual Studio installation" }
& "$vsPath\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -SkipAutomaticLocation
$bash = "${env:ProgramFiles}\Git\bin\bash.exe"
& $bash -c 'cd /c/Users/cored/djl-et && export JAVA_HOME="$(cygpath -u "$JAVA_HOME")" && ./native/build.sh'
if ($LASTEXITCODE -ne 0) { throw "build failed ($LASTEXITCODE)" }
PS
enc="$(iconv -f UTF-8 -t UTF-16LE < /tmp/winbuild.ps1 | base64 -w0)"
ssh winbox "powershell -NoProfile -EncodedCommand $enc"
```

Expected: `find_package` completes (a long list of `<X> library is not found` lines for unbuilt optional
components is **normal** — the runtime is a subset build; only the four linked targets matter), the link
succeeds, and the run ends with
`Artifact: src/main/resources/native/windows-x86_64/executorch_djl.dll`.

Ignore the trailing `#< CLIXML ... </Objs>` blob in the SSH reply — that is PowerShell serializing
`Write-Host` records over the channel, not an error.

- [ ] **Step 8: Commit**

```bash
git add native/build.sh
git commit -m "feat(native): build the shim on Windows via Git-Bash

Fork build.sh on uname -s: Windows takes JAVA_HOME from the runner instead of
extracting the Corretto RPM, skips the container chown trap and the pip Ninja
install (VS ships ninja), counts jobs from NUMBER_OF_PROCESSORS, and stages
executorch_djl.dll into windows-x86_64. The caller supplies the activated MSVC
environment.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Fork native/build_qa.sh for Windows

**Files:**
- Modify: `native/build_qa.sh`
- Test: the script itself, on Linux then winbox

**Interfaces:**
- Consumes: the host-fork idiom from Task 7.
- Produces: on Windows, a Catch2 run of `et_runtime_test` and **no** leak harness. This is Windows' XNNPACK coverage, standing in for the `nm` guard disabled in Task 3.

**`et_leak_harness` must never build or run on Windows.** MSVC provides ASan but has **no LeakSanitizer**,
so the harness is structurally Linux-only. Windows QA is the Catch2 suite with no sanitizers.

The suite is the XNNPACK gate: `native/spike/export_add.py:17` lowers `add.pte` through
`XnnpackPartitioner`, and `native/test/et_runtime_test.cpp:35` asserts `add(2,3) == 5` by executing it.
If the backend failed to register, that test fails.

- [ ] **Step 1: Add the host fork**

In `native/build_qa.sh`, after the `cd "${REPO_ROOT}"` line (line 22), insert:

```bash
# Host fork; see native/build.sh. Windows QA is Catch2-only: MSVC has ASan but no LeakSanitizer, so
# et_leak_harness is structurally Linux-only (design §6). The Catch2 suite is also Windows' XNNPACK
# gate, standing in for the nm post-link guard that cannot run under MSVC (design §3.4).
case "$(uname -s)" in
  MINGW*|MSYS*) ET_HOST_OS=windows ;;
  *)            ET_HOST_OS=linux ;;
esac
```

- [ ] **Step 2: Fork the configure, build and run**

Replace everything from the ASan-install block (line 25, `# QA is the only ASan consumer;`) through the
end of the file with:

```bash
ET_ARGS=(-DET_RUNTIME_VARIANT="${ET_RUNTIME_VARIANT:-logging}")
[ -n "${ET_INSTALL:-}" ] && ET_ARGS+=(-DET_INSTALL="${ET_INSTALL}")

if [ "${ET_HOST_OS}" = "windows" ]; then
  command -v cl >/dev/null 2>&1 || { echo "cl.exe not on PATH: activate the VS dev shell first"; exit 1; }
  JOBS="${JOBS:-${NUMBER_OF_PROCESSORS:-4}}"
  bash native/clean_stale_tree.sh native/asan native
  # No sanitizers: MSVC has no LeakSanitizer, and the leak harness is not built here at all.
  # RelWithDebInfo, NOT Debug: Debug would compile against the Debug CRT (/MDd) while the pinned runtime
  # is Release (/MD), and MSVC refuses to mix them — the same LNK2038/LNK1319 wall the shim build hits
  # (see build.sh). RelWithDebInfo gives us the matching /MD CRT while keeping symbols, so a Catch2
  # failure is still debuggable.
  cmake -B native/asan -S native -G Ninja "${ET_ARGS[@]}" -DET_BUILD_QA=ON \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
  cmake --build native/asan --target et_runtime_test -j"${JOBS}"

  echo "--- Catch2 unit suite (no sanitizers; MSVC has no LSan) ---"
  ./native/asan/et_runtime_test.exe
  echo "--- Leak harness SKIPPED: no LeakSanitizer under MSVC (Linux-only coverage) ---"
else
  # QA is the only ASan consumer; install the toolset's ASan runtime here (moved out of build.sh).
  if command -v dnf >/dev/null 2>&1; then
    echo "--- Installing ASan runtime (dnf), may appear to hang ---"
    TOOLSET_VER="$(gcc -dumpversion | cut -d. -f1)"
    dnf install -y -q "gcc-toolset-${TOOLSET_VER}-libasan-devel" || true
  fi

  JOBS="${JOBS:-$(nproc)}"
  # Drop the tree only if it was configured for a different source root (container vs host); a same-root
  # re-run keeps it so the cached Catch2 build (native/asan/_deps) is reused, not rebuilt. CLEAN=1 forces.
  bash native/clean_stale_tree.sh native/asan native
  cmake -B native/asan -S native -G "Unix Makefiles" "${ET_ARGS[@]}" -DET_BUILD_QA=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"
  cmake --build native/asan --target et_runtime_test et_leak_harness -j"${JOBS}"

  echo "--- Catch2 unit suite ---"
  ./native/asan/et_runtime_test

  echo "--- ASan/LSan leak harness (${ITERS} iterations) ---"
  ./native/asan/et_leak_harness native/spike/add.pte "${ITERS}"
fi
```

- [ ] **Step 3: Verify Linux QA is unregressed**

```bash
docker run --rm -v "$PWD":/workspace -w /workspace \
  quay.io/pypa/manylinux_2_28_x86_64 /bin/bash /workspace/native/build_qa.sh
```

Expected: `All tests passed` (6 assertions across 6 test cases), then the leak harness completing 1000
iterations with no LSan report.

- [ ] **Step 4: Verify Windows QA on winbox**

```bash
cat > /tmp/winqa.ps1 <<'PS'
$ErrorActionPreference = "Stop"
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath  = & $vswhere -latest -products * -property installationPath
if (-not $vsPath) { throw "vswhere found no Visual Studio installation" }
& "$vsPath\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -SkipAutomaticLocation
$bash = "${env:ProgramFiles}\Git\bin\bash.exe"
& $bash -c 'cd /c/Users/cored/djl-et && ./native/build_qa.sh'
if ($LASTEXITCODE -ne 0) { throw "QA failed ($LASTEXITCODE)" }
PS
enc="$(iconv -f UTF-8 -t UTF-16LE < /tmp/winqa.ps1 | base64 -w0)"
ssh winbox "powershell -NoProfile -EncodedCommand $enc"
```

Expected: `All tests passed`, including `add(2,3) == 5 with correct view metadata`. **That test passing is
the proof the XNNPACK backend registered under MSVC** — it is the whole Windows XNNPACK story. Then
`--- Leak harness SKIPPED ---` and exit 0.

If `add(2,3) == 5` fails while the others pass, the backend did not register: the MSVC link dropped the
`xnnpack_backend` static-init TU (`/OPT:REF` with no `/WHOLEARCHIVE:`). That is a real defect in the
runtime tarball's `ExecuTorchTargets.cmake`, not in this plan — escalate rather than paper over it.

- [ ] **Step 5: Commit**

```bash
git add native/build_qa.sh
git commit -m "feat(native): Catch2-only QA on Windows

MSVC has ASan but no LeakSanitizer, so et_leak_harness is structurally
Linux-only and is never built on Windows. The Catch2 suite runs unsanitized and
doubles as Windows' XNNPACK gate: add.pte is XNNPACK-delegated, so
'add(2,3) == 5' fails if the backend never registered — covering what the
nm post-link guard covers on Linux.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: Add the Windows CI job

**Files:**
- Modify: `.github/workflows/native-build-job.yml` (new sibling job), `native/tests/ci_workflow.sh`
- Test: `native/tests/ci_workflow.sh`, then the real runner

**Interfaces:**
- Consumes: `native/build.sh` (Task 7), `native/build_qa.sh` (Task 8).
- Produces: CI artifact **`executorch-libs-windows-x86_64`** containing `windows-x86_64/executorch_djl.dll`. Matched by the **existing** `executorch-libs-*` pattern in `publish.yml` and `native-build.yml` — **neither file changes**.

A **sibling job in the same file**, not a second workflow file. `native-build-job.yml` is a *reusable*
workflow called by both `native-build.yml` and `publish.yml` behind one `artifact-pattern` output; a
second file would force both callers to invoke two workflows and merge two outputs. This mirrors
`measly-java-learning/executorch-runtime-dist`'s `release.yml`, which pairs a containerized `build` job
with a native `build-windows` job.

> **`ci_workflow.sh` is already failing on `main`** — its `WF` points at `native-build.yml`, but the steps
> it asserts (`gh attestation verify`, `native/build.sh`) were moved into `native-build-job.yml` and the
> test was never repointed. Fix that first; you cannot add assertions to a suite that is already red
> without hiding your own regressions.

- [ ] **Step 1: Repair the pre-existing ci_workflow.sh breakage**

In `native/tests/ci_workflow.sh`, replace lines 7-12:

```bash
fail() { echo "FAIL: $1"; exit 1; }
WF=".github/workflows/native-build.yml"

grep -q 'pytorch/executorch' "${WF}" && fail "ExecuTorch checkout must be removed"
grep -q 'gh attestation verify' "${WF}" || fail "attestation verify step missing"
grep -q 'measly-java-learning/executorch-runtime-dist' "${WF}" || fail "attestation repo missing"
grep -q 'native/build.sh' "${WF}" || fail "shim build step missing"
```

with:

```bash
fail() { echo "FAIL: $1"; exit 1; }
# The caller workflow. The build steps themselves live in the reusable job below — asserting them
# against this file is what made this suite red on main.
WF=".github/workflows/native-build.yml"
# The reusable workflow that actually builds the shim (both platforms).
WFJOB=".github/workflows/native-build-job.yml"

grep -q 'pytorch/executorch' "${WFJOB}" && fail "ExecuTorch checkout must be removed"
grep -q 'gh attestation verify' "${WFJOB}" || fail "attestation verify step missing"
grep -q 'measly-java-learning/executorch-runtime-dist' "${WFJOB}" || fail "attestation repo missing"
grep -q 'native/build.sh' "${WFJOB}" || fail "shim build step missing"
grep -q 'native-build-job.yml' "${WF}" || fail "caller must delegate to the reusable job"
```

- [ ] **Step 2: Run the test to verify the repair**

Run: `bash native/tests/ci_workflow.sh`

Expected: `PASS: ci workflow`. (It was `FAIL: attestation verify step missing` before.)

- [ ] **Step 3: Write the failing test for the Windows job**

Append to `native/tests/ci_workflow.sh`, immediately before the YAML-validity block:

```bash
# Windows shim job: a sibling job in the reusable workflow (not a separate file), so publish.yml's
# single `uses:` + executorch-libs-* pattern keeps working unchanged.
grep -q 'build-executorch-shim-windows' "${WFJOB}" || fail "windows shim job missing"
grep -q 'runs-on: windows-2022'         "${WFJOB}" || fail "windows job must run on windows-2022"
grep -q 'vswhere'                       "${WFJOB}" || fail "windows job must discover VS via vswhere"
grep -q 'products \*'                   "${WFJOB}" || fail "vswhere must be edition-agnostic (-products *)"
grep -q 'executorch-libs-windows-x86_64' "${WFJOB}" || fail "windows artifact name missing"
grep -q 'build_qa.sh'                   "${WFJOB}" || fail "windows QA step missing"
# The aarch64 rows exist in the pin but are out of scope: the matrix entry must stay commented out.
grep -qE '^\s*- platform: linux-aarch64' "${WFJOB}" && fail "linux-aarch64 is out of scope for this PR"
```

Also extend the YAML-validity check to cover both files. Replace:

```bash
  python3 -c "import yaml,sys; yaml.safe_load(open('${WF}')); print('yaml ok')" || fail "workflow is not valid YAML"
```

with:

```bash
  python3 -c "import yaml,sys; yaml.safe_load(open('${WF}')); yaml.safe_load(open('${WFJOB}')); print('yaml ok')" \
    || fail "workflow is not valid YAML"
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `bash native/tests/ci_workflow.sh`

Expected: `FAIL: windows shim job missing`.

- [ ] **Step 5: Add the Windows job**

Append to `.github/workflows/native-build-job.yml`, as a sibling of `build-executorch-shim` (same
indentation level, i.e. two spaces under `jobs:`):

```yaml
  # Windows amd64: a sibling job rather than a matrix row, because the linux rows are container-based
  # (manylinux bakes the glibc floor into the image) and Windows has no container. Both jobs upload
  # under the executorch-libs-* pattern, so publish.yml and native-build.yml consume them unchanged.
  # A future macOS platform would follow this same shape. Mirrors executorch-runtime-dist's release.yml.
  build-executorch-shim-windows:
    runs-on: windows-2022
    permissions:
      contents: read
    steps:
      - name: Checkout djl-executorch-engine repository
        uses: actions/checkout@v7

      # Provenance gate: prove the pinned runtime tarball was built by Repo A's CI before we link
      # against it. URL_HASH in native/CMakeLists.txt covers integrity; this covers provenance.
      - name: Verify ExecuTorch runtime provenance
        shell: bash
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          url="$(grep -oE 'https://[^"]*logging-windows-x86_64\.tar\.gz' native/cmake/EtRuntimePin.cmake | head -1)"
          test -n "${url}" || { echo "no logging URL in EtRuntimePin.cmake"; exit 1; }
          curl -fL -o runtime.tgz "${url}"
          gh attestation verify runtime.tgz --repo measly-java-learning/executorch-runtime-dist
          rm -f runtime.tgz

      - name: Build the executorch_djl shim (MSVC)
        shell: pwsh
        run: |
          # Discover VS edition-agnostically (-products *) rather than hardcoding Enterprise, so a
          # runner-image edition change does not silently break the build.
          $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
          $vsPath = & $vswhere -latest -products * -property installationPath
          if (-not $vsPath) { throw "vswhere found no Visual Studio installation" }
          # Launch-VsDevShell mutates THIS process's env in place; the bash child inherits it.
          & "$vsPath\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -SkipAutomaticLocation
          # The windows-2022 image ships JDK 8 as JAVA_HOME_8_X64 (8.0.492+9). Bind it EXPLICITLY rather
          # than trusting the image's default JAVA_HOME: the default is an image property GitHub can change,
          # and we want the oldest supported jni.h for the widest runtime compatibility — matching the linux
          # arm's Corretto 8 RPM. We compile against jni.h only and never link libjvm.
          if (-not $env:JAVA_HOME_8_X64) { throw "JAVA_HOME_8_X64 not set on this runner image" }
          $env:JAVA_HOME = $env:JAVA_HOME_8_X64
          # Invoke Git-Bash by explicit path so PATH order cannot select WSL's System32\bash.exe, which
          # would run the build in a Linux environment with no MSVC toolchain. Non-login (`-c`, not
          # `-lc`): a login shell re-sources the profile and resets PATH, dropping the VS env.
          $bash = "${env:ProgramFiles}\Git\bin\bash.exe"
          & $bash -c './native/build.sh'
          if ($LASTEXITCODE -ne 0) { throw "shim build failed (exit $LASTEXITCODE)" }

      - name: QA the executorch_djl shim (MSVC)
        shell: pwsh
        run: |
          $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
          $vsPath = & $vswhere -latest -products * -property installationPath
          if (-not $vsPath) { throw "vswhere found no Visual Studio installation" }
          & "$vsPath\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -SkipAutomaticLocation
          $bash = "${env:ProgramFiles}\Git\bin\bash.exe"
          & $bash -c './native/build_qa.sh'
          if ($LASTEXITCODE -ne 0) { throw "shim QA failed (exit $LASTEXITCODE)" }

      - name: Store executorch_djl shim
        uses: actions/upload-artifact@v7
        with:
          name: executorch-libs-windows-x86_64
          path: ${{ github.workspace }}/src/main/resources/native/**/*.dll
          compression-level: 1
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `bash native/tests/ci_workflow.sh`

Expected: `PASS: ci workflow`.

- [ ] **Step 7: Run the full host test suite**

```bash
for t in native/tests/*.sh; do printf '%-32s ' "$(basename $t)"; \
  bash "$t" >/tmp/o 2>&1 && echo PASS || echo "FAIL -> $(tail -1 /tmp/o)"; done
```

Expected: everything PASS **except** `docs_present.sh`, which fails on `main` for unrelated `et-install/`
drift and is explicitly out of scope. If any other test fails, it is your regression.

- [ ] **Step 8: Commit**

```bash
git add .github/workflows/native-build-job.yml native/tests/ci_workflow.sh
git commit -m "ci: build and QA the windows-x86_64 shim on windows-2022

A sibling job rather than a matrix row: the linux rows are container-based and
Windows has no container. Both upload under executorch-libs-*, so publish.yml
and native-build.yml consume them unchanged.

Also repairs ci_workflow.sh, red on main since the build steps moved into the
reusable workflow but its assertions still pointed at the caller.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: Document the platform and open the PR

**Files:**
- Modify: `README.md`
- Test: `native/tests/docs_present.sh` (pre-existing failure is out of scope), full CI

**Interfaces:**
- Consumes: everything above.
- Produces: the merged feature.

- [ ] **Step 1: Document Windows support in the README**

`README.md:12-13` currently carries the status block:

```markdown
> **Status:** desktop **linux-x86_64** only, and a work in progress. These steps build what exists
> today — the ExecuTorch runtime, our JNI shim, and the JVM + native test suites.
```

Replace those two lines with:

```markdown
> **Status:** desktop **linux-x86_64** and **windows-x86_64**, and a work in progress. These steps
> build what exists today — the ExecuTorch runtime, our JNI shim, and the JVM + native test suites.
> The Docker prerequisite below applies to the Linux build only; Windows builds with MSVC 2022 and
> no container.
```

Then add this section immediately after the status block:

```markdown
### Supported platforms

| Platform | Artifact | Runtime variant | QA |
|---|---|---|---|
| `linux-x86_64` | `libexecutorch_djl.so` | `logging` | Catch2 + ASan/LSan leak harness |
| `windows-x86_64` | `executorch_djl.dll` | `logging` | Catch2 (MSVC has no LeakSanitizer) |

The native library ships in a per-platform classifier jar (`<artifact>-<platform>.jar`) and is extracted
on first load to a content-addressed cache — `%LOCALAPPDATA%\executorch-djl\<sha256>\` on Windows,
`$XDG_CACHE_HOME` (or `~/.cache`) `/executorch-djl/<sha256>/` elsewhere. Set `EXECUTORCH_LIBRARY_PATH` to
load a specific library instead and bypass extraction entirely.

Windows is built with MSVC 2022 against the `logging` runtime variant; `bare` and `devtools` are
Linux-only benchmarking builds (see `native/build_variants.sh`).
```

- [ ] **Step 2: Verify the full local suite**

```bash
for t in native/tests/*.sh; do printf '%-32s ' "$(basename $t)"; \
  bash "$t" >/tmp/o 2>&1 && echo PASS || echo "FAIL -> $(tail -1 /tmp/o)"; done
./gradlew build test
```

Expected: all native tests PASS except the out-of-scope `docs_present.sh`; `BUILD SUCCESSFUL` from Gradle.

- [ ] **Step 3: Commit and open the PR**

```bash
git add README.md
git commit -m "docs: document windows-x86_64 support

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push -u origin HEAD
gh pr create --title "feat: windows-x86_64 builds" --body "$(cat <<'EOF'
Makes windows-x86_64 a first-class published platform. Requires the v1.3.1-6 pin
bump (previous PR) to be merged first.

**Premise, verified against the artifact:** the v1.3.1-6 Windows tarball is
feature-equivalent to Linux — it ships xnnpack_backend, extension_module_static,
extension_tensor and portable_ops_lib, all exported by lib/cmake/ExecuTorch. The
external handoff note claiming a core-only, XNNPACK-less Windows build is stale
and describes an earlier planned scope.

**What changed**
- `ET_PLATFORM` is now a cache variable, so Windows pin resolution is asserted
  from a Linux host via the `ET_PRINT_RESOLUTION` seam — no Windows machine needed.
- `-DET_RUNTIME_VARIANT=bare` on Windows now fails with the real reason rather
  than a generic "no pin row".
- The JNI shim compiles under MSVC (`include/win32`); the `nm` XNNPACK guard is
  gated to non-Windows (it greps Itanium-ABI mangling out of ELF `nm` output).
- `LibUtils` gained `windows-x86_64` and a content-addressed cache. The old
  per-JVM temp file leaked 11.5 MB into `%TEMP%` on every Windows run, since the
  OS holds a loaded DLL open and `deleteOnExit` silently fails.
- New `build-executorch-shim-windows` job on `windows-2022`. `publish.yml` is
  untouched: its `executorch-libs-*` pattern already matches both platforms.
- Repairs `ci_workflow.sh`, red on `main` since the build steps moved into the
  reusable workflow while its assertions still pointed at the caller.

**Platform differences, by design**
- Windows ships the `logging` variant only; `bare`/`devtools` don't exist upstream.
- No leak coverage on Windows: MSVC has ASan but no LeakSanitizer. The XNNPACK
  property is covered instead by Catch2 executing the XNNPACK-delegated `add.pte`.

**Out of scope:** `linux-aarch64` (pin rows now exist and it's cheap to add on top
of this work, but it's a separate runner and QA story); `docs_present.sh`, which
fails on `main` for unrelated `et-install/` drift.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Confirm the acceptance gate**

winbox proved nothing about the real toolchain — it is VS 18 Community, the runner is VS 2022/17
Enterprise. **The `windows-2022` runner is the gate.**

Watch the run:

```bash
gh pr checks --watch
```

Expected: `build-executorch-shim` (linux) and `build-executorch-shim-windows` both green, and
`build-java-package` green with both platforms' libraries downloaded and merged into
`src/main/resources/native/`.
