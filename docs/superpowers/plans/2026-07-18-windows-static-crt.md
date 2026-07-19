# Windows Static CRT (`/MT`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Windows JNI DLL with the C runtime linked statically, so end users need no VC++ redistributable.

**Architecture:** Bump the runtime pin to `v1.3.1-8`, then split the overloaded `ET_PLATFORM` CMake variable into a platform *identity* and a pin *row*, letting Windows resolve the new `windows-x86_64-static` row while every existing comparison stays correct. Force the static CRT with a global cache variable rather than per-target properties, because Catch2 arrives via `FetchContent` and its targets cannot be reached with `set_property`. Verification is static (CRT directive scan + import table) because no redist-free Windows image is available.

**Tech Stack:** CMake 3.24+, MSVC 2022 / Ninja, bash test scripts, GitHub Actions, Gradle/JDK 17.

**Spec:** `docs/superpowers/specs/2026-07-18-windows-static-crt-design.md`

## Global Constraints

- Runtime pin target release: **`v1.3.1-8`**. There is no public `-7`.
- Windows pin row for the shim: **`logging_windows-x86_64-static`**. The `/MD` row `logging_windows-x86_64` stays in the pin file and stays selectable.
- `native/cmake/EtRuntimePin.cmake` is **generated — never hand-edit values**. Bump by replacing the whole file with the release asset, then re-applying the comment header block.
- The Java-side platform token `windows-x86_64` (resource path `/native/windows-x86_64/`, jar classifier) is a **different namespace** and must not change: `build.gradle.kts:56`, `src/main/java/org/measly/executorch/engine/LibUtils.java:60`, `native/build.sh` staging path, `native/tests/ci_workflow.sh`.
- Windows builds must keep passing a release-family `CMAKE_BUILD_TYPE` (`Release` for the shim, `RelWithDebInfo` for QA). This is unchanged by the CRT switch and is still required to avoid `LNK2038` `_ITERATOR_DEBUG_LEVEL` mismatches.
- No Linux behaviour changes. The `nm`-based `assert_xnnpack_registered.cmake` guard stays `if(NOT WIN32)`-gated.
- Do **not** file an issue against `pytorch/executorch` from this repo.
- CRT-checking scripts: always use dash-form flags (`-nologo -directives`, never `/nologo` — MSYS path-converts a leading `/`), and always assert on **presence of the expected marker**, never absence of the wrong one.

## Task ordering note

Task 2 is a **human-gated hardware task** on a Windows machine. Tasks 3–7 are designed against evidence already gathered and can be written before Task 2 completes, but **must not be merged** until Task 2's findings are recorded. If Task 2 finding **S1** (the Catch2 link) fails, stop and revise Task 4 before proceeding.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `native/cmake/EtRuntimePin.cmake` | Replace | Generated pin: URLs + SHA256 per variant/row |
| `native/CMakeLists.txt` | Modify (~17-25, 33-53, top) | Row resolution, platform identity, CRT selection |
| `native/tests/cmake_resolution.sh` | Modify | Host-fast assertions on row resolution |
| `native/build.sh` | Modify (97-100) | Comment correction only |
| `native/build_qa.sh` | Modify (38-42) | Comment correction only |
| `.github/workflows/native-build-job.yml` | Modify (99) | Windows provenance gate + CRT gate step |
| `native/tests/ci_workflow.sh` | Modify | Assertions on the workflow |
| `native/tests/check_windows_crt.sh` | Create | CRT directive + import table gate (runs on Windows, against both build trees) |
| `native/build_qa.sh` | Modify (Windows branch) | Comment correction + standing Catch2 CRT gate |
| `CLAUDE.md` | Modify | Stream B durable note |
| `docs/handover-windows-static-cxx17-findings.md` | Create | Report back to the producer repo |

---

### Task 1: Bump the runtime pin to `v1.3.1-8`

The pin bump is a supply-chain review gate in its own right. `v1.3.1-8` rebuilt **every** artifact — all six Linux hashes changed as well as Windows — so this task includes a Linux rebuild and full JVM test run, independent of any `/MT` work.

**Files:**
- Modify: `native/cmake/EtRuntimePin.cmake` (whole-file replace + header re-apply)
- Test: `native/tests/cmake_resolution.sh:30-32`

**Interfaces:**
- Consumes: nothing.
- Produces: cache variables `ET_RUNTIME_VERSION` = `"1.3.1-8"`, `ET_RUNTIME_ET_VERSION` = `"1.3.1"`; pin rows `ET_RUNTIME_URL_logging_windows-x86_64-static` and `ET_RUNTIME_SHA256_logging_windows-x86_64-static`. All URLs point at the `/download/v1.3.1-8/` path segment.

- [ ] **Step 1: Update the release-tag assertion to the new pin (this is the failing test)**

In `native/tests/cmake_resolution.sh`, replace lines 30-32:

```bash
# The pin must point at the v1.3.1-6 release: the ET version (1.3.1) is unchanged from v1.3.1-2, so
# the tarball stem alone cannot distinguish the two. Assert the release-tag path segment instead.
grep -q '/download/v1.3.1-6/'                                 <<<"${out}" || fail "pin is not at release v1.3.1-6"
```

with:

```bash
# The pin must point at the v1.3.1-8 release: the ET version (1.3.1) is unchanged across pkgrevs, so
# the tarball stem alone cannot distinguish them. Assert the release-tag path segment instead.
grep -q '/download/v1.3.1-8/'                                 <<<"${out}" || fail "pin is not at release v1.3.1-8"
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `bash native/tests/cmake_resolution.sh`
Expected: `FAIL: pin is not at release v1.3.1-8` (the committed pin is still `v1.3.1-6`).

- [ ] **Step 3: Download the generated pin asset**

```bash
gh release download v1.3.1-8 --repo measly-java-learning/executorch-runtime-dist \
  --pattern 'EtRuntimePin.cmake' --dir native/cmake/ --clobber
```

- [ ] **Step 4: Re-apply the comment header block**

The downloaded file starts with a single generated line. Replace that line:

```cmake
# Generated by executorch-runtime-dist release v1.3.1-8. Do not edit by hand.
```

with the full header, carried forward from the previous pin and extended with the two-Windows-row note:

```cmake
# Generated by executorch-runtime-dist release v1.3.1-8. Do not edit by hand.
# All three Linux variant rows are committed on purpose: `logging` is the SHIPPED/default runtime;
# `bare` and `devtools` exist ONLY for native/build_variants.sh benchmarking. Shipping only uses logging.
# windows ships `logging` ONLY — there is no bare/devtools Windows build upstream. It ships TWO rows:
# `windows-x86_64` is /MD (dynamic CRT, for CPython extensions) and `windows-x86_64-static` is /MT.
# This engine resolves the -static row so the shipped DLL needs no VC++ redistributable; the /MD row is
# kept because native/tests/cmake_resolution.sh asserts the row is a real choice, not a hardcode.
# The linux-aarch64 rows are currently INERT: no build resolves them yet (see native-build-job.yml).
# Bump procedure: replace this whole file with the EtRuntimePin.cmake asset from the next
# `v<etver>-<pkgrev>` Repo A release, then re-apply this comment block. The SHA256 change is the
# supply-chain review gate.
```

- [ ] **Step 5: Verify the static row landed and the hash matches the published checksum**

```bash
grep -A1 'ET_RUNTIME_URL_logging_windows-x86_64-static' native/cmake/EtRuntimePin.cmake
grep 'ET_RUNTIME_SHA256_logging_windows-x86_64-static' native/cmake/EtRuntimePin.cmake
```

Expected: the URL ends `executorch-runtime-1.3.1-logging-windows-x86_64-static.tar.gz` and the SHA256 is `c9d929e5f6724dcf03286526e93332acd7d75d7138bb5ade038064992324fa50`.

- [ ] **Step 6: Run the resolution test to confirm it passes**

Run: `bash native/tests/cmake_resolution.sh`
Expected: `PASS: cmake resolution`

- [ ] **Step 7: Rebuild Linux against the new pin and run the JVM suite**

```bash
./native/local_build_wrapper.sh
./gradlew test
```

Expected: build succeeds (the XNNPACK post-link guard passes), all tests green. This is the gate proving the Linux artifact rebuild in `-8` did not regress anything.

- [ ] **Step 8: Commit**

```bash
git add native/cmake/EtRuntimePin.cmake native/tests/cmake_resolution.sh
git commit -m "deps(runtime): bump ExecuTorch runtime pin to v1.3.1-8

Adds the windows-x86_64-static (/MT) row. Every Linux artifact was
rebuilt in this release, so all six Linux hashes changed too and the
Linux build/test was re-run as part of the review."
```

---

### Task 2: Windows spike — measure the three unknowns (HUMAN-GATED)

**This task runs on a Windows machine with an activated MSVC dev shell. It commits no source changes.** Its entire purpose is to convert three assumptions into measurements before the CMake change lands.

Everything is a command-line override — `ET_PLATFORM` and `ET_RUNTIME_VARIANT` are already `CACHE` variables and `CMAKE_MSVC_RUNTIME_LIBRARY` is a standard CMake cache variable — so **no source edits are needed for this task**.

**Files:**
- Create: `docs/handover-windows-static-cxx17-findings.md` (findings only; expanded in Task 7)

**Interfaces:**
- Consumes: `ET_RUNTIME_URL_logging_windows-x86_64-static` from Task 1.
- Produces: findings S1–S4, recorded in the findings doc. Task 4 depends on S1.

- [ ] **Step 1: Configure and build the shim against the static row**

From Git-Bash inside an activated VS dev shell, at the repo root:

```bash
rm -rf /tmp/spike-mt
cmake -B /tmp/spike-mt -S native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DET_PLATFORM=windows-x86_64-static \
  -DET_RUNTIME_VARIANT=logging \
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
cmake --build /tmp/spike-mt -j
```

Expected: `executorch_djl.dll` is produced with no `LNK2038`.

Note `-DET_PLATFORM=windows-x86_64-static` here is the *pre-split* variable — this spike runs against the Task 1 tree, where `ET_PLATFORM` is still the row key. Task 3 renames the row key; the spike deliberately does not depend on that.

- [ ] **Step 2: Record finding S2 — the import table**

```bash
dumpbin -nologo -dependents /tmp/spike-mt/executorch_djl.dll | grep -iE 'vcruntime|msvcp|api-ms-win-crt'
```

Expected: **no output** (exit 1 from grep). Any `VCRUNTIME140.dll` or `MSVCP140.dll` line means the CRT is still dynamic somewhere.

Contrast run, to prove the check can actually fail — this is what makes a "no output" result meaningful rather than a broken command:

```bash
rm -rf /tmp/spike-md
cmake -B /tmp/spike-md -S native -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DET_PLATFORM=windows-x86_64 -DET_RUNTIME_VARIANT=logging
cmake --build /tmp/spike-md -j
dumpbin -nologo -dependents /tmp/spike-md/executorch_djl.dll | grep -iE 'vcruntime|msvcp'
```

Expected: this one **does** list `VCRUNTIME140.dll`.

- [ ] **Step 3: Record finding S4 — the C++ standard flag (Stream B / B5)**

```bash
cmake --build /tmp/spike-mt --clean-first -- -v 2>&1 | grep -o '/std:c++[0-9a-z]*' | sort -u
```

Expected: exactly `/std:c++20`. An empty result means MSVC is on its C++14 default and the ET `#error` guard is being satisfied by nothing — which would contradict the spec's §6 conclusion.

- [ ] **Step 4: Record finding S1 — does the QA path link with Catch2 in the mix?**

This is the highest-risk item in the whole change. Catch2 is built from source via `FetchContent` and defaults to the dynamic CRT.

```bash
rm -rf native/asan
cmake -B native/asan -S native -G Ninja -DET_BUILD_QA=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DET_PLATFORM=windows-x86_64-static \
  -DET_RUNTIME_VARIANT=logging \
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
cmake --build native/asan --target et_runtime_test -j
```

Expected: links cleanly. **Per the spec's §2 and the work order's A3, a CRT mismatch links silently with no `LNK2005` and not even an `LNK4098`** — so a clean link is *not* sufficient evidence on its own. Confirm the CRT directly:

```bash
dumpbin -nologo -directives native/asan/_deps/catch2-build/src/Catch2.lib | grep -i defaultlib
```

Expected: `LIBCMT`/`LIBCPMT`, **not** `MSVCRT`/`MSVCPRT`. If Catch2 shows `MSVCRT`, the global cache variable did not propagate into the FetchContent subproject and Task 4 must be revised.

This manual check is the spike's one-shot answer. Task 6 makes it permanent by invoking `check_windows_crt.sh native/asan` from `build_qa.sh`, so propagation is re-verified on every QA run rather than trusted indefinitely from this single measurement.

- [ ] **Step 5: Record finding S3 — does it still work?**

```bash
./native/asan/et_runtime_test.exe
```

Expected: all Catch2 assertions pass. This also stands in for the XNNPACK registration guard, which cannot run on Windows (it greps Itanium-ABI mangling out of `nm`), because the suite executes an XNNPACK-delegated `add.pte`.

Then stage the DLL and run the JVM suite:

```bash
cp /tmp/spike-mt/executorch_djl.dll src/main/resources/native/windows-x86_64/
./gradlew test
```

Expected: green. Proves `System.load` succeeds and inference runs against a `/MT` DLL.

- [ ] **Step 6: Write the findings down**

Create `docs/handover-windows-static-cxx17-findings.md`:

```markdown
# Windows /MT adoption — measured findings

Spike run against runtime pin v1.3.1-8, MSVC 2022, on <machine/date>.

## Stream A

| ID | Question | Result |
|----|----------|--------|
| S1 | Does the QA path link all-`/MT` with Catch2 via FetchContent? | <PASS/FAIL + the Catch2.lib defaultlib line> |
| S2 | Does the DLL still import VCRUNTIME140/MSVCP140? | <result; include the /MD contrast run> |
| S3 | Catch2 suite + `./gradlew test` green? | <result> |

Not verified by execution: load on a Windows image that has never had a VC++
redistributable installed (no such machine available). The claim rests on S2's
import-table evidence.

## Stream B

| ID | Question | Result |
|----|----------|--------|
| S4 | Which `/std:` flag reaches the ET-including TUs on MSVC? | <output of the grep> |
```

- [ ] **Step 7: Commit the findings**

```bash
git add docs/handover-windows-static-cxx17-findings.md
git commit -m "docs: record measured findings from the Windows /MT spike"
```

**GATE:** If S1 failed, stop here and revise Task 4's mechanism before continuing.

---

### Task 3: Split `ET_PLATFORM` into platform identity + pin row

`ET_PLATFORM` currently serves two roles: the pin-row lookup key and the identity used in comparisons. The `-static` suffix separates them. Splitting keeps the variant guard at line 37 correct by construction — simply changing `ET_PLATFORM`'s value would make `ET_PLATFORM STREQUAL "windows-x86_64"` silently stop matching, downgrading its actionable error back to the generic "No pin row" message the guard exists to prevent.

**Files:**
- Modify: `native/CMakeLists.txt:17-25` (variable declarations), `:42-44` (lookup), `:51` (diagnostic)
- Test: `native/tests/cmake_resolution.sh:41-46`

**Interfaces:**
- Consumes: pin rows from Task 1.
- Produces: cache variable `ET_RUNTIME_ROW` (pin lookup key; defaults `windows-x86_64-static` when `ET_PLATFORM` is `windows-x86_64`, else the value of `ET_PLATFORM`). `ET_PLATFORM` keeps its existing values and meaning. The `ET_RESOLUTION` diagnostic line gains a `row=<value>` field between `platform=` and `stem=`.

- [ ] **Step 1: Write the failing tests**

In `native/tests/cmake_resolution.sh`, replace the Windows block at lines 41-46:

```bash
# Windows resolution, asserted from a Linux host: ET_PLATFORM is a cache var, so the ET_PRINT_RESOLUTION
# seam can resolve a foreign platform's pin row without that platform being present.
out="$(probe -DET_PLATFORM=windows-x86_64)"
grep -q 'platform=windows-x86_64'                                <<<"${out}" || fail "windows platform not echoed"
grep -q 'stem=executorch-runtime-1.3.1-logging-windows-x86_64'   <<<"${out}" || fail "windows stem wrong"
grep -q 'executorch-runtime-1.3.1-logging-windows-x86_64.tar.gz' <<<"${out}" || fail "windows url wrong"
```

with:

```bash
# Windows resolution, asserted from a Linux host: ET_PLATFORM is a cache var, so the ET_PRINT_RESOLUTION
# seam can resolve a foreign platform's pin row without that platform being present.
#
# Windows resolves the -static (/MT) row by default: the shipped DLL must not need a VC++ redist.
# ET_PLATFORM is the platform IDENTITY and stays 'windows-x86_64'; ET_RUNTIME_ROW is the pin key.
out="$(probe -DET_PLATFORM=windows-x86_64)"
grep -q 'platform=windows-x86_64'                                       <<<"${out}" || fail "windows platform not echoed"
grep -q 'row=windows-x86_64-static'                                     <<<"${out}" || fail "windows must default to the -static row"
grep -q 'stem=executorch-runtime-1.3.1-logging-windows-x86_64-static'   <<<"${out}" || fail "windows stem wrong"
grep -q 'executorch-runtime-1.3.1-logging-windows-x86_64-static.tar.gz' <<<"${out}" || fail "windows url wrong"

# The /MD row must remain selectable, not merely present in the pin file. Without this the row would be
# a hardcode with extra steps, and a typo'd row name would be indistinguishable from a deleted one.
out="$(probe -DET_PLATFORM=windows-x86_64 -DET_RUNTIME_ROW=windows-x86_64)"
grep -q 'row=windows-x86_64$'                                    <<<"${out}" || fail "dynamic row not selectable"
grep -q 'executorch-runtime-1.3.1-logging-windows-x86_64.tar.gz' <<<"${out}" || fail "dynamic row url wrong"

# Linux identity and row coincide; assert the split did not introduce a Linux-side divergence.
out="$(probe)"
grep -q 'row=linux-x86_64' <<<"${out}" || fail "linux row must equal linux platform"
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `bash native/tests/cmake_resolution.sh`
Expected: `FAIL: windows must default to the -static row` (no `row=` field is emitted yet).

- [ ] **Step 3: Declare the row variable**

In `native/CMakeLists.txt`, replace lines 17-25:

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

with:

```cmake
# ET_PLATFORM is the platform IDENTITY; ET_RUNTIME_ROW is the pin-row LOOKUP KEY. They coincide on
# Linux but not on Windows, which publishes two rows for one platform: `windows-x86_64` (/MD, dynamic
# CRT, built for CPython extensions) and `windows-x86_64-static` (/MT). The shim takes the static row
# so the shipped DLL needs no VC++ redistributable.
#
# Keeping identity separate from row is what lets the variant guard below stay a plain STREQUAL. If
# ET_PLATFORM itself carried the `-static` suffix, that guard would silently stop matching and its
# actionable error would degrade into the generic "No pin row" message it exists to pre-empt.
#
# Both are CACHE vars (not plain set) deliberately: it lets the ET_PRINT_RESOLUTION host-test seam
# force a foreign platform (-DET_PLATFORM=windows-x86_64) and assert the pin resolves, with no machine
# of that platform present. See native/tests/cmake_resolution.sh.
if(WIN32)
  set(_ET_PLATFORM_DEFAULT "windows-x86_64")
else()
  set(_ET_PLATFORM_DEFAULT "linux-x86_64")
endif()
set(ET_PLATFORM "${_ET_PLATFORM_DEFAULT}" CACHE STRING "Platform identity: windows-x86_64 | linux-x86_64")

# Derive the row default from ET_PLATFORM, NOT from WIN32: the host-test seam sets ET_PLATFORM on a
# Linux box, and deriving from WIN32 there would resolve a Linux row under a Windows identity.
if(ET_PLATFORM STREQUAL "windows-x86_64")
  set(_ET_ROW_DEFAULT "windows-x86_64-static")
else()
  set(_ET_ROW_DEFAULT "${ET_PLATFORM}")
endif()
set(ET_RUNTIME_ROW "${_ET_ROW_DEFAULT}" CACHE STRING "Pin row key in EtRuntimePin.cmake (may differ from ET_PLATFORM)")
```

- [ ] **Step 4: Point the lookup at the row**

In `native/CMakeLists.txt`, replace lines 42-44:

```cmake
  set(_ET_STEM "executorch-runtime-${ET_RUNTIME_ET_VERSION}-${ET_RUNTIME_VARIANT}-${ET_PLATFORM}")
  set(_ET_RESOLVED_URL "${ET_RUNTIME_URL_${ET_RUNTIME_VARIANT}_${ET_PLATFORM}}")
  set(_ET_RESOLVED_SHA "${ET_RUNTIME_SHA256_${ET_RUNTIME_VARIANT}_${ET_PLATFORM}}")
```

with:

```cmake
  set(_ET_STEM "executorch-runtime-${ET_RUNTIME_ET_VERSION}-${ET_RUNTIME_VARIANT}-${ET_RUNTIME_ROW}")
  set(_ET_RESOLVED_URL "${ET_RUNTIME_URL_${ET_RUNTIME_VARIANT}_${ET_RUNTIME_ROW}}")
  set(_ET_RESOLVED_SHA "${ET_RUNTIME_SHA256_${ET_RUNTIME_VARIANT}_${ET_RUNTIME_ROW}}")
```

Leave the variant guard at line 37 (`ET_PLATFORM STREQUAL "windows-x86_64"`) **unchanged** — that is the entire point of the split.

Also update the error message on line 46 to name the row, since that is what actually failed to resolve:

```cmake
    message(FATAL_ERROR "No pin row for variant='${ET_RUNTIME_VARIANT}' row='${ET_RUNTIME_ROW}' (platform='${ET_PLATFORM}') in EtRuntimePin.cmake")
```

- [ ] **Step 5: Emit the row in the diagnostic**

Replace line 51:

```cmake
  message(STATUS "ET_RESOLUTION resolution=${_ET_RESOLUTION} variant=${ET_RUNTIME_VARIANT} platform=${ET_PLATFORM} stem=${_ET_STEM} url=${_ET_RESOLVED_URL} et_install=${ET_INSTALL}")
```

with:

```cmake
  message(STATUS "ET_RESOLUTION resolution=${_ET_RESOLUTION} variant=${ET_RUNTIME_VARIANT} platform=${ET_PLATFORM} row=${ET_RUNTIME_ROW} stem=${_ET_STEM} url=${_ET_RESOLVED_URL} et_install=${ET_INSTALL}")
```

- [ ] **Step 6: Run the tests to confirm they pass**

Run: `bash native/tests/cmake_resolution.sh`
Expected: `PASS: cmake resolution`

- [ ] **Step 7: Confirm Linux is untouched end-to-end**

```bash
./native/local_build_wrapper.sh && ./gradlew test
```

Expected: green. `ET_RUNTIME_ROW` defaults to `linux-x86_64` there, so the resolved URL is byte-identical to Task 1's.

- [ ] **Step 8: Commit**

```bash
git add native/CMakeLists.txt native/tests/cmake_resolution.sh
git commit -m "feat(native): resolve the windows-x86_64-static pin row

Splits ET_PLATFORM (platform identity) from ET_RUNTIME_ROW (pin lookup
key) so Windows can take the /MT row without breaking the STREQUAL
variant guard. The /MD row stays selectable and is now covered by a test."
```

---

### Task 4: Force the static CRT across every target

**Files:**
- Modify: `native/CMakeLists.txt` (insert after line 4)
- Modify: `native/build.sh:97-100` (comment), `native/build_qa.sh:38-42` (comment)

**Interfaces:**
- Consumes: Task 2 finding S1 (proves the mechanism reaches Catch2); Task 3's `ET_RUNTIME_ROW`.
- Produces: `CMAKE_MSVC_RUNTIME_LIBRARY` = `MultiThreaded` on Windows for every target in the project and in every `FetchContent` subproject.

- [ ] **Step 1: Set the CRT globally, before any FetchContent**

In `native/CMakeLists.txt`, insert immediately after line 4 (`set(CMAKE_CXX_STANDARD_REQUIRED ON)`):

```cmake
# Static CRT (/MT) on Windows: folds the C runtime into the DLL so end users need no VC++
# redistributable. Windows consumers here are developers on potentially locked-down workstations who
# may not be able to run an installer. Pairs with the -static pin row selected below — the runtime
# tarball is itself built /MT (see its BUILDINFO cmake_flags), and every object linked into one
# binary must agree.
#
# A GLOBAL cache variable, not per-target set_property(... MSVC_RUNTIME_LIBRARY ...): Catch2 enters
# the QA build through FetchContent_MakeAvailable, so its targets are not declared here and cannot
# have a property set on them. This variable initialises the property on every target including those
# created in added subdirectories, which is the only mechanism that covers that case.
#
# NOT config-dependent (the CMake default is "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"): the pinned
# runtime is Release-only, so a Debug configure must still get the release static CRT rather than
# /MTd, which would not link against it.
#
# CMP0091 must be NEW for this to take effect; cmake_minimum_required(3.24) above guarantees it.
#
# Do NOT expect the linker to police this. Measured upstream: a /MD consumer linked a /MT ExecuTorch
# artifact with no error and not even an LNK4098 warning. The failure is at runtime — two CRTs, two
# heaps, corruption when an allocation crosses the boundary. native/tests/check_windows_crt.sh is the
# real gate.
if(WIN32)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded" CACHE STRING "Static CRT: ship without a VC++ redist")
endif()
```

- [ ] **Step 2: Correct the now-false comment in `native/build.sh`**

Replace lines 97-100:

```bash
# MSVC encodes the CRT flavour into every object and the linker refuses to mix them. The pinned runtime
# tarball is built Release (/MD — see its BUILDINFO cmake_flags), so the shim must be Release too or the
# link dies with LNK2038 'RuntimeLibrary' / '_ITERATOR_DEBUG_LEVEL' mismatches. GCC/ELF has no such ABI
# tag, so the Linux leg stays as-is (unset) and its artifact is unchanged.
```

with:

```bash
# MSVC encodes the CRT flavour into every object and the linker refuses to mix them. The pinned runtime
# tarball is built Release with the STATIC CRT (/MT — see its BUILDINFO cmake_flags:
# CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded), so the shim must be Release too or the link dies with
# LNK2038 '_ITERATOR_DEBUG_LEVEL' mismatches. Release is about the debug/release CRT split only; the
# static-vs-dynamic choice is made by CMAKE_MSVC_RUNTIME_LIBRARY in native/CMakeLists.txt, not here.
# GCC/ELF has no such ABI tag, so the Linux leg stays as-is (unset) and its artifact is unchanged.
```

- [ ] **Step 3: Correct the now-false comment in `native/build_qa.sh`**

Replace lines 38-42:

```bash
  # No sanitizers: MSVC has no LeakSanitizer, and the leak harness is not built here at all.
  # RelWithDebInfo, NOT Debug: Debug would compile against the Debug CRT (/MDd) while the pinned runtime
  # is Release (/MD), and MSVC refuses to mix them — the same LNK2038/LNK1319 wall the shim build hits
  # (see build.sh). RelWithDebInfo gives us the matching /MD CRT while keeping symbols, so a Catch2
  # failure is still debuggable.
```

with:

```bash
  # No sanitizers: MSVC has no LeakSanitizer, and the leak harness is not built here at all.
  # RelWithDebInfo, NOT Debug: Debug would compile against the Debug CRT while the pinned runtime is
  # Release, and MSVC refuses to mix them — the same LNK2038/LNK1319 wall the shim build hits (see
  # build.sh). RelWithDebInfo gives us the matching release CRT while keeping symbols, so a Catch2
  # failure is still debuggable. The static-vs-dynamic CRT choice is separate and is made by
  # CMAKE_MSVC_RUNTIME_LIBRARY in native/CMakeLists.txt, which also propagates into the FetchContent'd
  # Catch2 build — a /MD Catch2 inside a /MT test exe links silently and corrupts at runtime.
```

- [ ] **Step 4: Confirm Linux is unaffected**

```bash
bash native/tests/cmake_resolution.sh
./native/local_build_wrapper.sh && ./gradlew test
```

Expected: all green. The new block is `if(WIN32)`-gated, so Linux sees no change.

- [ ] **Step 5: Commit**

```bash
git add native/CMakeLists.txt native/build.sh native/build_qa.sh
git commit -m "feat(native): compile the Windows shim against the static CRT

Global CMAKE_MSVC_RUNTIME_LIBRARY rather than per-target properties:
Catch2 arrives via FetchContent and its targets cannot be reached with
set_property. Corrects two comments that described the runtime as /MD."
```

---

### Task 5: Point the Windows provenance gate at the tarball actually being linked

The CI provenance step extracts a URL with `grep -oE 'https://[^"]*logging-windows-x86_64\.tar\.gz'`. The `/MD` row remains in the pin file, so **this keeps matching and keeps passing while attesting a tarball the build no longer links.** That is a silent hole in a supply-chain gate, not a cosmetic issue.

**Files:**
- Modify: `.github/workflows/native-build-job.yml:99`
- Test: `native/tests/ci_workflow.sh`

**Interfaces:**
- Consumes: the `-static` row from Task 1; `ET_RUNTIME_ROW` semantics from Task 3.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the failing test**

Append to `native/tests/ci_workflow.sh`, immediately before the Python/YAML block at line 34:

```bash
# The Windows provenance gate must attest the row the build actually links (-static, /MT). The /MD row
# is still in the pin file, so a pattern without the suffix keeps matching and keeps PASSING while
# verifying the wrong tarball — a silent supply-chain hole, which is why this is asserted here.
awk '/^  build-executorch-shim-windows:/{f=1} f' "${WFJOB}" \
  | grep -q 'logging-windows-x86_64-static' \
  || fail "windows provenance gate must attest the -static tarball"
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `bash native/tests/ci_workflow.sh`
Expected: `FAIL: windows provenance gate must attest the -static tarball`

- [ ] **Step 3: Fix the workflow**

In `.github/workflows/native-build-job.yml`, replace line 99:

```bash
          url="$(grep -oE 'https://[^"]*logging-windows-x86_64\.tar\.gz' native/cmake/EtRuntimePin.cmake | head -1)"
```

with:

```bash
          # -static: the /MT row, which is what native/CMakeLists.txt resolves via ET_RUNTIME_ROW. The
          # /MD row is still in the pin file, so a pattern without the suffix would match it and attest
          # a tarball this build never links.
          url="$(grep -oE 'https://[^"]*logging-windows-x86_64-static\.tar\.gz' native/cmake/EtRuntimePin.cmake | head -1)"
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `bash native/tests/ci_workflow.sh`
Expected: `PASS: ci workflow`

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/native-build-job.yml native/tests/ci_workflow.sh
git commit -m "fix(ci): attest the -static runtime tarball on Windows

The old pattern still matched the /MD row left in the pin file, so the
provenance gate passed while verifying a tarball the build never links."
```

---

### Task 6: CRT verification gate

Replaces the work order's A5 clean-image test, which cannot be run (no redist-free Windows machine available). Two static checks substitute.

**Files:**
- Create: `native/tests/check_windows_crt.sh`
- Modify: `.github/workflows/native-build-job.yml` (new step after the QA step)
- Test: `native/tests/ci_workflow.sh`

**Interfaces:**
- Consumes: `executorch_djl.dll` staged at `src/main/resources/native/windows-x86_64/`, and the shim build tree at `native/build` (the default `NATIVE_BUILD_DIR` in `native/build.sh:28`).
- Note: `native/build` also contains the FetchContent'd runtime at `_deps/et_runtime-src/lib/`, so the scan sweeps ~20 runtime `.lib`s in addition to the shim's own. That is intentional — it verifies the whole link set, not just our objects — so expect ~21 `ok` lines, not one.
- Produces: `native/tests/check_windows_crt.sh <build-dir> [dll-path]`, exit 0 on pass, 1 on failure, 2 on usage error. Invoked twice: from CI against `native/build` **with** the DLL, and from `build_qa.sh` against `native/asan` **without** one.

- [ ] **Step 1: Write the gate script**

Create `native/tests/check_windows_crt.sh`:

```bash
#!/usr/bin/env bash
# Assert the shipped Windows DLL is fully statically linked against the CRT.
#
# Replaces the ideal test — loading the DLL on a Windows image that has never had a VC++
# redistributable installed — which we cannot run: no such machine is available, and a dev box proves
# nothing because it already has the runtime. These two static checks stand in for it.
#
# MSVC records the CRT choice per-object as /DEFAULTLIB directives:
#   /MT (static)  -> LIBCMT / LIBCPMT
#   /MD (dynamic) -> MSVCRT / MSVCPRT
# A mismatch is NOT reliably caught at link time: measured upstream, a /MD consumer linked a /MT
# ExecuTorch prefix with no LNK2005 and not even an LNK4098. The hazard is at runtime — two CRTs, two
# heaps, corruption when an allocation crosses the boundary. Hence a direct inspection.
#
# TWO TRAPS, both of which cost the producer repo real debugging time:
#  - Flags are passed as -nologo -directives, NOT /nologo. Under MSYS/Git-Bash a leading '/' is
#    path-converted (/nologo -> C:\Program Files\Git\nologo) and dumpbin fails on a garbage filename.
#  - Every assertion is POSITIVE: each lib must CARRY the expected marker. An absence-only check
#    ("no MSVCRT found") reports PASS when dumpbin failed to run at all — that exact bug once passed
#    18 libraries green while the tool was erroring on every one of them.
#
# Run inside Git-Bash from an activated VS dev shell (needs dumpbin).
#
# Usage: check_windows_crt.sh <build-dir> [dll-path]
#   <build-dir>  scanned for *.lib; every one must carry the static marker
#   [dll-path]   optional; when given, its import table is checked too
#
# The DLL argument is optional because this runs against TWO trees. The shim tree (native/build)
# produces a DLL. The QA tree (native/asan) does not — it produces a test executable — but it is the
# tree containing the FetchContent'd Catch2, which is the single most likely place for the static-CRT
# setting to stop propagating, and a /MD Catch2 inside a /MT test exe links with no diagnostic at all.
set -euo pipefail
usage_err() { echo "usage: check_windows_crt.sh <build-dir> [dll-path]" >&2; exit 2; }
BUILD_DIR="${1:-}"; [ -n "${BUILD_DIR}" ] || usage_err
DLL="${2:-}"

command -v dumpbin >/dev/null 2>&1 \
  || { echo "FAIL: dumpbin not on PATH — run inside an activated VS dev shell" >&2; exit 1; }
[ -d "${BUILD_DIR}" ] || { echo "FAIL: no build dir at ${BUILD_DIR}" >&2; exit 1; }
[ -z "${DLL}" ] || [ -f "${DLL}" ] || { echo "FAIL: no DLL at ${DLL}" >&2; exit 1; }

rc=0

echo "== 1/2 CRT directive scan: every .lib built here must carry LIBCMT/LIBCPMT =="
libs=0
while IFS= read -r lib; do
  libs=$((libs+1))
  name="$(basename "${lib}")"
  set +e
  out="$(dumpbin -nologo -directives "${lib}" 2>&1)"; drc=$?
  set -e
  # Do NOT swallow a dumpbin failure: it is exactly the condition this gate exists to notice.
  if [ "${drc}" -ne 0 ]; then
    echo "  ERROR ${name} -> dumpbin exited ${drc}:"
    head -5 <<< "${out}" | sed 's/^/          /'
    rc=1; continue
  fi
  # Positive assertion. A C-only lib carries LIBCMT with no LIBCPMT, so the pattern is an OR.
  if grep -qiE 'DEFAULTLIB:"?(LIBCMT|LIBCPMT)' <<< "${out}"; then
    if grep -qiE 'DEFAULTLIB:"?(MSVCRT|MSVCPRT)' <<< "${out}"; then
      echo "  MIXED ${name} -> carries BOTH static and dynamic CRT markers"; rc=1
    else
      echo "  ok    ${name}"
    fi
  else
    echo "  FAIL  ${name} -> no LIBCMT/LIBCPMT marker"; rc=1
  fi
done < <(find "${BUILD_DIR}" -name '*.lib' -type f)

# A scan that inspected zero libraries must not report success.
if [ "${libs}" -eq 0 ]; then
  echo "FAIL: no .lib files found under ${BUILD_DIR} — the scan proved nothing" >&2
  rc=1
fi

if [ -z "${DLL}" ]; then
  [ "${rc}" -eq 0 ] && echo "PASS: static CRT across ${libs} libs in ${BUILD_DIR} (no DLL to check)" \
                    || echo "FAIL: windows CRT check"
  exit "${rc}"
fi

echo "== 2/2 Import table: the DLL must not depend on a redistributable CRT =="
set +e
deps="$(dumpbin -nologo -dependents "${DLL}" 2>&1)"; drc=$?
set -e
[ "${drc}" -eq 0 ] || { echo "FAIL: dumpbin -dependents exited ${drc}"; echo "${deps}"; exit 1; }
# Positive assertion first: the output must actually look like a dependents listing, otherwise a
# grep that finds no VCRUNTIME below would be meaningless.
grep -qi 'KERNEL32.dll' <<< "${deps}" \
  || { echo "FAIL: dependents listing has no KERNEL32.dll — output is not what we think it is"; echo "${deps}"; exit 1; }
if grep -iE 'VCRUNTIME[0-9]*\.dll|MSVCP[0-9]*\.dll' <<< "${deps}"; then
  echo "FAIL: DLL imports a redistributable CRT (above) — the /MT switch did not fully apply"
  rc=1
else
  echo "  ok    no VCRUNTIME/MSVCP imports"
fi

[ "${rc}" -eq 0 ] && echo "PASS: windows CRT is static" || echo "FAIL: windows CRT check"
exit "${rc}"
```

- [ ] **Step 2: Make it executable and syntax-check it on Linux**

```bash
chmod +x native/tests/check_windows_crt.sh
bash -n native/tests/check_windows_crt.sh
```

Expected: no output (syntax OK).

- [ ] **Step 3: Verify the usage guard behaves**

```bash
bash native/tests/check_windows_crt.sh; echo "rc=$?"
```

Expected: `usage: check_windows_crt.sh <shim-build-dir> <dll-path>` and `rc=2`.

- [ ] **Step 4: Gate the QA tree from `build_qa.sh` itself**

Catch2 is the most likely place for the static-CRT setting to stop propagating, because it is the only target the project does not declare. Verifying it once by hand in Task 2 leaves no standing guard, and the failure mode is silent.

Put the check in `build_qa.sh` rather than only in the CI workflow, so it runs wherever QA runs — CI *and* a local winbox run — instead of only where someone remembered to wire it up.

In `native/build_qa.sh`, in the Windows branch, insert between the `cmake --build ... --target et_runtime_test` line (currently line 45) and the `echo "--- Catch2 unit suite ..."` line:

```bash
  # Catch2 comes in via FetchContent, so it is the one target whose CRT we do not set directly — the
  # global CMAKE_MSVC_RUNTIME_LIBRARY has to propagate into a subproject to reach it. A /MD Catch2
  # inside this /MT test exe links with no LNK2038 and not even an LNK4098, then corrupts the heap at
  # runtime. Assert it here, before running the suite, so the failure names its own cause instead of
  # surfacing as an inexplicable Catch2 crash.
  bash native/tests/check_windows_crt.sh native/asan
```

- [ ] **Step 5: Write the failing wiring tests**

Append to `native/tests/ci_workflow.sh`, immediately before the Python/YAML block:

```bash
# The CRT gate is what stands in for the (unavailable) clean-image test, so its absence from the
# windows job would silently remove the only evidence behind the "no VC++ redist needed" claim.
awk '/^  build-executorch-shim-windows:/{f=1} f' "${WFJOB}" \
  | grep -q 'check_windows_crt.sh' \
  || fail "windows CRT gate missing (check_windows_crt.sh not invoked in the windows job)"

# The QA tree needs its own scan: the workflow step above covers native/build (the shim), which does
# not contain Catch2. Scoped to build_qa.sh's windows branch — the linux branch neither has nor needs
# this, so an unscoped grep would stay green if the windows call were deleted.
awk '/ET_HOST_OS.*=.*"windows"/{f=1} f' native/build_qa.sh \
  | grep -q 'check_windows_crt.sh native/asan' \
  || fail "build_qa.sh windows branch must scan the QA tree (Catch2 CRT)"
```

- [ ] **Step 6: Run them to confirm they fail**

Run: `bash native/tests/ci_workflow.sh`
Expected: `FAIL: windows CRT gate missing (check_windows_crt.sh not invoked in the windows job)`

(After Step 7 wires the workflow, re-running surfaces the second failure if Step 4 was skipped.)

- [ ] **Step 7: Wire the shim-tree gate into CI**

In `.github/workflows/native-build-job.yml`, insert this step immediately **after** the `QA the executorch_djl shim (MSVC)` step and **before** the `Store executorch_djl shim` step, matching the surrounding steps' VS-discovery preamble:

```yaml
      - name: Assert the shim links the static CRT (MSVC)
        shell: pwsh
        run: |
          $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
          $vsPath = & $vswhere -latest -products * -property installationPath
          if (-not $vsPath) { throw "vswhere found no Visual Studio installation" }
          & "$vsPath\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -SkipAutomaticLocation
          $bash = "${env:ProgramFiles}\Git\bin\bash.exe"
          & $bash -c './native/tests/check_windows_crt.sh native/build src/main/resources/native/windows-x86_64/executorch_djl.dll'
          if ($LASTEXITCODE -ne 0) { throw "windows CRT check failed (exit $LASTEXITCODE)" }
```

- [ ] **Step 8: Run the tests to confirm they pass**

Run: `bash native/tests/ci_workflow.sh`
Expected: `PASS: ci workflow`

- [ ] **Step 9: Commit**

```bash
git add native/tests/check_windows_crt.sh native/tests/ci_workflow.sh \
        native/build_qa.sh .github/workflows/native-build-job.yml
git commit -m "test(native): gate the Windows build on a static-CRT check

Scans /DEFAULTLIB directives and the DLL import table. Runs against both
trees: the shim (native/build, from CI) and the QA tree (native/asan,
from build_qa.sh) — the latter is where the FetchContent'd Catch2 lives,
the one target whose CRT we cannot set directly.

Stands in for the clean-image redist test, which we have no machine to run."
```

---

### Task 7: Document the C++17 finding and report back

Stream B resolves to "no code change". The durable deliverable is the written record — it is what stops a future cleanup from deleting `set(CMAKE_CXX_STANDARD 20)` as apparently redundant and reintroducing an MSVC-only `C1189`.

**Files:**
- Modify: `CLAUDE.md` (the "The ExecuTorch runtime is NOT built here" section)
- Modify: `docs/handover-windows-static-cxx17-findings.md` (created in Task 2)

**Interfaces:**
- Consumes: Task 2 finding S4.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Add the durable note to `CLAUDE.md`**

In `CLAUDE.md`, in the `### The ExecuTorch runtime is NOT built here` section, append these bullets to the existing list (the one ending with the `etnp::lstm` bullet):

```markdown
- **`find_package(executorch)` supplies no language standard.** ExecuTorch's headers require C++17 and
  enforce it with a hard `#error` in `runtime/platform/compiler.h`, but the installed CMake package
  exports **no** `INTERFACE_COMPILE_FEATURES` on any target — verified on both Linux and Windows
  builds of v1.3.1. `native/CMakeLists.txt` therefore states the standard itself
  (`CMAKE_CXX_STANDARD 20` + `CMAKE_CXX_STANDARD_REQUIRED ON`); do **not** delete those as redundant.
  Removing them breaks MSVC only (GCC defaults to `gnu++17` and masks it), surfacing as
  `fatal error C1189: #error: "You need C++17 to compile ExecuTorch"`.
- **Windows links the `-static` (`/MT`) pin row** so the shipped DLL needs no VC++ redistributable.
  `ET_PLATFORM` is the platform identity (`windows-x86_64`); `ET_RUNTIME_ROW` is the pin-row key
  (`windows-x86_64-static`). The `/MD` row exists for CPython consumers and is not what we link.
  MSVC does **not** reliably diagnose a CRT mismatch — no `LNK2038`, not even an `LNK4098` — so
  `native/tests/check_windows_crt.sh` is the real gate.
```

- [ ] **Step 2: Complete the findings report**

Append to `docs/handover-windows-static-cxx17-findings.md`:

```markdown
## Stream B conclusion: nothing to change

The engine's C++17 requirement on Windows is satisfied by
`native/CMakeLists.txt:3-4` — `set(CMAKE_CXX_STANDARD 20)` with
`set(CMAKE_CXX_STANDARD_REQUIRED ON)` — declared at the top of the project's single
`CMakeLists.txt`, before any target is created.

That source is trustworthy on all four axes the work order asks about:

- **B1 explicit and first-party.** Not implicit, not inherited from a dependency, so it cannot
  disappear under an unrelated bump.
- **B2 `REQUIRED ON`.** CMake cannot silently decay to an older standard.
- **B3/B4 scope.** The setting precedes every `add_library`/`add_executable`, and the project has
  exactly one `CMakeLists.txt` — no sibling tree or earlier-added subdirectory can miss it. It also
  propagates into FetchContent subprojects.
- **B5 verified on MSVC.** See finding S4 above: `/std:c++20` reaches the ET-including TUs. ET's
  guard keys on `_MSVC_LANG`, which `/std:c++20` satisfies; `/Zc:__cplusplus` is not required.

C++20 rather than C++17 is a deliberate superset, not an accident.

Documented in `CLAUDE.md` so a future cleanup does not delete the setting as redundant.

## Note for the producer repo

Confirmed independently in the engine: the installed ExecuTorch CMake package exports no
`INTERFACE_COMPILE_FEATURES`, so `find_package` communicates no language standard to consumers. This
matches the producer repo's finding and supports its drafted upstream proposal. Per the work order, no
issue was filed against `pytorch/executorch` from this repo.
```

- [ ] **Step 3: Verify the docs are consistent with the code they describe**

```bash
sed -n '1,5p' native/CMakeLists.txt
grep -n 'ET_RUNTIME_ROW' native/CMakeLists.txt | head -5
```

Expected: lines 3-4 are the `CMAKE_CXX_STANDARD 20` / `CMAKE_CXX_STANDARD_REQUIRED ON` pair as the note claims, and `ET_RUNTIME_ROW` exists. Fix the note if the line numbers drifted.

- [ ] **Step 4: Run the full host-fast suite**

```bash
bash native/tests/cmake_resolution.sh
bash native/tests/ci_workflow.sh
./gradlew test
```

Expected: `PASS: cmake resolution`, `PASS: ci workflow`, tests green.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md docs/handover-windows-static-cxx17-findings.md
git commit -m "docs: record that find_package(executorch) exports no C++ standard

Stream B resolves to no code change; the note is the deliverable, since
it is what stops a cleanup from deleting CMAKE_CXX_STANDARD as redundant."
```

---

## Final verification

Run before opening the PR:

- [ ] `bash native/tests/cmake_resolution.sh` → `PASS: cmake resolution`
- [ ] `bash native/tests/ci_workflow.sh` → `PASS: ci workflow`
- [ ] `./native/local_build_wrapper.sh && ./gradlew test` → green (Linux, glibc floor preserved)
- [ ] Windows CI: shim build, Catch2 QA, CRT gate on **both** trees (`native/build` + `native/asan`) all green
- [ ] `git diff main --stat` touches no Java source and no `build.gradle.kts`
- [ ] Findings doc reflects what was actually measured in Task 2, not what was expected

**Known gap, to state in the PR description:** the "no VC++ redistributable required" claim rests on import-table evidence, not on execution on a redist-free Windows image. No such machine was available. Tracked as follow-up.
