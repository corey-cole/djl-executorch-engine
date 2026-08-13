# Gate A — UBSan over the native QA tree (PR 2 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compose UndefinedBehaviorSanitizer onto the existing ASan QA build so `et_runtime`, the Catch2 suite and the leak/stress harnesses run under both, with UB aborting the run rather than printing and continuing.

**Architecture:** An `ET_UBSAN` option in `native/CMakeLists.txt` that adds `-fsanitize=` compile and link options, enabled by `native/build_qa.sh`'s Linux branch alongside the ASan flags it already passes. No new build tree, no new script, no workflow change.

**Tech Stack:** CMake, GCC 14 (gcc-toolset-14 in the shared image), Bash.

**Spec:** `docs/superpowers/specs/2026-08-13-ubsan-and-jni-checking-design.md` (§6 Gate A, §8)
**Source material:** `docs/research/ubsan-jni-checking-port-handover.md` §4

## Global Constraints

- **Linux only.** MSVC has no UndefinedBehaviorSanitizer. `ET_UBSAN=ON` on WIN32 is a `FATAL_ERROR` at configure time, and `build_qa.sh`'s Windows branch never sets it. Every Windows path stays byte-identical.
- **Check set:** `undefined,float-cast-overflow,float-divide-by-zero`, with `-fno-sanitize=vptr`. The two float checks are added because GCC deliberately excludes both from its `undefined` umbrella; `vptr` is dropped because it needs every TU holding a polymorphic object instrumented and is the check most likely to misfire against the uninstrumented prebuilt runtime.
- **`-fno-sanitize-recover=undefined` is what makes this a gate.** UBSan's default is print-and-continue, under which CI stays green while diagnostics scroll past. Removing this flag silently converts the gate into a log.
- **GCC has no ignorelist.** `-fsanitize-ignorelist` and `-fsanitize-blacklist` are clang-only and rejected as unrecognized; `UBSAN_OPTIONS=suppressions=` does not suppress these checks (measured on gcc 13.3). To silence a diagnostic, in order of preference: `__attribute__((no_sanitize("undefined")))` on the specific function; a per-TU `set_source_files_properties(... COMPILE_OPTIONS "-fno-sanitize=<check>")`; `-fno-sanitize=<check>` program-wide as a last resort, always with a comment naming what was given up.
- **Known gap, do not try to close it here:** `implicit-signed-integer-truncation` is clang-only, so that class stays uncovered on GCC. The shared image ships no clang. A clang variant is a clean follow-on, not a prerequisite.
- **This lands as a CI gate on both arches automatically.** `native-build-job.yml` already runs `build_qa.sh` in the `linux-x86_64` and `linux-aarch64` rows, so no workflow edit is needed and a finding blocks CI on both.

---

### Task 1: Add the option and prove it catches UB

**Files:**
- Modify: `native/CMakeLists.txt` (immediately after the `ET_BUILD_QA` / `ET_BUILD_BENCH` options, ~line 38)
- Modify: `native/build_qa.sh` (the Linux `cmake -B native/asan` configure, ~line 74)
- Modify: `native/local_build_wrapper.sh` (the `-e` forwarding list, ~line 47)

**Interfaces:**
- Produces: CMake option `ET_UBSAN` (BOOL, default OFF) and cache variable `ET_UBSAN_CHECKS` (STRING). PR 3's Gate B consumes both — it enables `ET_UBSAN=ON` against a different build tree, which is the reason this is an option rather than another raw flag string.

**Note the asymmetry with ASan and keep it.** ASan reaches this tree as `-DCMAKE_CXX_FLAGS="-fsanitize=address …"` from `build_qa.sh`, not as a CMake option. Do **not** refactor ASan into an option to match — that is churn outside this PR. UBSan gets an option because PR 3 must enable it for the shim in a separate tree, and a shared seam beats duplicating a flag string across two scripts.

- [ ] **Step 1: Write the failing test — inject deliberate UB**

There is no unit test for a sanitizer; the test is that a known-bad expression is caught. Add this temporarily to `native/core/et_runtime.cpp`, inside the `EtRuntime` constructor body:

```cpp
  // TEMPORARY UBSAN PROBE -- revert in Step 7. Do not commit.
  { volatile int probe_shift = 1; (void)(probe_shift << 99); }
```

- [ ] **Step 2: Run QA to verify the UB goes UNDETECTED today**

```bash
./native/local_build_wrapper.sh native/build_qa.sh
```

Expected: BUILD SUCCESSFUL and the Catch2 suite **passes**. A shift past the width of `int` is undefined behaviour that the current ASan-only build does not see. That silence is the thing this task fixes.

Then hand ownership back — `build_qa.sh` does not do it for you:

```bash
sudo chown -R "$(id -u):$(id -g)" native/asan
```

- [ ] **Step 3: Add the CMake option**

In `native/CMakeLists.txt`, immediately after the `ET_BUILD_BENCH` option:

```cmake
option(ET_UBSAN "Build with UndefinedBehaviorSanitizer" OFF)

# The check set, as a cache variable so a one-off run can narrow it without editing this file. Two
# additions over GCC's `undefined` umbrella, which excludes both: float-cast-overflow and
# float-divide-by-zero, each reachable in element-count and scale arithmetic. vptr comes off below --
# it needs every TU holding a polymorphic object instrumented, and is the check most likely to
# misfire against the prebuilt runtime, which is not instrumented.
set(ET_UBSAN_CHECKS "undefined,float-cast-overflow,float-divide-by-zero"
    CACHE STRING "UBSan check set passed to -fsanitize=")

# UBSan composes with ASan, so this stacks on the -fsanitize=address flags build_qa.sh passes. It is
# per-TU and local -- each check is an inline test emitted at the operation, with no cross-module
# state -- so linking the uninstrumented prebuilt runtime produces no false positives and hides
# nothing in our own code.
#
# -fno-sanitize-recover is what makes this a gate rather than a log: UBSan's default is
# print-and-continue, under which CI stays green while diagnostics scroll past.
#
# GCC has no ignorelist: -fsanitize-ignorelist and -fsanitize-blacklist are unrecognized, and
# UBSAN_OPTIONS=suppressions= does not suppress these checks. To silence one, use
# __attribute__((no_sanitize("undefined"))) on the function, a per-TU
# set_source_files_properties COMPILE_OPTIONS override, or -fno-sanitize=<check> here -- each with a
# comment naming what was given up and why.
if(ET_UBSAN)
  if(WIN32)
    message(FATAL_ERROR "ET_UBSAN is unsupported on Windows: MSVC has no UndefinedBehaviorSanitizer")
  endif()
  add_compile_options(
      -fsanitize=${ET_UBSAN_CHECKS}
      -fno-sanitize=vptr
      -fno-sanitize-recover=undefined
      -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${ET_UBSAN_CHECKS})
endif()
```

Placement matters: `add_compile_options` is directory-scoped and applies only to targets defined *after* it, so this must sit above every `add_library`/`add_executable` in the file.

- [ ] **Step 4: Enable it from `build_qa.sh`**

In the Linux branch, add `-DET_UBSAN=ON` to the existing configure and export `UBSAN_OPTIONS` before the suite runs:

```bash
  export UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1
  cmake -B native/asan -S native -G "Unix Makefiles" "${ET_ARGS[@]}" -DET_BUILD_QA=ON \
    -DET_UBSAN=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"
```

Leave the Windows branch untouched — it must never pass `-DET_UBSAN=ON`.

- [ ] **Step 5: Forward `CLEAN` through the wrapper**

`build_qa.sh` reuses `native/asan` when it was configured for the same source root, and `CLEAN=1` is
its documented override — but `native/local_build_wrapper.sh` does not forward `CLEAN` into the
container, so setting it on the host is silently ignored. That failure mode is worse than an error:
the run appears to work while building the old configuration. Add it to the `-e` list beside
`ET_STRESS`:

```bash
    -e CLEAN \
```

- [ ] **Step 6: Run QA to verify the UB is now CAUGHT**

The configure changed, so the cached tree must go:

```bash
CLEAN=1 ./native/local_build_wrapper.sh native/build_qa.sh
```

Expected: **FAILURE**, with a line naming the probe, e.g.
`et_runtime.cpp:<line>: runtime error: shift exponent 99 is too large for 32-bit type 'int'`,
and a **nonzero exit**. A `runtime error:` line without a nonzero exit means `-fno-sanitize-recover` is not taking effect — fix that before continuing, because it is the whole gate.

```bash
sudo chown -R "$(id -u):$(id -g)" native/asan
```

- [ ] **Step 7: Revert the probe**

Delete the two probe lines from `native/core/et_runtime.cpp`. Confirm they are gone:

```bash
git diff --stat native/core/et_runtime.cpp
```

Expected: no output. **Never commit the probe.**

- [ ] **Step 8: Commit**

```bash
git add native/CMakeLists.txt native/build_qa.sh native/local_build_wrapper.sh
git commit -m "test: compose UBSan onto the native QA build"
```

---

### Task 2: Run the real QA suite under both sanitizers and fix findings

**Files:**
- Modify: `native/core/*.cpp`, `native/harness/*.cpp`, `native/test/et_runtime_test.cpp` — only in response to a diagnostic.

**Interfaces:** none.

**Expect this to find something, and expect some of it not to be ours.** Two distinct sources: real UB in `native/core`, which gets fixed; and diagnostics from FetchContent'd Catch2, which is built under this directory's compile options and is not our code.

- [ ] **Step 1: Run the full QA pass clean**

```bash
CLEAN=1 ./native/local_build_wrapper.sh native/build_qa.sh
sudo chown -R "$(id -u):$(id -g)" native/asan
```

Expected: Catch2 suite passes, then all three `et_leak_harness` runs pass, with no `runtime error:` lines.

- [ ] **Step 2: Triage any diagnostic by where it originates**

For each `runtime error:` line, read the file path in the message:

- **In `native/core/`, `native/jni/` or `native/harness/`** — ours. Fix the code. Do not reach for a suppression first: `-fno-sanitize-recover` means a suppression turns a red build green while leaving the defect in place.
- **In `native/asan/_deps/catch2-src/`** — not ours, and it arrives because `add_compile_options` is directory-scoped and Catch2 is declared under it. Prefer scoping the flags off that subproject; a program-wide `-fno-sanitize=<check>` is the last resort and must carry a comment naming the check it removes from the gate for every file.
- **Inside the prebuilt runtime** — should not happen. The runtime is not compiled here, and UBSan checks are per-TU inline tests, so an uninstrumented library emits none. If one appears, it means the check fired in *our* frame with a runtime symbol on the stack; treat it as ours.

- [ ] **Step 3: Re-run until clean**

```bash
CLEAN=1 ./native/local_build_wrapper.sh native/build_qa.sh
sudo chown -R "$(id -u):$(id -g)" native/asan
```

Expected: no `runtime error:` lines, zero exit.

- [ ] **Step 4: Commit any fixes**

Skip entirely if Step 1 was clean.

```bash
git add native/
git commit -m "fix: <the specific undefined behaviour UBSan reported>"
```

---

### Task 3: Confirm the shipping build is untouched, then document

**Files:**
- Modify: `CLAUDE.md` (the native QA section), `docs/building.md` (the native QA section)

**Interfaces:** none.

- [ ] **Step 1: Prove nothing instrumented reaches the shipping artifact**

`ET_UBSAN` defaults to OFF and `build.sh` never sets it, so the shipped `.so` must be unchanged. Verify rather than assume:

```bash
./native/local_build_wrapper.sh
nm -D --defined-only src/main/resources/native/linux-x86_64/libexecutorch_djl.so | grep -c ubsan
```

Expected: the build succeeds and the count is **0**. A nonzero count means the option leaked into the shipping configure.

- [ ] **Step 2: Run the host-fast shell suites**

```bash
bash native/tests/ci_workflow.sh && bash native/tests/build_config.sh && bash native/tests/docs_present.sh
```

Expected: three `PASS:` lines.

- [ ] **Step 3: Document the gate**

In `CLAUDE.md`'s native QA section and `docs/building.md`'s equivalent, state: `build_qa.sh` builds the QA tree under ASan **and** UBSan; UB aborts the run rather than printing, so a QA failure may be a `runtime error:` line rather than a failed assertion; the check set lives in `ET_UBSAN_CHECKS` and can be narrowed for a one-off run; GCC has no ignorelist, so `__attribute__((no_sanitize("undefined")))` is the way to exempt a function; and `implicit-signed-integer-truncation` is clang-only and therefore uncovered.

Note also that this runs in CI on both Linux arches, since `native-build-job.yml` already invokes `build_qa.sh` in both rows — and that UBSan adds compile and run time to both.

Describe the gate as it is. Do not narrate what the QA build did before this change.

- [ ] **Step 4: Commit and open the PR**

```bash
git add CLAUDE.md docs/building.md
git commit -m "docs: document the UBSan QA gate"
```

Push and open the PR. Watch the `linux-aarch64` row specifically: it is the arch nothing local covers, and it runs the same `build_qa.sh`.

---

## Known gaps

- **The shim is still uninstrumented.** `native/CMakeLists.txt` skips `executorch_djl` under `ET_BUILD_QA`, so this gate covers `et_runtime`, the Catch2 suite and the harnesses — not `native/jni/executorch_djl_jni.cpp`. That file is PR 3's entire purpose.
- **`implicit-signed-integer-truncation` stays uncovered** on GCC, as above.
- **A clean first run is weak evidence.** As with Gate C, this codebase's core has been debugged by hand for a long time; the exposure UBSan adds that nothing else covers is `alignment` over host buffers, and that lands mostly in the shim — again, PR 3.
