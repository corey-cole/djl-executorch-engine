# Gate B — UBSan over the JNI shim (PR 3 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Instrument `native/jni/executorch_djl_jni.cpp` with UBSan and exercise it with the JVM suite — the only configuration that reaches the file where JNI bugs live.

**Architecture:** A separate build tree (`native/ubsan`) configured with the existing `ET_UBSAN=ON` but *without* `ET_BUILD_QA`, so the shim is built rather than skipped, with `-static-libubsan` folding the UBSan runtime into the `.so` so a stock JVM can `dlopen` it. A new `native/ubsan_gate.sh` runs it in two phases — build in the container, test on the host — because the container's JDK cannot run Gradle. The instrumented library is reached through `EXECUTORCH_LIBRARY_PATH` and never staged into resources.

**Tech Stack:** CMake, GCC 14, Bash, Gradle 9.6.1 / JDK 17, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-13-ubsan-and-jni-checking-design.md` (§6 Gate B, §7, §8)
**Source material:** `docs/research/ubsan-jni-checking-port-handover.md` §6, §7

## Global Constraints

- **`ET_UBSAN` and `ET_UBSAN_CHECKS` already exist** (PR 2, `native/CMakeLists.txt:40-75`) and the `add_compile_options` block is directory-scoped above every target, so the shim is instrumented automatically once it is built in an `ET_UBSAN=ON` tree. Only the **link** side is missing.
- **`-static-libubsan` is not optional.** Without it the `.so` carries a dynamic `libubsan` dependency and `System.load` fails with a confusing linker error — possibly hours later, in a different CI job, on a different machine.
- **The instrumented library must never enter `src/main/resources/native/`.** It is reached via `EXECUTORCH_LIBRARY_PATH`, which `LibUtils.loadLibrary()` passes straight to `System.load()` as a full file path (`LibUtils.java:37-43`). This gate therefore stages nothing and requires no rebuild afterwards.
- **The CI artifact must not match `executorch-libs-*`.** `native-build.yml:36-38` downloads that pattern with `merge-multiple: true` directly into `src/main/resources/native/`. Name it `executorch-ubsan-linux-x86_64`.
- **A UB hit is a JVM hard crash**, not a test failure. The finding is the `runtime error:` line and its stack trace *above* the JVM's own crash output. Say so in the script header or it reads as a flake.
- **`--rerun-tasks` on the Gradle invocation.** A cached `UP-TO-DATE` reports a pass for a run that never loaded this library.
- **`--no-daemon`, always.** A pre-existing daemon lives in whatever cgroup it started in, so `./gradlew` would hand the work — including every forked test JVM — to a process outside any resource scope. `oomTest` exhausts a heap on purpose.
- **Test tasks: `test leakTest oomTest`.** The six config-variant and stress tasks vary Java-side configuration rather than shim behaviour, or are deliberately kept out of CI.
- **Linux only.** MSVC has no UBSan; every Windows path stays untouched.
- **No `sudo` in any step.** `native/container_env.sh` (PR 2) hands container outputs back on exit; to clear a root-owned tree, use the container, which is already root.

---

### Task 1: Link the instrumented shim statically and prove it

**Files:**
- Modify: `native/CMakeLists.txt` (the `executorch_djl` target block, ~line 187)

**Interfaces:**
- Produces: when configured with `-DET_UBSAN=ON` and without `ET_BUILD_QA`, a `native/ubsan/libexecutorch_djl.so` whose `ldd` output contains no `libubsan`. Task 2's script asserts exactly that.

- [ ] **Step 1: Write the failing test — build the tree and check the dependency**

No code change yet. Configure and build the shim under UBSan as it stands:

```bash
docker run --rm -v "$PWD":/workspace -w /workspace "$(cat .engine-build-image)" \
  bash -c 'cmake -S native -B native/ubsan -G "Unix Makefiles" -DET_UBSAN=ON \
    -DCMAKE_BUILD_TYPE=Debug && cmake --build native/ubsan --target executorch_djl -j"$(nproc)"'
ldd native/ubsan/libexecutorch_djl.so | grep -i ubsan
```

Expected: the build succeeds and `ldd` **prints a `libubsan.so` line**. That dynamic dependency is the defect — a stock JVM `System.load` of this file fails.

- [ ] **Step 2: Add the static link option**

In `native/CMakeLists.txt`, inside the `if(NOT ET_BUILD_QA AND NOT ET_BUILD_BENCH)` block, after `target_link_libraries(executorch_djl PRIVATE et_runtime)`:

```cmake
  # The shim is dlopen'd by a stock JVM that knows nothing about sanitizer runtimes, so UBSan's
  # runtime has to travel inside the .so. Without this the link succeeds and System.load fails
  # later -- in CI, possibly in a different job on a different machine than the build.
  if(ET_UBSAN)
    target_link_options(executorch_djl PRIVATE -static-libubsan)
  endif()
```

- [ ] **Step 3: Rebuild and verify the dependency is gone**

The tree must be dropped: link options changed, and this is the property being tested.

```bash
docker run --rm -v "$PWD":/workspace -w /workspace "$(cat .engine-build-image)" \
  bash -c 'rm -rf native/ubsan && cmake -S native -B native/ubsan -G "Unix Makefiles" -DET_UBSAN=ON \
    -DCMAKE_BUILD_TYPE=Debug && cmake --build native/ubsan --target executorch_djl -j"$(nproc)"'
ldd native/ubsan/libexecutorch_djl.so | grep -i ubsan
nm -D --defined-only native/ubsan/libexecutorch_djl.so | grep -c ubsan
```

Expected: `ldd` prints **nothing** for ubsan (grep exits 1), and `nm` reports a **nonzero** count of defined ubsan symbols. Together those mean the runtime is present and statically linked.

- [ ] **Step 4: Commit**

```bash
git add native/CMakeLists.txt
git commit -m "build: fold the UBSan runtime into the shim with -static-libubsan"
```

---

### Task 2: The two-phase gate script

**Files:**
- Create: `native/ubsan_gate.sh`
- Modify: `native/local_build_wrapper.sh` (the `-e` forwarding list)

**Interfaces:**
- Consumes: `ET_UBSAN_MODE` (`auto`|`build`|`test`|`all`, default `auto`), `TEST_TASKS`, `GRADLE_FLAGS`, `BUILD_DIR`.
- Produces: `native/ubsan/libexecutorch_djl.so` from the build phase; a Gradle run against it from the test phase.

**Why two phases.** The pinned image has the right toolchain and the wrong JDK — `JAVA_HOME` is Corretto 8, chosen for the oldest supported `jni.h`, while Gradle 9.6.1 with this project's JDK 17 toolchain cannot start there at all. Do **not** fix that by adding a modern JDK to the shared image: it would undo the compatibility floor the old JDK exists for, and that image is shared with another repo.

- [ ] **Step 1: Write the script**

Create `native/ubsan_gate.sh`, `chmod +x`:

```bash
#!/usr/bin/env bash
# UBSan gate for the JNI shim, driven by the JVM suite.
#
# This is the ONLY configuration in which native/jni/executorch_djl_jni.cpp is instrumented:
# native/build_qa.sh covers et_runtime, the Catch2 suite and the harnesses, but
# native/CMakeLists.txt skips the shim under ET_BUILD_QA so QA stays JVM-free.
#
# NOTE: a UB hit here presents as a JVM HARD CRASH mid-test, not a Java exception or an assertion
# failure. That is the gate working. Look for the "runtime error:" line and its stack trace above
# the JVM's own crash output.
#
# TWO PHASES, because they need different environments. The pinned image has the right toolchain and
# the wrong JDK (Corretto 8, for the oldest supported jni.h); Gradle 9.6.1 with this project's JDK 17
# toolchain cannot run there. So: build in the container, test on the host. ET_UBSAN_MODE selects a
# phase and defaults to `auto` -- build-only inside the image, both phases outside it.
#
# The instrumented .so is NEVER staged into src/main/resources: it is reached through
# EXECUTORCH_LIBRARY_PATH, which LibUtils honours ahead of the classpath copy and which
# build.gradle.kts already declares as a Test task input. The plain tree is left untouched.
#
# Linux only: MSVC has no UndefinedBehaviorSanitizer.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BUILD_DIR="${BUILD_DIR:-native/ubsan}"
JOBS="${JOBS:-$(nproc)}"

# The build phase runs as root under native/local_build_wrapper.sh, so without this the tree comes
# back root-owned and the next run's `rm -rf` dies with a bare "Permission denied". No-op on a host,
# where HOST_UID is unset.
# shellcheck source=native/container_env.sh
. "${REPO_ROOT}/native/container_env.sh"
et_chown_outputs_on_exit "${BUILD_DIR}"

MODE="${ET_UBSAN_MODE:-auto}"
if [ "${MODE}" = "auto" ]; then
  if [ -n "${MEASLY_DJL_PINNED_IMAGE:-}" ]; then MODE=build; else MODE=all; fi
fi
case "${MODE}" in
  build|test|all) ;;
  *) echo "ET_UBSAN_MODE must be build, test, all or auto (got '${MODE}')" >&2; exit 1 ;;
esac

# tasks.test excludes eight tags; oomTest and leakTest are where the marshalling loop meets
# allocation failure and memory pressure, which is the exposure this gate is for.
TEST_TASKS="${TEST_TASKS:-test leakTest oomTest}"

# --no-daemon is not a preference. A pre-existing daemon lives in whatever cgroup it was first
# started in, so ./gradlew would hand the work -- including every forked test JVM -- to a process
# outside any resource scope wrapping this script. oomTest exhausts a heap on purpose.
GRADLE_FLAGS="${GRADLE_FLAGS:---no-daemon}"

# -fno-sanitize-recover (native/CMakeLists.txt) makes UBSan abort; these make the abort legible.
export UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1

if [ "${MODE}" = "build" ] || [ "${MODE}" = "all" ]; then
  echo "--- Building the UBSan-instrumented shim ---"
  rm -rf "${BUILD_DIR}"
  # No ET_BUILD_QA: that is what makes CMakeLists build the shim rather than skip it. JAVA_HOME is
  # needed here for jni.h only -- we never link libjvm.
  cmake -S native -B "${BUILD_DIR}" -G "Unix Makefiles" \
    -DET_UBSAN=ON -DCMAKE_BUILD_TYPE=Debug
  cmake --build "${BUILD_DIR}" --target executorch_djl -j"${JOBS}"

  # A dynamic libubsan dependency means -static-libubsan did not apply and System.load would fail.
  # Assert here so the failure names its own cause: the build may happen in a container and the load
  # on a host hours later, in a different job.
  if ldd "${BUILD_DIR}/libexecutorch_djl.so" | grep -qi ubsan; then
    echo "FAIL: ${BUILD_DIR}/libexecutorch_djl.so has a dynamic libubsan dependency;" >&2
    echo "      -static-libubsan did not apply (native/CMakeLists.txt)" >&2
    exit 1
  fi
  echo "--- UBSan runtime is statically linked ---"
fi

if [ "${MODE}" = "build" ]; then
  echo "--- UBSan shim built at ${BUILD_DIR}/libexecutorch_djl.so; JVM phase skipped ---"
  echo "--- Run the JVM phase where a JDK 17 lives: ET_UBSAN_MODE=test ./native/ubsan_gate.sh ---"
  exit 0
fi

# Refuse the JVM phase rather than letting Gradle fail obscurely.
if [ -n "${MEASLY_DJL_PINNED_IMAGE:-}" ]; then
  echo "REFUSING the JVM phase inside the pinned image: JAVA_HOME is Corretto 8, and Gradle 9.6.1" >&2
  echo "with a JDK 17 toolchain cannot run there. Build here, test on the host:" >&2
  echo "  ./native/local_build_wrapper.sh native/ubsan_gate.sh   # build phase, in-container" >&2
  echo "  ET_UBSAN_MODE=test ./native/ubsan_gate.sh              # JVM phase, on the host" >&2
  exit 1
fi

_java_major="$("${JAVA_HOME:-/usr}/bin/java" -version 2>&1 | head -1 | sed -E 's/.*"([0-9]+).*/\1/')"
if [ "${_java_major:-0}" -lt 17 ]; then
  echo "JAVA_HOME points at Java ${_java_major}; Gradle 9.6.1 and this project need 17+." >&2
  exit 1
fi

if [ ! -f "${BUILD_DIR}/libexecutorch_djl.so" ]; then
  echo "no instrumented shim at ${BUILD_DIR}/libexecutorch_djl.so -- run the build phase first" >&2
  exit 1
fi

echo "--- JVM suite against the instrumented shim (${TEST_TASKS}) ---"
# --rerun-tasks: a cached UP-TO-DATE would report a pass for a run that never loaded this library.
EXECUTORCH_LIBRARY_PATH="${REPO_ROOT}/${BUILD_DIR}/libexecutorch_djl.so" \
  ./gradlew ${GRADLE_FLAGS} ${TEST_TASKS} --rerun-tasks

echo "--- UBSan gate PASS ---"
```

- [ ] **Step 2: Forward `ET_UBSAN_MODE` through the wrapper**

The wrapper forwards an explicit `-e` list, so an unlisted variable set on the host is silently dropped inside the container — the same trap `CLEAN` had. `auto` covers the normal path, but an explicit override must not be quietly ignored. Add beside `CLEAN`:

```bash
    -e ET_UBSAN_MODE \
```

- [ ] **Step 3: Run the build phase in the container**

```bash
./native/local_build_wrapper.sh native/ubsan_gate.sh
```

Expected: `--- UBSan runtime is statically linked ---`, then `JVM phase skipped` with the follow-up command. `auto` resolved to `build` because `MEASLY_DJL_PINNED_IMAGE` is set inside the image.

Confirm the tree came back yours, proving the `container_env.sh` registration works here too:

```bash
stat -c '%U %n' native/ubsan
```

- [ ] **Step 4: Confirm the in-container refusal is legible**

```bash
ET_UBSAN_MODE=test ./native/local_build_wrapper.sh native/ubsan_gate.sh
```

Expected: exit 1 with the `REFUSING the JVM phase inside the pinned image` message and both follow-up commands. This also proves Step 2's forwarding works — without it the variable would be dropped and the script would run the build phase instead.

- [ ] **Step 5: Run the JVM phase on the host**

```bash
./gradlew --stop
systemd-run --user --scope -p MemoryMax=4G taskset -c 0-3 timeout 1800 \
  env ET_UBSAN_MODE=test ./native/ubsan_gate.sh
```

Expected: `--- UBSan gate PASS ---`. If a UB hit occurs, that is Task 4's work, not a failure of this task.

- [ ] **Step 6: Commit**

```bash
git add native/ubsan_gate.sh native/local_build_wrapper.sh
git commit -m "test: add the two-phase UBSan gate for the JNI shim"
```

---

### Task 3: Prove the gate actually covers the shim

**Files:**
- Modify: `native/jni/executorch_djl_jni.cpp` (temporary probe, reverted in this task)

**Interfaces:** none.

**This is the most important task in the plan.** The gate exists solely to cover
`executorch_djl_jni.cpp`. A green run proves nothing until a deliberate fault in *that file* has been
shown to fail it — everything else here could be working while the shim is quietly uninstrumented.

- [ ] **Step 1: Inject UB into the shim**

In `native/jni/executorch_djl_jni.cpp`, inside the `forward` JNI entry point, at the top of the function body:

```cpp
  // TEMPORARY UBSAN PROBE -- revert in Step 4. Do not commit.
  { volatile int probe_shift = 1; (void)(probe_shift << 99); }
```

- [ ] **Step 2: Rebuild and run the gate**

```bash
./native/local_build_wrapper.sh native/ubsan_gate.sh
./gradlew --stop
systemd-run --user --scope -p MemoryMax=4G taskset -c 0-3 timeout 1800 \
  env ET_UBSAN_MODE=test ./native/ubsan_gate.sh
```

Expected: **FAILURE.** A line naming the shim —
`executorch_djl_jni.cpp:<line>: runtime error: shift exponent 99 is too large for 32-bit type 'int'` —
and a nonzero exit. Because `-fno-sanitize-recover` aborts, this surfaces as a JVM crash in whichever
test first calls `forward`, with the diagnostic **above** the crash dump.

If the suite passes instead, the shim is not instrumented; do not proceed. Check that the configure
omits `ET_BUILD_QA` and that `EXECUTORCH_LIBRARY_PATH` actually pointed at `native/ubsan`.

- [ ] **Step 3: If Step 2 passed, diagnose before proceeding**

Only if Step 2 did **not** fail. A pass means the JVM loaded something uninstrumented, so find out
what:

```bash
ldd native/ubsan/libexecutorch_djl.so | grep -i ubsan   # expect no output (static)
nm -D --defined-only native/ubsan/libexecutorch_djl.so | grep -c ubsan   # expect nonzero
```

A zero symbol count means the configure skipped the shim's instrumentation. A nonzero count with a
passing suite means `EXECUTORCH_LIBRARY_PATH` did not reach the test JVM, so the staged library was
loaded instead — check that the gate script exported it and that no task overrode it.

- [ ] **Step 4: Revert the probe**

```bash
git diff --stat native/jni/executorch_djl_jni.cpp
```

Expected after reverting: no output. **Never commit the probe.**

- [ ] **Step 5: No commit**

This task produces no commit — it is a proof, and its artifact is the knowledge that the gate bites.
Record the observed `runtime error:` line in the PR description.

---

### Task 4: Run the real suite and fix findings

**Files:**
- Modify: `native/jni/executorch_djl_jni.cpp`, `native/core/et_runtime.cpp` — only in response to a diagnostic.

**Interfaces:** none.

The exposure this gate adds that nothing else covers is `alignment` over host buffers whose alignment
the JVM does not guarantee — the marshalling loop reading `jbyte*`/`jlong*` regions. Expect findings
there before anywhere else.

- [ ] **Step 1: Full clean run**

```bash
./native/local_build_wrapper.sh native/ubsan_gate.sh
./gradlew --stop
systemd-run --user --scope -p MemoryMax=4G taskset -c 0-3 timeout 1800 \
  env ET_UBSAN_MODE=test ./native/ubsan_gate.sh
```

Expected: `--- UBSan gate PASS ---`.

- [ ] **Step 2: Triage each `runtime error:`**

Read the file and check named in the diagnostic:

- **`alignment`** — a load through a pointer whose alignment the JVM does not promise. Fix by copying through `memcpy` into a properly aligned local rather than casting the pointer.
- **`shift`, `null`, `bounds`, `return`** — ordinary C++ UB in our code. Fix at the site.
- **A float check** — inspect the arithmetic; a cast overflow in element-count or scale arithmetic is a real bug, not noise.

Do not reach for `__attribute__((no_sanitize(...)))` first. If one is genuinely required, it carries a
comment naming what was given up and why, per the constraint inherited from PR 2.

- [ ] **Step 3: Re-run until clean, then commit**

Skip if Step 1 was clean.

```bash
git add native/
git commit -m "fix: <the specific undefined behaviour UBSan reported in the shim>"
```

---

### Task 5: CI wiring

**Files:**
- Modify: `.github/workflows/native-build-job.yml` (build phase, `linux-x86_64` only)
- Modify: `.github/workflows/native-build.yml` (new JVM job)

**Interfaces:**
- Produces: artifact `executorch-ubsan-linux-x86_64` containing `native/ubsan/libexecutorch_djl.so`.

- [ ] **Step 1: Build and upload the instrumented shim**

In `native-build-job.yml`, after the existing QA step, scoped to the primary platform — a second
native build plus a `--rerun-tasks` suite is real CI time, and the aarch64 row would double it for no
new defect class:

```yaml
      # Gate B build phase. Runs only here: the JVM phase needs a JDK 17 the image does not have, so
      # it lives in a separate job that downloads this artifact.
      - name: Build the UBSan-instrumented shim
        if: matrix.platform == 'linux-x86_64'
        run: |
          docker run --rm \
            -v ${{ github.workspace }}:/workspace \
            -w /workspace \
            ${{ env.ET_BUILD_IMAGE }} \
            /bin/bash /workspace/native/ubsan_gate.sh

      # NOT executorch-libs-*: native-build.yml downloads that pattern with merge-multiple into
      # src/main/resources/native/, which would make this instrumented library what every ordinary
      # test run loads.
      - name: Store the UBSan-instrumented shim
        if: matrix.platform == 'linux-x86_64'
        uses: actions/upload-artifact@v7
        with:
          name: executorch-ubsan-linux-x86_64
          path: ${{ github.workspace }}/native/ubsan/libexecutorch_djl.so
          compression-level: 1
```

- [ ] **Step 2: Add the JVM phase job**

In `native-build.yml`, a sibling of `build-java-package`:

```yaml
  ubsan-jvm-gate:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    needs: build-executorch-shim
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-java@v5
        with:
          distribution: zulu
          java-version: 17

      # Deliberately NOT into src/main/resources/native/: the gate reaches this library through
      # EXECUTORCH_LIBRARY_PATH, and staging it would make it the default for every other job.
      - name: Download the UBSan-instrumented shim
        uses: actions/download-artifact@v8
        with:
          name: executorch-ubsan-linux-x86_64
          path: ubsan-lib

      - name: Setup Gradle
        uses: gradle/actions/setup-gradle@v6

      - name: Run the JVM suite against the instrumented shim
        env:
          ET_UBSAN_MODE: test
          BUILD_DIR: ubsan-lib
        run: ./native/ubsan_gate.sh
```

`BUILD_DIR=ubsan-lib` points the test phase at the downloaded artifact instead of `native/ubsan`,
which is why that variable is parameterised in the script.

- [ ] **Step 3: Validate the workflow YAML**

```bash
bash native/tests/ci_workflow.sh
```

Expected: `PASS: ci workflow` — its embedded `yaml.safe_load` proves both edited files still parse.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/native-build-job.yml .github/workflows/native-build.yml
git commit -m "ci: run the UBSan shim gate as a build phase plus a JVM job"
```

---

### Task 6: Document and close out

**Files:**
- Modify: `CLAUDE.md`, `docs/building.md`, `docs/README.md` if it indexes the gate

**Interfaces:** none.

- [ ] **Step 1: Document the gate**

State: `native/ubsan_gate.sh` is the only configuration instrumenting the JNI shim; it runs in two
phases because the pinned image's JDK cannot run Gradle, selected by `ET_UBSAN_MODE` (default `auto`);
a UB hit is a JVM hard crash with the `runtime error:` line above the crash dump; the instrumented
library is reached via `EXECUTORCH_LIBRARY_PATH` and never staged; and the local invocation is the two
commands the script prints. Note that it runs in CI on `linux-x86_64` only.

- [ ] **Step 2: Confirm the shipping path is untouched**

```bash
./native/local_build_wrapper.sh
nm -D --defined-only src/main/resources/native/linux-x86_64/libexecutorch_djl.so | grep -c ubsan
./gradlew clean build
```

Expected: the count is **0** and the build succeeds. `ET_UBSAN` defaults OFF and `build.sh` never
sets it, but this is the assertion that the two trees really are independent.

- [ ] **Step 3: Full regression**

```bash
bash native/tests/ci_workflow.sh && bash native/tests/build_config.sh && bash native/tests/docs_present.sh
./native/local_build_wrapper.sh native/build_qa.sh
```

Expected: three `PASS:` lines, and the PR-2 QA gate still green.

- [ ] **Step 4: Commit and open the PR**

```bash
git add CLAUDE.md docs/building.md
git commit -m "docs: document the UBSan shim gate"
```

Push and open the PR. In the description, record the `runtime error:` line observed in Task 3 — it is
the evidence the gate bites, and nothing in the merged tree demonstrates it.

---

## Known gaps

- **`implicit-signed-integer-truncation` remains uncovered**: clang-only, and the shared image ships no clang.
- **aarch64 is not gated.** Gate B runs on `linux-x86_64` only. Alignment behaviour is exactly the class where an arch difference could matter, so this is a real gap rather than a cost saving alone — revisit if the aarch64 row ever finds something the x86_64 row did not.
- **The JVM phase runs under `-Xcheck:jni` too**, inherited from PR 1's umbrella. That is a bonus, not a substitute: the two tools catch disjoint defect classes.
