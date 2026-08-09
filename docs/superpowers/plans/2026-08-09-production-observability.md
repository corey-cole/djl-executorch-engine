# Production Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give operators an always-on, fixed-cost view of inference throughput, native memory footprint, and effective configuration, readable from Java or any JMX console.

**Architecture:** Per-model counters are `volatile long` fields on a holder owned by `EtSymbolBlock` (single-writer by the engine's threading contract, so no CAS and no allocation on the forward path). A process-level registry in `EtEngineStats` maps live native handles to their blocks; `snapshot()` walks it on a cold path, pulling native byte counts through two new native accessors. An MXBean wrapping `snapshot()` auto-registers at first model load.

**Tech Stack:** Java 17, DJL 0.36.0, JUnit 5, slf4j, `javax.management` (JDK built-in — no new dependency), C++20, Catch2, CMake, JNI.

**Spec:** `docs/superpowers/specs/2026-08-09-production-observability-design.md`

## Global Constraints

- **No new runtime dependencies.** JMX is `java.management`, part of the JDK. Do not add Micrometer, OpenTelemetry, or Prometheus.
- **The forward path must not regress.** One `System.nanoTime()` pair and three `volatile long` stores per forward is the entire permitted budget. No allocation, no locking, no collection lookup inside `forwardInternal()`.
- **`snapshot()` never throws.** Unreadable values degrade to `-1` (bytes) or `"unknown"` (strings).
- **`-1` means "unavailable"; `0` means "genuinely zero".** This distinction is load-bearing for `stagingBytes`, which is legitimately `0` for memory-planned models.
- **Native changes require a rebuild before JVM tests pass.** Local iteration: `./native/build.sh`. Release/staging: `./native/local_build_wrapper.sh` (keeps the glibc-2.28 floor). Never ship a `build.sh` artifact.
- **`EtMethodMeta`'s JNI constructor signature is a hardcoded string literal** at `native/jni/executorch_djl_jni.cpp:82`. The Java field, the Java constructor, and that literal change together or class init fails.
- **JMX registration failure is never fatal.** One logged warning, no retry, load proceeds.
- **XNNPACK delegate workspace is out of scope** — `xnn_workspace_t` is opaque in the shipped `xnnpack.h`. Document the exclusion; do not approximate it.
- Java code style follows the existing files: 4-space indent, 100-column soft limit, javadoc on public members.
- **Every native build destroys the clangd index.** `.clangd` points at `native/build`, but
  `native/build.sh:92` does `rm -rf "${NATIVE_BUILD_DIR}"` unconditionally and its configure does
  **not** pass `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` (nor does `native/CMakeLists.txt` set it), so
  `compile_commands.json` is gone after *any* build — host fast path included, not just container
  runs. A container run additionally leaves the tree root-owned, so the regenerate fails until it is
  chowned. This plan runs native builds in Tasks 1, 2, 3, and 10; restore the database afterwards
  with:

  ```bash
  sudo chown -R "$(id -u):$(id -g)" native/build          # only needed after a container build
  cmake -S native -B native/build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  ```

  Configure only — no `cmake --build` needed, the database is written at configure time.

---

### Task 1: Native — capture the planned activation arena at load

ExecuTorch can report the size of its memory-planned activation arena, but we never read it. This adds it to the `MethodMeta` snapshot that is already built once at load and cached, so it costs nothing at steady state.

**Files:**
- Modify: `native/core/et_runtime.h` (the `MethodMeta` struct, around line 36-46)
- Modify: `native/core/et_runtime.cpp` (`buildMethodMeta`, around line 48-62)
- Test: `native/test/et_runtime_test.cpp`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `measly::et::MethodMeta::plannedArenaBytes` — a `size_t`, defaulted to `0`.

- [ ] **Step 1: Write the failing test**

Add to `native/test/et_runtime_test.cpp`, immediately after the existing `TEST_CASE("methodMeta: declared input byte counts are captured at load")`:

```cpp
TEST_CASE("methodMeta: the planned activation arena is captured at load") {
  EtRuntime rt(ADD_PTE_PATH);
  MethodMeta meta = rt.methodMeta();
  // add.pte is memory-planned (the export default), so ExecuTorch allocates a planned arena for
  // its activations. Exact bytes are an ExecuTorch planning detail we deliberately do not pin.
  REQUIRE(meta.plannedArenaBytes > 0);
}

TEST_CASE("methodMeta: the planned arena excludes the XNNPACK delegate workspace") {
  // Documents a known limitation as an executable fact: the number we report is the ExecuTorch
  // planned arena only. xnn_workspace_t is opaque in the shipped xnnpack.h (create/release only),
  // so the delegate workspace cannot be added here. See the runtime-dist issue in Task 9.
  EtRuntime rt(ADD_PTE_PATH);
  MethodMeta meta = rt.methodMeta();
  REQUIRE(meta.plannedArenaBytes < 64u * 1024u * 1024u);  // an arena, not a whole workspace
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./native/local_build_wrapper.sh native/build_qa.sh`

Expected: compile FAIL — `'struct measly::et::MethodMeta' has no member named 'plannedArenaBytes'`.

- [ ] **Step 3: Add the field**

In `native/core/et_runtime.h`, inside `struct MethodMeta`, after the `inputNbytes` member:

```cpp
  // Sum of MethodMeta::memory_planned_buffer_size(i) over num_memory_planned_buffers(), captured
  // once at load. This is ExecuTorch's planned activation arena for "forward". It does NOT include
  // the XNNPACK delegate workspace: xnn_workspace_t is opaque in the shipped xnnpack.h (create and
  // release only, no size accessor), and under the default `global` sharing mode that workspace is
  // not per-model in any case. Treat this as an exact lower bound on native footprint, not a total.
  size_t plannedArenaBytes = 0;
```

- [ ] **Step 4: Populate it**

In `native/core/et_runtime.cpp`, inside `buildMethodMeta`, after the `out.inputMemoryPlanned.resize(n, 0);` line:

```cpp
  // memory_planned_buffer_size returns Result<int64_t>; a failing entry contributes nothing rather
  // than failing the whole load, because an unreadable arena size must never break model loading.
  size_t arena = 0;
  for (size_t b = 0; b < meta->num_memory_planned_buffers(); ++b) {
    auto planned = meta->memory_planned_buffer_size(b);
    if (planned.ok()) {
      arena += static_cast<size_t>(*planned);
    }
  }
  out.plannedArenaBytes = arena;
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `./native/local_build_wrapper.sh native/build_qa.sh`

Expected: PASS, all Catch2 cases green under ASan/LSan.

If `plannedArenaBytes > 0` fails, do **not** weaken the assertion. Add a temporary `WARN(meta.plannedArenaBytes);` and re-run to see the real value — a zero there means `num_memory_planned_buffers()` returned 0, which contradicts `inputMemoryPlanned == {true, true}` on the same model and is a genuine bug to chase, not a test to relax.

- [ ] **Step 6: Fix container file ownership**

`build_qa.sh` runs as root and does not chown its outputs back.

```bash
sudo chown -R "$(id -u):$(id -g)" native/asan native/build
cmake -S native -B native/build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

The second command restores the clangd database that the build wiped (see Global Constraints).

- [ ] **Step 7: Commit**

```bash
git add native/core/et_runtime.h native/core/et_runtime.cpp native/test/et_runtime_test.cpp
git commit -m "feat(native): capture planned activation arena bytes in MethodMeta"
```

---

### Task 2: Native — expose current staging buffer bytes

Staging slots are the engine's own allocation: one per input, sized at load from the `.pte`'s declared bound, grow-only. This is the second exactly-measurable component of native footprint.

**Files:**
- Modify: `native/core/et_runtime.h` (the `EtRuntime` class, public section)
- Modify: `native/core/et_runtime.cpp` (after `EtRuntime::methodMeta`, around line 174)
- Test: `native/test/et_runtime_test.cpp`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `size_t measly::et::EtRuntime::stagingBytes() const` — total bytes across all staging slots.

- [ ] **Step 1: Write the failing test**

Add to `native/test/et_runtime_test.cpp`, after the existing `TEST_CASE("staging: slots are sized at load, so repeated forwards never grow")`:

```cpp
TEST_CASE("stagingBytes: zero for an all-planned model") {
  // add.pte's inputs are memory-planned, so no slot is ever allocated. Zero here is the correct
  // answer, not a missing measurement -- callers distinguish it from -1 ("unavailable").
  EtRuntime rt(ADD_PTE_PATH);
  REQUIRE(rt.stagingBytes() == 0);
}

TEST_CASE("stagingBytes: sums every slot of an unplanned model and is stable across forwards") {
  EtRuntime rt(ADD_UNPLANNED_PTE_PATH);
  // Each slot is ensure(nbytes + kStagingPadding), rounded up to a 64-byte multiple by StagingSlot.
  const size_t perSlot = ((sizeof(float) + kStagingPadding + 63) / 64) * 64;
  REQUIRE(rt.stagingBytes() == 2 * perSlot);

  float a = 2.0f, b = 3.0f;
  std::vector<InputDesc> inputs = {{&a, {1}, 6}, {&b, {1}, 6}};
  ForwardResult result = rt.forward(inputs);
  REQUIRE(*static_cast<const float*>(result.outputs()[0].data) == 5.0f);
  // Slots are sized at load, so a forward must not change the total. If this ever fails, the
  // grow-only invariant that makes steady-state allocation-free has been broken.
  REQUIRE(rt.stagingBytes() == 2 * perSlot);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./native/local_build_wrapper.sh native/build_qa.sh`

Expected: compile FAIL — `'class measly::et::EtRuntime' has no member named 'stagingBytes'`.

- [ ] **Step 3: Declare the accessor**

In `native/core/et_runtime.h`, in the `EtRuntime` public section, after `MethodMeta methodMeta() const;`:

```cpp
  // Total bytes currently held by this runtime's input staging slots. Returns 0 when every input
  // is memory-planned (the ExecuTorch export default) -- planned inputs are never staged, so their
  // slots stay at capacity 0. Cold path: O(numInputs), intended for a monitoring poll, not the
  // forward path.
  size_t stagingBytes() const;
```

- [ ] **Step 4: Implement it**

In `native/core/et_runtime.cpp`, immediately after `MethodMeta EtRuntime::methodMeta() const { return state_->meta; }`:

```cpp
size_t EtRuntime::stagingBytes() const {
  size_t total = 0;
  for (const auto& slot : state_->staging) {
    total += slot->capacity();
  }
  return total;
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `./native/local_build_wrapper.sh native/build_qa.sh`

Expected: PASS.

- [ ] **Step 6: Fix container file ownership**

```bash
sudo chown -R "$(id -u):$(id -g)" native/asan native/build
cmake -S native -B native/build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

The second command restores the clangd database that the build wiped (see Global Constraints).

- [ ] **Step 7: Commit**

```bash
git add native/core/et_runtime.h native/core/et_runtime.cpp native/test/et_runtime_test.cpp
git commit -m "feat(native): add EtRuntime::stagingBytes()"
```

---

### Task 3: JNI — bridge both native values to Java

This is the task with the cached-method-ID footgun. `g_metaCtor` is looked up once at `JNI_OnLoad` using a hardcoded signature string; changing `EtMethodMeta`'s constructor without changing that string fails at class init with a null method ID.

**Files:**
- Modify: `src/main/java/org/measly/executorch/jni/EtMethodMeta.java`
- Modify: `src/main/java/org/measly/executorch/jni/EtNative.java`
- Modify: `native/jni/executorch_djl_jni.cpp` (line 82 signature, line 171 `NewObject`, new function at end)
- Test: `src/test/java/org/measly/executorch/jni/EtMethodMetaTest.java`
- Test: `src/test/java/org/measly/executorch/jni/EtNativeStagingBytesTest.java` (create)

**Interfaces:**
- Consumes: `MethodMeta::plannedArenaBytes` (Task 1), `EtRuntime::stagingBytes()` (Task 2).
- Produces:
  - `public final long EtMethodMeta.plannedArenaBytes`
  - `public EtMethodMeta(int numInputs, int[] inputScalarTypes, boolean[] inputMemoryPlanned, long plannedArenaBytes)`
  - `public static native long EtNative.stagingBytes(long handle)`

- [ ] **Step 1: Write the failing tests**

Append to `src/test/java/org/measly/executorch/jni/EtMethodMetaTest.java`, inside the class:

```java
    @Test
    void capturesPlannedArenaBytes() {
        TestSupport.assumeNativeAvailable();
        long handle = EtNative.loadModule(TestSupport.addPtePath(), -1);
        try {
            EtMethodMeta meta = EtNative.methodMeta(handle);
            assertTrue(
                    meta.plannedArenaBytes > 0,
                    "add.pte is memory-planned, so it must report a planned arena");
        } finally {
            EtNative.destroy(handle);
        }
    }
```

Add the import `import static org.junit.jupiter.api.Assertions.assertTrue;` to that file.

Create `src/test/java/org/measly/executorch/jni/EtNativeStagingBytesTest.java`:

```java
package org.measly.executorch.jni;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.measly.executorch.TestSupport;
import org.junit.jupiter.api.Test;

class EtNativeStagingBytesTest {

    @Test
    void allPlannedModelStagesNothing() {
        TestSupport.assumeNativeAvailable();
        long handle = EtNative.loadModule(TestSupport.addPtePath(), -1);
        try {
            // 0 is the correct answer for a memory-planned model, not a failed measurement.
            assertEquals(0L, EtNative.stagingBytes(handle));
        } finally {
            EtNative.destroy(handle);
        }
    }

    @Test
    void unplannedModelReportsSlotBytes() {
        TestSupport.assumeUnplannedModelAvailable();
        long handle = EtNative.loadModule(TestSupport.addUnplannedPtePath(), -1);
        try {
            // Two f32 inputs, each slot padded and rounded to 64 bytes: 192 each, 384 total.
            assertEquals(384L, EtNative.stagingBytes(handle));
        } finally {
            EtNative.destroy(handle);
        }
    }

    @Test
    void stagingBytesIsStableAcrossMetadataQueries() {
        TestSupport.assumeUnplannedModelAvailable();
        long handle = EtNative.loadModule(TestSupport.addUnplannedPtePath(), -1);
        try {
            long first = EtNative.stagingBytes(handle);
            EtNative.methodMeta(handle);
            assertEquals(first, EtNative.stagingBytes(handle));
            assertTrue(first > 0);
        } finally {
            EtNative.destroy(handle);
        }
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./gradlew test --tests 'org.measly.executorch.jni.EtMethodMetaTest' \
               --tests 'org.measly.executorch.jni.EtNativeStagingBytesTest'
```

Expected: compile FAIL — `cannot find symbol: variable plannedArenaBytes` and `method stagingBytes`.

- [ ] **Step 3: Add the Java field and the native declaration**

In `src/main/java/org/measly/executorch/jni/EtMethodMeta.java`, add the field after `inputMemoryPlanned` and extend the constructor:

```java
    /**
     * ExecuTorch's planned activation arena for {@code forward}, in bytes, captured at load.
     *
     * <p>Excludes the XNNPACK delegate workspace, which cannot be sized from this layer:
     * {@code xnn_workspace_t} is opaque in the shipped {@code xnnpack.h}. Treat this as an exact
     * lower bound on native footprint, not a total.
     */
    public final long plannedArenaBytes;

    public EtMethodMeta(
            int numInputs,
            int[] inputScalarTypes,
            boolean[] inputMemoryPlanned,
            long plannedArenaBytes) {
        this.numInputs = numInputs;
        this.inputScalarTypes = inputScalarTypes;
        this.inputMemoryPlanned = inputMemoryPlanned;
        this.plannedArenaBytes = plannedArenaBytes;
    }
```

(Delete the old three-argument constructor body — do not overload. A stale overload would let the JNI signature drift undetected, which is the exact failure this task exists to avoid.)

In `src/main/java/org/measly/executorch/jni/EtNative.java`, after the `methodMeta` declaration:

```java
    /**
     * Total bytes currently held by the runtime's input staging slots.
     *
     * <p>Returns 0 when every input is memory-planned (the export default) — planned inputs are
     * never staged. Callers must not pass a destroyed handle; doing so is a use-after-free.
     *
     * @param handle the native handle
     * @return staging bytes, or 0 for an all-planned model
     */
    public static native long stagingBytes(long handle);
```

- [ ] **Step 4: Update the JNI shim — all three sites together**

In `native/jni/executorch_djl_jni.cpp`:

Line 82, the cached constructor signature. `J` is the JNI type code for `long`:

```cpp
  g_metaCtor = env->GetMethodID(g_etMethodMetaClass, "<init>", "(I[I[ZJ)V");
```

Line 171, the `NewObject` call:

```cpp
  return env->NewObject(g_etMethodMetaClass, g_metaCtor, static_cast<jint>(n), types, planned,
                        static_cast<jlong>(meta.plannedArenaBytes));
```

And a new function at the end of the file, next to the other `EtNative` entry points:

```cpp
extern "C" JNIEXPORT jlong JNICALL
Java_org_measly_executorch_jni_EtNative_stagingBytes(JNIEnv* env, jclass, jlong handle) {
  auto* rt = reinterpret_cast<EtRuntime*>(handle);
  try {
    return static_cast<jlong>(rt->stagingBytes());
  } catch (const std::exception& e) {
    throwJava(env, "stagingBytes failed", &e);
    return -1;
  }
}
```

- [ ] **Step 5: Rebuild the shim and stage it**

```bash
./native/build.sh
```

Expected: build succeeds and stages `libexecutorch_djl.so` into `src/main/resources/native/linux-x86_64/`.

This is the local fast path and breaks the glibc-2.28 floor — correct for running tests, never for a release.

It also wiped `native/build`, so restore the clangd database before doing any more C++ editing (no chown needed — this was a host build, not a container one):

```bash
cmake -S native -B native/build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
./gradlew test --tests 'org.measly.executorch.jni.EtMethodMetaTest' \
               --tests 'org.measly.executorch.jni.EtNativeStagingBytesTest'
```

Expected: PASS.

A `NoSuchMethodError` or a null-`g_metaCtor` abort at class init means Step 4's signature string does not match the Java constructor. Recheck `"(I[I[ZJ)V"` against the parameter list.

- [ ] **Step 7: Run the full suite to catch collateral damage**

```bash
./gradlew test
```

Expected: PASS. Any other caller of the old three-argument `EtMethodMeta` constructor surfaces here.

- [ ] **Step 8: Commit and push so winbox can fetch it**

```bash
git add src/main/java/org/measly/executorch/jni/ native/jni/executorch_djl_jni.cpp \
        src/test/java/org/measly/executorch/jni/
git commit -m "feat(jni): expose plannedArenaBytes and stagingBytes to Java"
git push -u origin feat/production-observability
```

- [ ] **Step 9: Verify the JNI signature on Windows (winbox, over SSH)**

**Why this step exists here and not in Task 10.** The Windows Catch2 suite links only the
**JNIEnv-free core** — it never touches `executorch_djl_jni.cpp`, so it **cannot** catch a
`g_metaCtor` signature mismatch. The only thing that proves the signature on MSVC is running the
**JVM** test suite on Windows. Deferring that to Task 10 would stack seven tasks on top of an
unverified ABI change. winbox has JDK 17 (Zulu 17.0.19), so it can run the real proof now.

Drive winbox in **short commands with `</dev/null`**, not one long-timeout script — a long remote
invocation over this link tends to stall rather than fail cleanly.

**The remote shell is `pwsh` 7.6.4 (PSEdition Core), not cmd and not Windows PowerShell 5.1.**
sshd's `DefaultShell` is pinned to
`C:\Program Files\WindowsApps\Microsoft.PowerShell_7.6.4.0_x64__8wekyb3d8bbwe\pwsh.exe`. Write
commands in PowerShell 7 syntax — cmd forms fail outright (`if exist ... (...)` produces
`Missing '(' after 'if'`). Two 7.x-vs-5.1 differences to keep in mind: `&&`/`||` chain operators
work in 7.x but not 5.1, and redirection defaults to UTF-8 no BOM in 7.x versus UTF-16LE in 5.1 —
so never generate a build input by redirecting through `powershell.exe`, which resolves to 5.1.

Fetch the branch. The checkout is behind and carries local modifications (`native/build_qa.sh`,
untracked `native/tests/check_windows_crt.sh`), so stash rather than discard:

```bash
ssh winbox "cd C:\\Users\\cored\\workspace\\djl-executorch-engine; git stash push -u -m preplan; git fetch origin; git checkout feat/production-observability; git pull --ff-only" </dev/null
```

Build the shim. `build.sh` does not activate Visual Studio itself — the caller must already have
the MSVC dev shell active, and Git-Bash must be invoked by explicit path (PATH order can otherwise
pick WSL's `bash.exe`) with `--noprofile` so the profile does not reset the VS environment:

```bash
ssh winbox "& 'C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\Common7\\Tools\\Launch-VsDevShell.ps1' -Arch amd64; cd C:\\Users\\cored\\workspace\\djl-executorch-engine; & 'C:\\Program Files\\Git\\bin\\bash.exe' --noprofile -c './native/build.sh'" </dev/null
```

Expected: compiles clean and stages `executorch_djl.dll` into
`src\main\resources\native\windows-x86_64\`.

Watch for two MSVC-specific failures that GCC does not produce: a narrowing warning on
`static_cast<jlong>(meta.plannedArenaBytes)` (`size_t` is 64-bit on win64, so this should be
silent — a warning here means the cast was dropped), and `C1189 "You need C++17 to compile
ExecuTorch"` (means the `CMAKE_CXX_STANDARD` lines in `native/CMakeLists.txt` were touched).

Then run the JVM tests — this is the actual signature proof:

```bash
ssh winbox "cd C:\\Users\\cored\\workspace\\djl-executorch-engine; .\\gradlew.bat test --tests 'org.measly.executorch.jni.*'" </dev/null
```

Expected: PASS. A `NoSuchMethodError` or a JVM abort during `EtNative` class init means
`GetMethodID(g_etMethodMetaClass, "<init>", "(I[I[ZJ)V")` returned null — the signature does not
match the Java constructor, and Linux masked it only because both were rebuilt together there.

Confirm the CRT gate still passes over the rebuilt tree:

```bash
ssh winbox "cd C:\\Users\\cored\\workspace\\djl-executorch-engine; & 'C:\\Program Files\\Git\\bin\\bash.exe' --noprofile -c './native/tests/check_windows_crt.sh'" </dev/null
```

Expected: PASS. MSVC does **not** reliably diagnose a CRT mismatch — no `LNK2038`, not even an
`LNK4098` — so this script is the only real gate on the `/MT` static-CRT requirement.

Record the outcome in the task notes. If any of the three fail, fix on the Linux host, push, and
re-run this step before starting Task 4 — do not proceed with an unverified ABI change underneath
seven dependent tasks.

---

### Task 4: Record the loaded native library path

The configuration group of the snapshot reports which library file is actually loaded. `LibUtils` resolves it through three different paths (env override, cache hit, cache miss + extract) and currently keeps none of them.

**Files:**
- Modify: `src/main/java/org/measly/executorch/engine/LibUtils.java`
- Test: `src/test/java/org/measly/executorch/engine/LibUtilsTest.java`

**Interfaces:**
- Consumes: nothing.
- Produces: `static String LibUtils.loadedPath()` — the absolute path passed to `System.load`, or `null` if the library is not loaded. Package-private to `org.measly.executorch.engine`.

- [ ] **Step 1: Write the failing test**

Append to `src/test/java/org/measly/executorch/engine/LibUtilsTest.java`, inside the class:

```java
    @Test
    void loadedPathIsRecordedAfterLoad() {
        TestSupport.assumeNativeLibraryAvailable();
        String path = LibUtils.loadedPath();
        assertNotNull(path, "loadedPath must be set once the library is loaded");
        assertTrue(
                path.endsWith(LibUtils.libName(LibUtils.platform())),
                "loadedPath must point at the platform's library file, got: " + path);
    }
```

Add the imports this needs: `import static org.junit.jupiter.api.Assertions.assertNotNull;`, `import static org.junit.jupiter.api.Assertions.assertTrue;`, and `import org.measly.executorch.TestSupport;` (skip any already present).

- [ ] **Step 2: Run the test to verify it fails**

```bash
./gradlew test --tests 'org.measly.executorch.engine.LibUtilsTest'
```

Expected: compile FAIL — `cannot find symbol: method loadedPath()`.

- [ ] **Step 3: Record the path at every load site**

In `src/main/java/org/measly/executorch/engine/LibUtils.java`, add the field next to `loaded`:

```java
    private static boolean loaded;
    // Absolute path handed to System.load, for the observability snapshot. Written under the same
    // synchronized(loadLibrary) as `loaded`, so a reader that sees loaded==true sees this too.
    private static String loadedPath;
```

Set it in the env-override branch, replacing `System.load(override); loaded = true;`:

```java
            System.load(override);
            loadedPath = override;
            loaded = true;
            return;
```

And in the extraction branch, replacing `System.load(target.toAbsolutePath().toString()); loaded = true;`:

```java
            String absolute = target.toAbsolutePath().toString();
            System.load(absolute);
            loadedPath = absolute;
            loaded = true;
```

Then add the accessor after `loadLibrary()`:

```java
    /**
     * The absolute path of the native library handed to {@code System.load}, or {@code null} if
     * the library has not been loaded. Reported by the observability snapshot so an operator can
     * tell an {@code EXECUTORCH_LIBRARY_PATH} override from a classpath extraction.
     *
     * @return the loaded library path, or {@code null}
     */
    static synchronized String loadedPath() {
        return loadedPath;
    }
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
./gradlew test --tests 'org.measly.executorch.engine.LibUtilsTest'
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/main/java/org/measly/executorch/engine/LibUtils.java \
        src/test/java/org/measly/executorch/engine/LibUtilsTest.java
git commit -m "feat: record the loaded native library path for introspection"
```

---

### Task 5: Per-model counters on the forward path

The hot-path change. `forwardInternal()` gains one `nanoTime` pair and three `volatile long` stores — nothing else.

**Files:**
- Create: `src/main/java/org/measly/executorch/engine/EtModelCounters.java`
- Test: `src/test/java/org/measly/executorch/engine/EtModelCountersTest.java` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: package-private `EtModelCounters` in `org.measly.executorch.engine` with:
  - `EtModelCounters(String name, String workspaceSharingMode, long plannedArenaBytes, long loadNanos)`
  - `void recordForward(long nanos)`
  - getters: `String name()`, `String workspaceSharingMode()`, `long plannedArenaBytes()`, `long loadNanos()`, `long forwardCount()`, `long forwardTotalNanos()`, `long forwardMaxNanos()`

- [ ] **Step 1: Write the failing test**

Create `src/test/java/org/measly/executorch/engine/EtModelCountersTest.java`:

```java
package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class EtModelCountersTest {

    private static EtModelCounters counters() {
        return new EtModelCounters("add", "global", 4096L, 1_000_000L);
    }

    @Test
    void startsAtZeroAndKeepsLoadTimeMetadata() {
        EtModelCounters c = counters();
        assertEquals("add", c.name());
        assertEquals("global", c.workspaceSharingMode());
        assertEquals(4096L, c.plannedArenaBytes());
        assertEquals(1_000_000L, c.loadNanos());
        assertEquals(0L, c.forwardCount());
        assertEquals(0L, c.forwardTotalNanos());
        assertEquals(0L, c.forwardMaxNanos());
    }

    @Test
    void accumulatesCountAndTotal() {
        EtModelCounters c = counters();
        c.recordForward(100L);
        c.recordForward(250L);
        c.recordForward(50L);
        assertEquals(3L, c.forwardCount());
        assertEquals(400L, c.forwardTotalNanos());
    }

    @Test
    void tracksTheMaximumNotTheLatest() {
        EtModelCounters c = counters();
        c.recordForward(100L);
        c.recordForward(900L);
        c.recordForward(200L); // a later smaller sample must not lower the peak
        assertEquals(900L, c.forwardMaxNanos());
    }

    @Test
    void recordsAZeroDurationSampleAsAnObservation() {
        // A clock with coarse resolution can legitimately report 0. It still counts as a forward.
        EtModelCounters c = counters();
        c.recordForward(0L);
        assertEquals(1L, c.forwardCount());
        assertEquals(0L, c.forwardTotalNanos());
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./gradlew test --tests 'org.measly.executorch.engine.EtModelCountersTest'
```

Expected: compile FAIL — `cannot find symbol: class EtModelCounters`.

- [ ] **Step 3: Implement the counters**

Create `src/main/java/org/measly/executorch/engine/EtModelCounters.java`:

```java
package org.measly.executorch.engine;

/**
 * Mutable per-model counters, updated on the forward path and read by the observability snapshot.
 *
 * <p><b>Single-writer by design.</b> {@code EtSymbolBlock.forward()} is not safe for concurrent
 * calls on the same model — the engine's contract is one {@code Model}/{@code Predictor} per
 * thread — so exactly one thread ever calls {@link #recordForward(long)} for a given instance.
 * That is what lets the accumulators be plain read-modify-writes with no CAS and no lock.
 *
 * <p>The fields are {@code volatile} for the reader's sake, not the writer's: a snapshot taken on
 * another thread must observe the updates and must never see a torn 64-bit value. A {@code
 * LongAdder} would be strictly worse here — it allocates cells and makes the read a summation, and
 * there is no write contention for it to relieve.
 */
final class EtModelCounters {

    private final String name;
    private final String workspaceSharingMode;
    private final long plannedArenaBytes;
    private final long loadNanos;

    private volatile long forwardCount;
    private volatile long forwardTotalNanos;
    private volatile long forwardMaxNanos;

    EtModelCounters(
            String name, String workspaceSharingMode, long plannedArenaBytes, long loadNanos) {
        this.name = name;
        this.workspaceSharingMode = workspaceSharingMode;
        this.plannedArenaBytes = plannedArenaBytes;
        this.loadNanos = loadNanos;
    }

    /**
     * Records one completed forward. Called only from the model's owning thread.
     *
     * @param nanos the measured wall duration of the native forward call
     */
    void recordForward(long nanos) {
        forwardCount = forwardCount + 1;
        forwardTotalNanos = forwardTotalNanos + nanos;
        if (nanos > forwardMaxNanos) {
            forwardMaxNanos = nanos;
        }
    }

    String name() {
        return name;
    }

    String workspaceSharingMode() {
        return workspaceSharingMode;
    }

    long plannedArenaBytes() {
        return plannedArenaBytes;
    }

    long loadNanos() {
        return loadNanos;
    }

    long forwardCount() {
        return forwardCount;
    }

    long forwardTotalNanos() {
        return forwardTotalNanos;
    }

    long forwardMaxNanos() {
        return forwardMaxNanos;
    }
}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
./gradlew test --tests 'org.measly.executorch.engine.EtModelCountersTest'
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/main/java/org/measly/executorch/engine/EtModelCounters.java \
        src/test/java/org/measly/executorch/engine/EtModelCountersTest.java
git commit -m "feat: add per-model forward counters"
```

---

### Task 6: Wire counters into EtSymbolBlock and produce per-model stats

`EtSymbolBlock` owns the native handle and makes the forward call, so it is the only place that can time the call and the only place that can safely read `stagingBytes` without racing `close()`.

**Files:**
- Create: `src/main/java/org/measly/executorch/engine/EtModelStats.java`
- Modify: `src/main/java/org/measly/executorch/engine/EtSymbolBlock.java`
- Test: `src/test/java/org/measly/executorch/engine/EtSymbolBlockStatsTest.java` (create)

**Interfaces:**
- Consumes: `EtModelCounters` (Task 5), `EtNative.stagingBytes` (Task 3).
- Produces:
  - `public final class EtModelStats` — immutable, MXBean-compatible bean with getters `getName()`, `getWorkspaceSharingMode()`, `getPlannedArenaBytes()`, `getStagingBytes()`, `getLoadNanos()`, `getForwardCount()`, `getForwardTotalNanos()`, `getForwardMaxNanos()`.
  - `void EtSymbolBlock.attachCounters(EtModelCounters counters)` — package-private.
  - `EtModelStats EtSymbolBlock.toStats()` — package-private; returns `null` if no counters are attached.

- [ ] **Step 1: Write the failing test**

Create `src/test/java/org/measly/executorch/engine/EtSymbolBlockStatsTest.java`:

```java
package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import java.nio.file.Paths;
import org.measly.executorch.TestSupport;
import org.junit.jupiter.api.Test;

class EtSymbolBlockStatsTest {

    @Test
    void countsForwardsAndReportsNativeBytes() throws Exception {
        TestSupport.assumeNativeAvailable();
        try (Model model = Model.newInstance("add", "ExecuTorch")) {
            model.load(Paths.get("native/spike"), "add");
            EtSymbolBlock block = (EtSymbolBlock) model.getBlock();

            EtModelStats before = block.toStats();
            assertNotNull(before, "counters must be attached at load");
            assertEquals(0L, before.getForwardCount());
            assertTrue(before.getPlannedArenaBytes() > 0);
            assertEquals(0L, before.getStagingBytes(), "add.pte is memory-planned");
            assertTrue(before.getLoadNanos() > 0);

            NDManager manager = model.getNDManager();
            for (int i = 0; i < 3; i++) {
                try (NDArray a = manager.create(new float[] {2.0f});
                        NDArray b = manager.create(new float[] {3.0f});
                        NDList in = new NDList(a, b);
                        NDList out = block.forward(null, in, false)) {
                    assertEquals(5.0f, out.head().toFloatArray()[0], 1e-6f);
                }
            }

            EtModelStats after = block.toStats();
            assertEquals(3L, after.getForwardCount());
            assertTrue(after.getForwardTotalNanos() > 0);
            assertTrue(after.getForwardMaxNanos() > 0);
            assertTrue(after.getForwardMaxNanos() <= after.getForwardTotalNanos());
        }
    }

    @Test
    void reportsUnavailableBytesAfterClose() throws Exception {
        TestSupport.assumeNativeAvailable();
        Model model = Model.newInstance("add", "ExecuTorch");
        model.load(Paths.get("native/spike"), "add");
        EtSymbolBlock block = (EtSymbolBlock) model.getBlock();
        model.close();
        // The handle is gone. Querying native staging bytes now would be a use-after-free, so the
        // guard must report -1 ("unavailable") rather than 0 ("genuinely zero") or crashing.
        assertEquals(-1L, block.toStats().getStagingBytes());
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./gradlew test --tests 'org.measly.executorch.engine.EtSymbolBlockStatsTest'
```

Expected: compile FAIL — `cannot find symbol: method toStats()`.

- [ ] **Step 3: Create the immutable stats bean**

Create `src/main/java/org/measly/executorch/engine/EtModelStats.java`:

```java
package org.measly.executorch.engine;

/**
 * An immutable point-in-time view of one loaded model's counters and native footprint.
 *
 * <p>Getters follow JavaBean naming because this type is exposed through an MXBean, which converts
 * it to {@code CompositeData} automatically. Do not add setters.
 *
 * <p><b>Byte fields use {@code -1} for "unavailable" and {@code 0} for "genuinely zero".</b> The
 * distinction matters: {@link #getStagingBytes()} is legitimately {@code 0} for a memory-planned
 * model, which is every model exported with ExecuTorch's defaults.
 */
public final class EtModelStats {

    private final String name;
    private final String workspaceSharingMode;
    private final long plannedArenaBytes;
    private final long stagingBytes;
    private final long loadNanos;
    private final long forwardCount;
    private final long forwardTotalNanos;
    private final long forwardMaxNanos;

    EtModelStats(
            String name,
            String workspaceSharingMode,
            long plannedArenaBytes,
            long stagingBytes,
            long loadNanos,
            long forwardCount,
            long forwardTotalNanos,
            long forwardMaxNanos) {
        this.name = name;
        this.workspaceSharingMode = workspaceSharingMode;
        this.plannedArenaBytes = plannedArenaBytes;
        this.stagingBytes = stagingBytes;
        this.loadNanos = loadNanos;
        this.forwardCount = forwardCount;
        this.forwardTotalNanos = forwardTotalNanos;
        this.forwardMaxNanos = forwardMaxNanos;
    }

    /** @return the DJL model name */
    public String getName() {
        return name;
    }

    /** @return the effective XNNPACK workspace sharing mode, or {@code unspecified} */
    public String getWorkspaceSharingMode() {
        return workspaceSharingMode;
    }

    /**
     * @return ExecuTorch's planned activation arena in bytes. Excludes the XNNPACK delegate
     *     workspace, which cannot be sized from this layer.
     */
    public long getPlannedArenaBytes() {
        return plannedArenaBytes;
    }

    /**
     * @return bytes held by the engine's input staging slots; {@code 0} when every input is
     *     memory-planned (the export default), {@code -1} if the model is closed
     */
    public long getStagingBytes() {
        return stagingBytes;
    }

    /** @return wall time spent loading this model, including delegate initialisation */
    public long getLoadNanos() {
        return loadNanos;
    }

    /** @return completed forward calls */
    public long getForwardCount() {
        return forwardCount;
    }

    /** @return summed wall time of all forward calls; divide by count for the mean */
    public long getForwardTotalNanos() {
        return forwardTotalNanos;
    }

    /** @return the slowest single forward observed */
    public long getForwardMaxNanos() {
        return forwardMaxNanos;
    }
}
```

- [ ] **Step 4: Time the forward and expose the stats**

In `src/main/java/org/measly/executorch/engine/EtSymbolBlock.java`, add the field next to `meta`:

```java
    // Attached by EtModel.load right after construction. Null only in the narrow window before
    // that, and in tests that build a block directly.
    private volatile EtModelCounters counters;
```

Replace the single line `EtTensor[] out = EtNative.forward(handle, in);` with:

```java
        final long startNanos = System.nanoTime();
        EtTensor[] out = EtNative.forward(handle, in);
        EtModelCounters c = counters;
        if (c != null) {
            c.recordForward(System.nanoTime() - startNanos);
        }
```

Then add these methods next to `isClosed()`:

```java
    /** Attaches the counters this block updates on each forward. Called once, at load. */
    void attachCounters(EtModelCounters counters) {
        this.counters = counters;
    }

    /**
     * Builds an immutable snapshot of this model's counters and native footprint.
     *
     * <p>Reads {@code handle} once into a local: a concurrent {@code close()} would otherwise let
     * a zero-check pass and then hand a freed pointer to native code. A closed block reports
     * {@code -1} staging bytes, meaning "unavailable" — distinct from a memory-planned model's
     * genuine {@code 0}.
     *
     * @return the snapshot, or {@code null} if no counters were ever attached
     */
    EtModelStats toStats() {
        EtModelCounters c = counters;
        if (c == null) {
            return null;
        }
        final long h = handle;
        long staging = -1L;
        if (h != 0) {
            try {
                staging = EtNative.stagingBytes(h);
            } catch (RuntimeException e) {
                staging = -1L; // a monitoring read must never propagate
            }
        }
        return new EtModelStats(
                c.name(),
                c.workspaceSharingMode(),
                c.plannedArenaBytes(),
                staging,
                c.loadNanos(),
                c.forwardCount(),
                c.forwardTotalNanos(),
                c.forwardMaxNanos());
    }
```

- [ ] **Step 5: Attach counters at load**

In `src/main/java/org/measly/executorch/engine/EtModel.java`, replace the block from `long handle = EtNative.loadModule(...)` through `block = new EtSymbolBlock(...)` with:

```java
        // Timed from here so loadNanos covers delegate initialisation: EtRuntime's constructor
        // calls Module::load_forward() unconditionally, so the XNNPACK setup cost lands in load,
        // not in the first forward.
        final long loadStartNanos = System.nanoTime();
        long handle = EtNative.loadModule(modelFile.toString(), workspaceSharingMode);
        EtMethodMeta meta;
        try {
            meta = EtNative.methodMeta(handle);
        } catch (RuntimeException e) {
            EtNative.destroy(handle); // don't leak the native module if metadata query fails
            throw e;
        }
        final long loadNanos = System.nanoTime() - loadStartNanos;
        EtSymbolBlock etBlock = new EtSymbolBlock(handle, (EtNDManager) manager, meta);
        etBlock.attachCounters(
                new EtModelCounters(
                        getName(),
                        EtWorkspaceSharing.name(workspaceSharingMode),
                        meta.plannedArenaBytes,
                        loadNanos));
        block = etBlock;
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
./gradlew test --tests 'org.measly.executorch.engine.EtSymbolBlockStatsTest'
```

Expected: PASS.

- [ ] **Step 7: Run the full suite**

```bash
./gradlew test
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/main/java/org/measly/executorch/engine/EtModelStats.java \
        src/main/java/org/measly/executorch/engine/EtSymbolBlock.java \
        src/main/java/org/measly/executorch/engine/EtModel.java \
        src/test/java/org/measly/executorch/engine/EtSymbolBlockStatsTest.java
git commit -m "feat: time forwards and expose per-model native footprint"
```

---

### Task 7: The registry and the snapshot API

The public entry point. A live-model registry plus a closed-model rollup, so a restart-on-error loop cannot erase throughput history.

**Files:**
- Create: `src/main/java/org/measly/executorch/engine/EtStatsSnapshot.java`
- Create: `src/main/java/org/measly/executorch/engine/EtEngineStats.java`
- Modify: `src/main/java/org/measly/executorch/engine/EtModel.java` (register at load)
- Modify: `src/main/java/org/measly/executorch/engine/EtSymbolBlock.java` (deregister at close)
- Test: `src/test/java/org/measly/executorch/engine/EtEngineStatsTest.java` (create)

**Interfaces:**
- Consumes: `EtModelStats`, `EtSymbolBlock.toStats()` (Task 6), `LibUtils.loadedPath()` (Task 4).
- Produces:
  - `public final class EtStatsSnapshot` with getters `getExecutorchVersion()`, `getPlatform()`, `getNativeLibraryPath()`, `getIntraOpThreads()`, `getDefaultWorkspaceSharingMode()`, `getModelsLoaded()`, `getModelsLive()`, `getTotalPlannedArenaBytes()`, `getTotalStagingBytes()`, `getClosedForwardCount()`, `getClosedForwardTotalNanos()`, `getModels()` returning `List<EtModelStats>`.
  - `public static EtStatsSnapshot EtEngineStats.snapshot()`
  - package-private `static void EtEngineStats.register(long handle, EtSymbolBlock block)` and `static void EtEngineStats.deregister(long handle)`

- [ ] **Step 1: Write the failing test**

Create `src/test/java/org/measly/executorch/engine/EtEngineStatsTest.java`:

```java
package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.djl.Model;
import java.nio.file.Paths;
import java.util.List;
import org.measly.executorch.TestSupport;
import org.junit.jupiter.api.Test;

class EtEngineStatsTest {

    @Test
    void reportsConfigurationWithoutAnyModelLoaded() {
        TestSupport.assumeNativeLibraryAvailable();
        EtStatsSnapshot s = EtEngineStats.snapshot();
        assertEquals("1.3.1", s.getExecutorchVersion());
        assertNotNull(s.getPlatform());
        assertNotNull(s.getNativeLibraryPath());
        assertTrue(s.getIntraOpThreads() >= 1);
        assertNotNull(s.getDefaultWorkspaceSharingMode());
    }

    @Test
    void tracksALiveModelThenRollsItUpOnClose() throws Exception {
        TestSupport.assumeNativeAvailable();
        long loadedBefore = EtEngineStats.snapshot().getModelsLoaded();

        Model model = Model.newInstance("add", "ExecuTorch");
        model.load(Paths.get("native/spike"), "add");

        EtStatsSnapshot live = EtEngineStats.snapshot();
        assertEquals(loadedBefore + 1, live.getModelsLoaded());
        assertTrue(live.getModelsLive() >= 1);
        assertTrue(live.getTotalPlannedArenaBytes() > 0);
        List<EtModelStats> models = live.getModels();
        assertTrue(
                models.stream().anyMatch(m -> "add".equals(m.getName())),
                "the live model must appear in the per-model list");

        long liveCount = live.getModelsLive();
        model.close();

        EtStatsSnapshot closed = EtEngineStats.snapshot();
        assertEquals(liveCount - 1, closed.getModelsLive());
        assertEquals(
                loadedBefore + 1,
                closed.getModelsLoaded(),
                "cumulative loads must not decrease when a model closes");
        assertTrue(
                closed.getModels().stream().noneMatch(m -> "add".equals(m.getName())),
                "a closed model must leave the per-model list");
    }

    @Test
    void snapshotIsAnIndependentCopy() throws Exception {
        TestSupport.assumeNativeAvailable();
        try (Model model = Model.newInstance("add", "ExecuTorch")) {
            model.load(Paths.get("native/spike"), "add");
            EtStatsSnapshot first = EtEngineStats.snapshot();
            int sizeBefore = first.getModels().size();
            try (Model second = Model.newInstance("add2", "ExecuTorch")) {
                second.load(Paths.get("native/spike"), "add");
                // The earlier snapshot must not observe the later load.
                assertEquals(sizeBefore, first.getModels().size());
                assertEquals(sizeBefore + 1, EtEngineStats.snapshot().getModels().size());
            }
        }
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./gradlew test --tests 'org.measly.executorch.engine.EtEngineStatsTest'
```

Expected: compile FAIL — `cannot find symbol: class EtEngineStats`.

- [ ] **Step 3: Create the snapshot type**

Create `src/main/java/org/measly/executorch/engine/EtStatsSnapshot.java`:

```java
package org.measly.executorch.engine;

import java.util.List;

/**
 * An immutable point-in-time view of the ExecuTorch engine: its effective configuration,
 * process-wide totals, and per-model detail for every live model.
 *
 * <p>Obtained from {@link EtEngineStats#snapshot()}. Getters follow JavaBean naming because this
 * type is exposed through an MXBean.
 */
public final class EtStatsSnapshot {

    private final String executorchVersion;
    private final String platform;
    private final String nativeLibraryPath;
    private final int intraOpThreads;
    private final String defaultWorkspaceSharingMode;
    private final long modelsLoaded;
    private final long modelsLive;
    private final long totalPlannedArenaBytes;
    private final long totalStagingBytes;
    private final long closedForwardCount;
    private final long closedForwardTotalNanos;
    private final List<EtModelStats> models;

    EtStatsSnapshot(
            String executorchVersion,
            String platform,
            String nativeLibraryPath,
            int intraOpThreads,
            String defaultWorkspaceSharingMode,
            long modelsLoaded,
            long modelsLive,
            long totalPlannedArenaBytes,
            long totalStagingBytes,
            long closedForwardCount,
            long closedForwardTotalNanos,
            List<EtModelStats> models) {
        this.executorchVersion = executorchVersion;
        this.platform = platform;
        this.nativeLibraryPath = nativeLibraryPath;
        this.intraOpThreads = intraOpThreads;
        this.defaultWorkspaceSharingMode = defaultWorkspaceSharingMode;
        this.modelsLoaded = modelsLoaded;
        this.modelsLive = modelsLive;
        this.totalPlannedArenaBytes = totalPlannedArenaBytes;
        this.totalStagingBytes = totalStagingBytes;
        this.closedForwardCount = closedForwardCount;
        this.closedForwardTotalNanos = closedForwardTotalNanos;
        this.models = models;
    }

    /** @return the pinned ExecuTorch runtime version */
    public String getExecutorchVersion() {
        return executorchVersion;
    }

    /** @return the resolved platform, e.g. {@code linux-x86_64}, or {@code unknown} */
    public String getPlatform() {
        return platform;
    }

    /**
     * @return the native library file actually loaded, or {@code unknown}. Distinguishes an
     *     {@code EXECUTORCH_LIBRARY_PATH} override from a classpath extraction.
     */
    public String getNativeLibraryPath() {
        return nativeLibraryPath;
    }

    /** @return the effective intra-op pool size as reported by the native pool */
    public int getIntraOpThreads() {
        return intraOpThreads;
    }

    /**
     * @return the JVM-wide default workspace sharing mode, or {@code unspecified} when no default
     *     is set and the runtime's compiled-in default ({@code global} for our pin) applies
     */
    public String getDefaultWorkspaceSharingMode() {
        return defaultWorkspaceSharingMode;
    }

    /** @return models loaded since JVM start, cumulative; never decreases */
    public long getModelsLoaded() {
        return modelsLoaded;
    }

    /** @return models currently loaded and not yet closed */
    public long getModelsLive() {
        return modelsLive;
    }

    /** @return summed planned activation arenas of all live models; excludes delegate workspace */
    public long getTotalPlannedArenaBytes() {
        return totalPlannedArenaBytes;
    }

    /** @return summed staging bytes of all live models; {@code 0} when all inputs are planned */
    public long getTotalStagingBytes() {
        return totalStagingBytes;
    }

    /** @return forwards completed by models that have since been closed */
    public long getClosedForwardCount() {
        return closedForwardCount;
    }

    /** @return summed forward wall time of models that have since been closed */
    public long getClosedForwardTotalNanos() {
        return closedForwardTotalNanos;
    }

    /** @return per-model detail for live models only; an unmodifiable, independent copy */
    public List<EtModelStats> getModels() {
        return models;
    }
}
```

- [ ] **Step 4: Create the registry**

Create `src/main/java/org/measly/executorch/engine/EtEngineStats.java`:

```java
package org.measly.executorch.engine;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import org.measly.executorch.jni.EtNative;

/**
 * The engine's production monitoring surface: an always-on, fixed-cost view of throughput, native
 * footprint, and effective configuration.
 *
 * <p>{@link #snapshot()} is a cold-path read — walk it from a scheduled poll, an HTTP health
 * endpoint, or a JMX console. It never throws: values that cannot be read degrade to {@code -1}
 * (bytes) or {@code unknown} (strings) rather than propagating a failure out of a monitoring call.
 *
 * <p><b>Relationship to DJL's {@code Metrics}.</b> {@code Predictor.setMetrics(...)} records
 * per-{@code predict} timings, but it is a time-series buffer built for benchmarking, not a
 * production counter: its {@code limit} defaults to 0 (uncapped, so samples are retained forever
 * unless you wire both {@code setLimit} and {@code setOnLimit}), its {@code addMetric} is a
 * check-then-act race at the flush boundary, and {@code percentile()} sorts the whole buffer on
 * every call. Use it for profiling; use this class for production monitoring.
 *
 * <p>Per-model detail covers live models only. A model's totals fold into the closed-model rollup
 * when it closes, so a restart-on-error loop cannot erase throughput history.
 */
public final class EtEngineStats {

    private static final Map<Long, EtSymbolBlock> LIVE = new ConcurrentHashMap<>();
    private static final AtomicLong MODELS_LOADED = new AtomicLong();
    private static final AtomicLong CLOSED_FORWARD_COUNT = new AtomicLong();
    private static final AtomicLong CLOSED_FORWARD_TOTAL_NANOS = new AtomicLong();

    private static final String UNKNOWN = "unknown";

    private EtEngineStats() {}

    /** Records a newly loaded model. Called from {@link EtModel#load}. */
    static void register(long handle, EtSymbolBlock block) {
        LIVE.put(handle, block);
        MODELS_LOADED.incrementAndGet();
    }

    /**
     * Removes a model and folds its totals into the closed-model rollup. Called from {@link
     * EtSymbolBlock#close()} <b>before</b> the native handle is released, so the counters are still
     * readable. Idempotent: a second close finds nothing to remove.
     */
    static void deregister(long handle) {
        EtSymbolBlock block = LIVE.remove(handle);
        if (block == null) {
            return;
        }
        EtModelStats stats = block.toStats();
        if (stats != null) {
            CLOSED_FORWARD_COUNT.addAndGet(stats.getForwardCount());
            CLOSED_FORWARD_TOTAL_NANOS.addAndGet(stats.getForwardTotalNanos());
        }
    }

    /**
     * Captures the engine's current state.
     *
     * @return an immutable snapshot; never {@code null}, never throws
     */
    public static EtStatsSnapshot snapshot() {
        List<EtModelStats> models = new ArrayList<>(LIVE.size());
        long arena = 0;
        long staging = 0;
        for (EtSymbolBlock block : LIVE.values()) {
            EtModelStats stats = block.toStats();
            if (stats == null) {
                continue; // registered but counters not yet attached; nothing to report
            }
            models.add(stats);
            if (stats.getPlannedArenaBytes() > 0) {
                arena += stats.getPlannedArenaBytes();
            }
            if (stats.getStagingBytes() > 0) {
                staging += stats.getStagingBytes(); // skips -1 so "unavailable" never sums in
            }
        }
        return new EtStatsSnapshot(
                EtEngine.EXECUTORCH_VERSION,
                safePlatform(),
                safeString(LibUtils.loadedPath()),
                safeIntraOpThreads(),
                sharingModeDefault(),
                MODELS_LOADED.get(),
                models.size(),
                arena,
                staging,
                CLOSED_FORWARD_COUNT.get(),
                CLOSED_FORWARD_TOTAL_NANOS.get(),
                Collections.unmodifiableList(models));
    }

    private static String safeString(String value) {
        return (value == null || value.isEmpty()) ? UNKNOWN : value;
    }

    private static String safePlatform() {
        try {
            return LibUtils.platform();
        } catch (RuntimeException e) {
            return UNKNOWN; // unsupported os.arch: reportable, not fatal to a monitoring read
        }
    }

    private static String sharingModeDefault() {
        String value = System.getProperty(EtEngine.WORKSPACE_SHARING_MODE_PROPERTY);
        // "unspecified" is meaningful here and distinct from "unknown": it means no spec is sent
        // and the runtime's compiled-in default (global for our pin) applies.
        return (value == null || value.isEmpty()) ? "unspecified" : value;
    }

    private static int safeIntraOpThreads() {
        try {
            return EtNative.intraOpThreads();
        } catch (RuntimeException | UnsatisfiedLinkError e) {
            return -1; // native library unavailable
        }
    }
}
```

- [ ] **Step 5: Wire registration and deregistration**

In `src/main/java/org/measly/executorch/engine/EtModel.java`, after the `block = etBlock;` line added in Task 6:

```java
        EtEngineStats.register(handle, etBlock);
```

In `src/main/java/org/measly/executorch/engine/EtSymbolBlock.java`, change `close()` to deregister **before** releasing the handle, so the rollup can still read the counters:

```java
    @Override
    public void close() {
        if (handle != 0) {
            // Before destroy: deregister reads this block's counters for the closed-model rollup,
            // and toStats() would report -1 staging bytes once the handle is zeroed.
            EtEngineStats.deregister(handle);
            EtNative.destroy(handle);
            handle = 0;
        }
    }
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
./gradlew test --tests 'org.measly.executorch.engine.EtEngineStatsTest'
```

Expected: PASS.

- [ ] **Step 7: Run the full suite**

```bash
./gradlew test
```

Expected: PASS. `EtSymbolBlockLifecycleTest.repeatedLoadCloseDoesNotDegrade` loops 100 load/close cycles and will catch a registry leak.

- [ ] **Step 8: Commit**

```bash
git add src/main/java/org/measly/executorch/engine/EtStatsSnapshot.java \
        src/main/java/org/measly/executorch/engine/EtEngineStats.java \
        src/main/java/org/measly/executorch/engine/EtModel.java \
        src/main/java/org/measly/executorch/engine/EtSymbolBlock.java \
        src/test/java/org/measly/executorch/engine/EtEngineStatsTest.java
git commit -m "feat: add EtEngineStats snapshot API with live-model registry"
```

---

### Task 8: JMX MXBean with auto-registration

An MXBean, not a plain MBean: the JMX runtime converts `EtStatsSnapshot` and `List<EtModelStats>` to `CompositeData`/`TabularData` automatically, so no `OpenType` code is needed.

**Files:**
- Create: `src/main/java/org/measly/executorch/engine/EtEngineStatsMXBean.java`
- Modify: `src/main/java/org/measly/executorch/engine/EtEngineStats.java`
- Modify: `src/main/java/org/measly/executorch/engine/EtModel.java` (trigger registration at load)
- Test: `src/test/java/org/measly/executorch/engine/EtEngineStatsJmxTest.java` (create)

**Interfaces:**
- Consumes: `EtEngineStats.snapshot()`, `EtStatsSnapshot` (Task 7).
- Produces:
  - `public interface EtEngineStatsMXBean { EtStatsSnapshot getSnapshot(); }`
  - `public static final String EtEngine.JMX_ENABLED_PROPERTY = "ai.djl.executorch.jmx_enabled"`
  - `public static void EtEngineStats.registerMBean()` / `unregisterMBean()`
  - package-private `static void EtEngineStats.registerMBeanOnce()`
  - `public static final String EtEngineStats.OBJECT_NAME = "org.measly.executorch:type=EtEngineStats"`

- [ ] **Step 1: Write the failing test**

Create `src/test/java/org/measly/executorch/engine/EtEngineStatsJmxTest.java`:

```java
package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.management.ManagementFactory;
import javax.management.MBeanServer;
import javax.management.ObjectName;
import javax.management.openmbean.CompositeData;
import org.measly.executorch.TestSupport;
import org.junit.jupiter.api.Test;

class EtEngineStatsJmxTest {

    private static ObjectName name() throws Exception {
        return new ObjectName(EtEngineStats.OBJECT_NAME);
    }

    @Test
    void registersAndExposesTheSnapshotAsCompositeData() throws Exception {
        TestSupport.assumeNativeLibraryAvailable();
        EtEngineStats.registerMBean();
        try {
            MBeanServer server = ManagementFactory.getPlatformMBeanServer();
            assertTrue(server.isRegistered(name()));
            // An MXBean converts the value type automatically; if EtStatsSnapshot is not a
            // conforming bean this throws instead of returning CompositeData.
            Object value = server.getAttribute(name(), "Snapshot");
            assertNotNull(value);
            CompositeData data = (CompositeData) value;
            assertEquals("1.3.1", data.get("executorchVersion"));
            assertNotNull(data.get("models"));
        } finally {
            EtEngineStats.unregisterMBean();
        }
    }

    @Test
    void registrationIsIdempotent() throws Exception {
        TestSupport.assumeNativeLibraryAvailable();
        EtEngineStats.registerMBean();
        try {
            // A second call must not throw InstanceAlreadyExistsException out to the caller.
            assertDoesNotThrow(EtEngineStats::registerMBean);
            assertTrue(ManagementFactory.getPlatformMBeanServer().isRegistered(name()));
        } finally {
            EtEngineStats.unregisterMBean();
        }
    }

    @Test
    void unregisterIsSafeWhenNotRegistered() {
        assertDoesNotThrow(EtEngineStats::unregisterMBean);
        assertDoesNotThrow(EtEngineStats::unregisterMBean);
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./gradlew test --tests 'org.measly.executorch.engine.EtEngineStatsJmxTest'
```

Expected: compile FAIL — `cannot find symbol: variable OBJECT_NAME`.

- [ ] **Step 3: Declare the MXBean interface**

Create `src/main/java/org/measly/executorch/engine/EtEngineStatsMXBean.java`:

```java
package org.measly.executorch.engine;

/**
 * JMX view of {@link EtEngineStats}, registered as {@value EtEngineStats#OBJECT_NAME}.
 *
 * <p>An <b>MX</b>Bean rather than a plain MBean: the JMX runtime converts {@link EtStatsSnapshot}
 * and its nested {@code List<EtModelStats>} to {@code CompositeData}/{@code TabularData}
 * automatically, so no hand-written {@code OpenType} mapping is needed. Keeping that conversion
 * working is why both value types are getter-only JavaBeans.
 */
public interface EtEngineStatsMXBean {

    /** @return a fresh snapshot of engine configuration, totals, and live models */
    EtStatsSnapshot getSnapshot();
}
```

- [ ] **Step 4: Add the property constant**

In `src/main/java/org/measly/executorch/engine/EtEngine.java`, after `WORKSPACE_SHARING_MODE_PROPERTY`:

```java
    /**
     * JVM flag controlling whether the engine registers its JMX MBean, e.g.
     * {@code -Dai.djl.executorch.jmx_enabled=false}. Registration happens once, at the first model
     * load, under the object name {@value EtEngineStats#OBJECT_NAME}. Any value other than
     * {@code false} (case-insensitive) leaves it enabled.
     *
     * <p>Registration failure — a name collision, a {@code SecurityManager}, a restricted
     * container — is a single logged warning and never fails a model load.
     */
    public static final String JMX_ENABLED_PROPERTY = "ai.djl.executorch.jmx_enabled";
```

- [ ] **Step 5: Implement registration**

Add to `src/main/java/org/measly/executorch/engine/EtEngineStats.java`. New imports:

```java
import java.lang.management.ManagementFactory;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.management.MBeanServer;
import javax.management.ObjectName;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
```

New members, and make the class implement the bean via a private holder:

```java
    /** The JMX object name this engine registers under. */
    public static final String OBJECT_NAME = "org.measly.executorch:type=EtEngineStats";

    private static final Logger logger = LoggerFactory.getLogger(EtEngineStats.class);
    // Guards the one-shot auto-registration attempt. A failed attempt is not retried: a
    // per-load retry would log on every load and re-run a failure we already reported.
    private static final AtomicBoolean JMX_ATTEMPTED = new AtomicBoolean();

    /** MXBean implementation. Separate from the static facade because an MXBean needs an instance. */
    private static final class Bean implements EtEngineStatsMXBean {
        @Override
        public EtStatsSnapshot getSnapshot() {
            return EtEngineStats.snapshot();
        }
    }

    /**
     * Registers the JMX MBean under {@value #OBJECT_NAME} on the platform MBean server.
     *
     * <p>Idempotent: registering an already-registered name is a no-op. Any JMX failure is logged
     * and swallowed — a monitoring surface must never be the thing that breaks the application.
     */
    public static void registerMBean() {
        try {
            MBeanServer server = ManagementFactory.getPlatformMBeanServer();
            ObjectName objectName = new ObjectName(OBJECT_NAME);
            if (server.isRegistered(objectName)) {
                return;
            }
            server.registerMBean(new Bean(), objectName);
            logger.info("registered JMX MBean {}", OBJECT_NAME);
        } catch (Exception e) {
            logger.warn(
                    "could not register JMX MBean {} ({}); set {}=false to silence this",
                    OBJECT_NAME,
                    e.toString(),
                    EtEngine.JMX_ENABLED_PROPERTY);
        }
    }

    /** Removes the JMX MBean if present. Safe to call when it was never registered. */
    public static void unregisterMBean() {
        try {
            MBeanServer server = ManagementFactory.getPlatformMBeanServer();
            ObjectName objectName = new ObjectName(OBJECT_NAME);
            if (server.isRegistered(objectName)) {
                server.unregisterMBean(objectName);
            }
        } catch (Exception e) {
            logger.warn("could not unregister JMX MBean {} ({})", OBJECT_NAME, e.toString());
        }
    }

    /**
     * One-shot auto-registration, driven by the first model load. Honours {@link
     * EtEngine#JMX_ENABLED_PROPERTY}; only the exact value {@code false} disables it.
     */
    static void registerMBeanOnce() {
        if (!JMX_ATTEMPTED.compareAndSet(false, true)) {
            return;
        }
        if ("false".equalsIgnoreCase(System.getProperty(EtEngine.JMX_ENABLED_PROPERTY))) {
            logger.info("JMX MBean disabled by {}=false", EtEngine.JMX_ENABLED_PROPERTY);
            return;
        }
        registerMBean();
    }
```

- [ ] **Step 6: Trigger it at the first model load**

In `src/main/java/org/measly/executorch/engine/EtModel.java`, immediately after the `EtEngineStats.register(handle, etBlock);` line added in Task 7:

```java
        // After registration so the first JMX read already sees this model. One-shot: later loads
        // return immediately.
        EtEngineStats.registerMBeanOnce();
```

- [ ] **Step 7: Run the test to verify it passes**

```bash
./gradlew test --tests 'org.measly.executorch.engine.EtEngineStatsJmxTest'
```

Expected: PASS.

A `javax.management.NotCompliantMBeanException` means `EtStatsSnapshot` or `EtModelStats` is not a conforming bean — check for a setter, a non-getter public method, or a getter returning a type MXBean cannot map.

- [ ] **Step 8: Add the opt-out test**

The opt-out must be checked in a JVM where auto-registration has not yet fired, so it needs its own forked task. Add to `build.gradle.kts`, following the existing `intraopTest` pattern:

```kotlin
val jmxDisabledTest by tasks.registering(Test::class) {
    description = "Verifies ai.djl.executorch.jmx_enabled=false suppresses MBean registration."
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("jmx-disabled") }
    jvmArgs("-Dai.djl.executorch.jmx_enabled=false")
}
```

Add `"jmx-disabled"` to the `excludeTags(...)` list in `tasks.test` at line 33.

Create `src/test/java/org/measly/executorch/engine/EtJmxDisabledIT.java`:

```java
package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertFalse;

import ai.djl.Model;
import java.lang.management.ManagementFactory;
import java.nio.file.Paths;
import javax.management.ObjectName;
import org.measly.executorch.TestSupport;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Runs in its own JVM with jmx_enabled=false, before auto-registration can fire. */
@Tag("jmx-disabled")
class EtJmxDisabledIT {

    @Test
    void loadDoesNotRegisterTheMBean() throws Exception {
        TestSupport.assumeNativeAvailable();
        try (Model model = Model.newInstance("add", "ExecuTorch")) {
            model.load(Paths.get("native/spike"), "add");
        }
        assertFalse(
                ManagementFactory.getPlatformMBeanServer()
                        .isRegistered(new ObjectName(EtEngineStats.OBJECT_NAME)),
                "jmx_enabled=false must suppress auto-registration");
    }
}
```

- [ ] **Step 9: Run both test tasks**

```bash
./gradlew test jmxDisabledTest
```

Expected: PASS for both.

- [ ] **Step 10: Commit**

```bash
git add src/main/java/org/measly/executorch/engine/ build.gradle.kts \
        src/test/java/org/measly/executorch/engine/EtEngineStatsJmxTest.java \
        src/test/java/org/measly/executorch/engine/EtJmxDisabledIT.java
git commit -m "feat: expose engine stats over JMX with opt-out"
```

---

### Task 9: Concurrency safety and hot-path overhead verification

Two claims the design makes that are worth nothing unless measured: that `snapshot()` is safe while forwards run, and that the counters do not move steady-state throughput.

**Files:**
- Create: `src/test/java/org/measly/executorch/stress/StatsConcurrencyIT.java`
- Test command: `./gradlew stressGate`, `./gradlew :example:jmh`

**Interfaces:**
- Consumes: `EtEngineStats.snapshot()` (Task 7), `PerThreadContext` (existing, `src/test/java/org/measly/executorch/stress/PerThreadContext.java`).
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Read the existing stress harness**

Read `src/test/java/org/measly/executorch/stress/StressGateIT.java` and `PerThreadContext.java`. `PerThreadContext` is the reference pattern: **one `ZooModel` per thread**, never a shared model behind a `ThreadLocal`. Follow it exactly — a shared model would make this test measure a contract violation rather than the snapshot.

- [ ] **Step 2: Write the concurrency test**

Create `src/test/java/org/measly/executorch/stress/StatsConcurrencyIT.java`:

```java
package org.measly.executorch.stress;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.measly.executorch.TestSupport;
import org.measly.executorch.engine.EtEngineStats;
import org.measly.executorch.engine.EtModelStats;
import org.measly.executorch.engine.EtStatsSnapshot;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Polls the stats snapshot while several threads forward on their own models. Tagged {@code
 * stress} and excluded from CI: it saturates every core for its duration.
 */
@Tag("stress")
class StatsConcurrencyIT {

    private static final int THREADS = 4;
    private static final int FORWARDS_PER_THREAD = 500;

    @Test
    void snapshotIsSafeWhileForwardsRun() throws Exception {
        TestSupport.assumeNativeAvailable();
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(THREADS);
        AtomicBoolean running = new AtomicBoolean(true);
        AtomicReference<Throwable> failure = new AtomicReference<>();
        List<Thread> workers = new ArrayList<>();

        for (int t = 0; t < THREADS; t++) {
            Thread worker =
                    new Thread(
                            () -> {
                                // One Model per thread: forward() is not safe on a shared model.
                                try (Model model = Model.newInstance("add", "ExecuTorch")) {
                                    model.load(Paths.get("native/spike"), "add");
                                    NDManager manager = model.getNDManager();
                                    start.await();
                                    for (int i = 0; i < FORWARDS_PER_THREAD; i++) {
                                        try (NDArray a = manager.create(new float[] {2.0f});
                                                NDArray b = manager.create(new float[] {3.0f});
                                                NDList in = new NDList(a, b);
                                                NDList out =
                                                        model.getBlock()
                                                                .forward(null, in, false)) {
                                            if (out.head().toFloatArray()[0] != 5.0f) {
                                                throw new AssertionError("wrong result");
                                            }
                                        }
                                    }
                                } catch (Throwable e) {
                                    failure.compareAndSet(null, e);
                                } finally {
                                    done.countDown();
                                }
                            });
            worker.start();
            workers.add(worker);
        }

        Thread poller =
                new Thread(
                        () -> {
                            while (running.get()) {
                                try {
                                    EtStatsSnapshot s = EtEngineStats.snapshot();
                                    assertNotNull(s.getModels());
                                    for (EtModelStats m : s.getModels()) {
                                        // Counters are monotonic and internally consistent: a
                                        // torn 64-bit read would surface as a negative or as a
                                        // max exceeding the total.
                                        assertTrue(m.getForwardCount() >= 0);
                                        assertTrue(m.getForwardTotalNanos() >= 0);
                                        assertTrue(
                                                m.getForwardMaxNanos()
                                                        <= m.getForwardTotalNanos());
                                    }
                                } catch (Throwable e) {
                                    failure.compareAndSet(null, e);
                                    return;
                                }
                            }
                        });
        poller.start();

        start.countDown();
        assertTrue(done.await(5, TimeUnit.MINUTES), "workers did not finish");
        running.set(false);
        poller.join(TimeUnit.SECONDS.toMillis(30));
        for (Thread worker : workers) {
            worker.join(TimeUnit.SECONDS.toMillis(30));
        }

        Throwable t = failure.get();
        if (t != null) {
            throw new AssertionError("concurrent snapshot/forward failed", t);
        }

        EtStatsSnapshot end = EtEngineStats.snapshot();
        assertTrue(
                end.getClosedForwardCount() >= (long) THREADS * FORWARDS_PER_THREAD,
                "closed-model rollup must retain the forwards of every closed model");
    }
}
```

- [ ] **Step 3: Run it**

```bash
./gradlew stressGate --tests 'org.measly.executorch.stress.StatsConcurrencyIT'
```

Expected: PASS. This saturates every core for its duration — expected and deliberate.

- [ ] **Step 4: Measure hot-path overhead**

```bash
./gradlew :example:exportModels
./gradlew :example:jmh
```

Record `MobilenetBenchmark.steadyState` for `ExecuTorch`. Compare against the pre-change baseline in `scratchpad.txt`: **~19.0–19.4 ms/op**.

The counters must not move this number beyond the reported error bars. If steady-state regresses measurably, stop and report it: the design's premise is that this measurement holds, and shipping a hot-path regression to gain a counter is the wrong trade.

- [ ] **Step 5: Record the result**

Append the before/after JMH table to the design spec under a new `## Measured overhead` section in `docs/superpowers/specs/2026-08-09-production-observability-design.md`, with the host's core count noted.

- [ ] **Step 6: Commit**

```bash
git add src/test/java/org/measly/executorch/stress/StatsConcurrencyIT.java \
        docs/superpowers/specs/2026-08-09-production-observability-design.md
git commit -m "test: concurrent snapshot safety and hot-path overhead measurement"
```

---

### Task 10: Rebuild all platforms and file the upstream issue

The native change from Tasks 1–3 has only been built on the local host so far. Every shipped platform needs a matching binary, and the documented limitation needs a real upstream ticket behind it.

**Files:**
- Modify: `src/main/resources/native/linux-x86_64/libexecutorch_djl.so` (regenerated)
- Modify: `src/main/java/org/measly/executorch/jni/EtMethodMeta.java` (issue link in javadoc)
- Modify: `src/main/java/org/measly/executorch/engine/EtModelStats.java` (issue link in javadoc)

**Interfaces:**
- Consumes: everything.
- Produces: shippable artifacts.

- [ ] **Step 1: Rebuild linux-x86_64 with the release-correct toolchain**

```bash
./native/local_build_wrapper.sh
```

This rebuilds inside `manylinux_2_28`, restoring the glibc-2.28 floor that the Task 3 `build.sh` artifact broke.

- [ ] **Step 2: Verify the glibc floor**

```bash
objdump -T src/main/resources/native/linux-x86_64/libexecutorch_djl.so \
  | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -3
```

Expected: nothing above `GLIBC_2.28`. A higher version means the container build did not take and the artifact is not shippable.

- [ ] **Step 3: Run the full JVM suite against the container-built library**

```bash
./gradlew test
```

Expected: PASS.

- [ ] **Step 4: Run the native QA suite**

```bash
./native/local_build_wrapper.sh native/build_qa.sh
sudo chown -R "$(id -u):$(id -g)" native/asan native/build
cmake -S native -B native/build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

Expected: PASS, clean under ASan/LSan. The last command restores the clangd database, which both
this run and the Step 1 container build destroyed.

- [ ] **Step 5: File the upstream issue**

```bash
gh issue create --repo measly-java-learning/executorch-runtime-dist \
  --title "Expose XNNPACK workspace size for host-side memory accounting" \
  --body "$(cat <<'EOF'
## Request

Expose the byte size of the XNNPACK workspace from the runtime distribution, so consumers can
account for it in host-side native memory reporting.

## Why

`djl-executorch-engine` now reports per-model native footprint (planned activation arena + our own
input staging buffers). Both are exact. The XNNPACK delegate workspace is the third component and
is the only one we cannot measure:

- `xnn_workspace_t` is opaque in the installed `xnnpack.h` — `xnn_create_workspace` and
  `xnn_release_workspace` only, no size accessor.
- ExecuTorch's `backends/xnnpack/runtime/XNNWorkspace.h` wrapper exposes `acquire()` and
  `unsafe_get_workspace()`, neither of which yields a size.

We therefore ship a documented lower bound rather than a total. An RSS-delta proxy was considered
and rejected as misleading under the default `global` sharing mode.

## Suggested shape

A size accessor on the `XNNWorkspace` wrapper, surfaced through the installed headers — enough to
read the current workspace allocation. A process-global figure is useful even without per-model
attribution, since under `global` sharing there is one workspace anyway.

## Context

- Consumer: https://github.com/measly-java-learning/djl-executorch-engine
- Runtime pin: v1.3.1-8
EOF
)"
```

Record the issue URL from the command output for the next step.

- [ ] **Step 6: Link the issue from the javadoc**

In `src/main/java/org/measly/executorch/jni/EtMethodMeta.java` and
`src/main/java/org/measly/executorch/engine/EtModelStats.java`, extend each sentence that mentions
the XNNPACK exclusion with the tracking link, e.g.:

```java
     * {@code xnn_workspace_t} is opaque in the shipped {@code xnnpack.h}. Tracked upstream at
     * <a href="https://github.com/measly-java-learning/executorch-runtime-dist/issues/N">
     * executorch-runtime-dist#N</a>.
```

Substitute the real issue number from Step 5.

- [ ] **Step 7: Full build**

```bash
./gradlew build
```

Expected: PASS, including the jacoco coverage report.

- [ ] **Step 8: Commit**

```bash
git add src/main/resources/native/ src/main/java/
git commit -m "chore: rebuild native shim in manylinux_2_28 and link upstream workspace issue"
```

- [ ] **Step 9: Re-verify Windows with the final code**

Task 3 proved the JNI signature on winbox against the JNI change alone. Re-run it now that the full
Java surface exists — `EtEngineStats`, the MXBean, and the registry all run on Windows too, and JMX
is the piece most likely to behave differently there:

```bash
git push
ssh winbox "cd C:\\Users\\cored\\workspace\\djl-executorch-engine; git pull --ff-only" </dev/null
ssh winbox "& 'C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\Common7\\Tools\\Launch-VsDevShell.ps1' -Arch amd64; cd C:\\Users\\cored\\workspace\\djl-executorch-engine; & 'C:\\Program Files\\Git\\bin\\bash.exe' --noprofile -c './native/build.sh'" </dev/null
ssh winbox "cd C:\\Users\\cored\\workspace\\djl-executorch-engine; .\\gradlew.bat test" </dev/null
ssh winbox "cd C:\\Users\\cored\\workspace\\djl-executorch-engine; .\\gradlew.bat jmxDisabledTest" </dev/null
```

Expected: PASS. `EtEngineStatsJmxTest` is the one to watch — it is the only coverage of the
platform MBean server outside Linux.

- [ ] **Step 10: Confirm the CI matrix**

`linux-aarch64` still cannot be built or tested on either host, so CI is its only proof:

```bash
gh run list --branch feat/production-observability --limit 5
```

Confirm all three platform legs are green before merging. The aarch64 leg is now the only
unverified one — Windows was covered directly in Task 3 and Step 9 above.

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: configuration group → Tasks 4 and 7; process totals and per-model detail → Tasks 5–7; planned arena → Task 1; staging bytes → Task 2; JNI bridge and the `g_metaCtor` footgun → Task 3; the closed-handle race → Task 6; JMX MXBean and opt-out → Task 8; `snapshot()` never throws and the `-1`/`0` convention → Tasks 6 and 7; Catch2, unit, integration, concurrency, and overhead testing → Tasks 1, 2, 3, 5, 6, 7, 8, 9; the upstream issue → Task 10; three-platform rebuild → Task 10.

**Deliberate ordering.** Native (1–2) precedes the JNI bridge (3) which precedes all Java work, because the Java tests cannot pass without a rebuilt shim. Task 10 rebuilds with the release toolchain at the end, because Task 3's `build.sh` artifact breaks the glibc floor and would otherwise ship.

**Windows verification is early by design.** The Windows Catch2 suite links only the JNIEnv-free core and therefore cannot catch a `g_metaCtor` signature mismatch — only the JVM suite can. Task 3 Step 9 runs that suite on winbox (JDK 17, VS 18 Community, both confirmed present) so the ABI change is proven before seven tasks are stacked on it; Task 10 Step 9 re-runs it against the finished surface. `linux-aarch64` remains CI-only.

**Naming consistency.** `plannedArenaBytes` and `stagingBytes` are used identically in C++ (`MethodMeta::plannedArenaBytes`, `EtRuntime::stagingBytes()`), Java fields (`EtMethodMeta.plannedArenaBytes`, `EtNative.stagingBytes`), and bean getters (`getPlannedArenaBytes()`, `getStagingBytes()`). `EtModelCounters` (mutable, package-private) and `EtModelStats` (immutable, public) are distinct types throughout and never interchanged.
