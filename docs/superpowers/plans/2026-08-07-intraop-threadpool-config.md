# Intra-op threadpool configuration — implementation plan

> Implements `docs/superpowers/specs/2026-08-06-intraop-threadpool-config-design.md` (approved, not yet implemented) on branch `feature/intraop-threadpool-config`. All design decisions in the spec are binding; this plan adds only the concrete edit steps, exact signatures, test seams, and verification. No changes to: `workspace_sharing_mode`, the forward path, `Criteria`, the runtime pin, or `example/`.

## Context

Expose ExecuTorch's intra-op (XNNPACK) threadpool size through the DJL engine. Today the pool sizes itself from the CPU's performance-core count with no external control surface, which oversubscribes the small, compute-light production hosts this engine targets. End state: one process-global setting (`-Dai.djl.executorch.num_threads=N` or `EtEngine.setIntraOpThreads(n)`) that applies exactly once, at the first `EtModel.load()`, with the applied native count observable via `EtEngine.getIntraOpThreads()`. The spec's benchmark caveat stands: this host (4-core/8-thread i7-1185G7) cannot produce a meaningful full sweep, so verification runs a small run that proves the harness executes and emits sane output through the NEW code path, not a threads×modes×reps sweep.

## Approach

Steps are ordered so the tree builds and existing tests pass after each. Dependencies: 1→2, 1→3, 1→4→5→6→7, 7→8→9; 10 (docs) is independent. Commit after each numbered group with the repo's house convention (message ends `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`); the first commit is the plan file itself, mirroring how the spec landed in `28d3037`.

**Commit 0 — plan file.** Copy this file to `docs/superpowers/plans/2026-08-07-intraop-threadpool-config.md` (house style: plans are committed alongside specs; all prior plans live there, dated). Commit message: `Plan: intra-op threadpool configuration`.

---

### Step 1 — Core: `measly::et::setIntraOpThreads` / `intraOpThreads`

**Edit `native/core/et_runtime.h`:** in `namespace measly::et`, after the `EtRuntime` class (before the closing brace), add:

```cpp
// Intra-op (XNNPACK) thread pool size, backed by ExecuTorch's process-global
// extension::threadpool singleton. The pool sizes itself to the performance-core count by
// default and nothing reads an env var (verified for v1.3.1: no getenv in extension/threadpool,
// the vendored pthreadpool, XNNPACK init, or cpuinfo), so this is the ONLY control surface.
//
// setIntraOpThreads returns the count in effect AFTER the attempt. Upstream's
// _unsafe_reset_threadpool always returns true (it early-returns for n == 0 and for
// n == get_thread_count()), so a bool status would be meaningless -- callers compare instead.
//
// Must be called before the first EtRuntime is constructed: delegate init during load submits
// work to the pool, and the reset must not race in-flight work.
uint32_t setIntraOpThreads(uint32_t n);
uint32_t intraOpThreads();
```

**Edit `native/core/et_runtime.cpp`:** add `#include <executorch/extension/threadpool/threadpool.h>` to the executorch include block (next to `module.h`), and before the closing `}  // namespace measly::et`:

```cpp
uint32_t setIntraOpThreads(uint32_t n) {
  executorch::extension::threadpool::ThreadPool* pool =
      executorch::extension::threadpool::get_threadpool();
  pool->_unsafe_reset_threadpool(n);  // documented to always return true; no-ops for 0/unchanged
  return static_cast<uint32_t>(pool->get_thread_count());
}

uint32_t intraOpThreads() {
  return static_cast<uint32_t>(
      executorch::extension::threadpool::get_threadpool()->get_thread_count());
}
```

`_unsafe_reset_threadpool` carries `[[deprecated]]` upstream — the existing harness already calls it and the build has no `-Werror`, so accept the warning (do not suppress it).

**Edit `native/CMakeLists.txt`:** in the `et_runtime` target block (currently lines ~184-189), add `extension_threadpool` to the PUBLIC link list with a one-line comment:

```cmake
target_link_libraries(et_runtime PUBLIC
  extension_module_static
  extension_tensor
  xnnpack_backend
  portable_ops_lib
  # Explicit, not transitive through xnnpack_backend: the core now calls
  # get_threadpool()/_unsafe_reset_threadpool itself (spec §5).
  extension_threadpool
)
```

**Edit `native/CMakeLists.txt`** (same file, `ET_BUILD_BENCH` block, lines ~197-201): remove `extension_threadpool` from `et_scaling_harness`'s link (et_runtime now supplies it PUBLIC) and delete the 3-line comment that justified the explicit link:

```cmake
  target_link_libraries(et_scaling_harness PRIVATE et_runtime Threads::Threads)
```

Note: nothing else in `native/` includes `threadpool.h` or calls the pool API (verified by grep — only the harness does), so Step 3 is the complete migration.

**Commit 1** (after Step 3, with the harness switch, so the tree stays coherent): `Core: expose intra-op threadpool size via measly::et`.

---

### Step 2 — Native Catch2 tests (pins upstream quirks)

**Edit `native/test/et_runtime_test.cpp`:** append two `TEST_CASE`s (the file already uses `using namespace measly::et;`):

```cpp
TEST_CASE("intraop: setIntraOpThreads resizes the shared pool and reports the applied count") {
  const uint32_t before = intraOpThreads();
  REQUIRE(setIntraOpThreads(1) == 1);
  REQUIRE(intraOpThreads() == 1);
  // The pool is process-global: restore so sibling tests run on the default pool.
  setIntraOpThreads(before);
  REQUIRE(intraOpThreads() == before);
}

TEST_CASE("intraop: upstream quirks -- 0 is silently ignored, same-count reset is a no-op") {
  const uint32_t cur = intraOpThreads();
  REQUIRE(setIntraOpThreads(0) == cur);   // upstream early-returns for 0: unchanged
  REQUIRE(intraOpThreads() == cur);
  REQUIRE(setIntraOpThreads(cur) == cur); // early-returns for the current count: unchanged
  REQUIRE(intraOpThreads() == cur);
}
```

These pin the upstream behaviors the Java layer relies on (the Java setter rejects `n < 1`, so 0 never reaches native from Java; the no-op case makes "requested ≠ applied" detectable only by comparison). If upstream ever makes these fail loudly, this test says so.

Verification of this step (and Steps 3): see Verification §V1 (QA) and §V3 (BUILD_ONLY).

---

### Step 3 — Harness switches to the core path

**Edit `native/harness/et_scaling_harness.cpp`:**
- Delete line 55: `#include <executorch/extension/threadpool/threadpool.h>` (the harness keeps the `runtime/backend/*` and `runtime/platform/runtime.h` includes).
- Replace lines 184-195 (the resize block) with:

```cpp
    // Resize the shared intra-op pool before anything runs on it, through the same core function
    // the engine ships (spec §5) -- measly::et::setIntraOpThreads cannot fail, so no status check.
    const uint32_t intraop_actual =
        intraop > 0 ? setIntraOpThreads(static_cast<uint32_t>(intraop)) : intraOpThreads();
```

  (Both functions are in `measly::et`, which the file already `using namespace`s.) The old `return 6` failure branch and the `const size_t intraop_actual = ...get_thread_count()` line disappear.
- In the final `printf`, change the `intraop=%zu` format + `size_t` arg to `intraop=%u` with the now-`uint32_t` `intraop_actual`.
- In the file-header comment, the sentence naming `get_threadpool()/_unsafe_reset_threadpool` (lines ~21-26 area) stays conceptually true; leave it — it describes the mechanism, not the call site. (No edit needed.)

**Commit 1** (with Step 1): `Core: expose intra-op threadpool size via measly::et` — covers et_runtime.h/.cpp, CMakeLists.txt, harness, and the new Catch2 tests in one commit.

---

### Step 4 — JNI shim

**Edit `native/jni/executorch_djl_jni.cpp`:** append before the closing of the file (after `Java_..._EtNative_destroy`), matching the existing `extern "C"` style:

```cpp
extern "C" JNIEXPORT jint JNICALL
Java_org_measly_executorch_jni_EtNative_setIntraOpThreads(JNIEnv* env, jclass, jint n) {
  return static_cast<jint>(measly::et::setIntraOpThreads(static_cast<uint32_t>(n)));
}

extern "C" JNIEXPORT jint JNICALL
Java_org_measly_executorch_jni_EtNative_intraOpThreads(JNIEnv* env, jclass) {
  return static_cast<jint>(measly::et::intraOpThreads());
}
```

No exception path: the core functions cannot fail. The `n < 1` guard lives in Java (Step 6), so `n` is already validated before this is reachable.

---

### Step 5 — `EtNative` declarations

**Edit `src/main/java/org/measly/executorch/jni/EtNative.java`:** add after `destroy`:

```java
/** Sizes ExecuTorch's intra-op (XNNPACK) pool; returns the count in effect after the attempt. */
public static native int setIntraOpThreads(int n);

/** Current intra-op pool size as reported by the native pool. */
public static native int intraOpThreads();
```

---

### Step 6 — `EtEngine` gate + `EtModel.load()` flush point

**Edit `src/main/java/org/measly/executorch/engine/EtEngine.java`** — add (plus `import org.slf4j.Logger; import org.slf4j.LoggerFactory;`):

```java
/**
 * JVM flag controlling the intra-op (XNNPACK) thread pool size, e.g.
 * {@code -Dai.djl.executorch.num_threads=2}. Process-global: ExecuTorch's pool is a process
 * singleton (extension::threadpool), so this is NOT a per-model option. Defaults to the
 * performance-core count (cpuinfo-derived), not nproc. There is deliberately no environment
 * variable: verified against the v1.3.1 runtime, nothing reads getenv in extension/threadpool,
 * the vendored pthreadpool, or XNNPACK init, and OMP_NUM_THREADS is inert (no OpenMP symbols).
 * The write window closes at the first model load; see {@link #setIntraOpThreads(int)}.
 */
public static final String NUM_THREADS_PROPERTY = "ai.djl.executorch.num_threads";

private static final Logger logger = LoggerFactory.getLogger(EtEngine.class);
private static final Object INTRAOP_LOCK = new Object();
// -1 = unset (leave the runtime default); >= 1 = requested. Written by the setter, read at seal.
private static int pendingIntraOpThreads = -1;
private static boolean intraOpSealed = false;
// Effective count decided at seal; -1 = runtime default. Set once, under INTRAOP_LOCK.
private static int sealedIntraOpThreads = -1;
```

Methods (all static, on the `EtEngine` class):

```java
/**
 * Sets the intra-op thread pool size. Process-global and write-once: the value is applied at
 * the first {@code EtModel.load()} (delegate init already submits work to the pool, so that is
 * the only provably safe window for upstream's reset) and later calls throw.
 *
 * @param n pool size; must be >= 1
 * @throws IllegalArgumentException if n &lt; 1 (before any JNI call)
 * @throws IllegalStateException if any model has already been loaded, naming the sealed value
 */
public static void setIntraOpThreads(int n) {
    if (n < 1) {
        throw new IllegalArgumentException(
                NUM_THREADS_PROPERTY + " must be >= 1, got " + n);
    }
    synchronized (INTRAOP_LOCK) {
        if (intraOpSealed) {
            throw new IllegalStateException(
                    "Intra-op thread pool is already sealed at "
                            + (sealedIntraOpThreads < 1 ? "the runtime default"
                                                        : String.valueOf(sealedIntraOpThreads))
                            + "; set " + NUM_THREADS_PROPERTY + " before the first model load");
        }
        pendingIntraOpThreads = n;
    }
}

/**
 * Effective intra-op pool size as reported by the native pool (get_thread_count), not the
 * requested value -- on a 40-core host the difference is the point. Triggers the native
 * library load.
 */
public static int getIntraOpThreads() {
    return EtNative.intraOpThreads();
}
```

Gate + resolution (package-private, so `EtModel` and the unit tests in the same package reach them without exposing a public API):

```java
/**
 * Pure precedence/resolution used by the seal: the setter wins over the property; a present
 * but unparseable or < 1 property is WARNed and ignored (fall back to the runtime default --
 * a typo'd JVM flag must not fail startup). Returns the effective count, or -1 for the
 * runtime default.
 */
static int resolveIntraOpThreads(int setterValue, String propertyValue) {
    if (setterValue > 0) {
        if (propertyValue != null) {
            logger.warn("{} property is ignored because setIntraOpThreads() was called",
                    NUM_THREADS_PROPERTY);
        }
        return setterValue;
    }
    if (propertyValue == null) {
        return -1;
    }
    try {
        int p = Integer.parseInt(propertyValue.trim());
        if (p >= 1) {
            return p;
        }
        logger.warn("{}={} is < 1; using the runtime default", NUM_THREADS_PROPERTY, propertyValue);
    } catch (NumberFormatException e) {
        logger.warn("{}='{}' is not an integer; using the runtime default",
                NUM_THREADS_PROPERTY, propertyValue);
    }
    return -1;
}

/**
 * Flush point, called by EtModel.load() immediately before the first native call. Under
 * INTRAOP_LOCK: resolves the pending value vs the property (the property is read HERE, at
 * first load -- not at class init -- so the precedence is testable and nothing races
 * EtNative's static initializer), applies it via JNI while still holding the lock (so
 * concurrent loadModel() calls cannot both seal, and no second load's delegate init can race
 * the reset), and marks the pool fixed. Logs the outcome at INFO on this first load.
 */
static void sealIntraOpThreads() {
    synchronized (INTRAOP_LOCK) {
        if (intraOpSealed) {
            return; // subsequent loads skip the apply; the pool is fixed
        }
        int n = resolveIntraOpThreads(
                pendingIntraOpThreads, System.getProperty(NUM_THREADS_PROPERTY));
        sealedIntraOpThreads = n;
        intraOpSealed = true;
        if (n >= 1) {
            int actual = EtNative.setIntraOpThreads(n);
            if (actual != n) {
                logger.warn("{}: requested {} but the pool reports {}; using {}",
                        NUM_THREADS_PROPERTY, n, actual, actual);
            }
            logger.info("intra-op thread pool sealed at {} ({}={})",
                    actual, NUM_THREADS_PROPERTY, n);
        } else {
            logger.info("intra-op thread pool left at the runtime default ({})",
                    EtNative.intraOpThreads());
        }
    }
}

/** Sealed effective count (-1 = runtime default). For the unit tests' ISE-message assertion. */
static int intraOpThreadCount() {
    synchronized (INTRAOP_LOCK) {
        return sealedIntraOpThreads;
    }
}

/**
 * Test seam (same package, no-native unit tests only): performs the seal bookkeeping without
 * the JNI apply. Idempotent -- safe whether or not an integration test already sealed this
 * JVM. The native pool is untouched; apply only ever happens in {@link #sealIntraOpThreads()}.
 */
static void sealIntraOpThreadsForTest() {
    synchronized (INTRAOP_LOCK) {
        if (!intraOpSealed) {
            sealedIntraOpThreads = -1;
            intraOpSealed = true;
        }
    }
}
```

Key behavior notes (decisions, not open questions):
- The property is read at seal time, not class init. Alternative (class-init parse) was rejected: it makes "setter beats property" untestable in the shared test JVM (EtEngine's class init would have already run) and changes nothing observable (JVM flags are fixed at process start).
- The JNI apply happens INSIDE the gate's critical section (apply → record → mark sealed), so concurrent `loadModel()` calls cannot both seal and no later load's delegate init can race the reset. The spec's diagram ("applies the pending value and marks the pool fixed") is satisfied with a strictly tighter window.
- A sealed engine is harmless to later loads (they skip the apply) and only makes future `setIntraOpThreads` calls throw — which is why the no-native unit tests are safe in the shared test JVM.
- Sealing with no property and no setter logs INFO with the runtime default; the JNI call is skipped (`n == -1`), so even the INFO path's `EtNative.intraOpThreads()` still triggers the native load — that is the desired lazy-load behavior.

**Edit `src/main/java/org/measly/executorch/engine/EtModel.java`:** in `load(Path, String, Map)`, immediately BEFORE `long handle = EtNative.loadModule(modelFile.toString());` (after the `FileNotFoundException` check), insert:

```java
// First load seals the process-global intra-op thread pool (applies pending/property value,
// logs the outcome); later loads are no-ops. Must precede loadModule: delegate init during
// load submits work to the pool.
EtEngine.sealIntraOpThreads();
```

`EtModel` is in the same package — no import needed.

**Commit 2** (with Steps 4-7): `Java: intra-op threadpool surface (EtEngine gate, JNI, seal on first load)`.

---

### Step 7 — Java unit tests (no native)

**Create `src/test/java/org/measly/executorch/engine/EtIntraOpThreadsTest.java`** (package `org.measly.executorch.engine`; JUnit 5, no tags, no native):

```java
package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Pure state machine for the intra-op thread pool gate; never touches the native library. */
class EtIntraOpThreadsTest {

    @Test
    void setIntraOpThreadsRejectsNonPositive() {
        assertThrows(IllegalArgumentException.class, () -> EtEngine.setIntraOpThreads(0));
        assertThrows(IllegalArgumentException.class, () -> EtEngine.setIntraOpThreads(-1));
        assertThrows(IllegalArgumentException.class, () -> EtEngine.setIntraOpThreads(-4));
    }

    @Test
    void setterAfterSealThrowsNamingSealedValue() {
        EtEngine.sealIntraOpThreadsForTest(); // idempotent: no-op if a test already sealed this JVM
        IllegalStateException e = assertThrows(
                IllegalStateException.class, () -> EtEngine.setIntraOpThreads(2));
        assertTrue(e.getMessage().contains("sealed"));
        assertTrue(e.getMessage().contains(String.valueOf(EtEngine.intraOpThreadCount())));
    }

    @Test
    void setterBeatsPropertyAndPropertyParsing() {
        assertEquals(3, EtEngine.resolveIntraOpThreads(3, "5"));     // setter wins
        assertEquals(5, EtEngine.resolveIntraOpThreads(-1, "5"));     // property alone
        assertEquals(-1, EtEngine.resolveIntraOpThreads(-1, "0"));    // < 1 -> ignore
        assertEquals(-1, EtEngine.resolveIntraOpThreads(-1, "abc"));  // unparseable -> ignore
        assertEquals(-1, EtEngine.resolveIntraOpThreads(-1, null));   // absent -> default
        assertEquals(7, EtEngine.resolveIntraOpThreads(7, "abc"));    // setter beats broken property
    }
}
```

All three tests are order-independent in the shared test JVM: the seam is idempotent, and `resolveIntraOpThreads` is pure. The WARN side effects are exercised by the integration tier / code review, not asserted here (no log-capture infra in this repo).

---

### Step 8 — Gradle: dedicated `intraOpTest` task

**Edit `build.gradle.kts`:**
1. In `tasks.test`, extend the exclusion: `useJUnitPlatform { excludeTags("leak", "oom", "intraop") }`.
2. After the `oomTest` block, add (mirroring `leakTest`):

```kotlin
tasks.register<Test>("intraOpTest") {
    description = "Intra-op threadpool configuration tests; forked JVM because the pool is process-global."
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("intraop") }
    jvmArgs("-Dai.djl.executorch.num_threads=2")
}
```

3. Wire it into the build (so `./gradlew build` includes it but `./gradlew test` does not):

```kotlin
tasks.check { dependsOn(tasks.named("intraOpTest")) }
```

The existing `tasks.withType<Test>().configureEach { inputs.property("executorchLibraryPath", ...) }` already covers the new task, so the `EXECUTORCH_LIBRARY_PATH` override participates in the up-to-date/cache key as it does for `leakTest`.

---

### Step 9 — Java integration test (property → native pool, end to end)

**Create `src/test/java/org/measly/executorch/IntraOpThreadsIT.java`** (package `org.measly.executorch`; only runs under the `intraopTest` task, which passes `-Dai.djl.executorch.num_threads=2`):

```java
package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import java.nio.file.Paths;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.measly.executorch.engine.EtEngine;

/**
 * The only test proving the property reaches the native pool end to end. Runs in a dedicated
 * forked JVM (intraOpTest task, -Dai.djl.executorch.num_threads=2): the pool is process-global,
 * so this cannot share a JVM with any other test.
 */
@Tag("intraop")
class IntraOpThreadsIT {

    @Test
    void propertySealsTheNativePoolAtRequestedSize() throws Exception {
        TestSupport.assumeNativeAvailable();
        Criteria<float[], Float> criteria =
                Criteria.builder()
                        .setTypes(float[].class, Float.class)
                        .optEngine("ExecuTorch")
                        .optModelPath(Paths.get("native/spike"))
                        .optModelName("add")
                        .optTranslator(new AddTranslator())
                        .build();
        try (ZooModel<float[], Float> model = criteria.loadModel()) {
            // The load sealed the pool: the native pool must report the property's value, not
            // the performance-core default (4 or 8 on this host -- never 2).
            assertEquals(2, EtEngine.getIntraOpThreads());
            // And the gate is closed: a setter after a real load throws, naming the sealed value.
            IllegalStateException e = assertThrows(
                    IllegalStateException.class, () -> EtEngine.setIntraOpThreads(4));
            assertEquals("2", e.getMessage().substring(
                    e.getMessage().indexOf("at ") + 3,
                    e.getMessage().indexOf("; set")));
        }
    }
}
```

(The message-extraction assertion is deliberately brittle-proof: it checks the ISE names the sealed value `2`.) `AddTranslator` and `TestSupport` already exist in the same package and are used identically by `LeakStressTest` and the `AddModelIT` family — reuse them as-is.

**Commit 3** (with Steps 8-9): `Test: intraOpTest task + end-to-end property->pool integration test`.

---

### Step 10 — Docs (spec §10, verbatim wording)

1. **`CLAUDE.md`** — in "Conventions worth knowing", immediately after the existing `EtSymbolBlock.forward()` safety-rule bullet, add the spec's §10 note (the "Threading, and why more threads is usually wrong" blockquote, verbatim from the spec, lines 201-207) plus one conventions bullet:

   ```
   - `ai.djl.executorch.num_threads` (JVM flag) or `EtEngine.setIntraOpThreads(n)` sizes ExecuTorch's intra-op (XNNPACK) threadpool. Process-global, write-once: applied and sealed at the first model load; the effective native count is `EtEngine.getIntraOpThreads()`.
   ```

2. **`src/main/java/org/measly/executorch/engine/EtSymbolBlock.java`** — in the class javadoc, after the existing concurrency paragraph, add the same §10 note (spec lines 201-207, verbatim).

3. **`EtEngine` javadoc** — already written in Step 6 (property + both methods, incl. the performance-core-count default and §7's no-environment-variable finding).

The existing safety sentence stays exactly as written (the spec says so explicitly).

**Commit 4**: `Docs: threading note (CLAUDE.md, EtSymbolBlock, EtEngine)`.

## Critical files & anchors

- `native/core/et_runtime.h` — `namespace measly::et` block; new free functions go after the `EtRuntime` class. JNIEnv-free core shared by shim, Catch2, harnesses.
- `native/core/et_runtime.cpp` — executorch include block (adds `extension/threadpool/threadpool.h`); function bodies before the closing namespace.
- `native/CMakeLists.txt:184-201` — `et_runtime` PUBLIC link list + `et_scaling_harness` link/comment.
- `native/harness/et_scaling_harness.cpp:55,184-195` — the only direct upstream-threadpool usage in the repo; migrated to the core path.
- `src/main/java/org/measly/executorch/engine/EtModel.java:48` (`long handle = EtNative.loadModule(...)`) — the seal hook point, immediately before it.
- `build.gradle.kts` — `tasks.test` `excludeTags` (line ~43), `oomTest` block (~line 60) as the `intraOpTest` template, plus a `tasks.check { dependsOn(...) }` line.
- `native/test/et_runtime_test.cpp` — Catch2 suite; new TEST_CASEs appended (existing precedent: `ProbeGuard` tests already mutate process-global state).

## Verification

All native runs go through the container wrapper (`./native/local_build_wrapper.sh`) — the harness/QA cannot link on the host (the runtime's exported config hardcodes the container's libm; see `scaling.sh` header). The wrapper leaves `native/asan/` and `native/bench/` root-owned; fix with `sudo chown -R "$(id -u):$(id -g)" native/asan native/bench` after each run. Working dir is the repo root for every command.

- **V1 — Native QA (Step 2 gate):** `./native/local_build_wrapper.sh native/build_qa.sh` → Catch2 suite must pass, including the two new `TEST_CASE`s ("intraop: setIntraOpThreads resizes...", "intraop: upstream quirks..."). This is the first proof the core functions link `extension_threadpool` and behave per spec (runs in CI via `native-build-job.yml` too).
- **V2 — Shim build (Steps 4-6 gate):** `./native/local_build_wrapper.sh` (default `native/build.sh`) → stages `libexecutorch_djl.so` into `src/main/resources/native/linux-x86_64/`. No build error from the two new JNI functions.
- **V3 — Harness compile+link check (Step 3 gate):** `BUILD_ONLY=1 ./native/local_build_wrapper.sh native/scaling.sh` → builds `native/bench/et_scaling_harness` (Release, `ET_BUILD_BENCH`) and exits before measuring. Cheap proof the harness still compiles after dropping the direct threadpool include.
- **V4 — JVM unit tests:** `./gradlew test --tests 'org.measly.executorch.engine.EtIntraOpThreadsTest'` → all three pass in the shared JVM without the native library.
- **V5 — NEW behavior, end to end (the primary acceptance check):** `./gradlew intraOpTest` (after V2) → forked JVM with `-Dai.djl.executorch.num_threads=2`; loads `native/spike/add.pte` via DJL; asserts `EtEngine.getIntraOpThreads() == 2` (native `get_thread_count()`), and that a post-load `setIntraOpThreads(4)` throws `IllegalStateException` naming `2`.
- **V6 — Regression + wiring:** `./gradlew test` (excludes `intraop`; must stay green — notably `EtEngineTest`, `AddModelIT`, `LstmModelIT`, `LoggingBridgeIT`) and `./gradlew build` (must now ALSO execute `intraOpTest` via `tasks.check` — confirm the task runs and passes in the build output).
- **V7 — Benchmark execution + sanity (the requested benchmark step; NOT a full sweep):** two small runs through the container, using the committed `native/spike/add.pte` fixture (the mobilenet fallback; `example/build/models/mobilenet_v2.pte` is absent unless `./gradlew :example:exportModels` ran):
  1. `INTRAOP=1 MODEL=native/spike/add.pte THREADS="1 2" MODES="default" ITERS=50 WARMUP=5 REPS=1 ./native/local_build_wrapper.sh native/scaling.sh`
  2. `MODEL=native/spike/add.pte THREADS="1" MODES="default" ITERS=50 WARMUP=5 REPS=1 ./native/local_build_wrapper.sh native/scaling.sh`
  Acceptance: exit 0; each row prints `et_scaling:` with `forwards_per_sec > 0`, `load_ms > 0`, `peak_rss_kb > 0`, and — the correctness check — run 1 reports `intraop=1` (the new `measly::et::setIntraOpThreads` path ran and took effect) with `parallelism` ≈ 1.0-1.5 per the spec §8 `Global`-mode rows (1.12/1.17), and run 2 reports `intraop=<default>` (4 or 8 on this host, NOT 1 or 2) with the pool untouched by the new code. A full threads×modes×reps sweep is explicitly NOT required on this host — the spec's §1 caveat ("4 physical cores is not the 18-40 core production target... ratios should not be extrapolated") applies; the full sweep command for a proper host is plain `./native/local_build_wrapper.sh native/scaling.sh` with `MODEL=example/build/models/mobilenet_v2.pte` after `./gradlew :example:exportModels`.

## Assumptions & contingencies

- **The host can run the container wrapper** (Docker + the pinned `djl-executorch-engine-build:linux-x86_64` image built from `docker/linux-x86_64.Dockerfile`). If Docker is unavailable, V1/V3/V7 cannot run on this host — fall back to CI (`native-build-job.yml` runs `build_qa.sh` on Linux+Windows; `native-build.yml` runs `./gradlew build` with the staged shim) plus `BUILD_ONLY=1` for the harness if the toolchain permits; JVM gates V4-V6 still run locally.
- **The staged shim is required for V2/V5/V6** (`src/main/resources/native/linux-x86_64/libexecutorch_djl.so`). Without it, the IT-style tests skip via `TestSupport.assumeNativeAvailable()` (existing behavior) and `intraOpTest` would report skipped — treat a skip as NOT a pass for V5.
- **Upstream `_unsafe_reset_threadpool` keeps its v1.3.1 semantics** (always true; early-return on 0/unchanged). If a future pin bump changes that, the Step 2 Catch2 tests fail loudly and the "compare not trust" design still holds.
- **`intraOpTest` naming the sealed value via string extraction** in `IntraOpThreadsIT` is tied to the exact ISE message in Step 6 — if the message changes, update both together.
