# Per-model XNNPACK Workspace Sharing Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let each loaded `.pte` choose its own XNNPACK workspace sharing mode, so a latency-sensitive model can run unserialised while other models keep sharing one arena.

**Architecture:** ExecuTorch 1.3.1 resolves the sharing mode per delegate at method-load time, preferring a per-load runtime spec over the process global. We pass that spec through `Module::load(const LoadBackendOptionsMap&)` at our single `module.load()` site. A DJL per-model option (`Criteria.optOption`) resolves to an int in Java, crosses JNI as a parameter on `loadModule`, and reaches `EtRuntime`'s constructor. The process-global `set_option` path is never used, so there is no seal, no load-order dependence, and no forked test JVM.

**Tech Stack:** C++20 (ExecuTorch 1.3.1 runtime, pin `1.3.1-8`), JNI, Java 17, DJL 0.36.0, Catch2, JUnit 5, Gradle, CMake.

**Spec:** `docs/superpowers/specs/2026-08-08-workspace-sharing-mode-design.md`

## Global Constraints

- The ExecuTorch runtime is **downloaded, not built** here. Do not edit `native/cmake/EtRuntimePin.cmake`. No pin bump is required by this work — every header used is already in the shipped tarball.
- `backends/xnnpack/runtime/XNNPACKBackend.h` is **not installed**. The backend id and option key must be hardcoded as string literals with a comment naming that upstream header.
- The backend id is exactly `"XnnpackBackend"` — not `"XNNPACKBackend"`. A mismatch is silently ignored by `Method::load` (empty span), not an error.
- The option key is exactly `"workspace_sharing_mode"`.
- Mode encoding: `0` = Disabled, `1` = PerModel, `2` = Global, `-1` = omit the spec entirely. `-1` is **not** a synonym for `2`.
- Public string values: `disabled`, `per_model`, `global` — case-insensitive, trimmed. Bare integers are **not** accepted in the Java API.
- Do **not** expose `weight_cache_enabled`. It reintroduces a process-global mutex across all of `execute()`.
- Native builds must go through `./native/local_build_wrapper.sh` (manylinux_2_28, preserves the glibc-2.28 floor). `native/build.sh` on the host is a local fast path only.
- `native/build_qa.sh` runs Catch2 with `--order decl`. Declaration order in `et_runtime_test.cpp` is load-bearing: the intra-op tests at the top of that file must run before any `EtRuntime` is constructed. **Append new test cases at the end of the file.**

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `native/core/et_runtime.h` | Modify | Add the sharing-mode parameter to the `EtRuntime` constructor |
| `native/core/et_runtime.cpp` | Modify | Build the `LoadBackendOptionsMap` and choose the `load()` overload |
| `native/test/et_runtime_test.cpp` | Modify (append) | Catch2 coverage, incl. the negative-control wiring proof |
| `src/main/java/org/measly/executorch/engine/EtWorkspaceSharing.java` | Create | Sole owner of the string↔int mapping and option/property precedence |
| `src/test/java/org/measly/executorch/engine/EtWorkspaceSharingTest.java` | Create | Pure unit tests for the above |
| `native/jni/executorch_djl_jni.cpp` | Modify | Pass the mode through `loadModule` |
| `src/main/java/org/measly/executorch/jni/EtNative.java` | Modify | `loadModule` signature |
| `src/main/java/org/measly/executorch/engine/EtModel.java` | Modify | Resolve at load and pass the int down |
| `src/test/java/org/measly/executorch/WorkspaceSharingIT.java` | Create | End-to-end load through `Criteria` |
| `CLAUDE.md`, `EtSymbolBlock.java` | Modify | Docs, incl. correcting the now-conditional threading note |

`EtWorkspaceSharing` is a separate class rather than more static methods on `EtEngine`: `EtEngine` already carries the intra-op seal state machine, and these two knobs have opposite lifetimes (process-global write-once vs. per-model). Keeping them apart stops a reader from assuming this one also seals.

---

### Task 1: Native core accepts a per-model sharing mode

**Files:**
- Modify: `native/core/et_runtime.h:68`
- Modify: `native/core/et_runtime.cpp:73-81`
- Test: `native/test/et_runtime_test.cpp` (append at end)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `measly::et::EtRuntime::EtRuntime(const std::string& ptePath, int workspaceSharingMode = -1)`. The default argument keeps every existing call site (harnesses, other tests) compiling unchanged.

- [ ] **Step 1: Write the failing tests**

Append to the **end** of `native/test/et_runtime_test.cpp` (after the existing `"intraop: a reset after a runtime exists is a logged no-op"` case — `--order decl` means position matters):

```cpp
// Per-model XNNPACK workspace sharing (spec 2026-08-08). add.pte is XNNPACK-delegated (its
// delegate id string is "XnnpackBackend"), so the runtime spec is actually consumed here.
TEST_CASE("workspace: every valid sharing mode loads") {
  REQUIRE_NOTHROW([] { EtRuntime rt(ADD_PTE_PATH, 0); }());  // Disabled
  REQUIRE_NOTHROW([] { EtRuntime rt(ADD_PTE_PATH, 1); }());  // PerModel
  REQUIRE_NOTHROW([] { EtRuntime rt(ADD_PTE_PATH, 2); }());  // Global
}

TEST_CASE("workspace: omitting the mode (-1) loads on the runtime default") {
  REQUIRE_NOTHROW([] { EtRuntime rt(ADD_PTE_PATH, -1); }());
  REQUIRE_NOTHROW([] { EtRuntime rt(ADD_PTE_PATH); }());  // same thing via the default argument
}

// THE WIRING PROOF. XnnpackBackendOptions::resolve_sharing_mode returns InvalidArgument for an
// out-of-range mode and XnnpackBackend::init propagates it, so the load fails -- but ONLY if the
// spec actually reached the XNNPACK backend. If the backend id or the option key were misspelled,
// Method::load would hand the backend an empty span, the mode would silently stay at the default,
// and this load would SUCCEED. There is no read-back API that would otherwise catch that: the
// backend's get_option returns the process-global value, not the per-model resolved one, and
// init does not log the mode it resolved. Do not delete this test as a mere bad-input check.
TEST_CASE("workspace: an out-of-range mode is rejected by the backend (proves the spec lands)") {
  REQUIRE_THROWS([] { EtRuntime rt(ADD_PTE_PATH, 99); }());
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./native/local_build_wrapper.sh native/build_qa.sh
```

Expected: **compile failure**, `no matching constructor for initialization of 'EtRuntime'` — the two-argument constructor does not exist yet.

- [ ] **Step 3: Add the parameter to the header**

In `native/core/et_runtime.h`, replace line 68:

```cpp
  explicit EtRuntime(const std::string& ptePath);
```

with:

```cpp
  // workspaceSharingMode: XNNPACK workspace sharing for THIS model, supplied as a per-load backend
  // runtime spec (Module::load(LoadBackendOptionsMap)). 0=Disabled, 1=PerModel, 2=Global, matching
  // executorch::backends::xnnpack::WorkspaceSharingMode in backends/xnnpack/runtime/XNNPACKBackend.h
  // -- a header the runtime tarball does NOT install, hence the hardcoded values.
  //
  // -1 omits the spec entirely, leaving whatever default the runtime was compiled with (Global for
  // our pin: EXECUTORCH_XNNPACK_SHARED_WORKSPACE=ON). Omitting is deliberately not the same as
  // passing 2 -- it follows the pin rather than pinning a value we would have to keep in sync.
  //
  // Any other value is passed through to ExecuTorch, which rejects it at delegate init and fails
  // the load. et_runtime_test.cpp depends on that to prove the spec reaches the backend.
  explicit EtRuntime(const std::string& ptePath, int workspaceSharingMode = -1);
```

- [ ] **Step 4: Implement the load path**

In `native/core/et_runtime.cpp`, add these two includes to the existing ExecuTorch include block (after the `module.h` line):

```cpp
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/options.h>
```

Add to the `using` block near the top of `namespace measly::et`:

```cpp
using executorch::runtime::BackendOptions;
using executorch::runtime::LoadBackendOptionsMap;
```

Then replace the constructor body's load (currently `EtRuntime::EtRuntime(const std::string& ptePath)` and its `if (state_->module.load() != ...)`):

```cpp
EtRuntime::EtRuntime(const std::string& ptePath, int workspaceSharingMode)
    : state_(std::make_unique<RuntimeState>(ptePath)) {
  // Set even when this ctor later throws: the pool is captured by XNNPACK at runtime creation,
  // so "has ever been constructed" is the safe boundary for the intra-op reset guard.
  g_etRuntimeConstructed.store(true);
  // Force-load now so a bad path/file throws at construction (the "load throws" contract).
  //
  // The options map and its BackendOptions storage are stack-local, which the non-owning-span
  // caveat on LoadBackendOptionsMap would normally forbid. It is correct here because
  // Module::load deep-copies into Module-owned storage before returning, and the lazy
  // load_method that forward() triggers consumes that copy -- see the doc comment on
  // Module::load(const LoadBackendOptionsMap&, Verification).
  executorch::runtime::Error loadErr;
  if (workspaceSharingMode >= 0) {
    BackendOptions<1> xnnOpts;
    // Key from backends/xnnpack/runtime/XNNPACKBackend.h (workspace_sharing_mode_option_key).
    xnnOpts.set_option("workspace_sharing_mode", workspaceSharingMode);
    LoadBackendOptionsMap optionsMap;
    // Backend id from the same header (xnnpack_backend_key). Spelled EXACTLY "XnnpackBackend":
    // Method::load matches it against the .pte's delegate id, and a mismatch is a SILENT no-op,
    // not an error.
    optionsMap.set_options("XnnpackBackend", xnnOpts.view());
    loadErr = state_->module.load(optionsMap);
  } else {
    loadErr = state_->module.load();
  }
  if (loadErr != executorch::runtime::Error::Ok) {
    throw std::runtime_error("EtRuntime: failed to load .pte: " + ptePath);
  }
  state_->meta = buildMethodMeta(state_->module);
```

Leave the rest of the constructor (the non-tensor input rejection loop and staging setup) exactly as it is.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
./native/local_build_wrapper.sh native/build_qa.sh
```

Expected: all Catch2 assertions pass, including the three new cases. ASan/LSan clean.

If `"workspace: an out-of-range mode is rejected"` **fails** (the load succeeded), the spec is not reaching the backend — check the `"XnnpackBackend"` and `"workspace_sharing_mode"` spellings before anything else. That is the exact bug this test exists to catch.

- [ ] **Step 6: Fix container-created root-owned output**

```bash
sudo chown -R "$(id -u):$(id -g)" native/asan native/qa_noasan
```

- [ ] **Step 7: Commit**

```bash
git add native/core/et_runtime.h native/core/et_runtime.cpp native/test/et_runtime_test.cpp
git commit -m "feat(native): accept a per-model XNNPACK workspace sharing mode"
```

---

### Task 2: Java resolution of option and property

Pure logic, no native library, no JNI. Runs under `./gradlew test` in the shared JVM — there is no global state to contaminate.

**Files:**
- Create: `src/main/java/org/measly/executorch/engine/EtWorkspaceSharing.java`
- Test: `src/test/java/org/measly/executorch/engine/EtWorkspaceSharingTest.java`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces, all package-private in `org.measly.executorch.engine`:
  - `static final String EtWorkspaceSharing.OPTION_KEY` = `"workspaceSharingMode"`
  - `static final String EtWorkspaceSharing.PROPERTY` = `"ai.djl.executorch.workspace_sharing_mode"`
  - `static final int UNSPECIFIED = -1, DISABLED = 0, PER_MODEL = 1, GLOBAL = 2`
  - `static int parse(String value)` — throws `IllegalArgumentException` on an unrecognised value
  - `static int resolve(Map<String, ?> options, String propertyValue)` — the precedence chain

- [ ] **Step 1: Write the failing test**

Create `src/test/java/org/measly/executorch/engine/EtWorkspaceSharingTest.java`:

```java
package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** Pure resolution logic for the per-model workspace sharing mode; never touches the native lib. */
class EtWorkspaceSharingTest {

    @Test
    void parseAcceptsTheThreeModesCaseInsensitivelyAndTrimmed() {
        assertEquals(EtWorkspaceSharing.DISABLED, EtWorkspaceSharing.parse("disabled"));
        assertEquals(EtWorkspaceSharing.PER_MODEL, EtWorkspaceSharing.parse("per_model"));
        assertEquals(EtWorkspaceSharing.GLOBAL, EtWorkspaceSharing.parse("global"));
        assertEquals(EtWorkspaceSharing.DISABLED, EtWorkspaceSharing.parse("DISABLED"));
        assertEquals(EtWorkspaceSharing.GLOBAL, EtWorkspaceSharing.parse("  Global  "));
    }

    @Test
    void parseRejectsUnrecognisedValuesAndNamesTheLegalOnes() {
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> EtWorkspaceSharing.parse("disabeld"));
        assertTrue(e.getMessage().contains("disabeld"), "message must quote the bad value");
        assertTrue(e.getMessage().contains("per_model"), "message must list the legal values");
    }

    @Test
    void parseRejectsBareIntegers() {
        // Ints are opaque at a call site and would let an out-of-range value reach the runtime.
        assertThrows(IllegalArgumentException.class, () -> EtWorkspaceSharing.parse("0"));
        assertThrows(IllegalArgumentException.class, () -> EtWorkspaceSharing.parse("99"));
    }

    @Test
    void optionWinsOverProperty() {
        Map<String, String> options =
                Collections.singletonMap(EtWorkspaceSharing.OPTION_KEY, "disabled");
        assertEquals(EtWorkspaceSharing.DISABLED, EtWorkspaceSharing.resolve(options, "global"));
    }

    @Test
    void propertyAppliesWhenNoOptionIsPresent() {
        assertEquals(
                EtWorkspaceSharing.GLOBAL,
                EtWorkspaceSharing.resolve(Collections.emptyMap(), "global"));
        assertEquals(EtWorkspaceSharing.GLOBAL, EtWorkspaceSharing.resolve(null, "global"));
    }

    @Test
    void nothingSpecifiedOmitsTheSpec() {
        assertEquals(
                EtWorkspaceSharing.UNSPECIFIED,
                EtWorkspaceSharing.resolve(Collections.emptyMap(), null));
        assertEquals(EtWorkspaceSharing.UNSPECIFIED, EtWorkspaceSharing.resolve(null, null));
    }

    @Test
    void aNullOptionValueCountsAsAbsent() {
        Map<String, String> options = new HashMap<>();
        options.put(EtWorkspaceSharing.OPTION_KEY, null);
        assertEquals(EtWorkspaceSharing.GLOBAL, EtWorkspaceSharing.resolve(options, "global"));
        assertEquals(EtWorkspaceSharing.UNSPECIFIED, EtWorkspaceSharing.resolve(options, null));
    }

    @Test
    void aBadOptionThrowsButABadPropertyIsIgnored() {
        // Asymmetric by design (spec section 6): a per-model option is explicit intent about one
        // model, so a silent fallback would be an invisible latency regression. The property is an
        // ambient default and a typo in a process-wide flag must not fail startup.
        Map<String, String> options =
                Collections.singletonMap(EtWorkspaceSharing.OPTION_KEY, "disabeld");
        assertThrows(
                IllegalArgumentException.class, () -> EtWorkspaceSharing.resolve(options, null));
        assertEquals(
                EtWorkspaceSharing.UNSPECIFIED,
                EtWorkspaceSharing.resolve(Collections.emptyMap(), "disabeld"));
    }

    @Test
    void nonStringOptionValuesAreCoerced() {
        // DJL's options map is Map<String, ?>; callers can put anything in it.
        Map<String, Object> options = new HashMap<>();
        options.put(EtWorkspaceSharing.OPTION_KEY, new StringBuilder("disabled"));
        assertEquals(EtWorkspaceSharing.DISABLED, EtWorkspaceSharing.resolve(options, null));
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./gradlew test --tests 'org.measly.executorch.engine.EtWorkspaceSharingTest'
```

Expected: **compile failure** — `cannot find symbol: class EtWorkspaceSharing`.

- [ ] **Step 3: Write the implementation**

Create `src/main/java/org/measly/executorch/engine/EtWorkspaceSharing.java`:

```java
package org.measly.executorch.engine;

import java.util.Locale;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Resolves the per-model XNNPACK workspace sharing mode from a DJL model option or a JVM-wide
 * default property.
 *
 * <p>Unlike {@code ai.djl.executorch.num_threads}, this is <b>not</b> process-global and is
 * <b>not</b> write-once. ExecuTorch resolves the mode per delegate at method-load time, preferring
 * the per-load runtime spec we supply over the backend's process global, so every model may choose
 * independently and the modes compose: a model electing {@code disabled} is isolated regardless of
 * what any other loaded model chose. Nothing is sealed and load order does not matter.
 *
 * <p>Values map to {@code executorch::backends::xnnpack::WorkspaceSharingMode}. That header is not
 * installed by the runtime tarball, so the ints are hardcoded here and again in
 * {@code native/core/et_runtime.cpp}; keep the two in sync.
 */
final class EtWorkspaceSharing {

    /** DJL per-model option key, e.g. {@code Criteria.optOption("workspaceSharingMode", ...)}. */
    static final String OPTION_KEY = "workspaceSharingMode";

    /** JVM-wide default for models that do not carry {@link #OPTION_KEY}. */
    static final String PROPERTY = "ai.djl.executorch.workspace_sharing_mode";

    /** Send no spec at all, leaving the runtime's compiled-in default. NOT a synonym for GLOBAL. */
    static final int UNSPECIFIED = -1;

    /** Every delegate instance gets its own workspace: maximum parallelism, maximum arena memory. */
    static final int DISABLED = 0;

    /** All delegate instances in one program share a workspace: one method at a time per model. */
    static final int PER_MODEL = 1;

    /** All delegate instances across all loaded methods share one workspace. The shipped default. */
    static final int GLOBAL = 2;

    private static final Logger logger = LoggerFactory.getLogger(EtWorkspaceSharing.class);

    private EtWorkspaceSharing() {}

    /**
     * Maps a mode name to its native int.
     *
     * @param value one of {@code disabled}, {@code per_model}, {@code global}; case-insensitive and
     *     trimmed
     * @return the native mode int
     * @throws IllegalArgumentException if the value is not one of the three names. Bare integers
     *     are rejected too: they are opaque at a call site and would let an out-of-range value
     *     through to a native load failure.
     */
    static int parse(String value) {
        String v = value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
        switch (v) {
            case "disabled":
                return DISABLED;
            case "per_model":
                return PER_MODEL;
            case "global":
                return GLOBAL;
            default:
                throw new IllegalArgumentException(
                        OPTION_KEY
                                + ": unrecognized value '"
                                + value
                                + "'; expected disabled|per_model|global");
        }
    }

    /**
     * Applies the precedence chain: per-model option, then JVM property, then unspecified.
     *
     * <p>A key present with a null value counts as absent. A bad option throws (explicit per-model
     * intent must not degrade silently); a bad property WARNs and is ignored (a typo in a
     * process-wide flag must not fail startup), matching the {@code num_threads} precedent.
     *
     * @param options the DJL model options map; may be null
     * @param propertyValue the value of {@link #PROPERTY}, passed in so this stays pure and
     *     testable; may be null
     * @return the native mode int, or {@link #UNSPECIFIED}
     * @throws IllegalArgumentException if the per-model option carries an unrecognized value
     */
    static int resolve(Map<String, ?> options, String propertyValue) {
        Object raw = options == null ? null : options.get(OPTION_KEY);
        if (raw != null) {
            return parse(String.valueOf(raw));
        }
        if (propertyValue == null) {
            return UNSPECIFIED;
        }
        try {
            return parse(propertyValue);
        } catch (IllegalArgumentException e) {
            logger.warn(
                    "{}='{}' is not a recognized mode; ignoring and using the runtime default",
                    PROPERTY,
                    propertyValue);
            return UNSPECIFIED;
        }
    }
}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
./gradlew test --tests 'org.measly.executorch.engine.EtWorkspaceSharingTest'
```

Expected: 8 tests, all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/main/java/org/measly/executorch/engine/EtWorkspaceSharing.java \
        src/test/java/org/measly/executorch/engine/EtWorkspaceSharingTest.java
git commit -m "feat: resolve per-model workspace sharing mode from option or property"
```

---

### Task 3: Wire the mode through JNI to the model load

**Files:**
- Modify: `native/jni/executorch_djl_jni.cpp:125-136`
- Modify: `src/main/java/org/measly/executorch/jni/EtNative.java:19`
- Modify: `src/main/java/org/measly/executorch/engine/EtModel.java:48-54`
- Test: `src/test/java/org/measly/executorch/WorkspaceSharingIT.java` (create)

**Interfaces:**
- Consumes: `EtRuntime(const std::string&, int)` from Task 1; `EtWorkspaceSharing.resolve(Map, String)`, `.OPTION_KEY`, `.PROPERTY`, `.UNSPECIFIED` from Task 2.
- Produces: `EtNative.loadModule(String ptePath, int workspaceSharingMode)` — the existing single-argument method is **replaced**, not overloaded. There is exactly one call site (`EtModel.load`), and an overload would mean two JNI entry points to keep in sync. `org.measly.executorch.jni` is public by visibility but internal by intent.

- [ ] **Step 1: Write the failing test**

Create `src/test/java/org/measly/executorch/WorkspaceSharingIT.java`:

```java
package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;

/**
 * End-to-end per-model workspace sharing. Safe in the shared test JVM and needs no dedicated
 * Gradle task: the mode is per model, so nothing here contaminates a later test. (Contrast
 * IntraOpThreadsIT, which needs a forked JVM because that pool is process-global.)
 */
class WorkspaceSharingIT {

    private static Criteria<float[], Float> criteriaWithMode(String mode) {
        Criteria.Builder<float[], Float> b =
                Criteria.builder()
                        .setTypes(float[].class, Float.class)
                        .optEngine("ExecuTorch")
                        .optModelPath(Paths.get("native/spike"))
                        .optModelName("add")
                        .optTranslator(new AddTranslator());
        if (mode != null) {
            b.optOption("workspaceSharingMode", mode);
        }
        return b.build();
    }

    @Test
    void everyModeLoadsAndPredicts() throws Exception {
        TestSupport.assumeNativeAvailable();
        for (String mode : new String[] {"disabled", "per_model", "global", "GLOBAL", " disabled "}) {
            try (ZooModel<float[], Float> model = criteriaWithMode(mode).loadModel();
                    Predictor<float[], Float> predictor = model.newPredictor()) {
                assertEquals(5f, predictor.predict(new float[] {2f, 3f}), 1e-6, "mode=" + mode);
            }
        }
    }

    @Test
    void modesComposeAcrossConcurrentlyLoadedModels() throws Exception {
        // A model electing "disabled" is isolated regardless of what other loaded models chose.
        // Both must remain independently usable while the other is open.
        TestSupport.assumeNativeAvailable();
        try (ZooModel<float[], Float> isolated = criteriaWithMode("disabled").loadModel();
                ZooModel<float[], Float> shared = criteriaWithMode("global").loadModel();
                Predictor<float[], Float> p1 = isolated.newPredictor();
                Predictor<float[], Float> p2 = shared.newPredictor()) {
            assertEquals(5f, p1.predict(new float[] {2f, 3f}), 1e-6);
            assertEquals(9f, p2.predict(new float[] {4f, 5f}), 1e-6);
            assertEquals(3f, p1.predict(new float[] {1f, 2f}), 1e-6);
        }
    }

    @Test
    void anUnrecognizedModeFailsTheLoad() {
        TestSupport.assumeNativeAvailable();
        // Explicit per-model intent must not degrade silently to the default.
        assertThrows(Exception.class, () -> criteriaWithMode("disabeld").loadModel());
    }

    @Test
    void noOptionStillLoads() throws Exception {
        TestSupport.assumeNativeAvailable();
        try (ZooModel<float[], Float> model = criteriaWithMode(null).loadModel();
                Predictor<float[], Float> predictor = model.newPredictor()) {
            assertEquals(5f, predictor.predict(new float[] {2f, 3f}), 1e-6);
        }
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./gradlew test --tests 'org.measly.executorch.WorkspaceSharingIT'
```

Expected: FAIL. `everyModeLoadsAndPredicts` passes vacuously today (the option is ignored) but `anUnrecognizedModeFailsTheLoad` FAILS — the load succeeds because nothing reads the option yet.

- [ ] **Step 3: Change the JNI entry point**

In `native/jni/executorch_djl_jni.cpp`, replace the `loadModule` function:

```cpp
extern "C" JNIEXPORT jlong JNICALL
Java_org_measly_executorch_jni_EtNative_loadModule(
    JNIEnv* env, jclass, jstring jpath, jint jworkspaceSharingMode) {
  const char* path = env->GetStringUTFChars(jpath, nullptr);
  std::string p(path);
  env->ReleaseStringUTFChars(jpath, path);
  try {
    // No range check here: the Java layer emits only -1/0/1/2, and any other value is deliberately
    // passed through so ExecuTorch rejects it at delegate init. et_runtime_test.cpp relies on that
    // to prove the runtime spec reaches the XNNPACK backend.
    return reinterpret_cast<jlong>(new EtRuntime(p, static_cast<int>(jworkspaceSharingMode)));
  } catch (const std::exception& e) {
    throwJava(env, "EtRuntime load failed", &e);
    return 0;
  }
}
```

- [ ] **Step 4: Change the Java declaration**

In `src/main/java/org/measly/executorch/jni/EtNative.java`, replace line 19:

```java
    public static native long loadModule(String ptePath);
```

with:

```java
    /**
     * Loads a .pte.
     *
     * @param ptePath path to the model file
     * @param workspaceSharingMode XNNPACK workspace sharing for this model: 0=Disabled, 1=PerModel,
     *     2=Global, -1 to send no spec and leave the runtime default in force
     * @return the native handle
     */
    public static native long loadModule(String ptePath, int workspaceSharingMode);
```

- [ ] **Step 5: Resolve and pass the mode at load**

In `src/main/java/org/measly/executorch/engine/EtModel.java`, replace lines 48-54 (the seal comment through the `loadModule` call):

```java
        // First load seals the process-global intra-op thread pool (applies pending/property value,
        // logs the outcome); later loads are no-ops. Must precede loadModule: delegate init during
        // load submits work to the pool.
        EtEngine.sealIntraOpThreads();
        // Per-model, by contrast: resolved fresh on every load, nothing is sealed, and load order
        // does not matter. Throws IllegalArgumentException for a bad per-model option.
        int workspaceSharingMode =
                EtWorkspaceSharing.resolve(options, System.getProperty(EtWorkspaceSharing.PROPERTY));
        // Not unit-tested below this point: loadModule/methodMeta/destroy require the native library
        // (integration-tested via EtModelTest#loadAndForwardAddModel).
        long handle = EtNative.loadModule(modelFile.toString(), workspaceSharingMode);
```

- [ ] **Step 6: Rebuild the native shim**

```bash
./native/local_build_wrapper.sh
```

Expected: builds and stages `libexecutorch_djl.so` into `src/main/resources/native/linux-x86_64/`.

Verify the symbol picked up the new signature (`JNI` name mangling is unaffected by parameters, so confirm by timestamp instead):

```bash
ls -l src/main/resources/native/linux-x86_64/libexecutorch_djl.so
```

Expected: modified within the last few minutes. A stale `.so` here produces an `UnsatisfiedLinkError` on `loadModule` at test time.

- [ ] **Step 7: Run the tests to verify they pass**

```bash
./gradlew test --tests 'org.measly.executorch.WorkspaceSharingIT'
```

Expected: 4 tests PASS.

Confirm they did not vacuously skip on `assumeNativeAvailable` — an aborted assumption reports as skipped, not failed:

```bash
grep -o 'tests="[0-9]*" skipped="[0-9]*" failures="[0-9]*" errors="[0-9]*"' \
    build/test-results/test/TEST-org.measly.executorch.WorkspaceSharingIT.xml
```

Expected: `tests="4" skipped="0" failures="0" errors="0"`.

- [ ] **Step 8: Run the full suite for regressions**

```bash
./gradlew test
```

Expected: PASS. `EtModelTest`, `AddModelIT`, `LstmModelIT`, `MultiDtypeIT`, `NamedParamsIT` and `LoggingBridgeIT` all go through `EtModel.load` and exercise the changed `loadModule` path.

- [ ] **Step 9: Commit**

```bash
git add native/jni/executorch_djl_jni.cpp \
        src/main/java/org/measly/executorch/jni/EtNative.java \
        src/main/java/org/measly/executorch/engine/EtModel.java \
        src/test/java/org/measly/executorch/WorkspaceSharingIT.java
git commit -m "feat: plumb per-model workspace sharing mode through JNI to model load"
```

---

### Task 4: Documentation, including a correction to the threading note

The existing "more threads is usually wrong" note is stated unconditionally, but its numbers were measured against the `Global` serialiser this feature removes. Left as-is it points people away from exactly the tuning they now need.

**Files:**
- Modify: `src/main/java/org/measly/executorch/engine/EtSymbolBlock.java:23-29`
- Modify: `CLAUDE.md:115-122`

**Interfaces:**
- Consumes: the option and property names from Task 2. Produces nothing.

- [ ] **Step 1: Correct and extend the javadoc note**

In `EtSymbolBlock.java`, replace the paragraph at lines 23-29:

```java
 * <p><b>Threading, and why more threads is usually wrong <i>under the default sharing mode</i>.</b>
 * The rule above is about <i>safety</i>, not throughput. XNNPACK-delegated models already
 * parallelize inside a single {@code forward()} on ExecuTorch's shared intra-op pool, and under the
 * shipped {@code global} workspace sharing mode concurrent delegate calls serialize on one
 * process-global workspace mutex — so N {@code Predictor}s on N threads is typically slower than
 * one, not N× faster. Tune {@code ai.djl.executorch.num_threads} before adding caller threads.
 * Measured on a 4-core/8-thread host with MobileNetV2: 1 thread 462 forwards/s, 4 threads 305, 8
 * threads 147 (peak RSS 33 MB → 224 MB). Ratios on larger hosts are unmeasured.
 *
 * <p><b>These figures are conditional on that mutex.</b> Loading a model with
 * {@code Criteria.optOption("workspaceSharingMode", "disabled")} gives it a private workspace, and
 * caller threads then scale: measured achieved parallelism at one intra-op thread was 1.12× / 1.12×
 * / 1.12× / 1.17× at 1/2/4/8 caller threads under {@code global}, versus 1.12× / 2.23× / 4.35× /
 * 7.13× under {@code disabled}. The cost is activation-arena memory per delegate instance. Prefer
 * intra-op tuning first; reach for {@code disabled} when several models with differing call rates
 * share a JVM and memory is not the binding constraint.
 */
```

- [ ] **Step 2: Mirror the correction in CLAUDE.md and add the conventions bullets**

In `CLAUDE.md`, replace the blockquote at lines 115-121 and add two bullets after the existing `num_threads` bullet (line 122):

```markdown
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
```

```markdown
- `Criteria.optOption("workspaceSharingMode", "disabled"|"per_model"|"global")` picks the XNNPACK workspace sharing mode **per model**; `ai.djl.executorch.workspace_sharing_mode` (JVM flag) is the default for models that don't specify. Unlike `num_threads` this is neither process-global nor write-once — ExecuTorch resolves it per delegate at load, so modes compose and load order is irrelevant. An unrecognized *option* fails the load; an unrecognized *property* warns and is ignored. Absent both, no spec is sent and the runtime default (`global` for our pin) applies. See `docs/superpowers/specs/2026-08-08-workspace-sharing-mode-design.md`.
- `weight_cache_enabled` is deliberately **not** exposed. `XnnpackBackend::execute()` holds a second process-global mutex (`weights_cache_mutex_`) for the whole delegate call whenever a model uses the cache, which would undo everything `workspaceSharingMode=disabled` buys. It is off in our pin (`EXECUTORCH_XNNPACK_ENABLE_WEIGHT_CACHE=OFF`), which is what makes the `disabled` numbers above real — treat a pin bump that flips it as a performance regression. To enable it anyway no rebuild is needed: the macro guards only the *default*, and `XNNWeightsCache` is compiled into the shipped `libxnnpack_backend.a`. Set `weight_cache_enabled` (a **bool**) in the same `LoadBackendOptionsMap` built in `native/core/et_runtime.cpp`, and keep those models off the hot path.
```

- [ ] **Step 3: Verify the javadoc still compiles**

```bash
./gradlew compileJava
```

Expected: BUILD SUCCESSFUL. (The `→` and `×` characters already appear in these files, so the source encoding handles them.)

- [ ] **Step 4: Commit**

```bash
git add src/main/java/org/measly/executorch/engine/EtSymbolBlock.java CLAUDE.md
git commit -m "docs: scope the threading note to the default sharing mode; document the new knob"
```

---

## Final Verification

- [ ] **Full JVM build**

```bash
./gradlew build
```

Expected: BUILD SUCCESSFUL. This runs `check`, which includes `test`, `intraOpTest`, and the jacoco report.

- [ ] **Full native QA**

```bash
./native/local_build_wrapper.sh native/build_qa.sh
sudo chown -R "$(id -u):$(id -g)" native/asan native/qa_noasan
```

Expected: Catch2 green, ASan/LSan clean.

- [ ] **Confirm no stray global state was introduced**

```bash
grep -rn "set_option" native/core native/jni
```

Expected: matches only inside the `BackendOptions<1>` block in `et_runtime.cpp`. Any call routed through a *backend interface* `set_option` (the process-global path) is out of scope and must not appear.

- [ ] **Push and open a PR**

```bash
git push -u origin feature/workspace-sharing-mode
```

## Out of Scope

- Measuring the parallelism win. That belongs in `native/scaling.sh`, is run on demand, and is the user's to invoke — not a pass/fail gate.
- Exposing `weight_cache_enabled` (spec section 8).
- The process-global backend `set_option` path (spec section 3).
- Any `EtRuntimePin.cmake` change.
