# Memory-Leak Quality Gates — Implementation Plan (tight)

> Inline execution. Builds the JVM-side leak gates (gates 1–3 from the leak-troubleshooting
> discussion). The native-`Module` leak gate (item 4) is recorded in
> [`docs/ci-native-build.md`](../../ci-native-build.md) as a deferred nightly/manual sanitizer job.

**Goal:** Add a JVM-side memory-leak quality gate covering the two regions JVM tools can see —
Java-heap wrapper objects and off-heap direct buffers — plus deterministic native-handle invariants.

**Scope:** `EtSymbolBlock.isClosed()` accessor; a native-handle lifecycle test; a constrained-memory
leak stress test behind a tagged `leakTest` Gradle task; `-XX:+HeapDumpOnOutOfMemoryError` on the
test JVMs.

---

## Task 1: Native-handle invariants (the region no profiler sees)

**Files:**
- Modify: `src/main/java/org/measly/executorch/engine/EtSymbolBlock.java`
- Create: `src/test/java/org/measly/executorch/engine/EtSymbolBlockLifecycleTest.java`

- [ ] **Step 1** — add a package-private accessor to `EtSymbolBlock` (after `close()`):
```java
    /** @return true once the native handle has been released by {@link #close()}. */
    boolean isClosed() {
        return handle == 0;
    }
```

- [ ] **Step 2** — failing test `EtSymbolBlockLifecycleTest.java`:
```java
package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.djl.Model;
import java.nio.file.Paths;
import org.measly.executorch.TestSupport;
import org.junit.jupiter.api.Test;

class EtSymbolBlockLifecycleTest {

    @Test
    void closeReleasesHandleAndIsIdempotent() throws Exception {
        TestSupport.assumeNativeAvailable();
        Model model = Model.newInstance("add", "ExecuTorch");
        model.load(Paths.get("native/spike"), "add");
        EtSymbolBlock block = (EtSymbolBlock) model.getBlock();
        assertFalse(block.isClosed());
        model.close();
        assertTrue(block.isClosed());
        block.close(); // idempotent: handle already 0, must not throw or double-destroy
    }

    @Test
    void repeatedLoadCloseDoesNotDegrade() throws Exception {
        TestSupport.assumeNativeAvailable();
        for (int i = 0; i < 100; i++) {
            try (Model model = Model.newInstance("add", "ExecuTorch")) {
                model.load(Paths.get("native/spike"), "add");
            } // close() must destroy the native Module each iteration
        }
    }
}
```

- [ ] **Step 3** — run `./gradlew test --tests '*EtSymbolBlockLifecycleTest'` → PASS.
- [ ] **Step 4** — commit.

## Task 2: Constrained-memory leak stress test + Gradle `leakTest` (+ heap dump)

**Files:**
- Create: `src/test/java/org/measly/executorch/LeakStressTest.java`
- Modify: `build.gradle.kts`

- [ ] **Step 1** — `LeakStressTest.java` (`@Tag("leak")`):
```java
package org.measly.executorch;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import java.nio.file.Paths;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Leak gates that turn a lifecycle leak into a deterministic OutOfMemoryError. Run via the
 * {@code leakTest} Gradle task under {@code -XX:MaxDirectMemorySize=64m -Xmx256m}; a correct
 * lifecycle survives the GC-reclaim retry, a leak exhausts memory and fails.
 */
@Tag("leak")
class LeakStressTest {

    /** Direct-buffer lifecycle (native-free): 200 x 4MB direct arrays, each freed before the next. */
    @Test
    void directBufferLifecycleUnderPressure() {
        try (NDManager base = NDManager.newBaseManager("ExecuTorch")) {
            for (int i = 0; i < 200; i++) {
                try (NDManager sub = base.newSubManager()) {
                    sub.create(new float[1_000_000], new Shape(1_000_000)); // 4 MB off-heap
                }
            }
        }
    }

    /** Inference path: many predictions; a leaked per-call input/output buffer accumulates. */
    @Test
    void inferencePathUnderPressure() throws Exception {
        TestSupport.assumeNativeAvailable();
        Criteria<float[], Float> criteria =
                Criteria.builder()
                        .setTypes(float[].class, Float.class)
                        .optEngine("ExecuTorch")
                        .optModelPath(Paths.get("native/spike"))
                        .optModelName("add")
                        .optTranslator(new AddTranslator())
                        .build();
        try (ZooModel<float[], Float> model = criteria.loadModel();
                Predictor<float[], Float> predictor = model.newPredictor()) {
            for (int i = 0; i < 20_000; i++) {
                predictor.predict(new float[] {1f, 2f});
            }
        }
    }
}
```

- [ ] **Step 2** — wire `build.gradle.kts`: exclude `leak` from `test`, add a `leakTest` task with the
constrained JVM args, and heap-dump-on-OOM on both. Replace `tasks.test { useJUnitPlatform() }` with:
```kotlin
tasks.test {
    useJUnitPlatform { excludeTags("leak") }
    jvmArgs("-XX:+HeapDumpOnOutOfMemoryError")
}

tasks.register<Test>("leakTest") {
    description = "Memory-leak stress tests under constrained heap/direct memory."
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("leak") }
    jvmArgs("-Xmx256m", "-XX:MaxDirectMemorySize=64m", "-XX:+HeapDumpOnOutOfMemoryError")
}
```

- [ ] **Step 3** — `./gradlew test` → green, leak tests NOT run (excluded). Then `./gradlew leakTest`
→ both leak tests pass under the constraints (a correct lifecycle survives; a leak would OOM).
- [ ] **Step 4** — commit.

## Self-review
- Gate 1 (direct buffers) → `LeakStressTest.directBufferLifecycleUnderPressure` (native-free) +
  `inferencePathUnderPressure` (native). Gate 2 (native handle) → `EtSymbolBlockLifecycleTest` +
  `isClosed()`. Gate 3 (diagnostics) → `-XX:+HeapDumpOnOutOfMemoryError` on both test JVMs.
- The native-`Module` region is explicitly out of scope here (recorded in `ci-native-build.md`).
