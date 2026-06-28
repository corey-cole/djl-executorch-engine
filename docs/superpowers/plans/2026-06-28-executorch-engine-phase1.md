# ExecuTorch DJL Engine — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working DJL engine named `ExecuTorch` that loads a `.pte` and runs float32 inference end-to-end through DJL's `Criteria`/`Predictor` API, backed by the Path-B `libexecutorch_djl.so`.

**Architecture:** Mirror DJL's ONNX Runtime engine structure (the closest precedent — limited NDArray, external runtime). `EtNDArray` is a Java-side data holder (`ByteBuffer`+`Shape`+`DataType`); tensors cross JNI only inside `forward()` via a small `EtNative`/`EtTensor` surface; ExecuTorch tensors are transient in native code. Native loading is **lazy** (in `EtNative`'s static initializer) so engine/manager/NDArray construction is native-free and unit-testable.

**Tech Stack:** Java 17, Gradle (Kotlin DSL), `ai.djl:api:0.36.0`, JUnit 5, the existing `native/` CMake/JNI shim.

**Spec:** [`docs/superpowers/specs/2026-06-28-executorch-engine-phase1-design.md`](../specs/2026-06-28-executorch-engine-phase1-design.md).

**Reference templates (read these alongside each task):** the ONNX engine at
`~/workspace/djl/engines/onnxruntime/onnxruntime-engine/src/main/java/ai/djl/onnxruntime/engine/`
(`OrtEngine`, `OrtEngineProvider`, `OrtModel`, `OrtNDManager`, `OrtNDArray`, `OrtSymbolBlock`).

---

## Notes on this plan

- **TDD with a native boundary.** Tasks 1–3 are fully unit-testable with no native library. Tasks 4–6 require the `.so`; their tests are guarded by a JUnit assumption so the suite stays green where the native build is absent. Build the native lib once before running guarded tests:
  ```bash
  ET_INSTALL="$HOME/workspace/executorch/cmake-out" JAVA_HOME=/usr/lib/jvm/zulu-17-amd64 ./native/build_desktop.sh
  ```
- **API version caveat.** Code targets `ai.djl:api:0.36.0`. A few signatures (e.g. `BaseModel.findModelFile`, `NDArrayAdapter` ctor, `SystemNDManager`) are taken from the DJL source; if the pinned version differs, the compile step in each task surfaces it immediately. Method names verified against DJL source: `Engine.getRank()`, `EngineProvider.getEngineRank()`, `NDArrayAdapter(NDManager, NDManager, Shape, DataType, String)`, `BaseNDManager.validateBuffer(Buffer, DataType, int)`, `forwardInternal(ParameterStore, NDList, boolean, PairList<String,Object>)`.

## File structure

| File | Responsibility |
|---|---|
| `settings.gradle.kts` (modify) | Single root module; drop the generated `lib` subproject |
| `build.gradle.kts` (create) | Java 17 toolchain, DJL `compileOnly` + test deps, JUnit |
| `src/main/java/org/measly/executorch/engine/EtEngineProvider.java` | SPI entry |
| `.../engine/EtEngine.java` | Engine singleton (native-free) |
| `.../engine/EtNDManager.java` | Tensor factory (`create(Buffer,Shape,DataType)`) |
| `.../engine/EtNDArray.java` | `ByteBuffer`-backed NDArray (`NDArrayAdapter`) |
| `.../engine/EtModel.java` | `.pte` load + lifecycle |
| `.../engine/EtSymbolBlock.java` | forward marshalling NDList↔EtTensor |
| `.../engine/LibUtils.java` | native lib resolution/loading |
| `src/main/java/org/measly/executorch/jni/EtNative.java` | native method surface (loads `.so` in static init) |
| `.../jni/EtTensor.java` | float32 tensor data carrier across JNI |
| `native/jni/executorch_djl_jni.cpp` (modify) | implement `forward(handle, EtTensor[])` |
| `src/main/resources/META-INF/services/ai.djl.engine.EngineProvider` | SPI registration |
| `src/test/java/org/measly/executorch/...` | tests |
| `src/test/java/org/measly/executorch/TestSupport.java` | native-availability assumption helper |

---

## Task 1: Single-module Gradle project with DJL on the classpath

**Files:**
- Modify: `settings.gradle.kts`
- Create: `build.gradle.kts`
- Test: `src/test/java/org/measly/executorch/ClasspathTest.java`

- [ ] **Step 1: Reduce settings to a single root module**

Replace `settings.gradle.kts` with:
```kotlin
plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}
rootProject.name = "djl-executorch-engine"
```
(Removes `include("lib")` — the engine lives in the root module. The generated `lib/` directory is now unused; leave it for the engineer to delete in the commit step.)

- [ ] **Step 2: Create the root build file**

Create `build.gradle.kts`:
```kotlin
plugins {
    `java-library`
}

group = "org.measly"
version = "0.1.0-SNAPSHOT"

java {
    toolchain { languageVersion = JavaLanguageVersion.of(17) }
}

repositories { mavenCentral() }

val djlVersion = "0.36.0"

dependencies {
    compileOnly("ai.djl:api:$djlVersion")

    testImplementation("ai.djl:api:$djlVersion")
    testImplementation(platform("org.junit:junit-bom:5.10.2"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.test { useJUnitPlatform() }
```

- [ ] **Step 3: Write a failing classpath test**

Create `src/test/java/org/measly/executorch/ClasspathTest.java`:
```java
package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.djl.engine.Engine;
import org.junit.jupiter.api.Test;

class ClasspathTest {
    @Test
    void djlApiIsOnClasspath() {
        // Compiles and runs only if ai.djl:api resolved.
        assertEquals("ai.djl.engine.Engine", Engine.class.getName());
    }
}
```

- [ ] **Step 4: Run — verify it passes (proves wiring)**

Run: `./gradlew test --tests '*ClasspathTest'`
Expected: `BUILD SUCCESSFUL`, 1 test passed. (If `./gradlew` is missing, run `gradle wrapper` first — the repo already has `gradlew` per the initial layout.)

- [ ] **Step 5: Commit**

```bash
git rm -r --cached lib 2>/dev/null; rm -rf lib
git add settings.gradle.kts build.gradle.kts src/test/java/org/measly/executorch/ClasspathTest.java
git commit -m "build: single-module gradle project with ai.djl:api"
```

---

## Task 2: Engine SPI discovery (native-free)

**Files:**
- Create: `src/main/java/org/measly/executorch/engine/EtEngine.java`
- Create: `src/main/java/org/measly/executorch/engine/EtEngineProvider.java`
- Create: `src/main/resources/META-INF/services/ai.djl.engine.EngineProvider`
- Test: `src/test/java/org/measly/executorch/engine/EtEngineTest.java`

> `EtEngine.newModel`/`newBaseManager` reference `EtModel`/`EtNDManager` (Tasks 3 & 5). To keep this task compiling standalone, `newModel`/`newBaseManager` are written now but the classes they call are created in later tasks — so this task's code will not compile until Task 3. **Implement Tasks 2 and 3 together if executing strictly task-by-task**, or stub `newBaseManager` to `throw new UnsupportedOperationException()` here and replace in Task 3. This plan writes the final form and relies on Task 3 following immediately.

- [ ] **Step 1: Write the failing discovery test**

Create `src/test/java/org/measly/executorch/engine/EtEngineTest.java`:
```java
package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

import ai.djl.engine.Engine;
import org.junit.jupiter.api.Test;

class EtEngineTest {
    @Test
    void engineIsDiscoverableViaSpi() {
        Engine engine = Engine.getEngine("ExecuTorch");
        assertNotNull(engine);
        assertEquals("ExecuTorch", engine.getEngineName());
        assertEquals(10, engine.getRank());
        assertNull(engine.getAlternativeEngine()); // no hybrid in Phase 1
    }
}
```

- [ ] **Step 2: Run — verify it fails**

Run: `./gradlew test --tests '*EtEngineTest'`
Expected: FAIL — `Engine.getEngine("ExecuTorch")` throws (no provider registered).

- [ ] **Step 3: Implement `EtEngine`**

Create `src/main/java/org/measly/executorch/engine/EtEngine.java`:
```java
package org.measly.executorch.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;

/** ExecuTorch implementation of {@link Engine}. CPU-only, limited NDArray support. */
public final class EtEngine extends Engine {

    public static final String ENGINE_NAME = "ExecuTorch";
    static final int RANK = 10;

    private EtEngine() {} // cheap: no native load here (lazy in EtNative)

    static Engine newInstance() {
        return new EtEngine();
    }

    @Override
    public Engine getAlternativeEngine() {
        return null; // Phase 1: no hybrid mode
    }

    @Override
    public String getEngineName() {
        return ENGINE_NAME;
    }

    @Override
    public int getRank() {
        return RANK;
    }

    @Override
    public String getVersion() {
        return "1.3.1"; // pinned ExecuTorch runtime version
    }

    @Override
    public boolean hasCapability(String capability) {
        return false; // no CUDA, no training
    }

    @Override
    public Model newModel(String name, Device device) {
        return new EtModel(name, newBaseManager(device));
    }

    @Override
    public NDManager newBaseManager() {
        return newBaseManager(null);
    }

    @Override
    public NDManager newBaseManager(Device device) {
        return EtNDManager.getSystemManager().newSubManager(device);
    }

    @Override
    public String toString() {
        return getEngineName() + ':' + getVersion();
    }
}
```

- [ ] **Step 4: Implement `EtEngineProvider` + SPI file**

Create `src/main/java/org/measly/executorch/engine/EtEngineProvider.java`:
```java
package org.measly.executorch.engine;

import ai.djl.engine.Engine;
import ai.djl.engine.EngineProvider;

/** ExecuTorch implementation of {@link EngineProvider}. */
public class EtEngineProvider implements EngineProvider {

    @Override
    public String getEngineName() {
        return EtEngine.ENGINE_NAME;
    }

    @Override
    public int getEngineRank() {
        return EtEngine.RANK;
    }

    @Override
    public Engine getEngine() {
        return InstanceHolder.INSTANCE;
    }

    private static final class InstanceHolder {
        static final Engine INSTANCE = EtEngine.newInstance();
    }
}
```

Create `src/main/resources/META-INF/services/ai.djl.engine.EngineProvider` containing exactly:
```
org.measly.executorch.engine.EtEngineProvider
```

- [ ] **Step 5: Run — verify pass (after Task 3 supplies `EtNDManager`/`EtModel`)**

Run: `./gradlew test --tests '*EtEngineTest'`
Expected: PASS. (Will not compile until `EtNDManager` and `EtModel` exist — proceed to Task 3, then re-run.)

- [ ] **Step 6: Commit**

```bash
git add src/main/java/org/measly/executorch/engine/EtEngine.java \
        src/main/java/org/measly/executorch/engine/EtEngineProvider.java \
        src/main/resources/META-INF/services/ai.djl.engine.EngineProvider \
        src/test/java/org/measly/executorch/engine/EtEngineTest.java
git commit -m "feat(engine): EtEngine + EtEngineProvider with SPI discovery"
```

---

## Task 3: EtNDManager + EtNDArray (data holder, native-free)

**Files:**
- Create: `src/main/java/org/measly/executorch/engine/EtNDArray.java`
- Create: `src/main/java/org/measly/executorch/engine/EtNDManager.java`
- Test: `src/test/java/org/measly/executorch/engine/EtNDArrayTest.java`

- [ ] **Step 1: Write the failing round-trip test**

Create `src/test/java/org/measly/executorch/engine/EtNDArrayTest.java`:
```java
package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.Test;

class EtNDArrayTest {
    @Test
    void floatRoundTrip() {
        try (NDManager manager = NDManager.newBaseManager("ExecuTorch")) {
            NDArray arr = manager.create(new float[] {1f, 2f, 3f}, new Shape(3));
            assertEquals(new Shape(3), arr.getShape());
            assertEquals(DataType.FLOAT32, arr.getDataType());
            assertArrayEquals(new float[] {1f, 2f, 3f}, arr.toFloatArray());
        }
    }
}
```

- [ ] **Step 2: Run — verify it fails**

Run: `./gradlew test --tests '*EtNDArrayTest'`
Expected: FAIL — no `EtNDManager`/`EtNDArray` (compile error or unsupported create).

- [ ] **Step 3: Implement `EtNDArray`**

Create `src/main/java/org/measly/executorch/engine/EtNDArray.java`:
```java
package org.measly.executorch.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrayAdapter;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** ExecuTorch {@link NDArray}: a Java-side data holder (no native handle). */
public class EtNDArray extends NDArrayAdapter {

    private ByteBuffer data;

    EtNDArray(
            NDManager manager,
            NDManager alternativeManager,
            ByteBuffer data,
            Shape shape,
            DataType dataType) {
        super(manager, alternativeManager, shape, dataType, NDManager.nextUid());
        this.data = data;
        manager.attachInternal(uid, this);
    }

    @Override
    public DataType getDataType() {
        return dataType;
    }

    @Override
    public Shape getShape() {
        return shape;
    }

    @Override
    public ByteBuffer toByteBuffer(boolean tryDirect) {
        data.rewind();
        return data.duplicate().order(ByteOrder.nativeOrder());
    }

    @Override
    public void intern(NDArray replaced) {
        throw new UnsupportedOperationException("ExecuTorch NDArray does not support intern");
    }

    @Override
    public void detach() {
        manager.detachInternal(getUid());
        manager = EtNDManager.getSystemManager();
    }

    @Override
    public void close() {
        data = null;
        super.close();
    }
}
```

- [ ] **Step 4: Implement `EtNDManager`**

Create `src/main/java/org/measly/executorch/engine/EtNDManager.java`:
```java
package org.measly.executorch.engine;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

/** ExecuTorch {@link NDManager}: minimal tensor factory. */
public class EtNDManager extends BaseNDManager {

    private static final EtNDManager SYSTEM_MANAGER = new SystemManager();

    private EtNDManager(NDManager parent, Device device) {
        super(parent, device);
    }

    static EtNDManager getSystemManager() {
        return SYSTEM_MANAGER;
    }

    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    @Override
    public EtNDArray from(NDArray array) {
        if (array == null || array instanceof EtNDArray) {
            return (EtNDArray) array;
        }
        EtNDArray result = create(array.toByteBuffer(), array.getShape(), array.getDataType());
        result.setName(array.getName());
        return result;
    }

    @Override
    public EtNDArray create(Buffer data, Shape shape, DataType dataType) {
        if (dataType == DataType.STRING) {
            throw new IllegalArgumentException("ExecuTorch does not support String NDArray");
        }
        int size = Math.toIntExact(shape.size());
        BaseNDManager.validateBuffer(data, dataType, size);
        ByteBuffer bb = allocateDirect(size * dataType.getNumOfBytes());
        copyInto(bb, data, dataType);
        bb.rewind();
        return new EtNDArray(this, alternativeManager, bb, shape, dataType);
    }

    @Override
    public NDManager newSubManager(Device device) {
        EtNDManager manager = new EtNDManager(this, device);
        attachInternal(manager.uid, manager);
        return manager;
    }

    @Override
    public Engine getEngine() {
        return Engine.getEngine(EtEngine.ENGINE_NAME);
    }

    /** Copies a typed input buffer into a native-order byte buffer. */
    private static void copyInto(ByteBuffer dst, Buffer src, DataType dataType) {
        if (src instanceof ByteBuffer) {
            dst.put((ByteBuffer) src);
            return;
        }
        switch (dataType) {
            case FLOAT32:
                dst.asFloatBuffer().put((FloatBuffer) src);
                break;
            case FLOAT64:
                dst.asDoubleBuffer().put((DoubleBuffer) src);
                break;
            case INT32:
                dst.asIntBuffer().put((IntBuffer) src);
                break;
            case INT64:
                dst.asLongBuffer().put((LongBuffer) src);
                break;
            default:
                throw new UnsupportedOperationException("Unsupported dtype: " + dataType);
        }
    }

    /** Root manager of which all others are children. */
    private static final class SystemManager extends EtNDManager
            implements BaseNDManager.SystemNDManager {
        SystemManager() {
            super(null, null);
        }
    }
}
```

> If `BaseNDManager.SystemNDManager` is not the exact nested-interface name in the pinned DJL version, check the type `OrtNDManager.SystemManager` implements (`grep "implements" OrtNDManager.java`) and use that. The compile step surfaces it.

- [ ] **Step 5: Run — verify pass; also re-run Task 2's engine test**

Run: `./gradlew test --tests '*EtNDArrayTest' --tests '*EtEngineTest'`
Expected: PASS (both). The engine test now compiles because `EtNDManager`/`EtModel` exist (`EtModel` arrives in Task 5; if running strictly in order, temporarily have `EtEngine.newModel` throw `UnsupportedOperationException` and restore in Task 5 — the `EtEngineTest` does not call `newModel`).

- [ ] **Step 6: Commit**

```bash
git add src/main/java/org/measly/executorch/engine/EtNDArray.java \
        src/main/java/org/measly/executorch/engine/EtNDManager.java \
        src/test/java/org/measly/executorch/engine/EtNDArrayTest.java
git commit -m "feat(engine): EtNDManager + EtNDArray data-holder NDArray"
```

---

## Task 4: Generalized native surface (EtTensor, EtNative, JNI `forward`)

**Files:**
- Create: `src/main/java/org/measly/executorch/jni/EtTensor.java`
- Create: `src/main/java/org/measly/executorch/jni/EtNative.java`
- Modify: `native/jni/executorch_djl_jni.cpp`
- Create: `src/test/java/org/measly/executorch/TestSupport.java`
- Test: `src/test/java/org/measly/executorch/jni/EtNativeTest.java`

- [ ] **Step 1: Native-availability helper**

Create `src/test/java/org/measly/executorch/TestSupport.java`:
```java
package org.measly.executorch;

import org.junit.jupiter.api.Assumptions;

/** Helpers for tests that require the native library. */
public final class TestSupport {

    private TestSupport() {}

    /** Skips the test (assumption) if libexecutorch_djl.so cannot be loaded. */
    public static void assumeNativeAvailable() {
        try {
            Class.forName("org.measly.executorch.jni.EtNative");
        } catch (Throwable t) { // UnsatisfiedLinkError, ExceptionInInitializerError, etc.
            Assumptions.abort("Native library not available: " + t.getMessage());
        }
    }

    /** Absolute path to the spike test model. */
    public static String addPtePath() {
        return new java.io.File("native/spike/add.pte").getAbsolutePath();
    }
}
```

- [ ] **Step 2: Write the failing native test**

Create `src/test/java/org/measly/executorch/jni/EtNativeTest.java`:
```java
package org.measly.executorch.jni;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.measly.executorch.TestSupport;
import org.junit.jupiter.api.Test;

class EtNativeTest {
    @Test
    void forwardAddsTwoScalars() {
        TestSupport.assumeNativeAvailable();
        long handle = EtNative.loadModule(TestSupport.addPtePath());
        try {
            EtTensor a = new EtTensor(new long[] {1}, new float[] {2f});
            EtTensor b = new EtTensor(new long[] {1}, new float[] {3f});
            EtTensor[] out = EtNative.forward(handle, new EtTensor[] {a, b});
            assertEquals(1, out.length);
            assertArrayEquals(new long[] {1}, out[0].shape);
            assertArrayEquals(new float[] {5f}, out[0].data);
        } finally {
            EtNative.destroy(handle);
        }
    }
}
```

- [ ] **Step 3: Run — verify it fails**

Run: `./gradlew test --tests '*EtNativeTest'`
Expected: FAIL to compile (no `EtNative`/`EtTensor`).

- [ ] **Step 4: Implement `EtTensor` and `EtNative`**

Create `src/main/java/org/measly/executorch/jni/EtTensor.java`:
```java
package org.measly.executorch.jni;

/** A float32 tensor crossing the JNI boundary. Phase 2 generalizes to ByteBuffer + dtype. */
public final class EtTensor {
    public final long[] shape;
    public final float[] data; // row-major, length == product(shape)

    public EtTensor(long[] shape, float[] data) {
        this.shape = shape;
        this.data = data;
    }
}
```

Create `src/main/java/org/measly/executorch/jni/EtNative.java`:
```java
package org.measly.executorch.jni;

import org.measly.executorch.engine.LibUtils;

/** JNI surface to the ExecuTorch native library. Loads the .so on class init. */
public final class EtNative {

    static {
        LibUtils.loadLibrary();
    }

    private EtNative() {}

    public static native long loadModule(String ptePath);

    public static native EtTensor[] forward(long handle, EtTensor[] inputs);

    public static native void destroy(long handle);
}
```

- [ ] **Step 5: Implement `LibUtils` (native resolution/loading)**

Create `src/main/java/org/measly/executorch/engine/LibUtils.java`:
```java
package org.measly.executorch.engine;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/** Resolves and loads libexecutorch_djl.so. */
public final class LibUtils {

    private static final String LIB = "libexecutorch_djl.so";
    private static boolean loaded;

    private LibUtils() {}

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
        String resource = "/native/" + platform + "/" + LIB;
        try (InputStream is = LibUtils.class.getResourceAsStream(resource)) {
            if (is == null) {
                throw new IllegalStateException(
                        "Native library not found on classpath: " + resource
                                + " (set EXECUTORCH_LIBRARY_PATH or run native/build_desktop.sh)");
            }
            Path tmp = Files.createTempFile("libexecutorch_djl", ".so");
            tmp.toFile().deleteOnExit();
            Files.copy(is, tmp, StandardCopyOption.REPLACE_EXISTING);
            System.load(tmp.toAbsolutePath().toString());
            loaded = true;
        } catch (IOException e) {
            throw new IllegalStateException("Failed to extract native library", e);
        }
    }

    private static String platform() {
        String os = System.getProperty("os.name").toLowerCase();
        String arch = System.getProperty("os.arch").toLowerCase();
        if (os.contains("linux") && (arch.equals("amd64") || arch.equals("x86_64"))) {
            return "linux-x86_64";
        }
        throw new UnsupportedOperationException(
                "ExecuTorch engine Phase 1 supports only linux-x86_64, got: " + os + "/" + arch);
    }
}
```

- [ ] **Step 6: Implement the JNI `forward` in C++**

Replace the body of `native/jni/executorch_djl_jni.cpp` with the generalized surface (keep `loadModule`/`destroy`, replace `forwardFloat` with `forward`):
```cpp
// Thin JNI shim over executorch::extension::Module. Raw JNI, no fbjni.
#include <jni.h>
#include <vector>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

using executorch::extension::Module;
using executorch::extension::from_blob;

extern "C" JNIEXPORT jlong JNICALL
Java_org_measly_executorch_jni_EtNative_loadModule(
    JNIEnv* env, jclass, jstring jpath) {
  const char* path = env->GetStringUTFChars(jpath, nullptr);
  auto* module = new Module(path);
  env->ReleaseStringUTFChars(jpath, path);
  return reinterpret_cast<jlong>(module);
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_org_measly_executorch_jni_EtNative_forward(
    JNIEnv* env, jclass, jlong handle, jobjectArray jinputs) {
  auto* module = reinterpret_cast<Module*>(handle);

  jclass tcls = env->FindClass("org/measly/executorch/jni/EtTensor");
  jfieldID fShape = env->GetFieldID(tcls, "shape", "[J");
  jfieldID fData = env->GetFieldID(tcls, "data", "[F");
  jmethodID ctor = env->GetMethodID(tcls, "<init>", "([J[F)V");

  jsize nIn = env->GetArrayLength(jinputs);

  // Backing storage must outlive forward(); from_blob does not copy.
  std::vector<std::vector<float>> inData(nIn);
  std::vector<std::vector<executorch::aten::SizesType>> inShape(nIn);
  std::vector<executorch::extension::TensorPtr> tensors;
  std::vector<executorch::runtime::EValue> evalues;
  tensors.reserve(nIn);
  evalues.reserve(nIn);

  for (jsize i = 0; i < nIn; ++i) {
    jobject jt = env->GetObjectArrayElement(jinputs, i);
    auto jshape = static_cast<jlongArray>(env->GetObjectField(jt, fShape));
    auto jdata = static_cast<jfloatArray>(env->GetObjectField(jt, fData));

    jsize nd = env->GetArrayLength(jshape);
    inShape[i].resize(nd);
    {
      std::vector<jlong> tmp(nd);
      env->GetLongArrayRegion(jshape, 0, nd, tmp.data());
      for (jsize k = 0; k < nd; ++k) {
        inShape[i][k] = static_cast<executorch::aten::SizesType>(tmp[k]);
      }
    }
    jsize nElem = env->GetArrayLength(jdata);
    inData[i].resize(nElem);
    env->GetFloatArrayRegion(jdata, 0, nElem, inData[i].data());

    tensors.push_back(from_blob(inData[i].data(), inShape[i]));
    evalues.emplace_back(tensors[i]);
  }

  auto result = module->forward(evalues);
  if (!result.ok()) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                  "ExecuTorch forward() failed");
    return nullptr;
  }

  const auto& outputs = result.get();
  jsize nOut = static_cast<jsize>(outputs.size());
  jobjectArray jout = env->NewObjectArray(nOut, tcls, nullptr);

  for (jsize i = 0; i < nOut; ++i) {
    auto t = outputs[i].toTensor();
    jsize ndim = static_cast<jsize>(t.dim());
    jlongArray jshape = env->NewLongArray(ndim);
    {
      std::vector<jlong> sh(ndim);
      for (jsize k = 0; k < ndim; ++k) {
        sh[k] = static_cast<jlong>(t.size(k));
      }
      env->SetLongArrayRegion(jshape, 0, ndim, sh.data());
    }
    jsize nElem = static_cast<jsize>(t.numel());
    jfloatArray jdata = env->NewFloatArray(nElem);
    env->SetFloatArrayRegion(jdata, 0, nElem, t.const_data_ptr<float>());

    jobject obj = env->NewObject(tcls, ctor, jshape, jdata);
    env->SetObjectArrayElement(jout, i, obj);
  }
  return jout;
}

extern "C" JNIEXPORT void JNICALL
Java_org_measly_executorch_jni_EtNative_destroy(
    JNIEnv*, jclass, jlong handle) {
  delete reinterpret_cast<Module*>(handle);
}
```

> The spike's `EtNative.java`/`EtSmoke.java` and `cpp_smoke` under `native/spike/` still reference the old `forwardFloat`. Leave `native/spike/` as-is (its `CMakeLists.txt` builds `cpp_smoke` + an old `executorch_djl`); the production build is `native/CMakeLists.txt`, which compiles `native/jni/executorch_djl_jni.cpp` — the file edited here. If `native/spike/` now fails to build, that's fine; it is not on the engine's path.

- [ ] **Step 7: Rebuild the native library**

Run:
```bash
ET_INSTALL="$HOME/workspace/executorch/cmake-out" JAVA_HOME=/usr/lib/jvm/zulu-17-amd64 \
  ./native/build_desktop.sh
```
Expected: builds and stages `src/main/resources/native/linux-x86_64/libexecutorch_djl.so`. If `executorch::aten::SizesType`, `TensorPtr`, or `EValue`/`forward` signatures differ in the pinned ExecuTorch headers, the compile error points to the exact line — adjust the type (e.g. the spike already proved `from_blob(ptr, {n})` and `outputs[i].toTensor().const_data_ptr<float>()` compile).

- [ ] **Step 8: Run — verify the native test passes**

Run: `./gradlew test --tests '*EtNativeTest'`
Expected: PASS — `out[0].data == [5.0]`.

- [ ] **Step 9: Commit**

```bash
git add src/main/java/org/measly/executorch/jni/EtTensor.java \
        src/main/java/org/measly/executorch/jni/EtNative.java \
        src/main/java/org/measly/executorch/engine/LibUtils.java \
        native/jni/executorch_djl_jni.cpp \
        src/test/java/org/measly/executorch/TestSupport.java \
        src/test/java/org/measly/executorch/jni/EtNativeTest.java \
        src/main/resources/native/linux-x86_64/libexecutorch_djl.so
git commit -m "feat(jni): generalized EtNative.forward(EtTensor[]) + lazy lib loading"
```

---

## Task 5: EtModel + EtSymbolBlock (wire load + forward)

**Files:**
- Create: `src/main/java/org/measly/executorch/engine/EtSymbolBlock.java`
- Create: `src/main/java/org/measly/executorch/engine/EtModel.java`
- Test: `src/test/java/org/measly/executorch/engine/EtModelTest.java`

- [ ] **Step 1: Write the failing model-forward test**

Create `src/test/java/org/measly/executorch/engine/EtModelTest.java`:
```java
package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import org.measly.executorch.TestSupport;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;

class EtModelTest {
    @Test
    void loadAndForwardAddModel() throws Exception {
        TestSupport.assumeNativeAvailable();
        try (Model model = Model.newInstance("add", "ExecuTorch")) {
            model.load(Paths.get("native/spike"), "add");
            try (NDManager manager = model.getNDManager().newSubManager()) {
                NDArray a = manager.create(new float[] {2f}, new Shape(1));
                NDArray b = manager.create(new float[] {3f}, new Shape(1));
                NDList out =
                        model.getBlock()
                                .forward(null, new NDList(a, b), false);
                assertArrayEquals(new float[] {5f}, out.singletonOrThrow().toFloatArray());
            }
        }
    }
}
```

- [ ] **Step 2: Run — verify it fails**

Run: `./gradlew test --tests '*EtModelTest'`
Expected: FAIL to compile (no `EtModel`/`EtSymbolBlock`).

- [ ] **Step 3: Implement `EtSymbolBlock`**

Create `src/main/java/org/measly/executorch/engine/EtSymbolBlock.java`:
```java
package org.measly.executorch.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractSymbolBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import org.measly.executorch.jni.EtNative;
import org.measly.executorch.jni.EtTensor;

/** Runs ExecuTorch forward() by marshalling NDList <-> EtTensor[]. */
public class EtSymbolBlock extends AbstractSymbolBlock implements AutoCloseable {

    private long handle;
    private EtNDManager manager;

    EtSymbolBlock(long handle, EtNDManager manager) {
        this.handle = handle;
        this.manager = manager;
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        EtTensor[] in = new EtTensor[inputs.size()];
        for (int i = 0; i < inputs.size(); ++i) {
            NDArray arr = inputs.get(i);
            in[i] = new EtTensor(arr.getShape().getShape(), arr.toFloatArray());
        }
        EtTensor[] out = EtNative.forward(handle, in);
        NDList ret = new NDList(out.length);
        for (EtTensor t : out) {
            ret.add(manager.create(t.data, new Shape(t.shape)));
        }
        ret.attach(inputs.head().getManager());
        return ret;
    }

    @Override
    public void removeLastBlock() {
        throw new UnsupportedOperationException("ExecuTorch does not support removeLastBlock");
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        throw new UnsupportedOperationException("ExecuTorch does not expose static output shapes");
    }

    @Override
    public void close() {
        if (handle != 0) {
            EtNative.destroy(handle);
            handle = 0;
        }
    }
}
```

> `manager.create(t.data, new Shape(t.shape))` uses the `NDManager` interface default `create(float[], Shape)` → `EtNDManager.create(Buffer, Shape, FLOAT32)`. Confirmed present as an interface default in DJL.

- [ ] **Step 4: Implement `EtModel`**

Create `src/main/java/org/measly/executorch/engine/EtModel.java`:
```java
package org.measly.executorch.engine;

import ai.djl.BaseModel;
import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import org.measly.executorch.jni.EtNative;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/** ExecuTorch {@link ai.djl.Model}: loads a .pte and owns its native handle via the block. */
public class EtModel extends BaseModel {

    EtModel(String name, NDManager manager) {
        super(name);
        this.manager = manager;
        this.manager.setName("etModel");
        dataType = DataType.FLOAT32;
    }

    @Override
    public void load(Path modelPath, String prefix, Map<String, ?> options)
            throws IOException, MalformedModelException {
        setModelDir(modelPath);
        if (block != null) {
            throw new UnsupportedOperationException("ExecuTorch does not support dynamic blocks");
        }
        Path modelFile;
        if (prefix != null) {
            modelFile = findModelFile(prefix);
        } else {
            modelFile = findModelFile(modelName, modelDir.toFile().getName(), "model.pte");
        }
        if (modelFile == null) {
            throw new FileNotFoundException(".pte file not found in: " + modelPath);
        }
        long handle = EtNative.loadModule(modelFile.toString());
        block = new EtSymbolBlock(handle, (EtNDManager) manager);
        wasLoaded = true;
    }

    @Override
    public void close() {
        if (block instanceof EtSymbolBlock) {
            ((EtSymbolBlock) block).close();
        }
        super.close();
    }
}
```

> `findModelFile(String...)` is a `protected` varargs helper on `BaseModel` (used identically by `OrtModel`). If `add.pte` isn't found, confirm `setModelDir`/`findModelFile` resolve `.pte` — `findModelFile(prefix)` checks `modelDir/prefix`, `modelDir/prefix.pte`, etc. The model dir is `native/spike`, prefix `add`, file `add.pte`.

- [ ] **Step 5: Run — verify pass (and the engine test now fully compiles)**

Run: `./gradlew test --tests '*EtModelTest' --tests '*EtEngineTest'`
Expected: PASS. (If `EtEngine.newModel` was temporarily stubbed in Task 2/3, restore it to `return new EtModel(name, newBaseManager(device));` now.)

- [ ] **Step 6: Commit**

```bash
git add src/main/java/org/measly/executorch/engine/EtSymbolBlock.java \
        src/main/java/org/measly/executorch/engine/EtModel.java \
        src/test/java/org/measly/executorch/engine/EtModelTest.java
git commit -m "feat(engine): EtModel load + EtSymbolBlock forward marshalling"
```

---

## Task 6: End-to-end Criteria acceptance test

**Files:**
- Test: `src/test/java/org/measly/executorch/AddModelIT.java`
- Create: `src/test/java/org/measly/executorch/AddTranslator.java`

- [ ] **Step 1: Write the translator the test uses**

Create `src/test/java/org/measly/executorch/AddTranslator.java`:
```java
package org.measly.executorch;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

/** Maps a 2-element float[] {a, b} to the add.pte inputs and back to a single float. */
public class AddTranslator implements Translator<float[], Float> {

    @Override
    public NDList processInput(TranslatorContext ctx, float[] input) {
        NDArray a = ctx.getNDManager().create(new float[] {input[0]}, new Shape(1));
        NDArray b = ctx.getNDManager().create(new float[] {input[1]}, new Shape(1));
        return new NDList(a, b);
    }

    @Override
    public Float processOutput(TranslatorContext ctx, NDList list) {
        return list.singletonOrThrow().toFloatArray()[0];
    }
}
```

- [ ] **Step 2: Write the failing end-to-end test**

Create `src/test/java/org/measly/executorch/AddModelIT.java`:
```java
package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;

class AddModelIT {
    @Test
    void predictThroughCriteria() throws Exception {
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
            assertEquals(5f, predictor.predict(new float[] {2f, 3f}), 1e-6);
        }
    }
}
```

- [ ] **Step 3: Run — verify it fails, then passes**

Run: `./gradlew test --tests '*AddModelIT'`
Expected: FAIL first if `AddTranslator` absent (compile) — then, with both files present and the native lib built, PASS: `predict({2,3}) == 5.0`. This is the **acceptance gate** — inference works through the public DJL API.

- [ ] **Step 4: Full suite green**

Run: `./gradlew test`
Expected: all tests pass (native-dependent ones run because the `.so` is staged; they'd be skipped, not failed, if it were absent).

- [ ] **Step 5: Commit**

```bash
git add src/test/java/org/measly/executorch/AddTranslator.java \
        src/test/java/org/measly/executorch/AddModelIT.java
git commit -m "test: end-to-end Criteria acceptance (add.pte -> 5.0)"
```

---

## Self-review against the spec

- **Spec §1 success test** → Task 6 (`AddModelIT` via `Criteria`, asserts 5.0). ✓
- **Spec §2 decisions** (Phase 1, data-holder, Path B) → Task 3 (data-holder `EtNDArray`), Task 4 (no fbjni/Android). ✓
- **Spec §3 native surface** (`EtTensor`, `loadModule`/`forward`/`destroy`) → Task 4. ✓
- **Spec §4 classes** → `EtEngineProvider`/`EtEngine` (Task 2), `EtModel`/`EtSymbolBlock` (Task 5), `EtNDManager`/`EtNDArray` (Task 3), `LibUtils` (Task 4). ✓ The §4 correction (forward in `EtSymbolBlock`, not `EtModel`) is realized in Task 5.
- **Spec §5 loading** (env override → bundled resource; lazy via `EtNative` static init) → `LibUtils` + `EtNative` (Task 4). ✓
- **Spec §6 Gradle** (single module, `compileOnly` DJL, SPI resource, staged `.so`, no PyTorch/fbjni) → Task 1 + Task 2 SPI file. ✓
- **Spec §7 API surface** → method set implemented across Tasks 2/3/5; names verified against DJL source. ✓
- **Spec §8 testing** (unit round-trip, SPI discovery, end-to-end, native guard) → Tasks 3, 2, 6, and `TestSupport` (Task 4). ✓

**Type consistency:** `EtNative.forward(long, EtTensor[]) -> EtTensor[]`, `EtTensor(long[] shape, float[] data)` (public fields), `EtNDManager.create(Buffer, Shape, DataType)`, `EtSymbolBlock(long, EtNDManager)`, `EtModel(String, NDManager)`, `LibUtils.loadLibrary()` — consistent across Tasks 4–6.

**Known cross-task ordering wrinkle (called out, not hidden):** `EtEngine.newModel`/`newBaseManager` reference `EtModel`/`EtNDManager` from later tasks. Mitigation stated in Tasks 2/3/5: either implement 2+3 together, or temporarily stub and restore in Task 5. `EtEngineTest` itself never calls those methods, so it passes as soon as the code compiles.

## Out of scope (later plans)

Named-parameter `MapTranslator` + `model_spec.json` (Phase 2), hybrid mode + PyTorch fallback and non-float dtypes (Phase 3), multi-platform/manylinux natives + release pipeline ([`ci-native-build.md`](../../ci-native-build.md)), Maven publication (Phase 4).
