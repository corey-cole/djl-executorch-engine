# ExecuTorch DJL Engine — Phase 2a Implementation Plan: Zero-Copy, Multi-Dtype Native Surface

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the engine's native surface from float32-only to a zero-copy, multi-dtype path supporting `{FLOAT32, FLOAT64, INT32, INT64, INT8, UINT8, BOOL}`, with model-metadata-driven arity/dtype validation.

**Architecture:** `EtTensor` carries a `ByteBuffer` + ExecuTorch ScalarType code instead of `float[]` — inputs are zero-copy direct buffers (`GetDirectBufferAddress`/`from_blob`), outputs are single-copy heap buffers (no native lifecycle). A new `method_meta` JNI call exposes input count + dtypes for validation. `EtSymbolBlock.forwardInternal` is rewritten to marshal any dtype and wrap outputs on the request manager with no extra copy.

**Tech Stack:** Java 17, `ai.djl:api:0.36.0`, JUnit 5, the existing `native/` CMake/JNI shim, ExecuTorch v1.3.1 C++ (`executorch::extension::Module`, `method_meta`, `from_blob`).

**Spec:** [`docs/superpowers/specs/2026-06-28-executorch-engine-phase2a-native-surface-design.md`](../specs/2026-06-28-executorch-engine-phase2a-native-surface-design.md).

---

## Starting point & build notes

- Branch off `main` (Phase 1 is merged). The working tree has an **uncommitted** minor tidy to `EtSymbolBlock.forwardInternal` (a hoisted `final int count`); Task 3 rewrites that method wholesale, so the tidy is subsumed — discard or ignore it.
- Native rebuilds use: `ET_INSTALL="$HOME/workspace/executorch/cmake-out" JAVA_HOME=/usr/lib/jvm/zulu-17-amd64 ./native/build_desktop.sh` (~1 min; stages `src/main/resources/native/linux-x86_64/libexecutorch_djl.so`).
- Native-dependent tests are guarded by `TestSupport.assumeNativeAvailable()` (exists).
- **Verified against installed ExecuTorch v1.3.1 headers:** `Module::method_meta(const std::string&) -> Result<MethodMeta>`; `MethodMeta::num_inputs() -> size_t`; `MethodMeta::input_tensor_meta(size_t) -> Result<TensorInfo>`; `TensorInfo::scalar_type() -> executorch::aten::ScalarType`. ScalarType codes (c10 canonical): Byte=0, Char=1, Int=3, Long=4, Float=6, Double=7, Bool=11.

## File structure

| File | Change |
|---|---|
| `src/main/java/org/measly/executorch/engine/EtDataTypes.java` | **Create** — `DataType` ↔ ScalarType code mapping |
| `src/main/java/org/measly/executorch/jni/EtMethodMeta.java` | **Create** — `{int numInputs, int[] inputScalarTypes}` |
| `src/main/java/org/measly/executorch/jni/EtTensor.java` | **Modify** — `float[]` → `ByteBuffer data` + `int scalarType` |
| `src/main/java/org/measly/executorch/jni/EtNative.java` | **Modify** — add `methodMeta`; `forward` signature unchanged (EtTensor changed) |
| `native/jni/executorch_djl_jni.cpp` | **Modify** — add `methodMeta`; rewrite `forward` for ByteBuffer/dtype; cache new IDs |
| `src/main/java/org/measly/executorch/engine/EtNDManager.java` | **Modify** — add `wrap(...)`; extend `copyInto` dtypes |
| `src/main/java/org/measly/executorch/engine/EtSymbolBlock.java` | **Modify** — store `EtMethodMeta`; arity check (Task 2); buffer/dtype forward (Task 3) |
| `src/main/java/org/measly/executorch/engine/EtModel.java` | **Modify** — query `methodMeta`, pass to block |
| `native/spike/export_dtypes.py`, `native/spike/dtypes.pte` | **Create** — mixed-dtype test fixture |
| `src/test/java/...` | tests per task |

---

## Task 1: `EtDataTypes` mapping (Java, native-free)

**Files:**
- Create: `src/main/java/org/measly/executorch/engine/EtDataTypes.java`
- Test: `src/test/java/org/measly/executorch/engine/EtDataTypesTest.java`

- [ ] **Step 1: Write the failing test**

Create `src/test/java/org/measly/executorch/engine/EtDataTypesTest.java`:
```java
package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import ai.djl.ndarray.types.DataType;
import org.junit.jupiter.api.Test;

class EtDataTypesTest {
    @Test
    void roundTripsSupportedTypes() {
        DataType[] types = {
            DataType.FLOAT32, DataType.FLOAT64, DataType.INT32,
            DataType.INT64, DataType.INT8, DataType.UINT8, DataType.BOOLEAN
        };
        for (DataType dt : types) {
            int code = EtDataTypes.toScalarType(dt);
            assertEquals(dt, EtDataTypes.fromScalarType(code), "round trip " + dt);
        }
    }

    @Test
    void knownCodes() {
        assertEquals(6, EtDataTypes.toScalarType(DataType.FLOAT32));
        assertEquals(4, EtDataTypes.toScalarType(DataType.INT64));
        assertEquals(0, EtDataTypes.toScalarType(DataType.UINT8));
        assertEquals(11, EtDataTypes.toScalarType(DataType.BOOLEAN));
    }

    @Test
    void rejectsUnsupported() {
        assertThrows(IllegalArgumentException.class,
                () -> EtDataTypes.toScalarType(DataType.FLOAT16));
        assertThrows(IllegalArgumentException.class,
                () -> EtDataTypes.fromScalarType(2)); // Short/INT16
    }
}
```

- [ ] **Step 2: Run — verify it fails**

Run: `./gradlew test --tests '*EtDataTypesTest'`
Expected: FAIL to compile (no `EtDataTypes`).

- [ ] **Step 3: Implement `EtDataTypes`**

Create `src/main/java/org/measly/executorch/engine/EtDataTypes.java`:
```java
package org.measly.executorch.engine;

import ai.djl.ndarray.types.DataType;

/** Maps DJL {@link DataType} to/from ExecuTorch ScalarType integer codes (c10 canonical). */
public final class EtDataTypes {

    private EtDataTypes() {}

    /** @return the ExecuTorch ScalarType code for {@code dataType}. */
    public static int toScalarType(DataType dataType) {
        switch (dataType) {
            case UINT8:   return 0;  // Byte
            case INT8:    return 1;  // Char
            case INT32:   return 3;  // Int
            case INT64:   return 4;  // Long
            case FLOAT32: return 6;  // Float
            case FLOAT64: return 7;  // Double
            case BOOLEAN: return 11; // Bool
            default:
                throw new IllegalArgumentException("Unsupported dtype for ExecuTorch: " + dataType);
        }
    }

    /** @return the DJL {@link DataType} for an ExecuTorch ScalarType code. */
    public static DataType fromScalarType(int scalarType) {
        switch (scalarType) {
            case 0:  return DataType.UINT8;
            case 1:  return DataType.INT8;
            case 3:  return DataType.INT32;
            case 4:  return DataType.INT64;
            case 6:  return DataType.FLOAT32;
            case 7:  return DataType.FLOAT64;
            case 11: return DataType.BOOLEAN;
            default:
                throw new IllegalArgumentException("Unsupported ExecuTorch ScalarType code: " + scalarType);
        }
    }
}
```

- [ ] **Step 4: Run — verify pass**

Run: `./gradlew test --tests '*EtDataTypesTest'`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/main/java/org/measly/executorch/engine/EtDataTypes.java \
        src/test/java/org/measly/executorch/engine/EtDataTypesTest.java
git commit -m "feat(engine): EtDataTypes DataType<->ScalarType mapping"
```
End the message body with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## Task 2: Model metadata + arity validation (EtTensor still float32)

This is **additive** — it adds the metadata query and arity check without touching the `float[]` forward path, so the existing float32 inference keeps working.

**Files:**
- Create: `src/main/java/org/measly/executorch/jni/EtMethodMeta.java`
- Modify: `src/main/java/org/measly/executorch/jni/EtNative.java`
- Modify: `native/jni/executorch_djl_jni.cpp`
- Modify: `src/main/java/org/measly/executorch/engine/EtModel.java`
- Modify: `src/main/java/org/measly/executorch/engine/EtSymbolBlock.java`
- Test: `src/test/java/org/measly/executorch/jni/EtMethodMetaTest.java`
- Test: modify `src/test/java/org/measly/executorch/engine/EtModelTest.java`

- [ ] **Step 1: Write the failing tests**

Create `src/test/java/org/measly/executorch/jni/EtMethodMetaTest.java`:
```java
package org.measly.executorch.jni;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.measly.executorch.TestSupport;
import org.junit.jupiter.api.Test;

class EtMethodMetaTest {
    @Test
    void readsAddModelMetadata() {
        TestSupport.assumeNativeAvailable();
        long handle = EtNative.loadModule(TestSupport.addPtePath());
        try {
            EtMethodMeta meta = EtNative.methodMeta(handle);
            assertEquals(2, meta.numInputs);
            assertArrayEquals(new int[] {6, 6}, meta.inputScalarTypes); // two float32 inputs
        } finally {
            EtNative.destroy(handle);
        }
    }
}
```

Add an arity test to `src/test/java/org/measly/executorch/engine/EtModelTest.java` (keep the existing `loadAndForwardAddModel` test):
```java
    @org.junit.jupiter.api.Test
    void wrongArityThrows() throws Exception {
        org.measly.executorch.TestSupport.assumeNativeAvailable();
        try (ai.djl.Model model = ai.djl.Model.newInstance("add", "ExecuTorch")) {
            model.load(java.nio.file.Paths.get("native/spike"), "add");
            try (ai.djl.ndarray.NDManager m = model.getNDManager().newSubManager()) {
                ai.djl.ndarray.NDArray only =
                        m.create(new float[] {2f}, new ai.djl.ndarray.types.Shape(1));
                org.junit.jupiter.api.Assertions.assertThrows(
                        IllegalArgumentException.class,
                        () -> model.getBlock().forward(null, new ai.djl.ndarray.NDList(only), false));
            }
        }
    }
```

- [ ] **Step 2: Run — verify they fail**

Run: `./gradlew test --tests '*EtMethodMetaTest' --tests '*EtModelTest'`
Expected: FAIL to compile (no `EtMethodMeta`/`EtNative.methodMeta`).

- [ ] **Step 3: Create `EtMethodMeta`**

Create `src/main/java/org/measly/executorch/jni/EtMethodMeta.java`:
```java
package org.measly.executorch.jni;

/** Static I/O metadata for a loaded module's "forward" method. */
public final class EtMethodMeta {
    public final int numInputs;
    public final int[] inputScalarTypes; // per-input ExecuTorch ScalarType code

    public EtMethodMeta(int numInputs, int[] inputScalarTypes) {
        this.numInputs = numInputs;
        this.inputScalarTypes = inputScalarTypes;
    }
}
```

- [ ] **Step 4: Add `methodMeta` to `EtNative`**

In `src/main/java/org/measly/executorch/jni/EtNative.java`, add the native method (keep the others):
```java
    public static native EtMethodMeta methodMeta(long handle);
```

- [ ] **Step 5: Implement `methodMeta` in JNI + cache its IDs**

In `native/jni/executorch_djl_jni.cpp`:

Add an include near the top (with the others):
```cpp
#include <executorch/runtime/executor/method_meta.h>
```

Add file-scope globals for `EtMethodMeta` (next to the existing `g_etTensorClass` group):
```cpp
static jclass g_etMethodMetaClass = nullptr;
static jmethodID g_metaCtor = nullptr;
```

In `JNI_OnLoad`, after the existing `EtTensor` caching, add (before `return JNI_VERSION_1_6;`):
```cpp
  jclass mlocal = env->FindClass("org/measly/executorch/jni/EtMethodMeta");
  if (mlocal == nullptr) {
    return JNI_ERR;
  }
  g_etMethodMetaClass = static_cast<jclass>(env->NewGlobalRef(mlocal));
  env->DeleteLocalRef(mlocal);
  g_metaCtor = env->GetMethodID(g_etMethodMetaClass, "<init>", "(I[I)V");
  if (g_metaCtor == nullptr) {
    return JNI_ERR;
  }
```

Add the JNI function (before the closing of the file, alongside `forward`):
```cpp
extern "C" JNIEXPORT jobject JNICALL
Java_org_measly_executorch_jni_EtNative_methodMeta(
    JNIEnv* env, jclass, jlong handle) {
  auto* module = reinterpret_cast<Module*>(handle);
  auto meta = module->method_meta("forward");
  if (!meta.ok()) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                  "method_meta(\"forward\") failed");
    return nullptr;
  }
  const auto n = static_cast<jsize>(meta->num_inputs());
  jintArray types = env->NewIntArray(n);
  std::vector<jint> tmp(n);
  for (jsize i = 0; i < n; ++i) {
    auto info = meta->input_tensor_meta(i);
    // Non-tensor inputs (rare) report an error; default such entries to -1.
    tmp[i] = info.ok() ? static_cast<jint>(info->scalar_type()) : -1;
  }
  env->SetIntArrayRegion(types, 0, n, tmp.data());
  return env->NewObject(g_etMethodMetaClass, g_metaCtor, static_cast<jint>(n), types);
}
```

- [ ] **Step 6: Cache metadata on the block; query it in `EtModel`**

In `EtSymbolBlock.java`: add a field and constructor parameter (leave `forwardInternal` body as-is except adding the arity check). Change the constructor and add the check at the top of `forwardInternal`:
```java
    private final EtMethodMeta meta;

    EtSymbolBlock(long handle, EtNDManager manager, EtMethodMeta meta) {
        this.handle = handle;
        this.manager = manager;
        this.meta = meta;
    }
```
Add as the first statement in `forwardInternal` (before building `in`):
```java
        if (inputs.size() != meta.numInputs) {
            throw new IllegalArgumentException(
                    "ExecuTorch model expects " + meta.numInputs + " inputs, got " + inputs.size());
        }
```
Add the import: `import org.measly.executorch.jni.EtMethodMeta;`

In `EtModel.java`, change the block construction in `load(...)`:
```java
        long handle = EtNative.loadModule(modelFile.toString());
        EtMethodMeta meta = EtNative.methodMeta(handle);
        block = new EtSymbolBlock(handle, (EtNDManager) manager, meta);
```
Add the import: `import org.measly.executorch.jni.EtMethodMeta;`

- [ ] **Step 7: Rebuild native + run tests**

Run:
```bash
ET_INSTALL="$HOME/workspace/executorch/cmake-out" JAVA_HOME=/usr/lib/jvm/zulu-17-amd64 ./native/build_desktop.sh
./gradlew test --tests '*EtMethodMetaTest' --tests '*EtModelTest' --tests '*EtNativeTest'
```
Expected: PASS — metadata `(2,[6,6])`, wrong-arity throws `IllegalArgumentException`, existing forward still 5.0.

- [ ] **Step 8: Commit**

```bash
git add src/main/java/org/measly/executorch/jni/EtMethodMeta.java \
        src/main/java/org/measly/executorch/jni/EtNative.java \
        native/jni/executorch_djl_jni.cpp \
        src/main/java/org/measly/executorch/engine/EtSymbolBlock.java \
        src/main/java/org/measly/executorch/engine/EtModel.java \
        src/test/java/org/measly/executorch/jni/EtMethodMetaTest.java \
        src/test/java/org/measly/executorch/engine/EtModelTest.java \
        src/main/resources/native/linux-x86_64/libexecutorch_djl.so
git commit -m "feat(engine): method_meta query + input-arity validation"
```
(+ Co-Authored-By trailer.)

---

## Task 3: Migrate `EtTensor` to `ByteBuffer` + ScalarType (zero-copy, multi-dtype core)

This is the coupled surface migration: `EtTensor`, the C++ `forward`, `EtNDManager.wrap`/`copyInto`, and `EtSymbolBlock.forwardInternal` change atomically.

**Files:**
- Modify: `src/main/java/org/measly/executorch/jni/EtTensor.java`
- Modify: `native/jni/executorch_djl_jni.cpp`
- Modify: `src/main/java/org/measly/executorch/engine/EtNDManager.java`
- Modify: `src/main/java/org/measly/executorch/engine/EtSymbolBlock.java`
- Modify: `src/test/java/org/measly/executorch/jni/EtNativeTest.java`

- [ ] **Step 1: Rewrite the native test to the ByteBuffer surface (failing)**

Replace the body of `src/test/java/org/measly/executorch/jni/EtNativeTest.java`:
```java
package org.measly.executorch.jni;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.measly.executorch.TestSupport;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;

class EtNativeTest {
    private static EtTensor floatScalar(float v) {
        ByteBuffer b = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder());
        b.putFloat(0, v);
        return new EtTensor(new long[] {1}, 6 /*Float*/, b);
    }

    @Test
    void forwardAddsTwoScalars() {
        TestSupport.assumeNativeAvailable();
        long handle = EtNative.loadModule(TestSupport.addPtePath());
        try {
            EtTensor[] out = EtNative.forward(
                    handle, new EtTensor[] {floatScalar(2f), floatScalar(3f)});
            assertEquals(1, out.length);
            assertArrayEquals(new long[] {1}, out[0].shape);
            assertEquals(6, out[0].scalarType);
            assertEquals(5f, out[0].data.order(ByteOrder.nativeOrder()).getFloat(0), 1e-6);
        } finally {
            EtNative.destroy(handle);
        }
    }
}
```

- [ ] **Step 2: Run — verify it fails**

Run: `./gradlew test --tests '*EtNativeTest'`
Expected: FAIL to compile (`EtTensor` still has `float[] data`, no `scalarType`).

- [ ] **Step 3: Generalize `EtTensor`**

Replace `src/main/java/org/measly/executorch/jni/EtTensor.java`:
```java
package org.measly.executorch.jni;

import java.nio.ByteBuffer;

/** A tensor crossing the JNI boundary: raw bytes + ExecuTorch ScalarType code + shape. */
public final class EtTensor {
    public final long[] shape;
    public final int scalarType;  // ExecuTorch ScalarType int code
    public final ByteBuffer data; // input: DIRECT (zero-copy); output: heap (single-copy)

    public EtTensor(long[] shape, int scalarType, ByteBuffer data) {
        this.shape = shape;
        this.scalarType = scalarType;
        this.data = data;
    }
}
```

- [ ] **Step 4: Rewrite the JNI `forward` for ByteBuffer/dtype**

In `native/jni/executorch_djl_jni.cpp`:

Update the cached `EtTensor` field/ctor IDs in `JNI_OnLoad` — the field types changed, so replace the `g_fData`/`g_ctor` lookups for `EtTensor` with:
```cpp
  g_fShape = env->GetFieldID(g_etTensorClass, "shape", "[J");
  g_fScalarType = env->GetFieldID(g_etTensorClass, "scalarType", "I");
  g_fData = env->GetFieldID(g_etTensorClass, "data", "Ljava/nio/ByteBuffer;");
  g_ctor = env->GetMethodID(g_etTensorClass, "<init>", "([JILjava/nio/ByteBuffer;)V");
  if (g_fShape == nullptr || g_fScalarType == nullptr || g_fData == nullptr || g_ctor == nullptr) {
    return JNI_ERR;
  }
```
Add the new global `static jfieldID g_fScalarType = nullptr;` (next to `g_fShape`/`g_fData`), and cache `java/nio/ByteBuffer` + its `wrap` method for building heap outputs:
```cpp
static jclass g_byteBufferClass = nullptr;
static jmethodID g_byteBufferWrap = nullptr;
```
In `JNI_OnLoad` (after the EtTensor block):
```cpp
  jclass bblocal = env->FindClass("java/nio/ByteBuffer");
  if (bblocal == nullptr) {
    return JNI_ERR;
  }
  g_byteBufferClass = static_cast<jclass>(env->NewGlobalRef(bblocal));
  env->DeleteLocalRef(bblocal);
  g_byteBufferWrap = env->GetStaticMethodID(g_byteBufferClass, "wrap", "([B)Ljava/nio/ByteBuffer;");
  if (g_byteBufferWrap == nullptr) {
    return JNI_ERR;
  }
```

Replace the entire `Java_..._forward` function body with the dtype-general, zero-copy-in/heap-out version:
```cpp
extern "C" JNIEXPORT jobjectArray JNICALL
Java_org_measly_executorch_jni_EtNative_forward(
    JNIEnv* env, jclass, jlong handle, jobjectArray jinputs) {
  auto* module = reinterpret_cast<Module*>(handle);

  jsize nIn = env->GetArrayLength(jinputs);
  std::vector<std::vector<executorch::aten::SizesType>> inShape(nIn);
  std::vector<executorch::extension::TensorPtr> tensors;
  std::vector<executorch::runtime::EValue> evalues;
  tensors.reserve(nIn);
  evalues.reserve(nIn);

  for (jsize i = 0; i < nIn; ++i) {
    jobject jt = env->GetObjectArrayElement(jinputs, i);
    auto jshape = static_cast<jlongArray>(env->GetObjectField(jt, g_fShape));
    jint st = env->GetIntField(jt, g_fScalarType);
    jobject jbuf = env->GetObjectField(jt, g_fData);

    jsize nd = env->GetArrayLength(jshape);
    inShape[i].resize(nd);
    {
      std::vector<jlong> tmp(nd);
      env->GetLongArrayRegion(jshape, 0, nd, tmp.data());
      for (jsize k = 0; k < nd; ++k) {
        inShape[i][k] = static_cast<executorch::aten::SizesType>(tmp[k]);
      }
    }
    // Zero-copy: read the direct ByteBuffer's memory in place (valid for the whole call).
    void* addr = env->GetDirectBufferAddress(jbuf);
    tensors.push_back(from_blob(
        addr, inShape[i], static_cast<executorch::aten::ScalarType>(st)));
    evalues.emplace_back(tensors[i]);

    env->DeleteLocalRef(jshape);
    env->DeleteLocalRef(jbuf);
    env->DeleteLocalRef(jt);
  }

  auto result = module->forward(evalues);
  if (!result.ok()) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                  "ExecuTorch forward() failed");
    return nullptr;
  }

  const auto& outputs = *result;
  jsize nOut = static_cast<jsize>(outputs.size());
  jobjectArray jout = env->NewObjectArray(nOut, g_etTensorClass, nullptr);

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

    // Single copy native -> heap byte[] -> heap ByteBuffer (no native memory outlives the call).
    jsize nbytes = static_cast<jsize>(t.nbytes());
    jbyteArray jbytes = env->NewByteArray(nbytes);
    env->SetByteArrayRegion(
        jbytes, 0, nbytes, reinterpret_cast<const jbyte*>(t.const_data_ptr()));
    jobject jbuf = env->CallStaticObjectMethod(g_byteBufferClass, g_byteBufferWrap, jbytes);

    jint st = static_cast<jint>(t.scalar_type());
    jobject obj = env->NewObject(g_etTensorClass, g_ctor, jshape, st, jbuf);
    env->SetObjectArrayElement(jout, i, obj);

    env->DeleteLocalRef(jshape);
    env->DeleteLocalRef(jbytes);
    env->DeleteLocalRef(jbuf);
    env->DeleteLocalRef(obj);
  }
  return jout;
}
```

- [ ] **Step 5: Add `EtNDManager.wrap` and extend `copyInto`**

In `src/main/java/org/measly/executorch/engine/EtNDManager.java`:

Add the no-copy wrap method (after `create(Buffer, Shape, DataType)`):
```java
    /** Wraps an existing buffer as an EtNDArray WITHOUT copying (for adopting forward() outputs). */
    EtNDArray wrap(java.nio.ByteBuffer data, Shape shape, DataType dataType) {
        return new EtNDArray(this, alternativeManager, data, shape, dataType);
    }
```
Extend `copyInto` to handle INT8/UINT8/BOOLEAN (they arrive as byte data via the `ByteBuffer` branch already, but add explicit cases so a typed `create(byte[]...)` path is covered; the `default` no longer throws for these). Replace the `switch` default arm so the three byte-width types pass through the ByteBuffer copy:
```java
            case INT8:
            case UINT8:
            case BOOLEAN:
                // 1-byte types: data is a ByteBuffer (handled above) or copied byte-wise.
                dst.put((ByteBuffer) src);
                break;
```
(Place these cases before `default`. The existing `if (src instanceof ByteBuffer) { dst.put(...); return; }` guard already covers the common path; these cases handle a typed-but-byte source defensively.)

- [ ] **Step 6: Rewrite `EtSymbolBlock.forwardInternal` for any dtype**

Replace the `forwardInternal` body in `src/main/java/org/measly/executorch/engine/EtSymbolBlock.java` (keep the arity check from Task 2 as step 1):
```java
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        final int count = inputs.size();
        if (count != meta.numInputs) {
            throw new IllegalArgumentException(
                    "ExecuTorch model expects " + meta.numInputs + " inputs, got " + count);
        }
        EtTensor[] in = new EtTensor[count];
        for (int i = 0; i < count; ++i) {
            EtNDArray et = manager.from(inputs.get(i));
            int st = EtDataTypes.toScalarType(et.getDataType());
            if (st != meta.inputScalarTypes[i]) {
                throw new IllegalArgumentException(
                        "Input " + i + " dtype " + et.getDataType()
                                + " != model's expected ScalarType " + meta.inputScalarTypes[i]);
            }
            in[i] = new EtTensor(et.getShape().getShape(), st, et.toByteBuffer());
        }
        EtTensor[] out = EtNative.forward(handle, in);
        NDManager rm = inputs.isEmpty() ? manager : inputs.head().getManager();
        EtNDManager target = (rm instanceof EtNDManager) ? (EtNDManager) rm : manager;
        NDList ret = new NDList(out.length);
        for (EtTensor t : out) {
            DataType dt = EtDataTypes.fromScalarType(t.scalarType);
            ret.add(target.wrap(t.data, new Shape(t.shape), dt));
        }
        return ret;
    }
```
Ensure imports include `ai.djl.ndarray.NDManager` and `org.measly.executorch.engine.EtDataTypes` (same package, no import needed) and `ai.djl.ndarray.types.DataType`. Remove the now-unused `DataType.FLOAT32` reference if the old check is gone.

- [ ] **Step 7: Rebuild native + run tests**

Run:
```bash
ET_INSTALL="$HOME/workspace/executorch/cmake-out" JAVA_HOME=/usr/lib/jvm/zulu-17-amd64 ./native/build_desktop.sh
./gradlew test --tests '*EtNativeTest' --tests '*EtModelTest' --tests '*AddModelIT'
```
Expected: PASS — `EtNativeTest` forward via ByteBuffers returns 5.0; the float32 `add.pte` model + Criteria paths still green. If `from_blob` lacks a 3-arg (ptr, sizes, ScalarType) overload in the pinned headers, check `executorch/extension/tensor/tensor.h` for the dtype parameter and adjust; `t.nbytes()`/`t.const_data_ptr()`/`t.scalar_type()` are on `executorch::aten::Tensor`.

- [ ] **Step 8: Commit**

```bash
git add src/main/java/org/measly/executorch/jni/EtTensor.java \
        native/jni/executorch_djl_jni.cpp \
        src/main/java/org/measly/executorch/engine/EtNDManager.java \
        src/main/java/org/measly/executorch/engine/EtSymbolBlock.java \
        src/test/java/org/measly/executorch/jni/EtNativeTest.java \
        src/main/resources/native/linux-x86_64/libexecutorch_djl.so
git commit -m "feat(jni): zero-copy multi-dtype EtTensor (ByteBuffer + ScalarType)"
```
(+ Co-Authored-By trailer.)

---

## Task 4: Multi-dtype acceptance — fixture + end-to-end + dtype validation

**Files:**
- Create: `native/spike/export_dtypes.py`, `native/spike/dtypes.pte`
- Create: `src/test/java/org/measly/executorch/PassthroughTranslator.java`
- Create: `src/test/java/org/measly/executorch/MultiDtypeIT.java`
- Test: native-free dtype round-trip in `src/test/java/org/measly/executorch/engine/EtNDArrayTest.java`

- [ ] **Step 1: Write the Python export for a mixed-dtype model**

Create `native/spike/export_dtypes.py`:
```python
"""Export a mixed-dtype model (int64 a, float32 b) -> (a+a int64, b+b float32) to dtypes.pte.

Exported WITHOUT the XNNPACK partitioner so the portable kernels handle int64 ops.
"""
import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower


class MixedDtypes(torch.nn.Module):
    def forward(self, a, b):
        return a + a, b + b


def main() -> None:
    model = MixedDtypes().eval()
    example_inputs = (torch.ones(1, dtype=torch.int64), torch.ones(1, dtype=torch.float32))
    exported = export(model, example_inputs)
    lowered = to_edge_transform_and_lower(exported).to_executorch()
    with open("dtypes.pte", "wb") as f:
        f.write(lowered.buffer)
    print("wrote dtypes.pte")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the fixture**

Run:
```bash
cd ~/workspace/djl-executorch-engine/native/spike && \
source ~/workspace/executorch/.venv/bin/activate && python3 export_dtypes.py
```
Expected: prints `wrote dtypes.pte`; `dtypes.pte` exists. (If `int64` add isn't supported by the lowered op set, change the model to `return a, b + b` — a passthrough int64 output still exercises multi-dtype marshalling. Note which you used.)

- [ ] **Step 3: Write a pass-through NDList translator (null batchifier)**

Create `src/test/java/org/measly/executorch/PassthroughTranslator.java`:
```java
package org.measly.executorch;

import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

/** Identity NDList translator; null batchifier (EtNDArray has no NDArrayInternal / stacking). */
public class PassthroughTranslator implements Translator<NDList, NDList> {
    @Override
    public NDList processInput(TranslatorContext ctx, NDList input) {
        return input;
    }

    @Override
    public NDList processOutput(TranslatorContext ctx, NDList list) {
        return list;
    }

    @Override
    public Batchifier getBatchifier() {
        return null;
    }
}
```

- [ ] **Step 4: Write the failing multi-dtype acceptance test**

Create `src/test/java/org/measly/executorch/MultiDtypeIT.java`:
```java
package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;

class MultiDtypeIT {
    @Test
    void int64AndFloat32ThroughPredictor() throws Exception {
        TestSupport.assumeNativeAvailable();
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optEngine("ExecuTorch")
                        .optModelPath(Paths.get("native/spike"))
                        .optModelName("dtypes")
                        .optTranslator(new PassthroughTranslator())
                        .build();
        try (ZooModel<NDList, NDList> model = criteria.loadModel();
                Predictor<NDList, NDList> predictor = model.newPredictor();
                NDManager m = model.getNDManager().newSubManager()) {
            NDArray a = m.create(new long[] {7L}, new Shape(1));   // int64
            NDArray b = m.create(new float[] {2.5f}, new Shape(1)); // float32
            NDList out = predictor.predict(new NDList(a, b));
            assertEquals(2, out.size());
            assertEquals(DataType.INT64, out.get(0).getDataType());
            assertArrayEquals(new long[] {14L}, out.get(0).toLongArray());   // a + a
            assertEquals(DataType.FLOAT32, out.get(1).getDataType());
            assertArrayEquals(new float[] {5.0f}, out.get(1).toFloatArray()); // b + b
        }
    }
}
```
(If Step 2 used the `return a, b + b` passthrough variant, change the int64 assertion to `new long[] {7L}`.)

- [ ] **Step 5: Add native-free dtype round-trips to `EtNDArrayTest`**

Add to `src/test/java/org/measly/executorch/engine/EtNDArrayTest.java`:
```java
    @org.junit.jupiter.api.Test
    void int64RoundTrip() {
        try (ai.djl.ndarray.NDManager manager =
                ai.djl.ndarray.NDManager.newBaseManager("ExecuTorch")) {
            ai.djl.ndarray.NDArray arr =
                    manager.create(new long[] {1L, 2L, 3L}, new ai.djl.ndarray.types.Shape(3));
            org.junit.jupiter.api.Assertions.assertEquals(
                    ai.djl.ndarray.types.DataType.INT64, arr.getDataType());
            org.junit.jupiter.api.Assertions.assertArrayEquals(
                    new long[] {1L, 2L, 3L}, arr.toLongArray());
        }
    }

    @org.junit.jupiter.api.Test
    void intAndDoubleRoundTrip() {
        try (ai.djl.ndarray.NDManager manager =
                ai.djl.ndarray.NDManager.newBaseManager("ExecuTorch")) {
            org.junit.jupiter.api.Assertions.assertArrayEquals(
                    new int[] {4, 5},
                    manager.create(new int[] {4, 5}, new ai.djl.ndarray.types.Shape(2)).toIntArray());
            org.junit.jupiter.api.Assertions.assertArrayEquals(
                    new double[] {1.5, 2.5},
                    manager.create(new double[] {1.5, 2.5}, new ai.djl.ndarray.types.Shape(2))
                            .toDoubleArray());
        }
    }
```

- [ ] **Step 6: Run — verify pass + full suite**

Run:
```bash
./gradlew test --tests '*MultiDtypeIT' --tests '*EtNDArrayTest'
./gradlew test
```
Expected: multi-dtype `(14 int64, 5.0 float32)`; dtype round-trips pass; **entire suite green** (Phase 1 tests + all Phase 2a tests).

- [ ] **Step 7: Commit**

```bash
git add native/spike/export_dtypes.py native/spike/dtypes.pte \
        src/test/java/org/measly/executorch/PassthroughTranslator.java \
        src/test/java/org/measly/executorch/MultiDtypeIT.java \
        src/test/java/org/measly/executorch/engine/EtNDArrayTest.java
git commit -m "test: multi-dtype end-to-end (int64 + float32) + dtype round-trips"
```
(+ Co-Authored-By trailer.)

---

## Self-review against the spec

- **§1 success tests** → multi-dtype end-to-end (Task 4 `MultiDtypeIT`), arity validation (Task 2 `EtModelTest.wrongArityThrows`), float32 regression (existing `AddModelIT`/`EtModelTest` rerun in Tasks 2–4). ✓
- **§2 decisions** → 2a-only scope; `EtTensor` ByteBuffer+code (Task 3); heap outputs no-lifecycle (Task 3 JNI); metadata (Task 2); 7 dtypes (Tasks 1/3/4); no C++ framework (tests are Java). ✓
- **§3 native surface** → `EtTensor` (T3), `EtMethodMeta`+`methodMeta` (T2), `EtDataTypes` (T1), asymmetric direct-in/heap-out (T3 JNI). ✓
- **§4 forwardInternal** → arity (T2) + buffer/dtype/wrap rewrite (T3). ✓
- **§5 touchpoints** → `EtNDManager.wrap`+`copyInto` (T3), `EtModel` metadata wiring (T2). `EtNDArray` unchanged; extraction via adapter validated by T4 round-trips. ✓
- **§6 testing** → integration + native-free units across T1–T4; `cpp_smoke` retained (no C++ framework). ✓

**Type consistency:** `EtTensor(long[] shape, int scalarType, ByteBuffer data)`, `EtMethodMeta(int numInputs, int[] inputScalarTypes)`, `EtNative.methodMeta(long)→EtMethodMeta`, `EtNative.forward(long, EtTensor[])→EtTensor[]`, `EtDataTypes.{toScalarType(DataType)→int, fromScalarType(int)→DataType}`, `EtNDManager.wrap(ByteBuffer, Shape, DataType)→EtNDArray`, `EtSymbolBlock(long, EtNDManager, EtMethodMeta)` — consistent across tasks. ScalarType codes (6/4/0/11 etc.) consistent between `EtDataTypes` and the JNI/tests.

**Known cross-task ordering:** Task 2 changes the `EtSymbolBlock` constructor (adds `EtMethodMeta`) and `EtModel`; Task 3 changes `EtSymbolBlock.forwardInternal` body. Both touch `EtSymbolBlock` — sequential, no conflict. The `native/spike/cpp_smoke.cpp` and old spike files still reference the float-only surface; they are off the engine build path (built only by `native/spike/CMakeLists.txt`) and may go stale — not a concern.

## Out of scope (later)

Named-parameter `MapTranslator` + `model_spec.json` + `ParamSpec`/`DType` (Phase 2b, builds on this), `FLOAT16`, multi-method modules, dtype widening/coercion, and direct-buffer/native-lifecycle outputs (the autoregressive revisit path).
