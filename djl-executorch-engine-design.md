# ExecuTorch Engine for DJL — Implementation Design

## Overview

This document describes the design for an out-of-tree DJL engine backed by ExecuTorch, PyTorch's on-device inference runtime. The engine loads `.pte` model artifacts (produced via `torch.export` → ExecuTorch compilation) and exposes them through DJL's standard `Model`, `Predictor`, and `Translator` interfaces.

### Why ExecuTorch as a DJL Engine

- ExecuTorch's runtime (`libexecutorch`) does **not** depend on `libtorch`. It can coexist in the same JVM process as DJL's PyTorch engine with zero symbol conflicts and no shared-library version coupling.
- The runtime is ~50KB base footprint (vs. hundreds of MB for libtorch).
- Models are exported via `torch.export`, the designated successor to TorchScript (which DJL's PyTorch engine currently requires and which is in maintenance mode).
- ExecuTorch already ships Java/JNI bindings (`org.pytorch.executorch`) for Android. The Java layer and native JNI code are portable to desktop JVMs with recompilation of the native `.so` against a standard JDK.
- DJL's engine SPI is explicitly designed for pluggable backends. The ONNX Runtime engine is the closest structural precedent — it also has limited NDArray support and runs in hybrid mode with a full engine for pre/post-processing.

### Out-of-Tree Strategy

Publish as a standalone Maven artifact (e.g., `org.measly:djl-executorch-engine`) that depends on `ai.djl:api` at compile time. Structure it identically to an in-tree engine so it can be submitted as a PR to `deepjavalibrary/djl` later. Users add it to their classpath alongside `ai.djl:api`, and DJL discovers it via Java SPI.

---

> **Status update (2026-06-28) — native build feasibility researched.**
> The show-stopper (build ExecuTorch natives for desktop Linux) is **practical**. Detailed, gated plan: [`docs/superpowers/plans/2026-06-28-executorch-desktop-linux-native-build.md`](docs/superpowers/plans/2026-06-28-executorch-desktop-linux-native-build.md).
> Key findings that **revise this design** (see corrected notes inline in §2, §5, §6):
> - The C++ runtime + XNNPACK build on desktop Linux is first-class (`cmake --preset linux`). Low risk.
> - **Option A (recompile the stock Android JNI) is NOT the recommended path.** `extension/android/CMakeLists.txt` hard-fails with `if(NOT ANDROID) → FATAL_ERROR "This directory is for Android build only."`, downloads a *prebuilt Android* `fbjni` AAR, and relies on the NDK for `jni.h`. The plan therefore drives **Option B (thin custom JNI shim over `executorch::extension::Module`, no fbjni)** as primary; Option A is a documented fallback only.
> - Consequence: the `org.pytorch.executorch.{Module,Tensor,EValue}` Java classes are **not** reused under Option B — the engine's `EtModel`/`EtNDArray` call a small custom native surface (`org.measly.executorch.jni.EtNative`) instead. §3c's `import org.pytorch.executorch.Module` reflects the old Option-A assumption.

---

## 1. Reference Architecture: How DJL Engines Work

### Engine Discovery

DJL discovers engines through `java.util.ServiceLoader`. Each engine provides a file at:

```
META-INF/services/ai.djl.engine.EngineProvider
```

containing the fully qualified class name of its `EngineProvider` implementation. `Engine.getEngine("ExecuTorch")` triggers loading.

### Key Interfaces and Classes to Implement

Study the ONNX Runtime engine (`engines/onnxruntime/onnxruntime-engine/`) as the primary reference. It is the closest precedent: limited NDArray support, uses an external runtime's Java API, supports hybrid mode.

| DJL Contract | OnnxRuntime Implementation | ExecuTorch Equivalent |
|---|---|---|
| `EngineProvider` | `OrtEngineProvider` | `EtEngineProvider` |
| `Engine` | `OrtEngine` | `EtEngine` |
| `Model` | `OrtModel` | `EtModel` |
| `NDManager` | `OrtNDManager` | `EtNDManager` |
| `NDArray` | `OrtNDArray` | `EtNDArray` |
| `Predictor` | (uses base class) | (uses base class) |
| `Translator` | (user implements) | Provide a `MapTranslator` for `Map<String, Number>` use case |

### Hybrid Mode

DJL's hybrid engine design allows engines with limited NDArray support to delegate unsupported operations (reshaping, slicing, arithmetic) to a full engine like PyTorch. This is documented at `docs/hybrid_engine.md` in the DJL repo. The ExecuTorch engine should declare limited NDArray support (basic creation methods only) and rely on hybrid mode for any pre/post-processing that needs richer tensor operations.

From the ONNX Runtime README:
> *"ONNX Runtime is a DL library with limited support for NDArray operations. Currently, it only covers the basic NDArray creation methods. To better support the necessary preprocessing and postprocessing, you can use one of the other Engines along with it to run in a hybrid mode."*

The ExecuTorch engine should follow this same pattern.

---

## 2. ExecuTorch Java API

### Existing Bindings

ExecuTorch publishes Android AAR artifacts to Maven Central: `org.pytorch:executorch-android:{version}`. The current stable version is 1.3 (latest release), with 1.0.1 as a recent patch release. The AAR contains:

- Java classes in `org.pytorch.executorch` package
- JNI shared library (`.so`) for `arm64-v8a` and `x86_64`
- Dependencies on `com.facebook.fbjni:fbjni-java-only` and `com.facebook.soloader:nativeloader`

### Core API Surface

From usage patterns across ExecuTorch GitHub issues and documentation:

```java
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;
import org.pytorch.executorch.EValue;

// Load model
Module module = Module.load("/path/to/model.pte");

// Create input tensor
float[] inputData = new float[]{1.0f, 2.0f, 3.0f};
Tensor inputTensor = Tensor.fromBlob(inputData, new long[]{1, 3});

// Run inference
EValue[] result = module.forward(EValue.from(inputTensor));

// Extract output
Tensor outputTensor = result[0].toTensor();
float[] outputData = outputTensor.getDataAsFloatArray();
```

Key classes:

- **`Module`** — Loads a `.pte` file and runs `forward()` or arbitrary `execute(methodName, ...)`. Wraps a `NativePeer` for JNI calls.
- **`Tensor`** — Created from Java arrays via `Tensor.fromBlob(data, shape)`. Supports `float[]`, `int[]`, `long[]`, `double[]`, `byte[]`. Data is copied into native memory.
- **`EValue`** — Tagged union wrapping `Tensor`, `bool`, `int`, `double`, `String`, or lists thereof. Method arguments and return values are `EValue[]`.
- **`NativePeer`** — JNI bridge. Handles lifecycle (`initHybrid`, `resetNative`). Uses Facebook's `fbjni` library for JNI binding.

### Desktop Portability Concern

The Android AAR's JNI `.so` is compiled against Android NDK. For desktop JVM use, two options:

**Option A: Recompile from source.** Build the ExecuTorch C++ runtime and JNI bindings against a standard JDK's `jni.h`. The JNI code in `executorch/extension/android/src/main/java/org/pytorch/executorch/` is standard Java — no Android framework dependencies. The native side at `executorch/extension/android/jni/` uses standard JNI plus `fbjni`. This is the recommended path.

**Option B: Rewrite JNI layer.** Write a new JNI (or JNA) bridge that calls ExecuTorch's C++ `Module` API directly, bypassing the Android-oriented `NativePeer`. Removes the `fbjni`/`soloader` dependencies entirely.

> **Corrected recommendation (2026-06-28):** **pursue Option B**, not Option A. Research showed Option A is more entangled with Android than originally assumed: `extension/android/CMakeLists.txt` aborts unless building for Android (`if(NOT ANDROID) → FATAL_ERROR`), pulls a *prebuilt Android* `fbjni` AAR from Maven, and depends on the NDK supplying `jni.h`. Option B is a single `.cpp` using `<jni.h>` (via host JDK `find_package(JNI)`) + `executorch::extension::Module`, with no fbjni and nothing Android-specific. A proof-of-concept shim is specified in the native-build plan. Reserve Option A only if reusing the upstream `org.pytorch.executorch.*` Java classes verbatim turns out to be worth the extra moving parts (host `fbjni` build, patching the Android guard).

---

## 3. Engine Implementation Classes

### 3a. EtEngineProvider

```java
package org.measly.executorch.engine;

import ai.djl.engine.Engine;
import ai.djl.engine.EngineProvider;

public class EtEngineProvider implements EngineProvider {

    @Override
    public String getEngineName() {
        return "ExecuTorch";
    }

    @Override
    public int getEngineRank() {
        // Lower rank = lower priority as default engine
        // OnnxRuntime uses 10. Use similar value so PyTorch/TF
        // remain the default when present.
        return 10;
    }

    @Override
    public Engine getEngine() {
        return EtEngine.newInstance();
    }
}
```

Register via SPI:

```
# META-INF/services/ai.djl.engine.EngineProvider
org.measly.executorch.engine.EtEngineProvider
```

### 3b. EtEngine

The engine is a singleton that manages the ExecuTorch runtime lifecycle.

```java
package org.measly.executorch.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;

public final class EtEngine extends Engine {

    public static final String ENGINE_NAME = "ExecuTorch";
    private static volatile EtEngine instance;

    private EtEngine() {
        // Load native libraries here.
        // For Android AAR reuse: trigger SoLoader/NativeLoader init.
        // For custom desktop build: System.load() the .so directly.
    }

    public static EtEngine newInstance() {
        if (instance == null) {
            synchronized (EtEngine.class) {
                if (instance == null) {
                    instance = new EtEngine();
                }
            }
        }
        return instance;
    }

    @Override
    public String getEngineName() { return ENGINE_NAME; }

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
        // ExecuTorch is CPU-only in initial implementation
        return new EtNDManager(this, Device.cpu());
    }

    @Override
    public String getVersion() {
        // Return the ExecuTorch version
        return "1.0.1";
    }

    @Override
    public boolean hasCapability(String capability) {
        // No CUDA, no training
        return false;
    }

    // toString, getAlternativeEngine (return "PyTorch" for hybrid mode),
    // and other Engine methods as needed.
}
```

Key design note: `getAlternativeEngine()` should return `"PyTorch"` so that DJL can use the PyTorch engine for NDArray operations that ExecuTorch's NDManager doesn't support (hybrid mode).

### 3c. EtModel

Wraps `org.pytorch.executorch.Module`:

```java
package org.measly.executorch.engine;

import ai.djl.BaseModel;
import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import org.pytorch.executorch.Module;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

public class EtModel extends BaseModel {

    private Module etModule;

    EtModel(String name, NDManager manager) {
        super(name);
        this.manager = manager;
        this.manager.setName("etModel");
    }

    @Override
    public void load(Path modelPath, String prefix, Map<String, ?> options)
            throws IOException {
        // Resolve the .pte file
        Path ptePath = resolveModelFile(modelPath, prefix);
        etModule = Module.load(ptePath.toString());
    }

    // Package-private accessor for Predictor/Translator
    Module getEtModule() {
        return etModule;
    }

    @Override
    public void close() {
        if (etModule != null) {
            etModule.destroy();
            etModule = null;
        }
        super.close();
    }

    private Path resolveModelFile(Path modelPath, String prefix) {
        // Look for {prefix}.pte or *.pte in the directory
        // Follow DJL's convention for model artifact resolution
        // ...
    }
}
```

### 3d. EtNDManager and EtNDArray

These provide minimal tensor creation for building model inputs and reading outputs. The ONNX Runtime engine (`OrtNDManager`, `OrtNDArray`) is the direct template.

**EtNDManager** creates `EtNDArray` instances from Java arrays. Only basic creation methods need implementation:

- `create(float[]/double[]/int[]/long[], Shape)` — wraps `Tensor.fromBlob(data, shape)`
- `create(Number)` — scalar tensor creation

**EtNDArray** wraps an ExecuTorch `Tensor` and implements the `NDArray` interface. Most NDArray operations (arithmetic, slicing, reshaping) should throw `UnsupportedOperationException` and rely on hybrid mode delegation to PyTorch. Must implement:

- `getDataType()` — map from ExecuTorch dtype to DJL `DataType`
- `getShape()` — read from underlying tensor
- `toFloatArray()` / `toDoubleArray()` / etc. — extract data
- `toByteBuffer()` — for raw data access

> **Batchifier constraint (discovered in Phase 1 — applies to every translator).** Because
> `EtNDArray` does not implement `NDArrayInternal`/`getAlternativeArray` (no hybrid engine in
> Phase 1), DJL's **default `StackBatchifier` throws** `UnsupportedOperationException` via
> `NDArrays.stack()`. Therefore **every `Translator` for this engine must override
> `getBatchifier()` to return `null`** (DJL's `Predictor` then runs each sample through
> `processInput`/`processOutput` individually — correct for single-sample inference). This applies
> to the Phase 2 `MapTranslator` too. The proper fix (real batching) requires hybrid-mode
> NDArray support and is deferred to Phase 3.

### 3e. EtNDArray dtype mapping

```java
// ExecuTorch Tensor has dtype codes; DJL has ai.djl.ndarray.types.DataType

static DataType fromEtDType(int etDType) {
    return switch (etDType) {
        case 0 -> DataType.UINT8;
        case 1 -> DataType.INT8;
        case 2 -> DataType.INT16;
        case 3 -> DataType.INT32;
        case 4 -> DataType.INT64;
        // 5 = float16
        case 6 -> DataType.FLOAT32;
        case 7 -> DataType.FLOAT64;
        // 11 = bool
        default -> DataType.UNKNOWN;
    };
}
```

---

## 4. Named Parameter Translator

### Problem

Users want to pass model inputs as `Map<String, Number>` rather than constructing `NDList` directly. The model spec (extracted at export time) provides the name-to-position mapping and dtype per input.

### Model Spec Generation (Python side)

Run alongside the ExecuTorch export:

```python
import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower

# Export
ep = export(model, example_inputs)

# Extract input metadata
input_spec = []
for spec in ep.graph_signature.input_specs:
    if spec.kind == torch.export.graph_signature.InputKind.USER_INPUT:
        input_spec.append({
            "name": spec.arg.name,
            "position": len(input_spec),
            "dtype": str(example_inputs[len(input_spec)].dtype)
        })

# Write spec alongside .pte artifact
import json
with open("model_spec.json", "w") as f:
    json.dump({"inputs": input_spec, "runtime": "executorch"}, f)

# Continue with ExecuTorch lowering
edge = to_edge_transform_and_lower(ep, ...)
et_program = edge.to_executorch()
with open("model.pte", "wb") as f:
    f.write(et_program.buffer)
```

### MapTranslator

A `Translator<Map<String, Number>, double[]>` that bridges named parameters to ExecuTorch tensors:

```java
package org.measly.executorch.translate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.util.List;
import java.util.Map;
import java.util.Set;

public class MapTranslator implements Translator<Map<String, Number>, double[]> {

    private final List<ParamSpec> paramSpecs;
    private final Set<String> expectedNames;

    public MapTranslator(List<ParamSpec> paramSpecs) {
        this.paramSpecs = paramSpecs;
        this.expectedNames = /* extract names from paramSpecs */;
    }

    public static MapTranslator fromSpec(Path specPath) throws IOException {
        // Parse model_spec.json → List<ParamSpec>
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Map<String, Number> input) {
        NDManager manager = ctx.getNDManager();

        // Validate input names
        if (!input.keySet().equals(expectedNames)) {
            throw new IllegalArgumentException(
                "Expected parameters " + expectedNames
                + ", got " + input.keySet());
        }

        NDList ndList = new NDList(paramSpecs.size());
        for (ParamSpec spec : paramSpecs) {
            Number val = input.get(spec.name());
            NDArray arr = spec.dtype().createScalar(manager, val);
            ndList.add(arr);
        }
        return ndList;
    }

    @Override
    public double[] processOutput(TranslatorContext ctx, NDList output) {
        NDArray result = output.singletonOrThrow();
        return result.toDoubleArray();
    }
}
```

### ParamSpec and DType

```java
public record ParamSpec(String name, int position, DType dtype) {

    public enum DType {
        FLOAT32 {
            NDArray createScalar(NDManager m, Number v) {
                return m.create(v.floatValue());
            }
        },
        FLOAT64 {
            NDArray createScalar(NDManager m, Number v) {
                return m.create(v.doubleValue());
            }
        },
        INT32 {
            NDArray createScalar(NDManager m, Number v) {
                int iv = v.intValue();
                if (iv != v.longValue()) {
                    throw new ArithmeticException(
                        "Value " + v + " does not fit in int32");
                }
                return m.create(iv);
            }
        },
        INT64 {
            NDArray createScalar(NDManager m, Number v) {
                return m.create(v.longValue());
            }
        };

        abstract NDArray createScalar(NDManager m, Number v);

        public static DType from(String name) {
            return switch (name) {
                case "float32", "torch.float32" -> FLOAT32;
                case "float64", "torch.float64" -> FLOAT64;
                case "int32", "torch.int32"     -> INT32;
                case "int64", "torch.int64"     -> INT64;
                default -> throw new IllegalArgumentException(
                    "Unsupported dtype: " + name);
            };
        }
    }
}
```

### End-to-End Usage

```java
// Load model via DJL API
Criteria<Map<String, Number>, double[]> criteria = Criteria.builder()
    .setTypes(Map.class, double[].class)
    .optEngine("ExecuTorch")
    .optModelPath(Path.of("/models/pricing"))
    .optTranslator(MapTranslator.fromSpec(
        Path.of("/models/pricing/model_spec.json")))
    .build();

try (ZooModel<Map<String, Number>, double[]> model = criteria.loadModel();
     Predictor<Map<String, Number>, double[]> predictor = model.newPredictor()) {

    double[] result = predictor.predict(Map.of(
        "volatility", 0.25,
        "rate",       0.05,
        "num_periods", 12,
        "contract_type", 1
    ));
}
```

---

## 5. Project Layout

```
djl-executorch-engine/
├── build.gradle.kts (or pom.xml)
├── src/main/java/org/measly/executorch/
│   ├── engine/
│   │   ├── EtEngine.java
│   │   ├── EtEngineProvider.java
│   │   ├── EtModel.java
│   │   ├── EtNDArray.java
│   │   ├── EtNDManager.java
│   │   └── LibUtils.java           # Native library loading
│   └── translate/
│       ├── DType.java               # (or nested in ParamSpec)
│       ├── MapTranslator.java
│       └── ParamSpec.java
├── src/main/resources/
│   └── META-INF/services/
│       └── ai.djl.engine.EngineProvider
├── src/test/java/...
└── native/                          # ExecuTorch desktop build scripts
    ├── CMakeLists.txt
    └── build_desktop.sh
```

### Dependencies (Gradle)

```kotlin
dependencies {
    // DJL API — compile only, users bring their own version
    compileOnly("ai.djl:api:0.36.0")

    // ExecuTorch Java bindings — repackaged for desktop or sourced from AAR
    implementation("org.measly:executorch-java:1.0.1")

    // Or, if using the AAR directly with extracted classes:
    // implementation("org.pytorch:executorch-android:1.0.1")

    // Facebook JNI helper (transitive from ExecuTorch)
    implementation("com.facebook.fbjni:fbjni-java-only:0.2.2")

    // Test
    testImplementation("ai.djl:api:0.36.0")
    testImplementation("ai.djl.pytorch:pytorch-engine:0.36.0")  // for hybrid mode tests
}
```

### Native Library Build

The `native/` directory contains scripts to build the ExecuTorch runtime and JNI bindings for desktop Linux (and optionally macOS/Windows). The build:

1. Clones the ExecuTorch source at the target (pinned) version, with submodules.
2. Builds the C++ runtime with the XNNPack backend enabled (CPU acceleration), with `-DCMAKE_POSITION_INDEPENDENT_CODE=ON` so the static libs can link into a shared object.
3. Compiles the **custom thin JNI shim** (`native/jni/executorch_djl_jni.cpp`, Option B) against the host JDK's `jni.h`, linking the runtime libs with whole-archive so XNNPACK/op registration resolves. (Original plan said "compile `extension/android/jni/`" — superseded; that path is Android-gated. See §2 corrected recommendation.)
4. Produces `libexecutorch_djl.so` (or `.dylib` / `.dll`).

This `.so` is what `EtEngine`'s constructor loads via `System.load()`. Full step-by-step build with go/no-go gates: [`docs/superpowers/plans/2026-06-28-executorch-desktop-linux-native-build.md`](docs/superpowers/plans/2026-06-28-executorch-desktop-linux-native-build.md).

---

## 6. Native Library Loading

Adapt patterns from DJL's `LibUtils.java` (in `engines/pytorch/pytorch-engine/src/main/java/ai/djl/pytorch/jni/`). Key patterns to reuse:

### Platform Detection

From DJL's `Platform.java` — detect OS and architecture, construct a string like `linux-x86_64`, `osx-aarch64`, `win-x86_64`.

### Cache and Download

Follow DJL's download-to-cache pattern:

- Default cache directory: `~/.djl.ai/executorch/{version}/`
- On first use, download the pre-built native library for the detected platform.
- On subsequent runs, load from cache.

For the initial implementation, prebuilt desktop natives may not be available on a public CDN. Two alternatives:

**Bundled in JAR:** Package the `.so` inside the Maven artifact under `native/{platform}/libexecutorch_jni.so`. Extract to a temp directory at runtime and `System.load()` it. This is how `com.microsoft.onnxruntime:onnxruntime` distributes its natives.

**Environment variable override:** Support `EXECUTORCH_LIBRARY_PATH` to point at a locally built native library, analogous to DJL's `PYTORCH_LIBRARY_PATH`.

### Load Order

ExecuTorch's native dependency chain is much simpler than libtorch. Under the **Option B thin shim** (recommended), there is a **single** library to load — the runtime, XNNPACK delegate, and op kernels are statically linked into it:

1. `libexecutorch_djl.so` (contains the ExecuTorch runtime, XNNPACK delegate, op kernels, and the custom JNI bindings)

No `libfbjni.so`, no SoLoader, no `libc10`, no `libtorch_cpu`, no OpenMP. (Under the abandoned Option A, `libfbjni.so` would have to load first — another reason Option B is simpler.) This is a major simplification over the PyTorch engine.

### No Conflict with DJL's PyTorch Engine

ExecuTorch's native libraries share no symbols with libtorch. Both can be loaded in the same process. No version convergence, no load ordering constraints between the two engines. This is the primary advantage over the AOTInductor approach discussed earlier.

---

## 7. Threading

ExecuTorch's threading model is simpler than libtorch:

- No internal thread pool by default. The XNNPack delegate may use pthreads for intra-op parallelism, but this is configurable at build time and generally less aggressive than libtorch's OpenMP pool.
- `Module.forward()` is safe to call from multiple Java threads (each with its own input tensors), but the `Module` instance itself may not support concurrent `forward()` calls. DJL's `Predictor` pattern (one Predictor per thread, each wrapping the same Model) handles this naturally.
- No global initialization race conditions comparable to libtorch's. The `fbjni` / `SoLoader` initialization is thread-safe.

DJL's existing recommendation from its performance optimization docs applies: create a `Predictor` per thread and reuse it for multiple predictions. Do not share a `Predictor` across threads.

---

## 8. Implementation Sequence

Suggested order for incremental development:

### Phase 1: Minimal engine with hardcoded model

1. Build ExecuTorch native libraries for desktop Linux. **(Done — feasibility plan complete; `native/build_desktop.sh` produces `src/main/resources/native/linux-x86_64/libexecutorch_djl.so`.)**
2. Implement `EtEngine`, `EtEngineProvider` — get SPI discovery working.
3. Implement `EtModel` — load a `.pte` file.
4. Implement minimal `EtNDManager` and `EtNDArray` — create float tensors, read float output.
5. Test: load a simple model via DJL's `Criteria` API. **Reuse the existing artifact `native/spike/add.pte` (2 float inputs → 1 float output, `a + b`) and its generator `native/spike/export_add.py` rather than creating a new one** — the JNI bridge is already proven against it (`RESULT=5.0`).

### Phase 2: Named parameter support

6. Implement `ParamSpec`, `DType`, `MapTranslator`.
7. Implement `model_spec.json` parsing.
8. Test: load a multi-parameter, mixed-type model; run via `Map<String, Number>`.

### Phase 3: Hybrid mode and robustness

9. Verify hybrid mode works — DJL falls back to PyTorch engine for unsupported NDArray ops.
10. Implement remaining `NDArray` data extraction methods (`toDoubleArray`, `toIntArray`, etc.).
11. Add proper error handling, resource cleanup, and lifecycle management.
12. Add native library caching / platform detection from DJL's `LibUtils` patterns.

### Phase 4: Packaging and distribution

13. Package native libraries in the JAR or set up a CDN.
14. Write user-facing documentation and README.
15. Publish to Maven Central under your own group ID.
16. Optionally: open a PR against `deepjavalibrary/djl` with the engine.

---

## 9. Reference Files in DJL Repository

When implementing, study these files in `https://github.com/deepjavalibrary/djl`: (cloned locally to `/home/corey/workspace/djl`)

### ONNX Runtime engine (primary reference)

```
engines/onnxruntime/onnxruntime-engine/src/main/java/ai/djl/onnxruntime/engine/
├── OrtEngine.java           # Engine singleton, runtime init
├── OrtEngineProvider.java   # SPI registration
├── OrtModel.java            # Model loading, wraps ORT session
├── OrtNDArray.java          # NDArray backed by ORT tensor
├── OrtNDManager.java        # Tensor factory, limited ops
└── OrtSymbolBlock.java      # Forward pass execution
```

### PyTorch engine (for LibUtils patterns)

```
engines/pytorch/pytorch-engine/src/main/java/ai/djl/pytorch/jni/
├── LibUtils.java            # Native lib download/cache/load ordering
└── (other files — skip JniUtils.java, PyTorchLibrary.java)
```

### DJL API (interfaces to implement)

```
api/src/main/java/ai/djl/
├── engine/Engine.java              # Abstract base
├── engine/EngineProvider.java      # SPI interface
├── BaseModel.java                  # Model base class
├── ndarray/NDManager.java          # Tensor lifecycle manager
├── ndarray/NDArray.java            # Tensor interface
├── ndarray/NDList.java             # List of tensors (model I/O)
├── translate/Translator.java       # Input/output mapping
├── inference/Predictor.java        # Thread-scoped inference handle
└── util/Platform.java              # OS/arch detection
```

### Docs

```
docs/hybrid_engine.md               # How limited engines delegate to full engines
docs/development/cache_management.md # Native library caching conventions
```
