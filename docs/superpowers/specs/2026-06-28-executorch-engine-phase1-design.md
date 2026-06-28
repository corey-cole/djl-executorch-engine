# ExecuTorch DJL Engine — Phase 1 Design Spec

> **Status:** approved design (2026-06-28). Scope: the minimal end-to-end engine.
> Builds on the completed [native-build feasibility](../reports/executorch-desktop-feasibility.md)
> (verdict: GO) and the overall [engine design doc](../../../djl-executorch-engine-design.md).
> Next step: a writing-plans implementation plan derived from this spec.

## 1. Goal & success criteria

Deliver a working DJL engine named **`ExecuTorch`** that loads a `.pte` artifact and runs
**float32** inference end-to-end through DJL's standard `Criteria`/`Predictor` API, backed by the
Path-B native library (`libexecutorch_djl.so`) produced in the feasibility phase.

**Success test (the acceptance gate):** a JUnit test resolves `Engine.getEngine("ExecuTorch")`
via Java SPI, loads `native/spike/add.pte` through `Criteria` with a small custom `Translator`,
and asserts `predict` of `(2.0f, 3.0f)` returns `5.0f`.

### Out of scope (each becomes its own later spec)
- Named-parameter `MapTranslator` + `model_spec.json` (Phase 2).
- Hybrid mode + PyTorch fallback for richer NDArray ops (Phase 3).
- Non-float dtypes beyond what the smoke test needs (Phase 2 generalizes).
- Multi-platform / manylinux natives + the CI release pipeline (see
  [`docs/ci-native-build.md`](../../ci-native-build.md)).
- Packaging/CDN/Maven publication (Phase 4).

## 2. Key decisions (settled during brainstorming)

1. **Scope = Phase 1 only** — the smallest vertical slice that runs inference through the public
   DJL API.
2. **NDArray = Java-side data holder** — `EtNDArray` holds a Java `ByteBuffer` + `Shape` +
   `DataType` with **no** native tensor handle. ExecuTorch tensors are created transiently inside
   the JNI `forward()` call and freed before it returns. This keeps the JNI surface tiny, avoids a
   per-tensor native lifecycle, and is forward-compatible with hybrid mode (data already lives in
   Java). Rejected alternative: native-handle NDArrays (libtorch-style), which add a native
   resource lifecycle and leak risk for no Phase 1 benefit.
3. **Path B confirmed** — no `fbjni`, no Android, no `org.pytorch.executorch.*` Java classes; the
   engine calls our own `org.measly.executorch.jni.EtNative` surface.

## 3. Generalized native surface

The spike's hardcoded `forwardFloat(handle, a, b)` is replaced with an N-tensor-general (still
float32) surface. A plain data class carries a tensor across JNI:

```java
package org.measly.executorch.jni;

/** A float32 tensor crossing the JNI boundary. Phase 2 generalizes to ByteBuffer + dtype. */
final class EtTensor {
    final long[] shape;
    final float[] data;   // row-major, length == product(shape)
    EtTensor(long[] shape, float[] data) { this.shape = shape; this.data = data; }
}
```

`EtNative` (static methods, loads the `.so` in its static initializer):
- `static native long loadModule(String ptePath);`
- `static native EtTensor[] forward(long handle, EtTensor[] inputs);`
  — builds transient ExecuTorch tensors from the inputs, runs `executorch::extension::Module::forward`,
  copies outputs into fresh `EtTensor[]`, frees the native tensors before returning.
- `static native void destroy(long handle);`

The existing `native/jni/executorch_djl_jni.cpp` is extended from the two-input float demo to this
general form (constructing `EtTensor[]` in JNI via cached `jclass`/`jmethodID` for
`EtTensor.<init>([J[F)`). No `fbjni`.

## 4. Engine classes

All under `org.measly.executorch.engine` unless noted. Method signatures target `ai.djl:api`
(see §7 for the abstract-method checklist verified against the DJL source).

| Class | Responsibility |
|---|---|
| `EtEngineProvider` | SPI entry. `getEngineName()="ExecuTorch"`, `getEngineRank()=10`, `getEngine()` → `EtEngine` singleton. |
| `EtEngine` (extends `Engine`) | Singleton, **cheap to construct (no native load)**. Implements `getEngineName`, `getRank()=10`, `getVersion()` (ExecuTorch version string, e.g. `"1.3.1"`), `hasCapability()=false`, **`getAlternativeEngine()=null`** (no hybrid in Phase 1), `newModel`, `newBaseManager()`/`newBaseManager(Device)` → `EtNDManager` on `Device.cpu()`. Native loading is **lazy** (see §5) so manager/NDArray construction stays native-free for testing. |
| `EtModel` (extends `BaseModel`) | `load(Path, String, Map)` resolves the `.pte`, calls `EtNative.loadModule`, sets the model block to `EtSymbolBlock(handle)`. `close()` → `EtNative.destroy` then `super.close()`. Owns load/lifecycle only — **not** forward. |
| `EtSymbolBlock` (**new**; extends `AbstractSymbolBlock` or implements `SymbolBlock`) | Holds the native handle. `forwardInternal(NDList)` marshals input `EtNDArray`s → `EtTensor[]` → `EtNative.forward` → wraps outputs back as `EtNDArray`s into an `NDList`. |
| `EtNDManager` (extends `BaseNDManager`) | Minimal factory: `create(float[], Shape)` and `create(Number)` → `EtNDArray`. Other creation methods may defer/throw for Phase 1. |
| `EtNDArray` (implements `NDArray`) | Holds `ByteBuffer` + `Shape` + `DataType`. Implements `getShape`, `getDataType`, `toFloatArray`, `toByteBuffer`, `getManager`, `getName/setName`, `close`. Arithmetic/slicing/reshaping throw `UnsupportedOperationException`. |
| `LibUtils` | Native library resolution/loading (§5). |
| `EtDataTypes` (or static helper) | ExecuTorch dtype-code ↔ DJL `DataType` mapping (per design-doc §3e; codes verified against ExecuTorch headers at implementation time). |

### Correction to the engine design doc
Design-doc §3c shows `EtModel` holding `org.pytorch.executorch.Module` with forward implied there.
Under Path B that Java class does not exist, and DJL runs inference through a model's **block**, so
forward lives in a new **`EtSymbolBlock`**. `EtModel` only owns load/lifecycle. (The design doc's
own reference table lists an `OrtSymbolBlock` precedent.)

## 5. Native library loading (`LibUtils`)

**Lazy loading:** `System.load()` happens in `EtNative`'s **static initializer** (as in the spike), so the native library is loaded only when a model is loaded or `forward()` runs — never just from constructing `EtEngine`/`EtNDManager`/`EtNDArray`. `LibUtils.loadLibrary()` (called from that static initializer) does the resolution.

One library, fully self-contained (`ldd` shows only libc/libstdc++/libm/libgcc). Resolution order:

1. **`EXECUTORCH_LIBRARY_PATH`** env var → `System.load()` it directly (mirrors DJL's
   `PYTORCH_LIBRARY_PATH`; this is how the spike's `EtSmoke` already runs).
2. Otherwise extract the bundled classpath resource `native/<platform>/libexecutorch_djl.so` to a
   temp file and `System.load()` it.

Platform detection is minimal: derive `linux-x86_64` from `os.name`/`os.arch`; throw a clear
`UnsupportedOperationException` for any other platform. No download/CDN logic (Phase 4). Load order
is trivial — a single `.so`, no `libfbjni`/SoLoader/`libc10`.

## 6. Gradle module

The repo is currently a stock `gradle init` (a generic `lib` subproject, no root
`build.gradle.kts`). Phase 1 restructures it into one engine module:

- Single module producing `org.measly:djl-executorch-engine`; Java toolchain **17**.
- Dependencies:
  - `compileOnly("ai.djl:api:0.36.0")` — users bring their own DJL version.
  - `testImplementation("ai.djl:api:0.36.0")` + JUnit 5.
  - **No** PyTorch, fbjni, or executorch-android dependencies.
- `src/main/resources/`:
  - `META-INF/services/ai.djl.engine.EngineProvider` containing
    `org.measly.executorch.engine.EtEngineProvider`.
  - `native/linux-x86_64/libexecutorch_djl.so` (already staged here by `native/build_desktop.sh`).
- The native build (`build_desktop.sh` / CMake) stays **outside** Gradle; Gradle just bundles the
  pre-staged `.so`. Wiring the native build into Gradle is deferred.

**Assumption:** `ai.djl:api:0.36.0` (from the engine design doc). Easily re-pinned.

## 7. DJL API surface to implement (verified)

`Engine` (abstract) requires: `getAlternativeEngine()`, `getEngineName()`, `getRank()`,
`getVersion()`, `hasCapability(String)`, `newModel(String, Device)`, `newBaseManager()`,
`newBaseManager(Device)`. `EngineProvider` requires: `getEngineName()`, `getEngineRank()`,
`getEngine()`. (Note `EngineProvider.getEngineRank()` vs `Engine.getRank()` — different names.)
Signatures must be re-confirmed against the exact published `ai.djl:api` version pinned in §6, as
DJL's API drifts between releases.

## 8. Testing

- **Unit (no native needed):** `EtNDArray` data round-trip (`create(float[])` →
  `toFloatArray`/`toByteBuffer`); dtype-code → `DataType` mapping.
- **SPI discovery:** `Engine.getEngine("ExecuTorch")` returns a non-null engine with the expected
  name and rank.
- **End-to-end integration (acceptance gate):** load `native/spike/add.pte` via `Criteria` with a
  small `Translator` (`processInput` builds a 2-element `NDList` of float scalars; `processOutput`
  reads the single float), `predict` → assert `5.0f`.
- **Native availability guard:** the integration + SPI-load tests require the `.so`. They run when
  the bundled resource or `EXECUTORCH_LIBRARY_PATH` is present, and are skipped otherwise so the
  unit suite stays green on machines without a native build. CI builds the `.so`
  (`native/build_desktop.sh`) before running the full suite.

## 9. Open items / assumptions to confirm at implementation time

- Pinned `ai.djl:api` version (assumed `0.36.0`).
- Exact `Engine`/`BaseModel`/`NDArray`/`BaseNDManager` method set for that published version.
- ExecuTorch `ScalarType` integer codes for the dtype mapping (verify against headers).
- `EtSymbolBlock` base: `AbstractSymbolBlock` vs implementing `SymbolBlock` directly — pick per the
  DJL version's available base classes.
