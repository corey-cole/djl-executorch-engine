# ExecuTorch DJL Engine — Phase 2a Design Spec: Zero-Copy, Multi-Dtype Native Surface

> **Status:** approved design (2026-06-28). Scope: the native-surface generalization that Phase 2
> builds named-parameter support on top of. Builds on the merged
> [Phase 1 engine](2026-06-28-executorch-engine-phase1-design.md) and the design doc
> ([`djl-executorch-engine-design.md`](../../../djl-executorch-engine-design.md), §3d / §4 / §8 Phase 2).
> Next step: a writing-plans implementation plan derived from this spec.

## 1. Goal & success criteria

Generalize the engine's native surface from **float32-only** to a **zero-copy, multi-dtype** path,
so DJL `Predictor` inference works for these seven tensor dtypes:

`FLOAT32, FLOAT64, INT32, INT64, INT8, UINT8, BOOL`

and so the inference hot path stops round-tripping tensor payloads through `float[]`.

### Success tests (acceptance gates)
1. **Multi-dtype end-to-end:** a `.pte` taking mixed-dtype inputs (e.g. `int64` + `float32`) and
   returning a tensor runs through `Criteria`/`Predictor` with a hand-written `NDList` translator,
   asserting the correct numeric result.
2. **Arity validation:** calling forward with the wrong number of inputs throws
   `IllegalArgumentException` (count comes from `method_meta`), before any native forward runs.
3. **Regression:** the Phase 1 float32 `add.pte` path still passes unchanged.

### Out of scope (separate specs)
- **Named-parameter layer** (`MapTranslator`, `model_spec.json`, `ParamSpec`/`DType`) — Phase **2b**,
  built directly on this surface.
- **`FLOAT16`** — needs `Float16Utils` half↔float conversion; not required by the named-param use
  case. `EtDataTypes` throws for it (clear failure).
- **Multi-method modules** — only the default `"forward"` method.
- **Direct-buffer / native-lifecycle outputs** — see §3 note; deferred as the autoregressive
  revisit path.

## 2. Key decisions (settled during brainstorming)

1. **Scope = 2a native surface only.** Named params are a separate later spec.
2. **`EtTensor` carries `ByteBuffer` + ScalarType code**, not `float[]`.
3. **Asymmetric payloads** — zero-copy direct buffers in, single-copy heap buffers out (details §3).
4. **Outputs are heap `ByteBuffer`s with no native lifecycle** (preserves Phase 1's "NDArray holds
   no native handle" principle). The direct-buffer + native-free alternative is the documented
   revisit path **if** an autoregressive/re-feed use case appears.
5. **Expose model I/O metadata** (`method_meta`: input count + per-input ScalarType) to drive
   arity (and optional dtype) validation with clear errors.
6. **Dtype set:** the seven above; `FLOAT16`/`INT16` excluded (mapping throws).
7. **Native C++ covered by the Java integration tests** (the JNI glue is best exercised across the
   real boundary); no C++ test framework in 2a. The deferred direct-buffer native-lifecycle code is
   the first candidate for a JNI-free Catch2 suite later; `native/spike/cpp_smoke.cpp` is kept as
   the seed.

## 3. Native surface

### `EtTensor` (generalized)
```java
package org.measly.executorch.jni;

/** A tensor crossing the JNI boundary: raw bytes + ExecuTorch ScalarType code + shape. */
public final class EtTensor {
    public final long[] shape;
    public final int scalarType;   // ExecuTorch ScalarType int code
    public final ByteBuffer data;  // input: DIRECT (zero-copy); output: heap (single-copy)
    public EtTensor(long[] shape, int scalarType, ByteBuffer data) { ... }
}
```

**Asymmetric payload — this is what makes the path zero-copy-in / no-lifecycle-out:**
- **Input** `data` is a **direct** `ByteBuffer`. Every `EtNDArray` already holds one (`allocateDirect`
  in `EtNDManager`). JNI reads it via `GetDirectBufferAddress` and `from_blob`s onto it — **0 copies**.
  (Assumes the buffer's content starts at position 0, which holds for engine-created buffers.)
- **Output** `data` is a **heap** `ByteBuffer`. ExecuTorch frees its tensor when `forward()` returns,
  so exactly one native→Java copy is required: JNI does `NewByteArray` + `SetByteArrayRegion`, wraps
  it (`ByteBuffer.wrap(...).order(nativeOrder)`), and stores it on the output `EtTensor` — **1 copy,
  nothing native to free** (the `byte[]` is GC'd normally).

### `EtMethodMeta` + metadata call
```java
package org.measly.executorch.jni;

/** Static I/O metadata for a loaded module's "forward" method. */
public final class EtMethodMeta {
    public final int numInputs;
    public final int[] inputScalarTypes;   // per-input ExecuTorch ScalarType code
    public EtMethodMeta(int numInputs, int[] inputScalarTypes) { ... }
}
```
Read once from `executorch::extension::Module::method_meta("forward")` (which exposes
`num_inputs()` and `input_tensor_meta(i).scalar_type()`), queried by `EtModel` right after
`loadModule` and cached on the block. Shapes are intentionally **not** exposed (dynamic dims add
complexity not needed here).

### `EtNative` surface (final)
```java
public static native long loadModule(String ptePath);
public static native EtMethodMeta methodMeta(long handle);
public static native EtTensor[] forward(long handle, EtTensor[] inputs);
public static native void destroy(long handle);
```
Raw JNI, no fbjni. `JNI_OnLoad` caches the (now two) class IDs + field/ctor IDs for `EtTensor` and
`EtMethodMeta`.

### `EtDataTypes` (new helper, `org.measly.executorch.engine`)
Bidirectional `DataType ↔ ScalarType` int code for the seven supported types; throws
`IllegalArgumentException` for anything else (e.g. FLOAT16, INT16) so unsupported dtypes fail
loudly rather than silently mis-reading bytes.

```
DataType.FLOAT32 ↔ 6
DataType.FLOAT64 ↔ 7
DataType.INT32   ↔ 3
DataType.INT64   ↔ 4
DataType.INT8    ↔ 1
DataType.UINT8   ↔ 0
DataType.BOOLEAN ↔ 11
```
> **Implementation note:** these ScalarType integer codes are from the design-doc §3e sketch and
> **must be re-confirmed against the pinned ExecuTorch headers** (`executorch::aten::ScalarType`)
> at implementation time.

## 4. `forwardInternal` (rewritten)

Replaces the float32-only body; folds in the improvements recorded in the design doc §8 Phase 2.
NOTE: Code below is missing hoist of inputs size method (`final int count = inputs.size(); // Use this in place of calls to inputs.size()`)

```java
@Override
protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training,
        PairList<String, Object> params) {
    // 1. Arity validation (allows numInputs == 0)
    if (inputs.size() != meta.numInputs) {
        throw new IllegalArgumentException(
                "ExecuTorch model expects " + meta.numInputs + " inputs, got " + inputs.size());
    }
    // 2. Inputs — zero-copy
    EtTensor[] in = new EtTensor[inputs.size()];
    for (int i = 0; i < in.length; ++i) {
        EtNDArray et = manager.from(inputs.get(i));               // direct buffer guaranteed
        int st = EtDataTypes.toScalarType(et.getDataType());      // throws on unsupported dtype
        if (st != meta.inputScalarTypes[i]) {
            throw new IllegalArgumentException(
                    "Input " + i + " dtype " + et.getDataType()
                            + " != model's expected ScalarType " + meta.inputScalarTypes[i]);
        }
        in[i] = new EtTensor(et.getShape().getShape(), st, et.toByteBuffer());
    }
    // 3. Native forward
    EtTensor[] out = EtNative.forward(handle, in);
    // 4. Outputs — wrap heap buffer with NO re-copy, on the request manager
    NDManager rm = inputs.isEmpty() ? manager : inputs.head().getManager();
    EtNDManager target = (rm instanceof EtNDManager) ? (EtNDManager) rm : manager;
    NDList ret = new NDList(out.length);
    for (EtTensor t : out) {
        DataType dt = EtDataTypes.fromScalarType(t.scalarType);
        ret.add(target.wrap(t.data, new Shape(t.shape), dt));    // no-copy; attaches to target
    }
    return ret;
}
```

Notes:
- `manager.from()` is a no-op for an `EtNDArray` input (the common case) and copies a foreign
  `NDArray` into a direct-buffer `EtNDArray` once — so the zero-copy path holds for engine-native
  inputs with a safe fallback otherwise.
- Outputs are wrapped via a new no-copy `EtNDManager.wrap(ByteBuffer, Shape, DataType)` (§5) on the
  **request manager**, eliminating Phase 1's create-then-reattach churn and its empty-input guard
  wart. The `instanceof` check is belt-and-suspenders — the request manager is always an
  `EtNDManager` in practice; the fallback to the block's `manager` only guards a theoretical foreign
  manager (outputs would then live with the model, an acceptable edge).
- The Phase 1 FLOAT32-only `IllegalArgumentException` is removed; dtype enforcement now comes from
  `EtDataTypes` (unsupported types throw) **and** a mandatory per-input check against
  `meta.inputScalarTypes` (clear error on a dtype the model doesn't expect). Phase 2a is **strict**
  (exact ScalarType match); **dtype widening/coercion (e.g. INT32→INT64) is deferred to Phase 2b**,
  where the named-param layer can add value-aware conversion.
- `meta` (an `EtMethodMeta`) is cached on the block, set by `EtModel` after `loadModule`.

## 5. NDArray / NDManager touchpoints

Small, because `EtNDArray` is already a `ByteBuffer`+`Shape`+`DataType` holder:

- **`EtNDArray`** — essentially unchanged. Data extraction (`toIntArray`/`toLongArray`/
  `toDoubleArray`/`toBooleanArray`/`toUint8Array`/…) is provided by `NDArrayAdapter` from
  `toByteBuffer()` + `getDataType()`, so it works for the new dtypes for free. Its package-private
  ctor is reused by `EtSymbolBlock` to wrap heap outputs without re-copying.
- **`EtNDManager.create(Buffer, Shape, DataType)`** — extend `copyInto` to the new dtypes. INT8/
  UINT8/BOOL arrive as byte data and fall through the existing `ByteBuffer` branch; the typed-buffer
  branches remain for the `create(float[]/int[]/long[]/double[])` convenience overloads. Inputs
  created here stay **direct** (`allocateDirect`), preserving zero-copy.
- **`EtNDManager.wrap(ByteBuffer, Shape, DataType)`** — new package-private method that wraps an
  existing buffer **without copying** (`new EtNDArray(this, alternativeManager, data, shape, dt)`),
  for `EtSymbolBlock` to adopt heap output buffers. Distinct from `create`, which copies into a
  fresh direct buffer. This keeps `alternativeManager` access inside the manager where it belongs.
- **No new native lifecycle** anywhere — outputs are heap `ByteBuffer`s, GC'd like any object.
- **`EtModel`** — after `loadModule`, calls `EtNative.methodMeta(handle)` and passes the
  `EtMethodMeta` into the `EtSymbolBlock` constructor (alongside the handle + manager).

## 6. Testing (Java integration + native-free units)

- **Multi-dtype end-to-end (acceptance):** new `native/spike/` fixture — a small `.pte` taking
  `int64` + `float32` and returning a tensor — exported by a Python script alongside `add.pte`. Run
  through `Criteria`/`Predictor` with an `NDList` translator; assert the result. Guarded by the
  native-availability assumption.
- **Arity validation:** wrong input count → `IllegalArgumentException`, no native forward attempted.
- **Metadata:** `methodMeta` returns the expected `numInputs` / `inputScalarTypes` for the fixtures.
- **Dtype round-trips (unit, native-free):** for each of the seven dtypes, `EtNDManager.create(...)`
  + the matching extraction (`toIntArray`, `toLongArray`, etc.) round-trips; `EtDataTypes` maps both
  directions and throws for an unsupported type (FLOAT16/INT16).
- **Regression:** the Phase 1 `add.pte` float32 path stays green.
- **No C++ test framework in 2a** (decision §2.7); the JNI glue is covered through the real
  `.so` + runtime by the integration tests above.

## 7. Open items / confirm at implementation time

- ExecuTorch `ScalarType` integer codes for the seven dtypes (verify against pinned headers; §3).
- `Module::method_meta` API shape in the pinned ExecuTorch version (`num_inputs()`,
  `input_tensor_meta(i).scalar_type()`).
- JNI mechanics for building a heap-backed `ByteBuffer` output (cache `ByteBuffer.wrap` static
  method id in `JNI_OnLoad`, or assemble parallel arrays and wrap Java-side — pick the simpler at
  implementation).
- `manager.from()` semantics for a foreign (non-Et) NDArray under no hybrid engine — confirm it
  copies via `toByteBuffer()` and yields a direct-buffer `EtNDArray`.
