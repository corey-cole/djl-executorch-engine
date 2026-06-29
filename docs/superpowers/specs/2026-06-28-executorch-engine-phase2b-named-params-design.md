# ExecuTorch DJL Engine — Phase 2b Design Spec: Named-Parameter Translator

> **Status:** approved design (2026-06-28). A pure-Java ergonomics layer on top of the merged
> [Phase 2a multi-dtype surface](2026-06-28-executorch-engine-phase2a-native-surface-design.md).
> No native/C++ changes. Builds toward the named-parameter use case sketched in the design doc §4.
> Next step: a writing-plans implementation plan derived from this spec.

## 1. Goal & success criteria

Let callers run inference with **named scalar parameters** instead of hand-building an `NDList`:
`predict(Map<String,Number>)` → `double[]`. An export-time `model_spec.json` maps each parameter
name to its input position, dtype, and (scalar) shape.

**Success test:** export a multi-parameter, mixed-dtype **scalar** model (`price: float32`,
`qty: int64` → `price * qty`) plus its `model_spec.json`; load via
`Criteria<Map<String,Number>, double[]>` with `MapTranslator.fromModelPath(...)`;
`predict(Map.of("price", 2.5, "qty", 4))` returns `double[]{10.0}`.

### Out of scope (later)
- **Tensor/array-valued named inputs** (`Map<String,Object>`). When wanted, study DJL's PyTorch
  engine complex-input handling first. The spec format already carries `shape` (below), so this is
  an extension, not a rewrite.
- **Named / multi-output** (Phase 2b is single-output → `double[]`).
- **Auto-discovery `TranslatorFactory`** (explicit factories only).
- **Param dtypes beyond the four numeric types** (`FLOAT32/FLOAT64/INT32/INT64`); BOOL/INT8/UINT8
  aren't natural `Number` scalars.
- **No native/C++ changes** — rides the 2a `forward(EtTensor[])` surface unchanged.

## 2. Key decisions (settled during brainstorming)

1. **Scalar-only named params** (`Map<String,Number>`).
2. **Output = dtype-aware `double[]`** (single output; widen any numeric dtype).
3. **Strict-but-sensible coercion** (integer targets reject fractional/out-of-range; float targets
   widen any `Number`).
4. **Explicit loading** (`fromSpec`/`fromModelPath`), **lenient JSON** (unknown fields ignored).
5. **Python under `tools/scripts/`**, fixture under `src/test/resources/models/`, not `native/spike`.

## 3. `model_spec.json` format + loading

Lenient format — **unknown top-level and per-input fields are ignored** (Gson default), so future
additions don't break older parsers:
```json
{
  "runtime": "executorch",
  "inputs": [
    {"name": "price", "position": 0, "dtype": "float32", "shape": [1]},
    {"name": "qty",   "position": 1, "dtype": "int64",   "shape": [1]}
  ]
}
```
- `shape` is recorded (the design-doc sketch omitted it) to remove the 0-dim-vs-`[1]` ambiguity —
  the translator builds each tensor at the **exact recorded shape**. For 2b, `product(shape)` must
  be `1` (scalar); a non-scalar shape is rejected with a clear message (that's the tensor-valued
  extension).
- Parsed with DJL's bundled `ai.djl.util.JsonUtils` (Gson — **no new dependency**) into a small DTO,
  then converted to `List<ParamSpec>`. Parsing validates: non-empty `inputs`, unique positions,
  each shape scalar, each dtype resolvable.
- Factories on `MapTranslator`: `fromSpec(Path jsonFile)` and `fromModelPath(Path modelDir)`
  (resolves `modelDir/model_spec.json`).

## 4. `ParamSpec` + `DType` (`org.measly.executorch.translate`)

```java
public record ParamSpec(String name, int position, DType dtype, long[] shape) {}
```

`DType` — the four numeric types `Number` maps to cleanly, each building a scalar `NDArray` of the
param's shape with **strict coercion**:
- **`FLOAT32` / `FLOAT64`:** accept any `Number`, widen — `m.create(new float[]{v.floatValue()}, new Shape(shape))` / `double[]`.
- **`INT64`:** require an integral value; `m.create(new long[]{requireIntegral(v)}, new Shape(shape))`.
- **`INT32`:** require integral **and** in `int` range; else throw.

```java
/** @throws IllegalArgumentException if v has a fractional part (e.g. 0.7) or is NaN/Inf. */
static long requireIntegral(Number v) {
    if (v instanceof Float || v instanceof Double) {
        double d = v.doubleValue();
        if (Double.isNaN(d) || Double.isInfinite(d) || d != Math.rint(d)) {
            throw new IllegalArgumentException("Value " + v + " is not an integer");
        }
    }
    return v.longValue();
}
```
`DType.from(String)` maps `"float32"`/`"torch.float32"`, `"float64"`/`"torch.float64"`,
`"int32"`/`"torch.int32"`, `"int64"`/`"torch.int64"`; unknown dtype → `IllegalArgumentException`.

## 5. `MapTranslator` (`org.measly.executorch.translate`)

`implements Translator<Map<String,Number>, double[]>`, holding position-ordered `List<ParamSpec>`
and the expected name `Set`.

- **`processInput(ctx, Map<String,Number> input)`:** validate `input.keySet()` equals the expected
  names — on mismatch throw `IllegalArgumentException` naming the **missing** and **unexpected**
  keys. **Report `missing` in position order** by walking the sorted `ParamSpec` list
  (`!input.containsKey(name)`), not via a `HashSet` diff — so messages and tests are deterministic;
  the input `Map`'s iteration order is irrelevant by design (any `Map` is accepted). Build the
  `NDList` in **position order**: for each `ParamSpec`, read `input.get(spec.name())`,
  **reject a `null` value** (`IllegalArgumentException` naming the param), then
  `spec.dtype().createScalar(ctx.getNDManager(), value, spec.shape())`.
- **`processOutput(ctx, NDList output)`:** `output.singletonOrThrow()` (clear error if multi-output),
  then a **dtype-aware widen** to `double[]` — a private helper `switch`ing on `DataType`
  (FLOAT32→`toFloatArray`→widen, FLOAT64→`toDoubleArray`, INT32→`toIntArray`→widen,
  INT64→`toLongArray`→widen; else throw). Fixes the sketch's `toDoubleArray()`-throws bug.
- **`getBatchifier()` returns `null`** — mandatory: `EtNDArray` has no `NDArrayInternal`, so the
  default `StackBatchifier` throws (design-doc §3d note; same as Phase 1's `AddTranslator`).

## 6. Python export helper + fixture

- **`tools/scripts/export_with_spec.py`** — a reusable helper that exports a `.pte` **and** writes
  `model_spec.json` together: pull `USER_INPUT` names from `ep.graph_signature.input_specs` and each
  input's `dtype` + `shape` from `example_inputs` (design §4 sketch). Exported **without** the
  XNNPACK partitioner (portable kernels handle int64), as with `dtypes.pte`.
- **Fixture:** `src/test/resources/models/named/` containing `priced.pte` + `model_spec.json` for a
  scalar `price: float32`, `qty: int64` → `price * qty` (float32 output) model. Committed like the
  other `.pte` fixtures. The existing `native/spike/` fixtures stay put (no churn in 2b).

## 7. Testing

- **Unit (native-free):**
  - `model_spec.json` parse → `List<ParamSpec>`, **including a JSON with unknown extra fields**
    (asserts leniency) and rejection of a non-scalar shape / unknown dtype / duplicate position.
  - `DType` coercion: float targets widen `12`, `12.0`, `2.5`; integer targets accept `12` and
    `12.0`, **reject `0.7`** and out-of-`int`-range with `IllegalArgumentException`.
  - `MapTranslator.processInput` name validation (missing key, extra key → `IllegalArgumentException`
    naming them) and correct **position-ordered** `NDList` with right dtypes/shapes (uses the
    native-free `EtNDManager.create`).
  - `MapTranslator.processOutput` dtype-aware widen for a single FLOAT32 output and a single INT64
    output (build the output `NDArray` via the manager; assert the `double[]`).
- **Integration (native, acceptance):** load `src/test/resources/models/named` via
  `Criteria<Map<String,Number>, double[]>` + `MapTranslator.fromModelPath`, guarded by the
  native-availability assumption; `predict(Map.of("price", 2.5, "qty", 4))` → `double[]{10.0}`.
- Full suite (Phase 1 + 2a + 2b) stays green; leak-tagged tests remain in their own task.

## 8. Open items / confirm at implementation time

- The exact `Criteria`/`Translator` generic wiring for `Map<String,Number>` → `double[]` against
  `ai.djl:api:0.36.0` (the `setTypes(Map.class, double[].class)` erasure form, as in the design §4
  usage).
- `ep.graph_signature.input_specs` / `InputKind.USER_INPUT` shape in the pinned `torch`/`executorch`
  version (the export-helper metadata extraction).
- How the integration test resolves `src/test/resources/models/named` as a filesystem `Path`
  (cwd = project root under Gradle, as the existing `native/spike` tests assume) vs. copying the two
  classpath resources to a temp dir — pick the simpler at implementation.
