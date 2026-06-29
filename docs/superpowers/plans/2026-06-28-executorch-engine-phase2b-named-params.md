# ExecuTorch DJL Engine — Phase 2b Implementation Plan: Named-Parameter Translator

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `MapTranslator` so callers run inference with named scalar params — `predict(Map<String,Number>)` → `double[]` — driven by an export-time `model_spec.json`.

**Architecture:** Pure Java in a new `org.measly.executorch.translate` package on top of the 2a multi-dtype surface (no native/C++ changes). `model_spec.json` (name/position/dtype/shape, parsed leniently via DJL's Gson) → `List<ParamSpec>`; `MapTranslator` validates names, builds the `NDList` in position order with strict `Number`→dtype coercion, and widens the single output to `double[]`. A Python helper exports a fixture.

**Tech Stack:** Java 17, `ai.djl:api:0.36.0` (incl. `ai.djl.util.JsonUtils` Gson), JUnit 5, Python (`torch`/`executorch` export).

**Spec:** [`docs/superpowers/specs/2026-06-28-executorch-engine-phase2b-named-params-design.md`](../specs/2026-06-28-executorch-engine-phase2b-named-params-design.md).

---

## Notes
- Start on a new branch off `main`. No `build.gradle.kts` changes (Gson ships in `ai.djl:api`; tests use the existing source sets).
- **Verified against the DJL source:** `ai.djl.util.JsonUtils.GSON` (a `Gson`); `Translator<I,O>` with `processInput(TranslatorContext,I)` / `processOutput(TranslatorContext,NDList)` / default `getBatchifier()`; `NDManager.create(float[]/int[]/long[]/double[], Shape)`; `NDArray.toIntArray()/toLongArray()`. Gson ignores unknown JSON fields by default (the leniency the spec requires).
- Tasks 1–3 are native-free (they use `NDManager.newBaseManager("ExecuTorch")` + `create`, which don't load the `.so`). Task 4 is native-gated.

## File structure

| File | Responsibility |
|---|---|
| `src/main/java/org/measly/executorch/translate/DType.java` | scalar dtype enum: strict `Number`→typed-scalar `NDArray`; `from(String)` |
| `src/main/java/org/measly/executorch/translate/ParamSpec.java` | record `(name, position, dtype, shape)` |
| `src/main/java/org/measly/executorch/translate/ModelSpec.java` | parse `model_spec.json` (lenient) → position-sorted `List<ParamSpec>` |
| `src/main/java/org/measly/executorch/translate/MapTranslator.java` | `Translator<Map<String,Number>, double[]>` |
| `tools/scripts/export_with_spec.py` | export `.pte` + `model_spec.json` together |
| `src/test/resources/models/named/{priced.pte, model_spec.json}` | mixed-dtype scalar fixture |
| `src/test/java/org/measly/executorch/translate/*Test.java`, `.../NamedParamsIT.java` | tests |

---

## Task 1: `DType` + `ParamSpec` (strict coercion)

**Files:**
- Create: `src/main/java/org/measly/executorch/translate/DType.java`
- Create: `src/main/java/org/measly/executorch/translate/ParamSpec.java`
- Test: `src/test/java/org/measly/executorch/translate/DTypeTest.java`

- [ ] **Step 1: failing test** — `DTypeTest.java`:
```java
package org.measly.executorch.translate;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import org.junit.jupiter.api.Test;

class DTypeTest {
    @Test
    void fromMapsNamesAndAliases() {
        assertEquals(DType.FLOAT32, DType.from("float32"));
        assertEquals(DType.FLOAT32, DType.from("torch.float32"));
        assertEquals(DType.INT64, DType.from("int64"));
        assertThrows(IllegalArgumentException.class, () -> DType.from("bfloat16"));
    }

    @Test
    void intTargetsRejectFractionalAndOverflow() {
        try (NDManager m = NDManager.newBaseManager("ExecuTorch")) {
            assertThrows(IllegalArgumentException.class,
                    () -> DType.INT32.createScalar(m, 0.7, new long[] {1}));
            assertThrows(IllegalArgumentException.class,
                    () -> DType.INT32.createScalar(m, 3_000_000_000L, new long[] {1}));
        }
    }

    @Test
    void intTargetsAcceptIntegral() {
        try (NDManager m = NDManager.newBaseManager("ExecuTorch")) {
            NDArray a = DType.INT64.createScalar(m, 12.0, new long[] {1});
            assertEquals(DataType.INT64, a.getDataType());
            assertArrayEquals(new long[] {12L}, a.toLongArray());
        }
    }

    @Test
    void floatTargetsWidenAnyNumber() {
        try (NDManager m = NDManager.newBaseManager("ExecuTorch")) {
            NDArray a = DType.FLOAT32.createScalar(m, 2, new long[] {1}); // int -> float
            assertEquals(DataType.FLOAT32, a.getDataType());
            assertArrayEquals(new float[] {2f}, a.toFloatArray());
        }
    }
}
```

- [ ] **Step 2: run → fail** — `./gradlew test --tests '*DTypeTest'` (no `DType`).

- [ ] **Step 3: implement** — `ParamSpec.java`:
```java
package org.measly.executorch.translate;

/** One named scalar input: name, input position, dtype, and (scalar) shape. */
public record ParamSpec(String name, int position, DType dtype, long[] shape) {}
```
`DType.java`:
```java
package org.measly.executorch.translate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/** Scalar parameter dtype: maps a Number to a typed scalar NDArray with strict coercion. */
public enum DType {
    FLOAT32 {
        @Override
        NDArray createScalar(NDManager m, Number v, long[] shape) {
            return m.create(new float[] {v.floatValue()}, new Shape(shape));
        }
    },
    FLOAT64 {
        @Override
        NDArray createScalar(NDManager m, Number v, long[] shape) {
            return m.create(new double[] {v.doubleValue()}, new Shape(shape));
        }
    },
    INT32 {
        @Override
        NDArray createScalar(NDManager m, Number v, long[] shape) {
            long l = requireIntegral(v);
            if (l < Integer.MIN_VALUE || l > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Value " + v + " does not fit in int32");
            }
            return m.create(new int[] {(int) l}, new Shape(shape));
        }
    },
    INT64 {
        @Override
        NDArray createScalar(NDManager m, Number v, long[] shape) {
            return m.create(new long[] {requireIntegral(v)}, new Shape(shape));
        }
    };

    /** Builds a scalar NDArray of {@code shape} from {@code v}. Package-private. */
    abstract NDArray createScalar(NDManager m, Number v, long[] shape);

    /** @throws IllegalArgumentException if v has a fractional part or is NaN/Inf. */
    static long requireIntegral(Number v) {
        if (v instanceof Float || v instanceof Double) {
            double d = v.doubleValue();
            if (Double.isNaN(d) || Double.isInfinite(d) || d != Math.rint(d)) {
                throw new IllegalArgumentException("Value " + v + " is not an integer");
            }
        }
        return v.longValue();
    }

    public static DType from(String name) {
        switch (name) {
            case "float32":
            case "torch.float32":
                return FLOAT32;
            case "float64":
            case "torch.float64":
                return FLOAT64;
            case "int32":
            case "torch.int32":
                return INT32;
            case "int64":
            case "torch.int64":
                return INT64;
            default:
                throw new IllegalArgumentException("Unsupported dtype: " + name);
        }
    }
}
```

- [ ] **Step 4: run → pass** — `./gradlew test --tests '*DTypeTest'`.
- [ ] **Step 5: commit** — `git add src/main/java/org/measly/executorch/translate/DType.java src/main/java/org/measly/executorch/translate/ParamSpec.java src/test/java/org/measly/executorch/translate/DTypeTest.java` then `git commit -m "feat(translate): DType strict scalar coercion + ParamSpec"` (+ Co-Authored-By trailer).

---

## Task 2: `ModelSpec` parsing (lenient)

**Files:**
- Create: `src/main/java/org/measly/executorch/translate/ModelSpec.java`
- Test: `src/test/java/org/measly/executorch/translate/ModelSpecTest.java`

- [ ] **Step 1: failing test** — `ModelSpecTest.java`:
NOTE: Java 17 introduced the text block (`String foo = """\nsomething else\n""";)
```java
package org.measly.executorch.translate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.StringReader;
import java.util.List;
import org.junit.jupiter.api.Test;

class ModelSpecTest {
    @Test
    void parsesAndIgnoresUnknownFields() {
        String json =
                "{ \"runtime\":\"executorch\", \"extra\":123, \"inputs\":["
                    + "{\"name\":\"price\",\"position\":0,\"dtype\":\"float32\",\"shape\":[1],\"desc\":\"x\"},"
                    + "{\"name\":\"qty\",\"position\":1,\"dtype\":\"int64\",\"shape\":[1]}]}";
        List<ParamSpec> specs = ModelSpec.parse(new StringReader(json));
        assertEquals(2, specs.size());
        assertEquals("price", specs.get(0).name());
        assertEquals(DType.FLOAT32, specs.get(0).dtype());
        assertEquals(1, specs.get(1).position());
        assertEquals(DType.INT64, specs.get(1).dtype());
    }

    @Test
    void sortsByPosition() {
        String json =
                "{\"inputs\":["
                    + "{\"name\":\"b\",\"position\":1,\"dtype\":\"int64\",\"shape\":[1]},"
                    + "{\"name\":\"a\",\"position\":0,\"dtype\":\"float32\",\"shape\":[1]}]}";
        List<ParamSpec> specs = ModelSpec.parse(new StringReader(json));
        assertEquals("a", specs.get(0).name());
        assertEquals("b", specs.get(1).name());
    }

    @Test
    void rejectsNonScalarShapeUnknownDtypeAndEmpty() {
        assertThrows(IllegalArgumentException.class, () -> ModelSpec.parse(new StringReader(
                "{\"inputs\":[{\"name\":\"v\",\"position\":0,\"dtype\":\"float32\",\"shape\":[3]}]}")));
        assertThrows(IllegalArgumentException.class, () -> ModelSpec.parse(new StringReader(
                "{\"inputs\":[{\"name\":\"v\",\"position\":0,\"dtype\":\"bf16\",\"shape\":[1]}]}")));
        assertThrows(IllegalArgumentException.class, () -> ModelSpec.parse(new StringReader(
                "{\"inputs\":[]}")));
    }
}
```

- [ ] **Step 2: run → fail** — `./gradlew test --tests '*ModelSpecTest'`.

- [ ] **Step 3: implement** — `ModelSpec.java`:
```java
package org.measly.executorch.translate;

import ai.djl.util.JsonUtils;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** Parses model_spec.json (lenient: unknown fields ignored) into position-sorted ParamSpecs. */
final class ModelSpec {

    private ModelSpec() {}

    static List<ParamSpec> parse(Path jsonFile) throws IOException {
        try (Reader r = Files.newBufferedReader(jsonFile)) {
            return parse(r);
        }
    }

    static List<ParamSpec> parse(Reader reader) {
        Dto dto = JsonUtils.GSON.fromJson(reader, Dto.class);
        if (dto == null || dto.inputs == null || dto.inputs.isEmpty()) {
            throw new IllegalArgumentException("model_spec.json has no inputs");
        }
        List<ParamSpec> specs = new ArrayList<>(dto.inputs.size());
        Set<Integer> positions = new HashSet<>();
        for (InputDto in : dto.inputs) {
            if (in.name == null || in.dtype == null || in.shape == null) {
                throw new IllegalArgumentException("model_spec.json input missing name/dtype/shape");
            }
            if (!positions.add(in.position)) {
                throw new IllegalArgumentException("Duplicate input position " + in.position);
            }
            long product = 1;
            for (long d : in.shape) {
                product *= d;
            }
            if (product != 1) {
                throw new IllegalArgumentException(
                        "Phase 2b supports scalar params only; '" + in.name + "' has shape "
                                + Arrays.toString(in.shape));
            }
            specs.add(new ParamSpec(in.name, in.position, DType.from(in.dtype), in.shape));
        }
        specs.sort(Comparator.comparingInt(ParamSpec::position));
        return specs;
    }

    /** Gson DTO matching the JSON; unknown fields are ignored by Gson. */
    private static final class Dto {
        List<InputDto> inputs;
    }

    private static final class InputDto {
        String name;
        int position;
        String dtype;
        long[] shape;
    }
}
```

- [ ] **Step 4: run → pass** — `./gradlew test --tests '*ModelSpecTest'`.
- [ ] **Step 5: commit** — `git add src/main/java/org/measly/executorch/translate/ModelSpec.java src/test/java/org/measly/executorch/translate/ModelSpecTest.java` then `git commit -m "feat(translate): lenient model_spec.json parsing"` (+ trailer).

---

## Task 3: `MapTranslator`

**Files:**
- Create: `src/main/java/org/measly/executorch/translate/MapTranslator.java`
- Test: `src/test/java/org/measly/executorch/translate/MapTranslatorTest.java`

- [ ] **Step 1: failing test** — `MapTranslatorTest.java`:
```java
package org.measly.executorch.translate;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class MapTranslatorTest {

    private static MapTranslator translator() {
        return new MapTranslator(List.of(
                new ParamSpec("price", 0, DType.FLOAT32, new long[] {1}),
                new ParamSpec("qty", 1, DType.INT64, new long[] {1})));
    }

    @Test
    void buildsInputsInPositionOrderRegardlessOfMapOrder() {
        try (NDManager m = NDManager.newBaseManager("ExecuTorch")) {
            NDList in = translator().buildInputs(m, Map.of("qty", 4, "price", 2.5));
            assertEquals(2, in.size());
            assertArrayEquals(new float[] {2.5f}, in.get(0).toFloatArray()); // price @ pos 0
            assertArrayEquals(new long[] {4L}, in.get(1).toLongArray());     // qty   @ pos 1
        }
    }

    @Test
    void rejectsMissingUnexpectedAndNull() {
        try (NDManager m = NDManager.newBaseManager("ExecuTorch")) {
            assertTrue(assertThrows(IllegalArgumentException.class,
                    () -> translator().buildInputs(m, Map.of("price", 1.0)))
                    .getMessage().contains("qty"));
            assertTrue(assertThrows(IllegalArgumentException.class,
                    () -> translator().buildInputs(m, Map.of("price", 1.0, "qty", 1, "foo", 9)))
                    .getMessage().contains("foo"));
        }
    }

    @Test
    void outputWidensFloat32AndInt64() {
        try (NDManager m = NDManager.newBaseManager("ExecuTorch")) {
            NDArray f = m.create(new float[] {10f}, new Shape(1));
            assertArrayEquals(new double[] {10.0}, MapTranslator.toDoubleArray(f), 0.0);
            NDArray l = m.create(new long[] {7L}, new Shape(1));
            assertArrayEquals(new double[] {7.0}, MapTranslator.toDoubleArray(l), 0.0);
        }
    }

    @Test
    void getBatchifierIsNull() {
        assertNull(translator().getBatchifier());
    }
}
```

- [ ] **Step 2: run → fail** — `./gradlew test --tests '*MapTranslatorTest'`.

- [ ] **Step 3: implement** — `MapTranslator.java`:
```java
package org.measly.executorch.translate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Named-scalar-parameter translator: {@code Map<String,Number>} in, {@code double[]} out. */
public class MapTranslator implements Translator<Map<String, Number>, double[]> {

    private final List<ParamSpec> paramSpecs; // position-ordered
    private final Set<String> expectedNames;

    public MapTranslator(List<ParamSpec> paramSpecs) {
        this.paramSpecs = paramSpecs;
        this.expectedNames = new LinkedHashSet<>();
        for (ParamSpec s : paramSpecs) {
            expectedNames.add(s.name());
        }
    }

    public static MapTranslator fromSpec(Path jsonFile) throws IOException {
        return new MapTranslator(ModelSpec.parse(jsonFile));
    }

    public static MapTranslator fromModelPath(Path modelDir) throws IOException {
        return fromSpec(modelDir.resolve("model_spec.json"));
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Map<String, Number> input) {
        return buildInputs(ctx.getNDManager(), input);
    }

    /** Validates names and builds the position-ordered NDList. Package-private for testing. */
    NDList buildInputs(NDManager manager, Map<String, Number> input) {
        if (!input.keySet().equals(expectedNames)) {
            List<String> missing = new ArrayList<>();
            for (ParamSpec s : paramSpecs) { // position order
                if (!input.containsKey(s.name())) {
                    missing.add(s.name());
                }
            }
            List<String> unexpected = new ArrayList<>();
            for (String k : input.keySet()) {
                if (!expectedNames.contains(k)) {
                    unexpected.add(k);
                }
            }
            throw new IllegalArgumentException(
                    "Parameter mismatch. Missing: " + missing + ", unexpected: " + unexpected
                            + ". Expected (position order): " + expectedNames);
        }
        NDList ndList = new NDList(paramSpecs.size());
        for (ParamSpec spec : paramSpecs) {
            Number value = input.get(spec.name());
            if (value == null) {
                throw new IllegalArgumentException("Null value for parameter '" + spec.name() + "'");
            }
            ndList.add(spec.dtype().createScalar(manager, value, spec.shape()));
        }
        return ndList;
    }

    @Override
    public double[] processOutput(TranslatorContext ctx, NDList output) {
        return toDoubleArray(output.singletonOrThrow());
    }

    /** Widens any numeric NDArray to double[]. Package-private for testing. */
    static double[] toDoubleArray(NDArray array) {
        DataType dt = array.getDataType();
        switch (dt) {
            case FLOAT64:
                return array.toDoubleArray();
            case FLOAT32:
                return widen(array.toFloatArray());
            case INT32:
                return widen(array.toIntArray());
            case INT64:
                return widen(array.toLongArray());
            default:
                throw new IllegalArgumentException("Unsupported output dtype: " + dt);
        }
    }

    private static double[] widen(float[] a) {
        double[] d = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            d[i] = a[i];
        }
        return d;
    }

    private static double[] widen(int[] a) {
        double[] d = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            d[i] = a[i];
        }
        return d;
    }

    private static double[] widen(long[] a) {
        double[] d = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            d[i] = a[i];
        }
        return d;
    }

    @Override
    public Batchifier getBatchifier() {
        return null; // EtNDArray has no NDArrayInternal; default StackBatchifier would throw.
    }
}
```

- [ ] **Step 4: run → pass** — `./gradlew test --tests '*MapTranslatorTest'`.
- [ ] **Step 5: commit** — `git add src/main/java/org/measly/executorch/translate/MapTranslator.java src/test/java/org/measly/executorch/translate/MapTranslatorTest.java` then `git commit -m "feat(translate): MapTranslator (named scalar params -> double[])"` (+ trailer).

---

## Task 4: Python export helper + fixture + end-to-end acceptance

**Files:**
- Create: `tools/scripts/export_with_spec.py`
- Create: `src/test/resources/models/named/{priced.pte, model_spec.json}` (generated)
- Test: `src/test/java/org/measly/executorch/NamedParamsIT.java`

- [ ] **Step 1: export helper** — `tools/scripts/export_with_spec.py`:
```python
"""Export a torch model to <name>.pte AND write model_spec.json (named scalar params).

Parameter names come from the forward() signature; dtype/shape from example_inputs.
Exported without the XNNPACK partitioner so the portable kernels handle int64.
Run from the output directory: it writes priced.pte + model_spec.json into the cwd.
"""
import inspect
import json
import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower


class Priced(torch.nn.Module):
    def forward(self, price, qty):
        return price * qty.to(torch.float32)


def main() -> None:
    model = Priced().eval()
    example_inputs = (torch.ones(1, dtype=torch.float32), torch.ones(1, dtype=torch.int64))

    names = list(inspect.signature(model.forward).parameters)  # ['price', 'qty']
    inputs_meta = [
        {
            "name": name,
            "position": pos,
            "dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(t.shape),
        }
        for pos, (name, t) in enumerate(zip(names, example_inputs))
    ]

    lowered = to_edge_transform_and_lower(export(model, example_inputs)).to_executorch()
    with open("priced.pte", "wb") as f:
        f.write(lowered.buffer)
    with open("model_spec.json", "w") as f:
        json.dump({"runtime": "executorch", "inputs": inputs_meta}, f, indent=2)
    print("wrote priced.pte and model_spec.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: generate the fixture**

Run:
```bash
cd ~/workspace/djl-executorch-engine && mkdir -p src/test/resources/models/named && \
( cd src/test/resources/models/named && \
  source ~/workspace/executorch/.venv/bin/activate && \
  python3 "$HOME/workspace/djl-executorch-engine/tools/scripts/export_with_spec.py" )
```
Expected: prints `wrote priced.pte and model_spec.json`; both files exist under `src/test/resources/models/named/`. Confirm the JSON:
```bash
cat src/test/resources/models/named/model_spec.json
```
Expected: `inputs` = `price`(pos 0, float32, [1]) and `qty`(pos 1, int64, [1]). If the lowered op set rejects `qty.to(float32) * price`, simplify the model body to `price + qty.to(torch.float32)` and adjust the Step-4 assertion accordingly (report which you used).

- [ ] **Step 3: failing acceptance test** — `NamedParamsIT.java`:
```java
package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import org.measly.executorch.translate.MapTranslator;
import org.junit.jupiter.api.Test;

class NamedParamsIT {
    @Test
    void namedScalarParamsThroughPredictor() throws Exception {
        TestSupport.assumeNativeAvailable();
        Path dir = Paths.get("src/test/resources/models/named");
        assumeTrue(Files.exists(dir.resolve("priced.pte")), "named fixture missing");
        Criteria<Map<String, Number>, double[]> criteria =
                Criteria.builder()
                        .setTypes(Map.class, double[].class)
                        .optEngine("ExecuTorch")
                        .optModelPath(dir)
                        .optModelName("priced")
                        .optTranslator(MapTranslator.fromModelPath(dir))
                        .build();
        try (ZooModel<Map<String, Number>, double[]> model = criteria.loadModel();
                Predictor<Map<String, Number>, double[]> predictor = model.newPredictor()) {
            double[] out = predictor.predict(Map.of("price", 2.5, "qty", 4));
            assertArrayEquals(new double[] {10.0}, out, 1e-6); // 2.5 * 4
        }
    }
}
```
Note: `setTypes(Map.class, double[].class)` is the design's raw-`Class` form; if 0.36.0's generics reject it, add `@SuppressWarnings({"unchecked","rawtypes"})` on the method or use raw `Criteria` locals — keep the runtime types `Map<String,Number>`→`double[]`.

- [ ] **Step 4: run → pass** — `./gradlew test --tests '*NamedParamsIT'`, then `./gradlew test` (whole suite green; leak-tagged tests stay excluded). Expected: `predict({price:2.5, qty:4}) == [10.0]`.

- [ ] **Step 5: commit** — `git add tools/scripts/export_with_spec.py src/test/resources/models/named/priced.pte src/test/resources/models/named/model_spec.json src/test/java/org/measly/executorch/NamedParamsIT.java` then `git commit -m "test: named-parameter end-to-end (Map<String,Number> -> double[])"` (+ trailer).

---

## Self-review against the spec
- **§1 success test** → Task 4 `NamedParamsIT` (`predict({price,qty}) == [10.0]`). ✓
- **§3 model_spec format/loading** (shape, lenient, scalar-only, Gson, `fromSpec`/`fromModelPath`) → Task 2 `ModelSpec` + Task 3 factories. ✓
- **§4 ParamSpec/DType strict coercion** → Task 1. ✓
- **§5 MapTranslator** (name validation reporting missing in position order, null-value reject, position-ordered NDList, dtype-aware `double[]`, `getBatchifier()=null`) → Task 3. ✓
- **§6 Python helper + fixture** (`tools/scripts/`, `src/test/resources/models/named/`) → Task 4. ✓
- **§7 testing** (native-free units T1–T3; native acceptance T4) ✓
- **Type consistency:** `DType.createScalar(NDManager,Number,long[])`, `ParamSpec(String,int,DType,long[])`, `ModelSpec.parse(Reader|Path)→List<ParamSpec>`, `MapTranslator.{buildInputs(NDManager,Map),toDoubleArray(NDArray),fromSpec,fromModelPath}` — consistent across tasks.

## Out of scope (later)
Tensor/array-valued named inputs (`Map<String,Object>` — study DJL's PyTorch engine first), named/multi-output, auto-discovery `TranslatorFactory`, dtypes beyond the four numeric types.
