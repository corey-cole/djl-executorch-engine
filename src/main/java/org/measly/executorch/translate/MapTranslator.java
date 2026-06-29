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
