package org.measly.executorch.stress;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

/**
 * Maps {@code {v1, v2}} to the stress model's two (batch, hidden) f32 inputs and back to the flat
 * output tensor.
 *
 * <p>The inputs are reconstructed from two scalars rather than stored, which is what keeps the
 * golden file at kilobytes instead of megabytes.
 */
public class StressTranslator implements Translator<float[], float[]> {

    private final int batch;
    private final int hidden;
    private final float ramp;

    public StressTranslator(int batch, int hidden, float ramp) {
        this.batch = batch;
        this.hidden = hidden;
        this.ramp = ramp;
    }

    /**
     * Deterministic input tensor from one scalar, bit-identical to {@code build_input} in
     * tools/scripts/export_stress_model.py.
     *
     * <p>Every step is float32, in the same order as the Python side: {@code (float) i * ramp + v}.
     * Accumulating the ramp in double and narrowing at the end would differ by an ulp and silently
     * break the bitwise self-reference check in StressGateIT — which is exactly the kind of drift
     * that check exists to catch, so it must not be introduced by the harness itself.
     *
     * <p>Element 0 is exactly {@code v} (index 0 contributes a zero ramp term); that element steers
     * the bucket lookup.
     */
    public static float[] buildInput(float v, int batch, int hidden, float ramp) {
        float[] out = new float[batch * hidden];
        for (int i = 0; i < out.length; i++) {
            out[i] = (float) i * ramp + v;
        }
        return out;
    }

    @Override
    public Batchifier getBatchifier() {
        // Null, for the same reason as AddTranslator: EtNDArray does not implement NDArrayInternal,
        // so DJL's default StackBatchifier fails in NDArrays.stack(). The model's own batch
        // dimension is baked into the exported shape, not produced by DJL batching.
        return null;
    }

    @Override
    public NDList processInput(TranslatorContext ctx, float[] input) {
        Shape shape = new Shape(batch, hidden);
        NDArray a = ctx.getNDManager().create(buildInput(input[0], batch, hidden, ramp), shape);
        NDArray b = ctx.getNDManager().create(buildInput(input[1], batch, hidden, ramp), shape);
        return new NDList(a, b);
    }

    @Override
    public float[] processOutput(TranslatorContext ctx, NDList list) {
        return list.singletonOrThrow().toFloatArray();
    }
}
