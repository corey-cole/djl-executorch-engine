package org.measly.executorch;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

/**
 * Maps a 2-element float[] {a, b} to the add.pte inputs and back to a single float.
 *
 * <p>Uses a null batchifier because EtNDArray does not implement NDArrayInternal (required by the
 * default StackBatchifier via NDArrays.stack). add.pte is always called with a single sample, so
 * no actual batching is needed.
 */
public class AddTranslator implements Translator<float[], Float> {

    @Override
    public Batchifier getBatchifier() {
        // Return null: EtNDArray does not implement NDArrayInternal, so DJL's default
        // StackBatchifier would fail in NDArrays.stack(). A null batchifier makes
        // Predictor run each sample through processInput/processOutput individually
        // (correct for this inference-only, single-sample engine; hybrid mode is Phase 3).
        return null;
    }

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
