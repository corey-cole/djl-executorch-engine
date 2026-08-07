package org.measly.example;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

/**
 * Pass-through translator for the W5 output-size sweep. The benchmark builds the input NDList
 * itself, and {@code processOutput} hands the engine's NDArray back untouched: a
 * {@code toFloatArray()} here would add a second user-side heap copy and pollute the measurement
 * of the engine's own output marshalling (copy 3).
 */
public class AddOutputTranslator implements Translator<NDList, NDArray> {

    @Override
    public NDList processInput(TranslatorContext ctx, NDList input) {
        return input;
    }

    @Override
    public NDArray processOutput(TranslatorContext ctx, NDList list) {
        return list.singletonOrThrow();
    }

    @Override
    public Batchifier getBatchifier() {
        // Return null: EtNDArray does not implement NDArrayInternal (same reason as the test
        // AddTranslator). The sweep is single-sample, so no actual batching is needed.
        return null;
    }
}
