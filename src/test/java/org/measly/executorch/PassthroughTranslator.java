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
