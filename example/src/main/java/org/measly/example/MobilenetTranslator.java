package org.measly.example;

import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.util.List;

/**
 * Image classification translator for the ExecuTorch engine.
 *
 * <p>The ExecuTorch {@code NDArray} (see {@code EtNDArray}/{@code EtEngine}) is a minimal,
 * Java-side data holder with no {@code NDArrayEx} support and no alternative-engine fallback
 * ({@code EtEngine#getAlternativeEngine()} returns {@code null} by design in this phase), so
 * image ops such as resize/to-tensor/normalize/softmax cannot run directly on it — {@link
 * ai.djl.modality.cv.translator.ImageClassificationTranslator}'s built-in pipeline throws {@code
 * UnsupportedOperationException} when handed the model's own manager. This translator instead
 * does that numeric work on a separate PyTorch-backed {@link NDManager} (PyTorch is already a
 * declared dependency of the example, used as the CPU preprocessing baseline) and only hands the
 * ExecuTorch manager a plain tensor via {@link NDManager#from(NDArray)} for the forward pass, then
 * adopts the raw output back for softmax/top-k.
 */
final class MobilenetTranslator implements Translator<Image, Classifications>, AutoCloseable {

    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    private final List<String> synset;
    private final NDManager preManager;

    MobilenetTranslator(List<String> synset) {
        this.synset = synset;
        this.preManager = NDManager.newBaseManager("PyTorch");
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        try (NDManager sub = preManager.newSubManager()) {
            NDArray array = input.toNDArray(sub, Image.Flag.COLOR);
            array = new Resize(224, 224).transform(array);
            array = new ToTensor().transform(array);
            array = new Normalize(MEAN, STD).transform(array);
            array = array.expandDims(0); // add the batch dim ourselves; see getBatchifier()
            NDArray adopted = ctx.getNDManager().from(array);
            return new NDList(adopted);
        }
    }

    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list) {
        NDArray output = list.singletonOrThrow();
        try (NDManager sub = preManager.newSubManager()) {
            NDArray adopted = sub.from(output);
            NDArray probabilities = adopted.softmax(-1);
            return new Classifications(synset, probabilities);
        }
    }

    @Override
    public void close() {
        preManager.close();
    }

    @Override
    public Batchifier getBatchifier() {
        // Batchifier.STACK calls NDArrays.stack, which needs NDArrayEx support EtNDArray doesn't
        // have. processInput() already adds the batch dim itself, so batching here is a no-op
        // (this example always predicts one image at a time).
        return new Batchifier() {
            @Override
            public NDList batchify(NDList[] inputs) {
                if (inputs.length != 1) {
                    throw new UnsupportedOperationException(
                            "MobilenetTranslator only supports a batch size of 1");
                }
                return inputs[0];
            }

            @Override
            public NDList[] unbatchify(NDList inputs) {
                return new NDList[] {inputs};
            }
        };
    }
}
