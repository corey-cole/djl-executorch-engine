package org.measly.example;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * The smallest useful ExecuTorch-under-DJL program: load a {@code .pte} and run one prediction.
 *
 * <p>This class is the source of the README's quickstart. It is compiled by the build, so an API
 * change breaks CI rather than leaving the README quietly wrong. Keep the two in sync.
 *
 * <p>Defaults to the two-input float32 {@code add} model committed at {@code native/spike/add.pte},
 * so it runs from a fresh clone with no model export step. Pass a directory and model name to run
 * a different {@code .pte}.
 */
public final class QuickStart {

    private QuickStart() {}

    /** Turns a {@code float[]} into the model's input list and its output back into a float. */
    private static final class AddTranslator implements Translator<float[], Float> {
        // EtNDArray does not support NDArrays.stack (used by the default STACK batchifier),
        // so we provide a no-op batchifier. This example always predicts one input at a time.
        private static final Batchifier BATCHIFIER =
                new Batchifier() {
                    @Override
                    public NDList batchify(NDList[] inputs) {
                        if (inputs.length != 1) {
                            throw new UnsupportedOperationException("Batch size 1 only");
                        }
                        return inputs[0];
                    }

                    @Override
                    public NDList[] unbatchify(NDList inputs) {
                        return new NDList[] {inputs};
                    }
                };

        @Override
        public NDList processInput(TranslatorContext ctx, float[] input) {
            // One NDArray per model input. The add model takes two 1-element float32 tensors.
            NDArray a = ctx.getNDManager().create(new float[] {input[0]});
            NDArray b = ctx.getNDManager().create(new float[] {input[1]});
            return new NDList(a, b);
        }

        @Override
        public Float processOutput(TranslatorContext ctx, NDList list) {
            return list.singletonOrThrow().toFloatArray()[0];
        }

        @Override
        public Batchifier getBatchifier() {
            return BATCHIFIER;
        }
    }

    /**
     * Runs one prediction and prints the result.
     *
     * @param args optionally the model directory and model name; defaults to {@code
     *     native/spike} and {@code add}
     * @throws Exception if the model cannot be loaded or the prediction fails
     */
    public static void main(String[] args) throws Exception {
        Path modelDir = Paths.get(args.length > 0 ? args[0] : "native/spike");
        String modelName = args.length > 1 ? args[1] : "add";

        Criteria<float[], Float> criteria =
                Criteria.builder()
                        .setTypes(float[].class, Float.class)
                        .optEngine("ExecuTorch") // this engine, by name
                        .optModelPath(modelDir)
                        .optModelName(modelName)
                        .optTranslator(new AddTranslator())
                        .build();

        // One ZooModel and one Predictor per thread: forward() is not safe to share.
        try (ZooModel<float[], Float> model = criteria.loadModel();
                Predictor<float[], Float> predictor = model.newPredictor()) {
            System.out.println("2 + 3 = " + predictor.predict(new float[] {2f, 3f}));
        }
    }
}
