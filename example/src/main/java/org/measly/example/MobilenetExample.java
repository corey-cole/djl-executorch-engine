package org.measly.example;

import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/** Classifies a bundled image with MobileNetV2 through the ExecuTorch engine and prints top-5. */
public final class MobilenetExample {
    private MobilenetExample() {}

    public static void main(String[] args) throws Exception {
        Variant variant;
        try {
            variant = args.length > 0 ? Variant.valueOf(args[0]) : Variant.ET_HYBRID;
        } catch (IllegalArgumentException e) {
            System.err.println("Unknown variant: " + args[0]);
            System.err.println("Valid variants: " + Arrays.toString(Variant.values()));
            System.err.println("Usage: run [--args=\"<VARIANT>\"]   (default ET_HYBRID)");
            System.exit(2);
            return;
        }

        String artifact = variant == Variant.PYTORCH ? "mobilenet_v2.pt" : "mobilenet_v2.pte";
        Path models = ModelArtifacts.require(artifact).getParent();
        List<String> synset = loadSynset();

        try (CloseableImageTranslator translator = variant.newTranslator(synset);
                InputStream imageStream =
                        MobilenetExample.class.getResourceAsStream("/kitten.jpg");
                ZooModel<Image, Classifications> model =
                        Criteria.builder()
                                .setTypes(Image.class, Classifications.class)
                                .optEngine(variant.engine)
                                .optModelPath(models)
                                .optModelName("mobilenet_v2")
                                .optTranslator(translator)
                                .build()
                                .loadModel();
                Predictor<Image, Classifications> predictor = model.newPredictor()) {
            Image image = ImageFactory.getInstance().fromInputStream(imageStream);
            Classifications result = predictor.predict(image);
            System.out.println("Top-5 (" + variant + " / MobileNetV2):");
            for (Classifications.Classification c : result.topK(5)) {
                System.out.printf("  %-30s %.4f%n", c.getClassName(), c.getProbability());
            }
        }
    }

    private static List<String> loadSynset() throws Exception {
        try (InputStream in = MobilenetExample.class.getResourceAsStream("/synset.txt")) {
            if (in == null) {
                throw new IllegalStateException("synset.txt not found on classpath");
            }
            return Arrays.asList(new String(in.readAllBytes()).trim().split("\\R"));
        }
    }
}
