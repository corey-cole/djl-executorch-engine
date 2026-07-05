package org.measly.example;

import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Translator;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/** Classifies a bundled image with MobileNetV2 through the ExecuTorch engine and prints top-5. */
public final class MobilenetExample {
    private MobilenetExample() {}

    public static void main(String[] args) throws Exception {
        Path models = ModelArtifacts.require("mobilenet_v2.pte").getParent();

        List<String> synset = loadSynset();
        Translator<Image, Classifications> translator = new MobilenetTranslator(synset);

        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optEngine("ExecuTorch")
                        .optModelPath(models)
                        .optModelName("mobilenet_v2")
                        .optTranslator(translator)
                        .build();

        try (InputStream imageStream =
                        MobilenetExample.class.getResourceAsStream("/kitten.jpg");
                ZooModel<Image, Classifications> model = criteria.loadModel();
                Predictor<Image, Classifications> predictor = model.newPredictor()) {
            Image image = ImageFactory.getInstance().fromInputStream(imageStream);
            Classifications result = predictor.predict(image);
            System.out.println("Top-5 (ExecuTorch / MobileNetV2):");
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
