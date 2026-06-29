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
        try (ZooModel<Map<String, Number>, double[]> model = namedCriteria(dir).loadModel();
                Predictor<Map<String, Number>, double[]> predictor = model.newPredictor()) {
            double[] out = predictor.predict(Map.of("price", 2.5, "qty", 4));
            assertArrayEquals(new double[] {10.0}, out, 1e-6); // 2.5 * 4
        }
    }

    // setTypes(Map.class, ...) erases the input type to raw Map; MapTranslator is
    // Translator<Map<String,Number>, double[]>. The unchecked/rawtypes warnings are confined here.
    @SuppressWarnings({"unchecked", "rawtypes"})
    private static Criteria<Map<String, Number>, double[]> namedCriteria(Path dir) throws Exception {
        return (Criteria<Map<String, Number>, double[]>)
                Criteria.builder()
                        .setTypes(Map.class, double[].class)
                        .optEngine("ExecuTorch")
                        .optModelPath(dir)
                        .optModelName("priced")
                        .optTranslator((ai.djl.translate.Translator) MapTranslator.fromModelPath(dir))
                        .build();
    }
}
