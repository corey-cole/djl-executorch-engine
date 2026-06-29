package org.measly.executorch;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import java.nio.file.Paths;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Leak gates that turn a lifecycle leak into a deterministic OutOfMemoryError. Run via the
 * {@code leakTest} Gradle task under {@code -XX:MaxDirectMemorySize=64m -Xmx256m}; a correct
 * lifecycle survives the GC-reclaim retry, a leak exhausts memory and fails.
 */
@Tag("leak")
class LeakStressTest {

    /** Direct-buffer lifecycle (native-free): 200 x 4MB direct arrays, each freed before the next. */
    @Test
    void directBufferLifecycleUnderPressure() {
        try (NDManager base = NDManager.newBaseManager("ExecuTorch")) {
            for (int i = 0; i < 200; i++) {
                try (NDManager sub = base.newSubManager()) {
                    sub.create(new float[1_000_000], new Shape(1_000_000)); // 4 MB off-heap
                }
            }
        }
    }

    /** Inference path: many predictions; a leaked per-call input/output buffer accumulates. */
    @Test
    void inferencePathUnderPressure() throws Exception {
        TestSupport.assumeNativeAvailable();
        Criteria<float[], Float> criteria =
                Criteria.builder()
                        .setTypes(float[].class, Float.class)
                        .optEngine("ExecuTorch")
                        .optModelPath(Paths.get("native/spike"))
                        .optModelName("add")
                        .optTranslator(new AddTranslator())
                        .build();
        try (ZooModel<float[], Float> model = criteria.loadModel();
                Predictor<float[], Float> predictor = model.newPredictor()) {
            for (int i = 0; i < 20_000; i++) {
                predictor.predict(new float[] {1f, 2f});
            }
        }
    }
}
