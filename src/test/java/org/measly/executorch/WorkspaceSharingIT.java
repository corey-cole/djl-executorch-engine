package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;

/**
 * End-to-end per-model workspace sharing. Safe in the shared test JVM and needs no dedicated
 * Gradle task: the mode is per model, so nothing here contaminates a later test. (Contrast
 * IntraOpThreadsIT, which needs a forked JVM because that pool is process-global.)
 */
class WorkspaceSharingIT {

    private static Criteria<float[], Float> criteriaWithMode(String mode) {
        Criteria.Builder<float[], Float> b =
                Criteria.builder()
                        .setTypes(float[].class, Float.class)
                        .optEngine("ExecuTorch")
                        .optModelPath(Paths.get("native/spike"))
                        .optModelName("add")
                        .optTranslator(new AddTranslator());
        if (mode != null) {
            b.optOption("workspaceSharingMode", mode);
        }
        return b.build();
    }

    @Test
    void everyModeLoadsAndPredicts() throws Exception {
        TestSupport.assumeNativeAvailable();
        for (String mode : new String[] {"disabled", "per_model", "global", "GLOBAL", " disabled "}) {
            try (ZooModel<float[], Float> model = criteriaWithMode(mode).loadModel();
                    Predictor<float[], Float> predictor = model.newPredictor()) {
                assertEquals(5f, predictor.predict(new float[] {2f, 3f}), 1e-6, "mode=" + mode);
            }
        }
    }

    @Test
    void modesComposeAcrossConcurrentlyLoadedModels() throws Exception {
        // A model electing "disabled" is isolated regardless of what other loaded models chose.
        // Both must remain independently usable while the other is open.
        TestSupport.assumeNativeAvailable();
        try (ZooModel<float[], Float> isolated = criteriaWithMode("disabled").loadModel();
                ZooModel<float[], Float> shared = criteriaWithMode("global").loadModel();
                Predictor<float[], Float> p1 = isolated.newPredictor();
                Predictor<float[], Float> p2 = shared.newPredictor()) {
            assertEquals(5f, p1.predict(new float[] {2f, 3f}), 1e-6);
            assertEquals(9f, p2.predict(new float[] {4f, 5f}), 1e-6);
            assertEquals(3f, p1.predict(new float[] {1f, 2f}), 1e-6);
        }
    }

    @Test
    void anUnrecognizedModeFailsTheLoad() {
        TestSupport.assumeNativeAvailable();
        // Explicit per-model intent must not degrade silently to the default.
        assertThrows(Exception.class, () -> criteriaWithMode("disabeld").loadModel());
    }

    @Test
    void noOptionStillLoads() throws Exception {
        TestSupport.assumeNativeAvailable();
        try (ZooModel<float[], Float> model = criteriaWithMode(null).loadModel();
                Predictor<float[], Float> predictor = model.newPredictor()) {
            assertEquals(5f, predictor.predict(new float[] {2f, 3f}), 1e-6);
        }
    }
}
