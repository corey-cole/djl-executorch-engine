package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import java.nio.file.Paths;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.measly.executorch.jni.EtNative;

/**
 * Leak gates that turn a lifecycle leak into a failure. Heap/direct-buffer lifecycle leaks are
 * caught by the OOM caps under {@code -XX:MaxDirectMemorySize=64m -Xmx256m} (a correct lifecycle
 * survives the GC-reclaim retry, a leak exhausts memory and fails). Since W6, output buffers are
 * JNI-allocated and do not count against those caps, so the output leak gate is the native
 * alive-counter ({@link EtNative#aliveOutputBuffers()}), asserted to drain once unreachable.
 * Run via the {@code leakTest} Gradle task under {@code -XX:MaxDirectMemorySize=64m -Xmx256m}.
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
        assertOutputBuffersDrained();
    }

    /**
     * Inference path over the unplanned (borrowed-input) variant of the same add model: W7 stages
     * each input into an engine-owned slot, so this exercises the staging path under the OOM caps.
     * Native staging memory is not counted against {@code -Xmx256m} / {@code -XX:MaxDirectMemorySize=64m},
     * so the assertion is the same "does not OOM/crash" as the planned variant — the point is that
     * the staging path must not regress the existing gates.
     */
    @Test
    void inferencePathUnderPressureUnplanned() throws Exception {
        TestSupport.assumeUnplannedModelAvailable(); // calls loadNativeLibrary(); implies assumeNativeAvailable
        Criteria<float[], Float> criteria =
                Criteria.builder()
                        .setTypes(float[].class, Float.class)
                        .optEngine("ExecuTorch")
                        .optModelPath(Paths.get("native/spike"))
                        .optModelName("add_unplanned")
                        .optTranslator(new AddTranslator())
                        .build();
        try (ZooModel<float[], Float> model = criteria.loadModel();
                Predictor<float[], Float> predictor = model.newPredictor()) {
            for (int i = 0; i < 20_000; i++) {
                predictor.predict(new float[] {1f, 2f});
            }
        }
        assertOutputBuffersDrained();
    }

    /**
     * W6 leak gate: JNI-allocated output buffers do not count against the heap/direct-memory caps,
     * so the alive counter — not memory pressure — is the output-leak signal. Every predict's
     * output wraps on the PredictorContext manager, which predict() closes, making the buffer
     * unreachable; the Cleaner must free it, or this poll fails after 5 s.
     */
    private static void assertOutputBuffersDrained() throws InterruptedException {
        long deadline = System.currentTimeMillis() + 5_000;
        while (System.currentTimeMillis() < deadline && EtNative.aliveOutputBuffers() != 0) {
            System.gc();
            Thread.sleep(25);
        }
        assertEquals(0, EtNative.aliveOutputBuffers(),
                "Cleaner must free every JNI-allocated output buffer once unreachable");
    }
}
