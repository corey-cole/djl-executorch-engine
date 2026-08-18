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
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;

/**
 * Races four arms on the same MobileNetV2 weights, all in DJL {@code Criteria}/{@code Predictor}:
 * {@code ET_HYBRID} (ExecuTorch/XNNPACK forward, PyTorch-backed preprocessing), {@code PYTORCH}
 * (LibTorch), {@code ET_NATIVE} (ExecuTorch forward, plain-Java preprocessing — no LibTorch at
 * all), and {@code ET_OPENVINO} (ExecuTorch forward through the OpenVINO delegate, same
 * preprocessing as ET_HYBRID so the difference is attributable to the delegate). Each arm's engine,
 * artifact, and translator come from {@link Variant}. JMH runs each {@code @Param} value in its own
 * fork, so the {@code ET_NATIVE} fork — which never constructs a PyTorch manager — loads no
 * LibTorch, and the {@code ET_OPENVINO} fork gets its own once-per-process delegate dlopen.
 * Every state closes its translator last, after predictor and model.
 *
 * <p>{@code ET_OPENVINO} needs {@code :example:exportOpenVinoModel} and runs on linux-x86_64 only;
 * elsewhere its cells fail at setup rather than silently measuring something else. Select the other
 * arms with {@code -p variant=...} on a platform without an OpenVINO runtime.
 */
public class MobilenetBenchmark {

    /** Engine choice and shared, read-only fixtures (image + synset), reused by both states. */
    @State(Scope.Benchmark)
    public static class Config {
        @Param public Variant variant;
        /**
         * Selects the ExecuTorch .pte export mode. Arms with no unplanned counterpart (PYTORCH,
         * ET_OPENVINO) resolve both values to the same artifact, so those cells duplicate.
         */
        @Param({"planned", "unplanned"}) public String exportMode;

        Path modelsDir;
        Image image;
        List<String> synset;

        /** Model name for the arm under the selected export mode. */
        String modelName() {
            return variant.modelName(exportMode);
        }

        @Setup(Level.Trial)
        public void setup() throws Exception {
            modelsDir = ModelArtifacts.require(variant.artifact(exportMode)).getParent();
            try (InputStream in = MobilenetBenchmark.class.getResourceAsStream("/kitten.jpg")) {
                image = ImageFactory.getInstance().fromInputStream(in);
            }
            try (InputStream in = MobilenetBenchmark.class.getResourceAsStream("/synset.txt")) {
                synset = Arrays.asList(new String(in.readAllBytes()).trim().split("\\R"));
            }
        }
    }

    /** Builds the Criteria for the arm under test, parameterized by variant. */
    static Criteria<Image, Classifications> criteria(
            Config cfg, CloseableImageTranslator translator, String modelName) {
        return Criteria.builder()
                .setTypes(Image.class, Classifications.class)
                .optEngine(cfg.variant.engine)
                .optModelPath(cfg.modelsDir)
                .optModelName(modelName)
                .optTranslator(translator)
                .build();
    }

    /**
     * Warm predictor held across invocations: measures steady-state inference. Close order on
     * teardown is predictor, then model, then translator (translator owns the native preManager
     * and must outlive both).
     */
    @State(Scope.Benchmark)
    public static class Warm {
        CloseableImageTranslator translator;
        ZooModel<Image, Classifications> model;
        Predictor<Image, Classifications> predictor;

        @Setup(Level.Trial)
        public void setup(Config cfg) throws Exception {
            translator = cfg.variant.newTranslator(cfg.synset);
            try {
                model = criteria(cfg, translator, cfg.modelName()).loadModel();
                predictor = model.newPredictor();
                predictor.predict(cfg.image); // warm once so first measured op is steady-state
            } catch (Throwable t) {
                // JMH won't call @TearDown if @Setup throws, so close whatever was already
                // opened here ourselves (same order as tearDown: predictor, model, translator).
                if (predictor != null) predictor.close();
                if (model != null) model.close();
                translator.close();
                throw t;
            }
        }

        @TearDown(Level.Trial)
        public void tearDown() {
            if (predictor != null) predictor.close();
            if (model != null) model.close();
            if (translator != null) translator.close();
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public Classifications steadyState(Config cfg, Warm warm) throws Exception {
        return warm.predictor.predict(cfg.image);
    }

    @Benchmark
    @BenchmarkMode(Mode.SingleShotTime)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public Classifications coldStart(Config cfg) throws Exception {
        try (CloseableImageTranslator translator = cfg.variant.newTranslator(cfg.synset);
                ZooModel<Image, Classifications> model = criteria(cfg, translator, cfg.modelName()).loadModel();
                Predictor<Image, Classifications> predictor = model.newPredictor()) {
            return predictor.predict(cfg.image); // load + first forward, per invocation
        }
    }
}
