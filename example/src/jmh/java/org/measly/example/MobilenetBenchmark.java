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
 * Races the ExecuTorch engine (.pte) against the DJL PyTorch engine (.pt) on the same MobileNetV2
 * weights, reusing {@link MobilenetTranslator} for both arms.
 *
 * <p>{@link MobilenetTranslator} is engine-agnostic by construction (Task 4): it does all image
 * preprocessing on its own PyTorch-backed {@code preManager} and only adopts a plain tensor into
 * whichever engine's {@code NDManager} the predictor is using, so it works unchanged for the
 * PyTorch arm too. It owns that native {@code preManager} and must be closed - see {@link
 * MobilenetTranslator#close()} - so every state below closes the translator last, after the
 * predictor and model that use it.
 */
public class MobilenetBenchmark {

    /** Engine choice and shared, read-only fixtures (image + synset), reused by both states. */
    @State(Scope.Benchmark)
    public static class Config {
        @Param({"ExecuTorch", "PyTorch"})
        public String engine;

        Path modelsDir;
        Image image;
        List<String> synset;

        @Setup(Level.Trial)
        public void setup() throws Exception {
            modelsDir = ModelArtifacts.require("mobilenet_v2.pte").getParent();
            try (InputStream in = MobilenetBenchmark.class.getResourceAsStream("/kitten.jpg")) {
                image = ImageFactory.getInstance().fromInputStream(in);
            }
            try (InputStream in = MobilenetBenchmark.class.getResourceAsStream("/synset.txt")) {
                synset = Arrays.asList(new String(in.readAllBytes()).trim().split("\\R"));
            }
        }
    }

    /** Builds the Criteria exactly as {@code MobilenetExample} does, parameterized by engine. */
    static Criteria<Image, Classifications> criteria(Config cfg, MobilenetTranslator translator) {
        return Criteria.builder()
                .setTypes(Image.class, Classifications.class)
                .optEngine(cfg.engine)
                .optModelPath(cfg.modelsDir)
                .optModelName("mobilenet_v2")
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
        MobilenetTranslator translator;
        ZooModel<Image, Classifications> model;
        Predictor<Image, Classifications> predictor;

        @Setup(Level.Trial)
        public void setup(Config cfg) throws Exception {
            translator = new MobilenetTranslator(cfg.synset);
            try {
                model = criteria(cfg, translator).loadModel();
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
        try (MobilenetTranslator translator = new MobilenetTranslator(cfg.synset);
                ZooModel<Image, Classifications> model = criteria(cfg, translator).loadModel();
                Predictor<Image, Classifications> predictor = model.newPredictor()) {
            return predictor.predict(cfg.image); // load + first forward, per invocation
        }
    }
}
