package org.measly.example;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import java.nio.file.Path;
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
 * Measures the engine's output marshalling (copy 3: native arena → JVM-visible buffer) against a
 * trivial add model at four output sizes, from MobileNet's 4 KB logits to a 64 MB tensor. The
 * kernel is a single elementwise add, so the measured time is marshalling + GC, not compute —
 * exactly the W5 output/GC arm. Inputs are built once in {@link Warm} and reused: the engine
 * copies inputs per forward, and {@code Predictor.predict} does not close the caller's input
 * NDList (only its own {@code PredictorContext} manager closes).
 */
public class AddOutputBenchmark {

    /** Model name → N of the (1, N) f32 output; must match export_w5_add_models.py. */
    private static int outputLength(String model) {
        switch (model) {
            case "add_4kb":
                return 1024;
            case "add_256kb":
                return 65536;
            case "add_4mb":
                return 1048576;
            case "add_64mb":
                return 16777216;
            default:
                throw new IllegalArgumentException("unknown model: " + model);
        }
    }

    @State(Scope.Benchmark)
    public static class Config {
        @Param({"add_4kb", "add_256kb", "add_4mb", "add_64mb"}) public String model;

        Path modelsDir;

        @Setup(Level.Trial)
        public void setup() throws Exception {
            modelsDir = ModelArtifacts.require(model + ".pte").getParent();
        }
    }

    /** Warm predictor + input NDList held across invocations; close order is predictor, model, manager. */
    @State(Scope.Benchmark)
    public static class Warm {
        NDManager manager;
        ZooModel<NDList, NDArray> model;
        Predictor<NDList, NDArray> predictor;
        NDList input;

        @Setup(Level.Trial)
        public void setup(Config cfg) throws Exception {
            manager = NDManager.newBaseManager();
            int n = outputLength(cfg.model);
            NDArray a = manager.create(new float[n], new Shape(1, n));
            NDArray b = manager.create(new float[n], new Shape(1, n));
            input = new NDList(a, b);
            AddOutputTranslator translator = new AddOutputTranslator();
            try {
                model = Criteria.builder()
                        .setTypes(NDList.class, NDArray.class)
                        .optEngine("ExecuTorch")
                        .optModelPath(cfg.modelsDir)
                        .optModelName(cfg.model)
                        .optTranslator(translator)
                        .build()
                        .loadModel();
                predictor = model.newPredictor();
                predictor.predict(input); // warm once so the first measured op is steady-state
            } catch (Throwable t) {
                // JMH won't call @TearDown if @Setup throws, so close whatever was already
                // opened here ourselves (same order as tearDown: predictor, model, manager).
                if (predictor != null) predictor.close();
                if (model != null) model.close();
                if (manager != null) manager.close();
                throw t;
            }
        }

        @TearDown(Level.Trial)
        public void tearDown() {
            if (predictor != null) predictor.close();
            if (model != null) model.close();
            if (manager != null) manager.close();
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public NDArray steadyState(Config cfg, Warm warm) throws Exception {
        NDArray out = warm.predictor.predict(warm.input);
        // Required: DJL does not close the output NDList, and the output EtNDArray wraps on the
        // input's manager (EtSymbolBlock.forwardInternal), so without the per-op close the
        // buffers would accumulate for the whole trial — the IREE OOM profile. EtNDArray.close()
        // is idempotent, so a DJL-side close later would be a no-op.
        out.close();
        return out;
    }
}
