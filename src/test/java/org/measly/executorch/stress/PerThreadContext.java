package org.measly.executorch.stress;

import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import java.util.concurrent.atomic.AtomicInteger;
import org.measly.executorch.TestSupport;
import org.measly.executorch.engine.EtEngine;

/**
 * One thread's model and predictor, as a single {@link AutoCloseable} unit. <b>This is the
 * reference pattern for using this engine from multiple threads.</b>
 *
 * <p><b>Why a whole model per thread, not just a predictor.</b> {@code EtSymbolBlock.forward()} is
 * not thread-safe <i>on the same model</i>. The shape most DJL users reach for first — one shared
 * {@link ZooModel}, a {@code ThreadLocal<Predictor>} over it — is therefore <b>wrong on this
 * engine</b>: the predictors would share one native handle. Each thread needs its own model.
 *
 * <p>This also makes the workspace sharing mode expressible per thread, since
 * {@code workspaceSharingMode} is a per-model load option.
 *
 * <p><b>Why no ThreadLocal here.</b> Each worker is a dedicated {@link Thread} whose {@code run()}
 * body is a single try-with-resources, so the thread's lifetime <i>is</i> the resource's lifetime
 * and a plain local is strictly better. {@code ThreadLocal} earns its keep only when a context must
 * outlive the block that created it — a pooled executor. Using it here would publish ceremony as if
 * it were safety.
 *
 * <p><b>Two things to know before adapting this to a thread pool.</b> First,
 * {@code ThreadLocal.remove()} drops the reference without calling {@code close()}, so the native
 * handle leaks until GC; it is the most commonly cargo-culted teardown and it is wrong here. Second,
 * a pool needs an explicit drain phase — submit exactly one close-task per pool thread, held apart
 * by a barrier so each lands on a distinct thread, before {@code shutdown()}. That variant is a
 * welcome contribution; it is deliberately not implemented here, because pool-thread affinity would
 * make the sweep's thread-count axis mushy.
 *
 * <p>Never {@code close()} a model with a forward in flight. Try-with-resources on the owning
 * thread makes that impossible by construction, which is the whole point.
 */
public final class PerThreadContext implements AutoCloseable {

    private static final AtomicInteger OPENED = new AtomicInteger();
    private static final AtomicInteger CLOSED = new AtomicInteger();

    private final ZooModel<float[], float[]> model;
    private final Predictor<float[], float[]> predictor;

    private PerThreadContext(ZooModel<float[], float[]> model, Predictor<float[], float[]> predictor) {
        this.model = model;
        this.predictor = predictor;
    }

    /**
     * Loads a private model and predictor for the calling thread.
     *
     * @param sharingMode one of {@code disabled}, {@code per_model}, {@code global}, or {@code null}
     *     to send no option at all and let the runtime default apply
     */
    public static PerThreadContext open(String sharingMode) throws Exception {
        StressGolden.Config cfg = StressGolden.load(TestSupport.stressGoldenPath()).config();
        Criteria.Builder<float[], float[]> b =
                Criteria.builder()
                        .setTypes(float[].class, float[].class)
                        .optEngine("ExecuTorch")
                        .optModelPath(TestSupport.stressModelDir())
                        .optModelName("stress_mlp")
                        .optTranslator(new StressTranslator(cfg.batch, cfg.hidden, cfg.ramp));
        if (sharingMode != null) {
            // The engine publishes the key; do not hardcode the string.
            b.optOption(EtEngine.WORKSPACE_SHARING_MODE_OPTION, sharingMode);
        }

        ZooModel<float[], float[]> model = b.build().loadModel();
        Predictor<float[], float[]> predictor;
        try {
            predictor = model.newPredictor();
        } catch (RuntimeException e) {
            model.close(); // do not leak the native handle when only the predictor failed
            throw e;
        }
        OPENED.incrementAndGet();
        return new PerThreadContext(model, predictor);
    }

    public float[] predict(float v1, float v2) throws Exception {
        return predictor.predict(new float[] {v1, v2});
    }

    @Override
    public void close() {
        // Reverse acquisition order: the predictor borrows from the model, so it goes first.
        try {
            predictor.close();
        } finally {
            model.close();
            CLOSED.incrementAndGet();
        }
    }

    public static int opened() {
        return OPENED.get();
    }

    public static int closed() {
        return CLOSED.get();
    }

    public static void resetCounters() {
        OPENED.set(0);
        CLOSED.set(0);
    }
}
