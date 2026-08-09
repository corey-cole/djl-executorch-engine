package org.measly.executorch.engine;

import java.nio.ByteBuffer;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractSymbolBlock;
import ai.djl.nn.ParameterList;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import org.measly.executorch.jni.EtMethodMeta;
import org.measly.executorch.jni.EtNative;
import org.measly.executorch.jni.EtTensor;

/**
 * Runs ExecuTorch forward() by marshalling NDList &lt;-&gt; EtTensor[].
 *
 * <p>Not safe for concurrent {@code forward()} calls on the same model: the underlying
 * ExecuTorch {@code Module} may not support concurrent execution. Use one {@code Model} /
 * {@code Predictor} per thread, and do not close the model while a forward call is in flight.
 *
 * <p><b>Threading, and why more threads is usually wrong <i>under the default sharing mode</i>.</b>
 * The rule above is about <i>safety</i>, not throughput. XNNPACK-delegated models already
 * parallelize inside a single {@code forward()} on ExecuTorch's shared intra-op pool, and under the
 * shipped {@code global} workspace sharing mode concurrent delegate calls serialize on one
 * process-global workspace mutex — so N {@code Predictor}s on N threads is typically slower than
 * one, not N× faster. Tune {@code ai.djl.executorch.num_threads} before adding caller threads.
 * Measured on a 4-core/8-thread host with MobileNetV2: 1 thread 462 forwards/s, 4 threads 305, 8
 * threads 147 (peak RSS 33 MB → 224 MB). Ratios on larger hosts are unmeasured.
 *
 * <p><b>These figures are conditional on that mutex.</b> Loading a model with
 * {@code Criteria.optOption("workspaceSharingMode", "disabled")} gives it a private workspace, and
 * caller threads then scale: measured achieved parallelism at one intra-op thread was 1.12× / 1.12×
 * / 1.12× / 1.17× at 1/2/4/8 caller threads under {@code global}, versus 1.12× / 2.23× / 4.35× /
 * 7.13× under {@code disabled}. The cost is activation-arena memory per delegate instance. Prefer
 * intra-op tuning first; reach for {@code disabled} when several models with differing call rates
 * share a JVM and memory is not the binding constraint.
 */
public class EtSymbolBlock extends AbstractSymbolBlock implements AutoCloseable {

    private volatile long handle;
    private EtNDManager manager;
    private final EtMethodMeta meta;
    // Serializes the stats cold path against close(): toStats() reads the native handle and
    // queries staging bytes, close() destroys the handle, and a destroy between the read and the
    // JNI call would be a use-after-free. The lock is taken only by close()/toStats()/deregister
    // (reentrant), never by forwardInternal — the hot path stays lock-free.
    private final Object statsLock = new Object();
    // Attached by EtModel.load right after construction. Null only in the narrow window before
    // that, and in tests that build a block directly.
    private volatile EtModelCounters counters;

    EtSymbolBlock(long handle, EtNDManager manager, EtMethodMeta meta) {
        this.handle = handle;
        this.manager = manager;
        this.meta = java.util.Objects.requireNonNull(meta, "meta");
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        final int count = inputs.size();
        if (count != meta.numInputs) {
            throw new IllegalArgumentException(
                    "ExecuTorch model expects " + meta.numInputs + " inputs, got " + count);
        }
        EtTensor[] in = new EtTensor[count];
        for (int i = 0; i < count; ++i) {
            EtNDArray et = manager.from(inputs.get(i));
            int st = EtDataTypes.toScalarType(et.getDataType());
            if (st != meta.inputScalarTypes[i]) {
                throw new IllegalArgumentException(
                        "Input " + i + " dtype " + et.getDataType()
                                + " != model's expected ScalarType " + meta.inputScalarTypes[i]);
            }
            ByteBuffer buf = et.toByteBuffer();
            if (!buf.isDirect()) {
                ByteBuffer direct = manager.allocateDirect(buf.remaining());
                direct.put(buf);
                direct.rewind();
                buf = direct;
            }
            in[i] = new EtTensor(et.getShape().getShape(), st, buf);
        }
        final long startNanos = System.nanoTime();
        EtTensor[] out = EtNative.forward(handle, in);
        EtModelCounters c = counters;
        if (c != null) {
            c.recordForward(System.nanoTime() - startNanos);
        }
        NDManager rm = inputs.isEmpty() ? manager : inputs.head().getManager();
        EtNDManager target = (rm instanceof EtNDManager) ? (EtNDManager) rm : manager;
        NDList ret = new NDList(out.length);
        for (EtTensor t : out) {
            DataType dt = EtDataTypes.fromScalarType(t.scalarType);
            ret.add(target.wrap(t.data, new Shape(t.shape), dt));
        }
        return ret;
    }

    @Override
    public void removeLastBlock() {
        throw new UnsupportedOperationException("ExecuTorch does not support removeLastBlock");
    }

    @Override
    public ParameterList getDirectParameters() {
        return new ParameterList();
    }

    @Override
    public void close() {
        // Mutual exclusion with toStats(): a concurrent snapshot poll must never observe the
        // handle between destroy() freeing it and the handle read. Reentrant — deregister() below
        // calls toStats() on this block, which takes the same monitor.
        synchronized (statsLock) {
            if (handle != 0) {
                // Before destroy: deregister reads this block's counters for the closed-model
                // rollup, and toStats() would report -1 staging bytes once the handle is zeroed.
                EtEngineStats.deregister(handle);
                EtNative.destroy(handle);
                handle = 0;
            }
        }
    }

    /** @return true once the native handle has been released by {@link #close()}. */
    boolean isClosed() {
        return handle == 0;
    }

    /** Attaches the counters this block updates on each forward. Called once, at load. */
    void attachCounters(EtModelCounters counters) {
        this.counters = counters;
    }

    /**
     * Builds an immutable snapshot of this model's counters and native footprint.
     *
     * <p>Runs under {@link #statsLock}, the same monitor {@link #close()} holds while it destroys
     * the native handle: a concurrent {@code close()} cannot free the handle between the non-zero
     * check and {@code EtNative.stagingBytes(...)}, so a poll can never hand a freed pointer to
     * native code. A closed block reports {@code -1} staging bytes, meaning "unavailable" —
     * distinct from a memory-planned model's genuine {@code 0}.
     *
     * @return the snapshot, or {@code null} if no counters were ever attached
     */
    EtModelStats toStats() {
        EtModelCounters c = counters;
        if (c == null) {
            return null;
        }
        synchronized (statsLock) {
            final long h = handle;
            long staging = -1L;
            if (h != 0) {
                try {
                    staging = EtNative.stagingBytes(h);
                } catch (RuntimeException e) {
                    staging = -1L; // a monitoring read must never propagate
                }
            }
            return new EtModelStats(
                    c.name(),
                    c.workspaceSharingMode(),
                    c.plannedArenaBytes(),
                    staging,
                    c.loadNanos(),
                    c.forwardCount(),
                    c.forwardTotalNanos(),
                    c.forwardMaxNanos());
        }
    }

    /** @return the metadata captured at load (test seam, mirrors {@link #isClosed()}). */
    EtMethodMeta meta() {
        return meta;
    }
}
