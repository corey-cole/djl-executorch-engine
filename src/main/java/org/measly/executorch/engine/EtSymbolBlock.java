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
 */
public class EtSymbolBlock extends AbstractSymbolBlock implements AutoCloseable {

    private volatile long handle;
    private EtNDManager manager;
    private final EtMethodMeta meta;

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
        EtTensor[] out = EtNative.forward(handle, in);
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
        if (handle != 0) {
            EtNative.destroy(handle);
            handle = 0;
        }
    }

    /** @return true once the native handle has been released by {@link #close()}. */
    boolean isClosed() {
        return handle == 0;
    }
}
