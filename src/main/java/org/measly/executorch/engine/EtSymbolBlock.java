package org.measly.executorch.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractSymbolBlock;
import ai.djl.nn.ParameterList;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
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

    EtSymbolBlock(long handle, EtNDManager manager) {
        this.handle = handle;
        this.manager = manager;
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        EtTensor[] in = new EtTensor[inputs.size()];
        for (int i = 0; i < inputs.size(); ++i) {
            NDArray arr = inputs.get(i);
            if (arr.getDataType() != DataType.FLOAT32) {
                throw new IllegalArgumentException(
                        "ExecuTorch Phase 1 supports only FLOAT32 inputs, got "
                                + arr.getDataType() + " at index " + i);
            }
            in[i] = new EtTensor(arr.getShape().getShape(), arr.toFloatArray());
        }
        EtTensor[] out = EtNative.forward(handle, in);
        NDList ret = new NDList(out.length);
        for (EtTensor t : out) {
            // Outputs are created on the model-scoped manager, then re-attached below to the
            // request's manager so they are freed with the inference call, not the model.
            ret.add(manager.create(t.data, new Shape(t.shape)));
        }
        if (!inputs.isEmpty()) {
            ret.attach(inputs.head().getManager());
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
}
