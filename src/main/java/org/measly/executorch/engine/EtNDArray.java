package org.measly.executorch.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrayAdapter;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** ExecuTorch {@link NDArray}: a Java-side data holder (no native handle). */
public class EtNDArray extends NDArrayAdapter {

    private ByteBuffer data;

    EtNDArray(
            NDManager manager,
            NDManager alternativeManager,
            ByteBuffer data,
            Shape shape,
            DataType dataType) {
        super(manager, alternativeManager, shape, dataType, NDManager.nextUid());
        this.data = data;
        manager.attachInternal(uid, this);
    }

    @Override
    public DataType getDataType() {
        if (isClosed) {
            throw new IllegalStateException("EtNDArray has been closed");
        }
        return dataType;
    }

    @Override
    public Shape getShape() {
        if (isClosed) {
            throw new IllegalStateException("EtNDArray has been closed");
        }
        return shape;
    }

    @Override
    public ByteBuffer toByteBuffer(boolean tryDirect) {
        if (isClosed) {
            throw new IllegalStateException("EtNDArray has been closed");
        }
        return data.duplicate().order(ByteOrder.nativeOrder()).rewind();
    }

    @Override
    public void intern(NDArray replaced) {
        throw new UnsupportedOperationException("ExecuTorch NDArray does not support intern");
    }

    @Override
    public void detach() {
        manager.detachInternal(getUid());
        manager = EtNDManager.getSystemManager();
    }

    @Override
    public void close() {
        data = null;
        super.close();
    }
}
