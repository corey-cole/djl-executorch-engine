package org.measly.executorch.engine;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

/** ExecuTorch {@link NDManager}: minimal tensor factory. */
public class EtNDManager extends BaseNDManager {

    private static final EtNDManager SYSTEM_MANAGER = new SystemManager();

    private EtNDManager(NDManager parent, Device device) {
        super(parent, device);
    }

    static EtNDManager getSystemManager() {
        return SYSTEM_MANAGER;
    }

    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    @Override
    public EtNDArray from(NDArray array) {
        if (array == null || array instanceof EtNDArray) {
            return (EtNDArray) array;
        }
        EtNDArray result = create(array.toByteBuffer(), array.getShape(), array.getDataType());
        result.setName(array.getName());
        return result;
    }

    @Override
    public EtNDArray create(Buffer data, Shape shape, DataType dataType) {
        if (dataType == DataType.STRING) {
            throw new IllegalArgumentException("ExecuTorch does not support String NDArray");
        }
        int size = Math.toIntExact(shape.size());
        BaseNDManager.validateBuffer(data, dataType, size);
        ByteBuffer bb = allocateDirect(size * dataType.getNumOfBytes());
        copyInto(bb, data, dataType);
        bb.rewind();
        return new EtNDArray(this, alternativeManager, bb, shape, dataType);
    }

    @Override
    public NDManager newSubManager(Device device) {
        EtNDManager manager = new EtNDManager(this, device);
        attachInternal(manager.uid, manager);
        return manager;
    }

    @Override
    public Engine getEngine() {
        return Engine.getEngine(EtEngine.ENGINE_NAME);
    }

    /** Copies a typed input buffer into a native-order byte buffer. */
    private static void copyInto(ByteBuffer dst, Buffer src, DataType dataType) {
        if (src instanceof ByteBuffer) {
            dst.put((ByteBuffer) src);
            return;
        }
        switch (dataType) {
            case FLOAT32:
                dst.asFloatBuffer().put((FloatBuffer) src);
                break;
            case FLOAT64:
                dst.asDoubleBuffer().put((DoubleBuffer) src);
                break;
            case INT32:
                dst.asIntBuffer().put((IntBuffer) src);
                break;
            case INT64:
                dst.asLongBuffer().put((LongBuffer) src);
                break;
            default:
                throw new UnsupportedOperationException("Unsupported dtype: " + dataType);
        }
    }

    /** Root manager of which all others are children. */
    private static final class SystemManager extends EtNDManager
            implements NDManager.SystemNDManager {
        SystemManager() {
            super(null, null);
        }
    }
}
