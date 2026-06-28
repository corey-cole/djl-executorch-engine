package org.measly.executorch.jni;

/** A float32 tensor crossing the JNI boundary. Phase 2 generalizes to ByteBuffer + dtype. */
public final class EtTensor {
    public final long[] shape;
    public final float[] data; // row-major, length == product(shape)

    public EtTensor(long[] shape, float[] data) {
        this.shape = shape;
        this.data = data;
    }
}
