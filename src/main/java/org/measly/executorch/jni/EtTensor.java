package org.measly.executorch.jni;

import java.nio.ByteBuffer;

/** A tensor crossing the JNI boundary: raw bytes + ExecuTorch ScalarType code + shape. */
public final class EtTensor {
    public final long[] shape;
    public final int scalarType;  // ExecuTorch ScalarType int code
    public final ByteBuffer data; // input: DIRECT (zero-copy); output: heap (single-copy)

    public EtTensor(long[] shape, int scalarType, ByteBuffer data) {
        this.shape = shape;
        this.scalarType = scalarType;
        this.data = data;
    }
}
