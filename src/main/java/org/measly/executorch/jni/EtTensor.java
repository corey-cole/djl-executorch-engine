package org.measly.executorch.jni;

import java.nio.ByteBuffer;

/** A tensor crossing the JNI boundary: raw bytes + ExecuTorch ScalarType code + shape. */
public final class EtTensor {
    /** Tensor dimensions, outermost first. */
    public final long[] shape;

    /** ExecuTorch {@code ScalarType} integer code; see {@code EtDataTypes} for the mapping. */
    public final int scalarType;

    /**
     * The tensor's raw bytes in native order. Inputs are direct buffers, which the native side
     * reads without copying; outputs are heap buffers holding a single copy out of ExecuTorch's
     * arena, because the arena's contents are invalidated by the next forward.
     */
    public final ByteBuffer data;

    /**
     * @param shape tensor dimensions, outermost first
     * @param scalarType ExecuTorch {@code ScalarType} integer code
     * @param data raw bytes in native order; direct for an input, heap for an output
     */
    public EtTensor(long[] shape, int scalarType, ByteBuffer data) {
        this.shape = shape;
        this.scalarType = scalarType;
        this.data = data;
    }
}
