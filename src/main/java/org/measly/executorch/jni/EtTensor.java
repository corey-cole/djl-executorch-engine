package org.measly.executorch.jni;

import java.nio.ByteBuffer;

/** A tensor crossing the JNI boundary: raw bytes + ExecuTorch ScalarType code + shape. */
public final class EtTensor {
    public final long[] shape;
    public final int scalarType;  // ExecuTorch ScalarType int code

    /**
     * Input: DIRECT (zero-copy), borrowed for the call.
     *
     * <p>Output: DIRECT, JNI-allocated, freed by a {@link java.lang.ref.Cleaner} once this buffer
     * is unreachable — its memory dies with the ByteBuffer, not the JVM heap. The engine's own
     * consumers (EtNDArray holds the original buffer) are safe by construction; holding a
     * {@code toByteBuffer()} duplicate past the original's collection is now UB where it used to
     * be safe (the same contract IREE shipped).
     */
    public final ByteBuffer data;

    public EtTensor(long[] shape, int scalarType, ByteBuffer data) {
        this.shape = shape;
        this.scalarType = scalarType;
        this.data = data;
    }
}
