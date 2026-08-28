package org.measly.executorch.jni;

import java.nio.ByteBuffer;

/**
 * Test helper: builds {@link EtNative#forward}'s struct-of-arrays input layout from a small list
 * of {@link EtTensor}s, so tests can keep writing {@code floatScalar(2f)}-style fixtures instead
 * of hand-rolling flat arrays. A {@code null} element becomes a {@code null} buffer at that
 * position -- the flat layout's equivalent of a {@code null} {@code EtTensor} array element.
 */
final class FlatInputs {

    final long[] flatShapes;
    final int[] shapeOffsets;
    final int[] scalarTypes;
    final ByteBuffer[] buffers;

    private FlatInputs(
            long[] flatShapes, int[] shapeOffsets, int[] scalarTypes, ByteBuffer[] buffers) {
        this.flatShapes = flatShapes;
        this.shapeOffsets = shapeOffsets;
        this.scalarTypes = scalarTypes;
        this.buffers = buffers;
    }

    static FlatInputs of(EtTensor... tensors) {
        int n = tensors.length;
        int[] offsets = new int[n + 1];
        int total = 0;
        for (int i = 0; i < n; i++) {
            total += tensors[i] == null ? 0 : tensors[i].shape.length;
            offsets[i + 1] = total;
        }
        long[] flat = new long[total];
        int[] types = new int[n];
        ByteBuffer[] bufs = new ByteBuffer[n];
        for (int i = 0; i < n; i++) {
            EtTensor t = tensors[i];
            if (t == null) {
                continue;
            }
            System.arraycopy(t.shape, 0, flat, offsets[i], t.shape.length);
            types[i] = t.scalarType;
            bufs[i] = t.data;
        }
        return new FlatInputs(flat, offsets, types, bufs);
    }
}
