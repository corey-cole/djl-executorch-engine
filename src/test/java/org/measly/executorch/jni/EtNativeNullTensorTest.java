package org.measly.executorch.jni;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import org.measly.executorch.TestSupport;

/**
 * A null buffer element must throw IllegalArgumentException with a pinned message.
 *
 * <p>The forward()/EtTensor[] predecessor of this test also covered a null {@code shape} field on
 * a non-null EtTensor -- that scenario has no equivalent under the struct-of-arrays input layout:
 * there is no per-input shape object to be null, only an always-present slice of the flat shape
 * array (a zero-length slice is a legitimate rank-0 shape, not an error).
 */
class EtNativeNullTensorTest {

    @Test
    void forwardWithNullBufferThrows() {
        TestSupport.assumeNativeAvailable();
        long handle = EtNative.loadModule(TestSupport.addPtePath(), -1, false);
        try {
            FlatInputs in = FlatInputs.of((EtTensor) null);
            IllegalArgumentException ex = assertThrows(
                    IllegalArgumentException.class,
                    () -> EtNative.forward(
                            handle, in.flatShapes, in.shapeOffsets, in.scalarTypes, in.buffers));
            assertEquals("buffers[0] is null", ex.getMessage());
        } finally {
            EtNative.destroy(handle);
        }
    }
}
