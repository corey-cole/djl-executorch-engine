package org.measly.executorch.jni;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;
import org.measly.executorch.TestSupport;

/** Null EtTensor elements / null shape fields must throw IllegalArgumentException with a pinned message. */
class EtNativeNullTensorTest {

    @Test
    void forwardWithNullElementThrows() {
        TestSupport.assumeNativeAvailable();
        long handle = EtNative.loadModule(TestSupport.addPtePath(), -1);
        try {
            IllegalArgumentException ex = assertThrows(
                    IllegalArgumentException.class,
                    () -> EtNative.forward(handle, new EtTensor[] {null}));
            assertEquals("EtTensor[0] is null", ex.getMessage());
        } finally {
            EtNative.destroy(handle);
        }
    }

    @Test
    void forwardWithNullShapeThrows() {
        TestSupport.assumeNativeAvailable();
        long handle = EtNative.loadModule(TestSupport.addPtePath(), -1);
        try {
            ByteBuffer direct = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder());
            EtTensor tensor = new EtTensor(null, 6 /*Float*/, direct);
            IllegalArgumentException ex = assertThrows(
                    IllegalArgumentException.class,
                    () -> EtNative.forward(handle, new EtTensor[] {tensor}));
            assertEquals("EtTensor.shape is null", ex.getMessage());
        } finally {
            EtNative.destroy(handle);
        }
    }
}
