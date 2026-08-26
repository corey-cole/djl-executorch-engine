package org.measly.executorch.jni;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;
import org.measly.executorch.TestSupport;

/**
 * An EtTensor whose declared shape/dtype implies more bytes than its direct buffer's real
 * capacity must be rejected with IllegalArgumentException, before the native memcpy that would
 * otherwise read past the end of the JVM-owned buffer. See executorch_djl_jni.cpp's forward():
 * this is the read-side guard, distinct from and complementary to kStagingPadding, which only
 * protects the write side (XNNPACK reading past the end of the engine's own staging slot).
 */
class EtNativeUndersizedBufferTest {

    @Test
    void forwardWithUndersizedBufferThrows() {
        TestSupport.assumeNativeAvailable();
        long handle = EtNative.loadModule(TestSupport.addPtePath(), -1, false);
        try {
            // add.pte's forward(a, b) declares two 4-byte float32 scalars (shape [1]); this buffer
            // is paired with a shape claiming the same [1]-float32 layout but has only 1 real byte.
            ByteBuffer undersized = ByteBuffer.allocateDirect(1).order(ByteOrder.nativeOrder());
            EtTensor bad = new EtTensor(new long[] {1}, 6 /*Float*/, undersized);
            ByteBuffer ok = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder());
            ok.putFloat(0, 3f);
            EtTensor good = new EtTensor(new long[] {1}, 6 /*Float*/, ok);

            IllegalArgumentException ex = assertThrows(
                    IllegalArgumentException.class,
                    () -> EtNative.forward(handle, new EtTensor[] {bad, good}));
            assertTrue(
                    ex.getMessage().contains("EtTensor[0].data")
                            && ex.getMessage().contains("capacity 1")
                            && ex.getMessage().contains("implies 4 bytes"),
                    "unexpected message: " + ex.getMessage());
        } finally {
            EtNative.destroy(handle);
        }
    }
}
