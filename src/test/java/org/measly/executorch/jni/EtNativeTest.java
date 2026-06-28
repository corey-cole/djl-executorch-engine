package org.measly.executorch.jni;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.measly.executorch.TestSupport;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;

class EtNativeTest {
    private static EtTensor floatScalar(float v) {
        ByteBuffer b = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder());
        b.putFloat(0, v);
        return new EtTensor(new long[] {1}, 6 /*Float*/, b);
    }

    @Test
    void forwardAddsTwoScalars() {
        TestSupport.assumeNativeAvailable();
        long handle = EtNative.loadModule(TestSupport.addPtePath());
        try {
            EtTensor[] out = EtNative.forward(
                    handle, new EtTensor[] {floatScalar(2f), floatScalar(3f)});
            assertEquals(1, out.length);
            assertArrayEquals(new long[] {1}, out[0].shape);
            assertEquals(6, out[0].scalarType);
            assertEquals(5f, out[0].data.order(ByteOrder.nativeOrder()).getFloat(0), 1e-6);
        } finally {
            EtNative.destroy(handle);
        }
    }
}
