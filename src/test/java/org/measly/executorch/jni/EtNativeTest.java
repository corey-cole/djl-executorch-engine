package org.measly.executorch.jni;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.measly.executorch.TestSupport;
import org.junit.jupiter.api.Test;

class EtNativeTest {
    @Test
    void forwardAddsTwoScalars() {
        TestSupport.assumeNativeAvailable();
        long handle = EtNative.loadModule(TestSupport.addPtePath());
        try {
            EtTensor a = new EtTensor(new long[] {1}, new float[] {2f});
            EtTensor b = new EtTensor(new long[] {1}, new float[] {3f});
            EtTensor[] out = EtNative.forward(handle, new EtTensor[] {a, b});
            assertEquals(1, out.length);
            assertArrayEquals(new long[] {1}, out[0].shape);
            assertArrayEquals(new float[] {5f}, out[0].data);
        } finally {
            EtNative.destroy(handle);
        }
    }
}
