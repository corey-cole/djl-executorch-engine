package org.measly.executorch.jni;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.measly.executorch.TestSupport;
import org.junit.jupiter.api.Test;

class EtMethodMetaTest {
    @Test
    void readsAddModelMetadata() {
        TestSupport.assumeNativeAvailable();
        long handle = EtNative.loadModule(TestSupport.addPtePath());
        try {
            EtMethodMeta meta = EtNative.methodMeta(handle);
            assertEquals(2, meta.numInputs);
            assertArrayEquals(new int[] {6, 6}, meta.inputScalarTypes); // two float32 inputs
        } finally {
            EtNative.destroy(handle);
        }
    }
}
