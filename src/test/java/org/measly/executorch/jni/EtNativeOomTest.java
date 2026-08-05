package org.measly.executorch.jni;

import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.measly.executorch.TestSupport;

/**
 * OOM contract for the output-marshalling path: a 512 MiB output must surface as a clean
 * OutOfMemoryError, never an unchecked native crash. Run via the {@code oomTest} Gradle task
 * under {@code -Xmx128m}: the test heap cannot hold the marshalled byte[], so
 * {@code NewByteArray} fails and the JNI code must observe the failure instead of writing into
 * a null array.
 */
@Tag("oom")
class EtNativeOomTest {

    @Test
    void oversizedOutputThrowsOutOfMemory() {
        TestSupport.assumeMedOutputModelAvailable();
        long handle = EtNative.loadModule(TestSupport.medOutputPtePath());
        try {
            ByteBuffer input = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder());
            input.putFloat(0, 1f);
            EtTensor tensor = new EtTensor(new long[] {1}, 6 /*Float*/, input);
            assertThrows(
                    OutOfMemoryError.class,
                    () -> EtNative.forward(handle, new EtTensor[] {tensor}));
        } finally {
            EtNative.destroy(handle);
        }
    }
}
