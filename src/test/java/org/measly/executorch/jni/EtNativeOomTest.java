package org.measly.executorch.jni;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.measly.executorch.TestSupport;

/**
 * Output-marshalling contract under a constrained heap. Repurposed for W6: the 512 MiB output is
 * now marshalled into a JNI-allocated block, so the forward no longer allocates on the
 * {@code -Xmx128m} test heap at all — the test proves the output path is heap-independent and
 * exercises the alive counter (allocate → observe 1 → free → observe 0). Run via the
 * {@code oomTest} Gradle task under {@code -Xmx128m}.
 */
@Tag("oom")
class EtNativeOomTest {

    @Test
    void oversizedOutputAllocatesOffHeap() {
        TestSupport.assumeMedOutputModelAvailable();
        long handle = EtNative.loadModule(TestSupport.medOutputPtePath());
        try {
            ByteBuffer input = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder());
            input.putFloat(0, 1f);
            EtTensor tensor = new EtTensor(new long[] {1}, 6 /*Float*/, input);
            EtTensor[] out = EtNative.forward(handle, new EtTensor[] {tensor});
            assertNotNull(out);
            assertEquals(1, out.length);
            assertEquals(1, EtNative.aliveOutputBuffers());
            EtNative.freeOutputBuffer(EtNative.bufferAddress(out[0].data));
            assertEquals(0, EtNative.aliveOutputBuffers());
        } finally {
            EtNative.destroy(handle);
        }
    }
}
