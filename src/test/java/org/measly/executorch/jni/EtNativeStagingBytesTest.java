package org.measly.executorch.jni;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.measly.executorch.TestSupport;
import org.junit.jupiter.api.Test;

class EtNativeStagingBytesTest {

    @Test
    void allPlannedModelStagesNothing() {
        TestSupport.assumeNativeAvailable();
        long handle = EtNative.loadModule(TestSupport.addPtePath(), -1, false);
        try {
            // 0 is the correct answer for a memory-planned model, not a failed measurement.
            assertEquals(0L, EtNative.stagingBytes(handle));
        } finally {
            EtNative.destroy(handle);
        }
    }

    @Test
    void unplannedModelReportsSlotBytes() {
        TestSupport.assumeUnplannedModelAvailable();
        long handle = EtNative.loadModule(TestSupport.addUnplannedPtePath(), -1, false);
        try {
            // Two f32 inputs, each slot padded and rounded to 64 bytes: 192 each, 384 total.
            assertEquals(384L, EtNative.stagingBytes(handle));
        } finally {
            EtNative.destroy(handle);
        }
    }

    @Test
    void stagingBytesIsStableAcrossMetadataQueries() {
        TestSupport.assumeUnplannedModelAvailable();
        long handle = EtNative.loadModule(TestSupport.addUnplannedPtePath(), -1, false);
        try {
            long first = EtNative.stagingBytes(handle);
            EtNative.methodMeta(handle);
            assertEquals(first, EtNative.stagingBytes(handle));
            assertTrue(first > 0);
        } finally {
            EtNative.destroy(handle);
        }
    }
}
