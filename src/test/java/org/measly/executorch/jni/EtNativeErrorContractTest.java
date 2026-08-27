package org.measly.executorch.jni;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import org.junit.jupiter.api.Test;
import org.measly.executorch.TestSupport;

/**
 * Characterization pins for the error-reporting paths issue #12 hardens: FindClass results must
 * never reach ThrowNew unchecked. No deterministic red exists (java.lang classes always load in a
 * healthy JVM; the real failure mode is only reachable from native code), so these pin the
 * protected behavior; the fix is defense-in-depth on the path every other fix relies on.
 */
class EtNativeErrorContractTest {

    @Test
    void loadModuleWithMissingFileThrowsRuntimeException() {
        TestSupport.assumeNativeLibraryAvailable();
        assertThrows(
                RuntimeException.class, () -> EtNative.loadModule("/nonexistent/foo.pte", -1, false));
    }

    @Test
    void forwardWithHeapInputBufferThrowsIllegalArgumentException() {
        TestSupport.assumeNativeLibraryAvailable();
        long handle = EtNative.loadModule(TestSupport.addPtePath(), -1, false);
        try {
            // Heap ByteBuffer: GetDirectBufferAddress returns null, exercising the exact
            // forward() data-site check that issue #12 fixes.
            EtTensor tensor = new EtTensor(new long[] {1}, 6 /*Float*/, ByteBuffer.allocate(4));
            FlatInputs in = FlatInputs.of(tensor);
            IllegalArgumentException ex = assertThrows(
                    IllegalArgumentException.class,
                    () -> EtNative.forward(
                            handle, in.flatShapes, in.shapeOffsets, in.scalarTypes, in.buffers));
            assertEquals("buffers[0] must be a direct ByteBuffer", ex.getMessage());
        } finally {
            EtNative.destroy(handle);
        }
    }

    @Test
    void etDumpOnUnprofiledModelReturnsNonNullEmptyArray() {
        TestSupport.assumeNativeLibraryAvailable();
        long handle = EtNative.loadModule(TestSupport.addPtePath(), -1, false);
        try {
            // Not profiled: no dump is an answer, not an error -- pins the native empty-array
            // (never null) contract for an unprofiled model.
            byte[] dump = EtNative.etDump(handle);
            assertNotNull(dump);
            assertEquals(0, dump.length);
        } finally {
            EtNative.destroy(handle);
        }
    }

    @Test
    void devtoolsAvailableIsCallable() {
        TestSupport.assumeNativeLibraryAvailable();
        // The value is a build property, so nothing is asserted: the call exercises the new native
        // signature under -Xcheck:jni, where a mismatch surfaces as a warning, not a test failure.
        EtNative.devtoolsAvailable();
    }
}
