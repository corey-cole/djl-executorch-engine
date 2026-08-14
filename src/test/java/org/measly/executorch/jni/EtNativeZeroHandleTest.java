package org.measly.executorch.jni;

import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import org.junit.jupiter.api.Test;
import org.measly.executorch.TestSupport;

/**
 * The JNI backstop for issue #36: a zero handle must be rejected at the boundary rather than
 * dereferenced. Every entry point that takes a handle is covered, not just {@code forward} — the
 * Java layer guards the paths it owns, but {@link EtNative} is public and these are the last line
 * before an unchecked {@code reinterpret_cast}. Without the check each of these calls is a null
 * dereference in native code, i.e. a JVM crash rather than a test failure.
 */
class EtNativeZeroHandleTest {

    @Test
    void forwardWithZeroHandleThrowsIllegalState() {
        TestSupport.assumeNativeLibraryAvailable();
        ByteBuffer data = ByteBuffer.allocateDirect(4);
        EtTensor tensor = new EtTensor(new long[] {1}, 6 /*Float*/, data);
        assertThrows(
                IllegalStateException.class,
                () -> EtNative.forward(0L, new EtTensor[] {tensor}));
    }

    @Test
    void methodMetaWithZeroHandleThrowsIllegalState() {
        TestSupport.assumeNativeLibraryAvailable();
        assertThrows(IllegalStateException.class, () -> EtNative.methodMeta(0L));
    }

    @Test
    void stagingBytesWithZeroHandleThrowsIllegalState() {
        TestSupport.assumeNativeLibraryAvailable();
        assertThrows(IllegalStateException.class, () -> EtNative.stagingBytes(0L));
    }

    @Test
    void destroyWithZeroHandleIsANoOp() {
        TestSupport.assumeNativeLibraryAvailable();
        // delete on a null pointer is already well-defined; pinned so the guard added to the other
        // entry points is not "helpfully" copied here and turned into a throw.
        EtNative.destroy(0L);
    }
}
