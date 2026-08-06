package org.measly.executorch.engine;

import java.lang.ref.Cleaner;
import java.nio.ByteBuffer;
import org.measly.executorch.jni.EtNative;

/**
 * Registers a {@link Cleaner} action that frees JNI-allocated {@code forward()} output buffers.
 *
 * <p>JNI-allocated buffers are not counted against {@code -XX:MaxDirectMemorySize} (the W6 OOM
 * mechanism), so the leak signal for outputs is the native alive counter
 * ({@link EtNative#aliveOutputBuffers()}), asserted by {@code LeakStressTest} — not memory
 * pressure. Registration must capture ONLY the address primitive: capturing {@code buffer} (or
 * any holder of it) keeps the buffer strongly reachable and the Cleaner would never fire.
 * {@code 0} is a no-op on the native side, so a mis-registration can never double-free.
 */
final class EtOutputBuffers {
    private static final Cleaner CLEANER = Cleaner.create();

    private EtOutputBuffers() {}

    /**
     * Frees {@code buffer}'s native backing block once the buffer is unreachable. Captures ONLY
     * the address primitive: capturing {@code buffer} (or any holder of it) keeps it strongly
     * reachable and the Cleaner would never fire. 0 is a no-op on the native side, so a
     * mis-registration can never double-free.
     */
    static void register(ByteBuffer buffer) {
        long address = EtNative.bufferAddress(buffer);
        if (address == 0) {
            return; // non-direct: not engine-allocated, nothing to free
        }
        CLEANER.register(buffer, () -> EtNative.freeOutputBuffer(address));
    }
}
