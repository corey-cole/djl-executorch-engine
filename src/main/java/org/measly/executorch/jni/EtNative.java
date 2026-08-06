package org.measly.executorch.jni;

import java.nio.ByteBuffer;
import org.measly.executorch.engine.LibUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** JNI surface to the ExecuTorch native library. Loads the .so on class init. */
public final class EtNative {

    /** Sink for ExecuTorch's native ET_LOG output, forwarded by the JNI PAL bridge. */
    private static final Logger NATIVE_LOG = LoggerFactory.getLogger("org.measly.executorch.native");

    static {
        LibUtils.loadLibrary();
    }

    private EtNative() {}

    public static native long loadModule(String ptePath);

    public static native EtMethodMeta methodMeta(long handle);

    public static native EtTensor[] forward(long handle, EtTensor[] inputs);

    public static native void destroy(long handle);

    /**
     * Returns the native address of a direct buffer's backing memory, or 0 for a non-direct
     * buffer (JNI spec; never throws). Used to wire engine-allocated output buffers into the
     * Cleaner.
     */
    public static native long bufferAddress(ByteBuffer buffer);

    /**
     * Frees a JNI-allocated output buffer by address. {@code 0} is a no-op (idempotent by
     * contract, so a mis-registration can never double-free). Called exactly once per buffer,
     * from the Cleaner registered by {@code EtOutputBuffers}; plain Java code never calls it.
     */
    public static native void freeOutputBuffer(long address);

    /** Live JNI-allocated output buffers (leak probe for tests — see LeakStressTest). */
    public static native long aliveOutputBuffers();

    /**
     * Called from native code (the ExecuTorch PAL sink) to route an ET_LOG message to slf4j.
     * Level codes match {@code measly::et::Slf4jLevel}: 0=debug, 1=info, 2=warn, 3=error
     * (unknown → info).
     */
    static void nativeLog(int level, String message) {
        switch (level) {
            case 0:
                NATIVE_LOG.debug(message);
                break;
            case 2:
                NATIVE_LOG.warn(message);
                break;
            case 3:
                NATIVE_LOG.error(message);
                break;
            case 1:
            default:
                NATIVE_LOG.info(message);
                break;
        }
    }
}
