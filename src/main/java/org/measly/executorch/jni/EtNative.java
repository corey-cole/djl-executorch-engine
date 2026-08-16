package org.measly.executorch.jni;

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

    /**
     * Loads a .pte.
     *
     * @param ptePath path to the model file
     * @param workspaceSharingMode XNNPACK workspace sharing for this model: 0=Disabled, 1=PerModel,
     *     2=Global, -1 to send no spec and leave the runtime default in force
     * @return the native handle
     */
    public static native long loadModule(String ptePath, int workspaceSharingMode);

    /**
     * Reads the static I/O metadata ExecuTorch captured for {@code forward} at load time.
     *
     * @param handle the native handle
     * @return per-input scalar types, memory-planned flags, and the planned arena size
     * @throws IllegalStateException if {@code handle} is 0, i.e. the model has been closed
     */
    public static native EtMethodMeta methodMeta(long handle);

    /**
     * Total bytes currently held by the runtime's input staging slots.
     *
     * <p>Returns 0 when every input is memory-planned (the export default) — planned inputs are
     * never staged. Callers must not pass a destroyed handle; doing so is a use-after-free.
     *
     * @param handle the native handle
     * @return staging bytes, or 0 for an all-planned model
     * @throws IllegalStateException if {@code handle} is 0, i.e. the model has been closed
     */
    public static native long stagingBytes(long handle);

    /**
     * Runs the model's {@code forward} method on {@code inputs}.
     *
     * @param handle the native handle
     * @param inputs one {@link EtTensor} per declared input, in position order
     * @return one {@link EtTensor} per model output, each a heap-buffer single copy out of
     *     ExecuTorch's arena
     * @throws IllegalStateException if {@code handle} is 0, i.e. the model has been closed
     */
    public static native EtTensor[] forward(long handle, EtTensor[] inputs);

    /**
     * Releases the native module and its arena. Idempotent on the Java side is not guaranteed;
     * callers must not pass this handle to any other native method afterward.
     *
     * @param handle the native handle
     */
    public static native void destroy(long handle);

    /**
     * Sets the process-global intra-op (XNNPACK) thread pool size.
     *
     * @param n requested thread count; must be at least 1
     * @return the count in effect after the attempt, which may differ from {@code n}
     */
    public static native int setIntraOpThreads(int n);

    /** @return the current intra-op pool size as reported by the native pool */
    public static native int intraOpThreads();

    /**
     * Reads the XNNPACK delegate's memory arena size, for native-memory accounting.
     *
     * <p>Three properties a caller must not paper over. The figure is <b>process-wide</b> — the
     * pinned runtime sets {@code EXECUTORCH_XNNPACK_SHARED_WORKSPACE=ON}, so every delegate shares
     * one arena and this is already the total; never sum it per model. It is a <b>high-water
     * mark</b> including allocator alignment padding, so it is the peak rather than the live
     * footprint. And it stays {@code 0} until an XNNPACK-delegated method actually <b>executes</b>
     * — loading one is not enough, and neither is a delegated model that allocates nothing, which
     * is the case for an elementwise add or a Linear.
     *
     * <p>The backing option is a vendored patch in the pinned distribution, not upstream
     * ExecuTorch. Against a stock runtime the key does not resolve and this reports {@code -1}.
     *
     * @return arena size in bytes, or {@code -1} when unavailable; {@code 0} is a real answer
     */
    public static native long xnnpackWorkspaceBytes();

    /**
     * Reports whether a {@code .pte}'s {@code forward} method is delegated to a given backend,
     * reading metadata only.
     *
     * <p>Deliberately separate from {@link #loadModule}: that call constructs the native runtime,
     * whose constructor calls {@code load_forward()} — which for a delegated model is delegate
     * init. OpenVINO's delegate init resolves its C API with a {@code dlopen} under
     * {@code std::call_once} and never retries, so a caller that must configure something first has
     * to ask before {@code loadModule}, not after.
     *
     * @param ptePath absolute path to the model file
     * @param backend backend id, e.g. {@code OpenvinoBackend} (lowercase {@code v})
     * @return true if {@code forward} is delegated to that backend
     */
    public static native boolean pteUsesBackend(String ptePath, String backend);

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
