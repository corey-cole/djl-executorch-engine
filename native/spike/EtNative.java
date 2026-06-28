package org.measly.executorch.jni;

/** Minimal native surface for the desktop ExecuTorch JNI shim (spike). */
public final class EtNative {
    static {
        String lib = System.getenv("EXECUTORCH_LIBRARY_PATH");
        if (lib == null) {
            throw new IllegalStateException("set EXECUTORCH_LIBRARY_PATH to libexecutorch_djl.so");
        }
        System.load(lib);
    }

    private EtNative() {}

    /** Loads a .pte and returns an opaque native handle. */
    public static native long loadModule(String ptePath);

    /** Runs forward(a, b) on a 2-input float model, returns the output as a float[]. */
    public static native float[] forwardFloat(long handle, float[] a, float[] b);

    /** Frees the native module. */
    public static native void destroy(long handle);
}