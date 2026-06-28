package org.measly.executorch.jni;

import org.measly.executorch.engine.LibUtils;

/** JNI surface to the ExecuTorch native library. Loads the .so on class init. */
public final class EtNative {

    static {
        LibUtils.loadLibrary();
    }

    private EtNative() {}

    public static native long loadModule(String ptePath);

    public static native EtTensor[] forward(long handle, EtTensor[] inputs);

    public static native void destroy(long handle);
}
