package org.measly.executorch;

import org.junit.jupiter.api.Assumptions;

/** Helpers for tests that require the native library. */
public final class TestSupport {

    private TestSupport() {}

    /** Skips the test (assumption) if libexecutorch_djl.so cannot be loaded. */
    public static void assumeNativeAvailable() {
        try {
            Class.forName("org.measly.executorch.jni.EtNative");
        } catch (Throwable t) { // UnsatisfiedLinkError, ExceptionInInitializerError, etc.
            Assumptions.abort("Native library not available: " + t.getMessage());
        }
    }

    /** Absolute path to the spike test model. */
    public static String addPtePath() {
        return new java.io.File("native/spike/add.pte").getAbsolutePath();
    }
}
