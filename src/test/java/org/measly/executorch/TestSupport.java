package org.measly.executorch;

import org.junit.jupiter.api.Assumptions;

/** Helpers for tests that require the native library. */
public final class TestSupport {

    private TestSupport() {}

    /** Skips the test (assumption) if the native lib or the test model fixture is unavailable. */
    public static void assumeNativeAvailable() {
        try {
            Class.forName("org.measly.executorch.jni.EtNative");
        } catch (Throwable t) { // UnsatisfiedLinkError, ExceptionInInitializerError, etc.
            Assumptions.abort("Native library not available: " + t.getMessage());
        }
        if (!new java.io.File("native/spike/add.pte").isFile()) {
            Assumptions.abort(
                    "Test model native/spike/add.pte not found"
                            + " (build it via native/spike/export_add.py).");
        }
    }

    /** Absolute path to the spike test model. */
    public static String addPtePath() {
        return new java.io.File("native/spike/add.pte").getAbsolutePath();
    }

    /**
     * Skips the test (assumption) if the native lib or the dtypes multi-dtype fixture is
     * unavailable.
     */
    public static void assumeDtypesModelAvailable() {
        try {
            Class.forName("org.measly.executorch.jni.EtNative");
        } catch (Throwable t) {
            Assumptions.abort("Native library not available: " + t.getMessage());
        }
        if (!new java.io.File("native/spike/dtypes.pte").isFile()) {
            Assumptions.abort(
                    "Test model native/spike/dtypes.pte not found"
                            + " (build it via native/spike/export_dtypes.py).");
        }
    }
}
