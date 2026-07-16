package org.measly.executorch;

import java.io.File;
import org.junit.jupiter.api.Assumptions;

/** Helpers for tests that require the native library. */
public final class TestSupport {

    private TestSupport() {}

    private static void loadNativeLibrary() {
        try {
            Class.forName("org.measly.executorch.jni.EtNative");
        } catch (Throwable t) { // UnsatisfiedLinkError, ExceptionInInitializerError, etc.
            Assumptions.abort("Native library not available: " + t.getMessage());
        }
    }

    private static boolean isModelArtifactAvailable(String path) {
        return new File(path).isFile();
    }

    /** Skips the test (assumption) if the native lib or the test model fixture is unavailable. */
    public static void assumeNativeAvailable() {
        loadNativeLibrary();
        if (!isModelArtifactAvailable("native/spike/add.pte")) {
            Assumptions.abort(
                    "Test model native/spike/add.pte not found"
                            + " (build it via native/spike/export_add.py).");
        }
    }

    /** Absolute path to the spike test model. */
    public static String addPtePath() {
        return new File("native/spike/add.pte").getAbsolutePath();
    }

    /**
     * Skips the test (assumption) if the native lib or the dtypes multi-dtype fixture is
     * unavailable.
     */
    public static void assumeDtypesModelAvailable() {
        loadNativeLibrary();
        if (!isModelArtifactAvailable("native/spike/dtypes.pte")) {
            Assumptions.abort(
                    "Test model native/spike/dtypes.pte not found"
                            + " (build it via native/spike/export_dtypes.py).");
        }
    }

    /**
     * Skips the test (assumption) unless the native lib is loadable AND we are on linux-x86_64.
     * The etnp::lstm custom op ships linux-only (no ETNPExtras.cmake in the Windows tarball), so
     * on any other platform the shim legitimately lacks the op — a skip, not a failure.
     */
    public static void assumeLstmModelAvailable() {
        loadNativeLibrary();
        String os = System.getProperty("os.name").toLowerCase();
        String arch = System.getProperty("os.arch").toLowerCase();
        boolean linuxX64 = os.contains("linux") && (arch.equals("amd64") || arch.equals("x86_64"));
        if (!linuxX64) {
            Assumptions.abort("etnp::lstm op is linux-x86_64 only; skipping on " + os + "/" + arch);
        }
    }
}
