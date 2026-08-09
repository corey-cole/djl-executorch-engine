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

    /** Skips the test if the native library itself is unavailable (no model fixture needed). */
    public static void assumeNativeLibraryAvailable() {
        loadNativeLibrary();
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

    /** Absolute path to the borrowed-input add model (exported with alloc_graph_input=False). */
    public static String addUnplannedPtePath() {
        return new File("native/spike/add_unplanned.pte").getAbsolutePath();
    }

    /**
     * Skips the test (assumption) if the native lib or the unplanned add fixture is unavailable.
     */
    public static void assumeUnplannedModelAvailable() {
        loadNativeLibrary();
        if (!isModelArtifactAvailable("native/spike/add_unplanned.pte")) {
            Assumptions.abort(
                    "Test model native/spike/add_unplanned.pte not found"
                            + " (build it via native/spike/export_add_unplanned.py).");
        }
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

    /** Skips the test (assumption) if the native lib or the large-output fixture is unavailable. */
    public static void assumeMedOutputModelAvailable() {
        loadNativeLibrary();
        if (!isModelArtifactAvailable("native/spike/med_output.pte")) {
            Assumptions.abort(
                    "Test model native/spike/med_output.pte not found"
                            + " (build it via native/spike/export_med_output.py).");
        }
    }

    /** Absolute path to the large-output spike test model. */
    public static String medOutputPtePath() {
        return new File("native/spike/med_output.pte").getAbsolutePath();
    }

    /** Directory holding the stress fixture (.pte + goldens), which are committed together. */
    public static java.nio.file.Path stressModelDir() {
        return java.nio.file.Paths.get("src/test/resources/models/stress");
    }

    /** Path to the committed golden digest file. */
    public static java.nio.file.Path stressGoldenPath() {
        return stressModelDir().resolve("stress_golden.json");
    }

    /**
     * Skips the test (assumption) if the native lib or the stress fixture is unavailable. The .pte
     * and its goldens are committed together on purpose — a regenerated model with stale goldens is
     * a silent wrong-answer bug — so both are checked here.
     */
    public static void assumeStressModelAvailable() {
        loadNativeLibrary();
        if (!isModelArtifactAvailable(stressModelDir().resolve("stress_mlp.pte").toString())
                || !isModelArtifactAvailable(stressGoldenPath().toString())) {
            Assumptions.abort(
                    "Stress fixture not found in "
                            + stressModelDir()
                            + " (build it via tools/scripts/export_stress_model.py).");
        }
    }
}
