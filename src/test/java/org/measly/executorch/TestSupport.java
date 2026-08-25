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

    /** Skips the test (assumption) if the native library itself is unavailable (no model fixture needed). */
    public static void assumeNativeLibraryAvailable() {
        loadNativeLibrary();
    }

    /** Skips when the OpenVINO bundle jar is not on the classpath. */
    public static void assumeOpenVinoBundleAvailable() {
        assumeNativeLibraryAvailable();
        org.junit.jupiter.api.Assumptions.assumeTrue(
                org.measly.executorch.engine.OpenVinoRuntime.bundleAvailable(),
                "OpenVINO bundle jar not on the classpath");
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

    /**
     * Concatenates {@code t}'s message with each of its causes' messages, so an assertion can
     * explain a wrapped failure (e.g. a DJL load error) instead of showing only the outermost
     * message.
     */
    public static String messageChain(Throwable t) {
        StringBuilder sb = new StringBuilder();
        for (Throwable cur = t; cur != null; cur = cur.getCause()) {
            if (sb.length() > 0) {
                sb.append("Caused by: ");
            }
            String message = cur.getMessage();
            sb.append(message == null ? cur.getClass().getName() : message);
        }
        return sb.toString();
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
     * Skips the test (assumption) unless the native lib is loadable AND the shim links the
     * {@code etnp::lstm} custom op. The op arrives with {@code lib/cmake/ETNPExtras/}, which the
     * Linux runtime tarballs ship and the Windows one does not, so on Windows the shim legitimately
     * lacks the op — a skip, not a failure.
     *
     * <p>Both Linux architectures qualify: the op is whole-archived wherever the tarball ships
     * ETNPExtras, and CMake reports it as {@code etnp extras: LSTM op whole-archived}. Verified on
     * linux-aarch64 at pin v1.4.1-3, where the golden vector matches.
     *
     * <p>This is a platform test standing in for a capability query, which is the wrong shape by
     * this project's own rule — see issue for exposing custom-op registration the way
     * {@code EtNative.backendRegistered} exposes backends (issue #64).
     */
    public static void assumeLstmModelAvailable() {
        loadNativeLibrary();
        String os = System.getProperty("os.name").toLowerCase();
        if (!os.contains("linux")) {
            Assumptions.abort(
                    "etnp::lstm ships with ETNPExtras, which the Windows runtime tarball does not"
                            + " carry; skipping on " + os);
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
