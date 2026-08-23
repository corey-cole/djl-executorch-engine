package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.djl.Model;
import ai.djl.engine.EngineException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.measly.executorch.TestSupport;
import org.measly.executorch.jni.EtNative;

@Tag("openvino")
// Deterministic order is load-bearing here: the openvinoTest JVM is forked per CLASS (forkEvery=1),
// so the first test that extracts/configures sets process-global state the later tests observe.
// anOperatorSetLibPathIsHonouredUntouched asserts resolvedLibPath() is still null, which can only
// hold while it runs before any test that legitimately extracts the bundle.
@TestMethodOrder(MethodOrderer.MethodName.class)
class OpenVinoRuntimeTest {

    @Test
    void extractsTheBundleToAFlatDirectoryAndResolvesTheDeclaredLibrary() throws Exception {
        TestSupport.assumeOpenVinoBundleAvailable();

        Path dir = OpenVinoRuntime.ensureExtracted();
        assertNotNull(dir);
        assertTrue(Files.isDirectory(dir), "bundle must extract to a directory: " + dir);

        // Flat, not nested: RPATH=$ORIGIN only resolves siblings.
        try (var entries = Files.list(dir)) {
            assertTrue(
                    entries.noneMatch(Files::isDirectory),
                    "the library directory must be flat; $ORIGIN does not search subdirectories");
        }

        String lib = OpenVinoRuntime.resolvedLibPath();
        assertNotNull(lib);
        assertTrue(Files.isRegularFile(Paths.get(lib)), "must name a file, not a directory: " + lib);
        // The library the BUNDLE declared, not one this test reconstructs: that is the whole point
        // of c_library, and it is what makes this assertion identical on Windows.
        java.util.Properties man = new java.util.Properties();
        try (var is = OpenVinoRuntime.class.getResourceAsStream(
                "/native/" + LibUtils.platform() + "/openvino/MANIFEST")) {
            man.load(new java.io.InputStreamReader(is, java.nio.charset.StandardCharsets.UTF_8));
        }
        assertEquals(dir.resolve(man.getProperty("c_library")).toAbsolutePath().toString(), lib);
        assertFalse(Files.isSymbolicLink(Paths.get(lib)), "must never resolve through a symlink");
    }

    @Test
    void repeatedExtractionIsIdempotentAndReturnsTheSameDirectory() throws Exception {
        TestSupport.assumeOpenVinoBundleAvailable();
        assertEquals(OpenVinoRuntime.ensureExtracted(), OpenVinoRuntime.ensureExtracted());
    }

    @Test
    void probingForABackendDoesNotBurnTheDelegatesOneShotDlopen() throws Exception {
        TestSupport.assumeOpenVinoBundleAvailable();

        Path pte = Paths.get("src/test/resources/models/openvino/openvino_tiny.pte");

        // Probe FIRST, with OPENVINO_LIB_PATH deliberately unresolved. If pteUsesBackend loaded the
        // method rather than just the program, this would run delegate init unconfigured -- and the
        // delegate's dlopen is std::call_once with no retry, so the load below would then fail
        // forever in this JVM no matter how correctly we configure afterwards.
        assertTrue(EtNative.pteUsesBackend(pte.toString(), OpenVinoRuntime.BACKEND));

        // Now configure and load for real. Success here proves the probe consumed nothing.
        try (Model model = Model.newInstance("openvino_tiny", "ExecuTorch")) {
            model.load(pte.getParent(), "openvino_tiny");
        }
    }

    @Test
    void anOperatorSetLibPathIsHonouredUntouched() throws Exception {
        // Cannot be asserted by mutating this JVM's environment -- Java cannot -- so this asserts
        // the decision function instead: given a non-empty existing value, resolution must return
        // it unchanged and must not extract anything.
        String existing = System.getenv("OPENVINO_LIB_PATH");
        Assumptions.assumeTrue(
                existing == null || existing.isEmpty(),
                "this asserts the default path; an inherited OPENVINO_LIB_PATH would mask it");
        // With no override set, a non-OpenVINO model must leave configuration untouched: the
        // bundle is not extracted and no lib path is resolved for a model that never needs one.
        OpenVinoRuntime.ensureReady(Paths.get(TestSupport.addPtePath()));
        assertNull(
                OpenVinoRuntime.resolvedLibPath(),
                "a non-OpenVINO model must not trigger bundle resolution");
    }

    @Test
    void anUnusableLibPathOverrideIsRejectedWithTheValueThatCausedIt() throws Exception {
        // Tested as a pure function because a JVM cannot set its own environment. The end-to-end
        // env path is covered natively in et_runtime_test.cpp, which can call setenv in-process.
        Path realFile = Files.createTempFile("not-a-library", ".so");
        Path dir = Files.createTempDirectory("openvino-dir");
        try {
            EngineException nonexistent = assertThrows(
                    EngineException.class, () -> OpenVinoRuntime.validateOverride("XXX"));
            assertTrue(
                    nonexistent.getMessage().contains("XXX"),
                    "the message must quote the offending value: " + nonexistent.getMessage());

            EngineException directory = assertThrows(
                    EngineException.class,
                    () -> OpenVinoRuntime.validateOverride(dir.toString()));
            assertTrue(
                    directory.getMessage().contains("directory"),
                    "a directory must be called out by name, because the error the delegate would "
                            + "otherwise give mentions LD_LIBRARY_PATH and misleads: "
                            + directory.getMessage());

            // Any readable regular file passes. Validation deliberately stops at "could this be
            // dlopen'd at all" -- proving it is really OpenVINO would mean loading it, which is
            // the once-only operation this check exists to protect.
            assertDoesNotThrow(() -> OpenVinoRuntime.validateOverride(realFile.toString()));
        } finally {
            Files.deleteIfExists(realFile);
            Files.deleteIfExists(dir);
        }
    }

    @Test
    void reportsBundleAvailabilityFromTheClasspathRatherThanThePlatform() {
        TestSupport.assumeNativeLibraryAvailable();
        // A boolean either way is correct -- what must NOT happen is a throw. This runs on every
        // platform, including ones with no bundle, because that is the case whose error path
        // matters most.
        assertDoesNotThrow(OpenVinoRuntime::bundleAvailable);
    }

    @Test
    void reportsTheInferencePrecisionOpenVinoWillUseOnThisHost() throws Exception {
        TestSupport.assumeOpenVinoBundleAvailable();
        OpenVinoRuntime.ensureExtracted();

        String precision = EtEngine.openVinoInferencePrecision();
        // The VALUE is not asserted -- it is a property of the CPU this happens to run on, and
        // asserting it would assert the hardware. What is asserted is that the read succeeded, i.e.
        // the vendored C API loaded and answered. "unavailable" means it did not.
        assertNotEquals("unavailable", precision, "the C API should have loaded and answered");
        assertTrue(
                precision.equals("f32") || precision.equals("bf16") || precision.equals("f16"),
                "unexpected precision: " + precision);
    }

    @Test
    void theNoDelegateErrorDirectsTheUserToReExport() {
        // The condition guarding this message (!backendRegistered) is false on every SHIPPED
        // platform: all three runtime tarballs carry the delegate. It stays reachable through the
        // ET_INSTALL escape hatch, which links a caller-supplied runtime tree that may have been
        // built without OpenVINO -- so the message must stay correct, and a test gated on the
        // condition would skip everywhere and prove nothing. The message is the asset; test it.
        String msg = OpenVinoRuntime.noDelegateMessage().getMessage();
        assertTrue(msg.contains(OpenVinoRuntime.BACKEND), "must name the backend: " + msg);
        assertTrue(msg.contains(LibUtils.platform()), "must name the platform: " + msg);
        // The remedy is the whole point of keeping this distinct from the no-runtime error: one
        // says re-export the model, the other says add a runtime artifact. Asserting the remedy is
        // what stops the two from converging.
        assertTrue(msg.contains("Re-export"), "must direct the user to re-export: " + msg);
        assertFalse(
                msg.contains("-openvino artifact"),
                "must not offer the runtime artifact; no runtime can help here: " + msg);
    }
}
