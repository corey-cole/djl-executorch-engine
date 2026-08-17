package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.djl.Model;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.measly.executorch.engine.OpenVinoRuntime;
import org.measly.executorch.jni.EtNative;

/**
 * The inverse of {@link OpenVinoModelIT}: this runs wherever the OpenVINO runtime is NOT resolvable,
 * so both matrix legs assert something real instead of one of them merely skipping.
 *
 * <p>Its earlier name said "unsupported platform", which was built on a premise that turned out to
 * be false. The claim inherited from upstream — that the OpenVINO delegate ships on
 * {@code linux-x86_64} only — is wrong: {@code libopenvino_backend.a} is in the
 * {@code linux-aarch64} tarball too, and CI logs {@code OpenVINO delegate: linked} on that leg.
 * What is {@code linux-x86_64}-only is the OpenVINO <b>runtime bundle</b>. Windows is the only
 * platform where the delegate itself is absent.
 *
 * <p>So there are two distinct failures with opposite remedies, and this asserts whichever one the
 * running platform actually has:
 *
 * <ul>
 *   <li><b>Delegate absent</b> (Windows) — the model cannot run here at all; re-export is the fix.
 *   <li><b>Delegate present, no runtime</b> (linux-aarch64) — the model runs fine once a runtime is
 *       supplied; pointing the user at re-export would be actively wrong.
 * </ul>
 */
@Tag("openvino-unsupported")
class OpenVinoMissingRuntimeIT {

    private static final Path DIR = Paths.get("src/test/resources/models/openvino");

    @Test
    void anOpenVinoModelWithNoResolvableRuntimeNamesTheRealProblem() {
        TestSupport.assumeNativeAvailable();
        Assumptions.assumeFalse(
                OpenVinoRuntime.bundleAvailable(),
                "this asserts the no-resolvable-runtime message; the bundle is present here");

        Exception e = assertThrows(Exception.class, () -> {
            try (Model model = Model.newInstance("openvino_tiny", "ExecuTorch")) {
                model.load(DIR, "openvino_tiny");
            }
        });
        String message = String.valueOf(e.getMessage()) + String.valueOf(e.getCause());

        // Both branches must name the backend. Without any guard this falls through the generic
        // load-failure path and reports a corrupt or version-mismatched model, which actively
        // misdirects: the .pte is fine, the environment cannot run it.
        assertTrue(message.contains("OpenvinoBackend"), "must name the backend: " + message);

        if (EtNative.backendRegistered("OpenvinoBackend")) {
            // Delegate linked, runtime missing. The remedy is to supply a runtime, so the message
            // must NOT tell the user to re-export -- that would send them to rebuild a model that
            // is already correct.
            assertTrue(
                    message.contains("no OpenVINO runtime is available"),
                    "must say the runtime is what is missing: " + message);
            assertTrue(
                    message.contains("OPENVINO_LIB_PATH"),
                    "must name the escape hatch that works without a published bundle: " + message);
            assertTrue(
                    !message.toLowerCase().contains("re-export"),
                    "must NOT advise re-export when the delegate is present: " + message);
        } else {
            // No delegate at all. Re-export really is the only fix, and promising that a runtime
            // would help would be a lie.
            assertTrue(
                    message.contains("does not provide"),
                    "must say this build lacks the delegate: " + message);
            assertTrue(
                    message.toLowerCase().contains("re-export"),
                    "must say what to do about it: " + message);
        }
    }
}
