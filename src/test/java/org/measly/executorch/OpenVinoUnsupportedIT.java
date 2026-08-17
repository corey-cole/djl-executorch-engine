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

/**
 * The inverse of {@link OpenVinoModelIT}: this runs where the delegate is ABSENT, so both matrix
 * legs assert something real instead of one of them merely skipping.
 */
@Tag("openvino-unsupported")
class OpenVinoUnsupportedIT {

    private static final Path DIR = Paths.get("src/test/resources/models/openvino");

    @Test
    void anOpenVinoModelOnAPlatformWithoutTheDelegateNamesTheRealProblem() {
        TestSupport.assumeNativeAvailable();
        Assumptions.assumeFalse(
                OpenVinoRuntime.bundleAvailable(), "this asserts the UNSUPPORTED platform's message");

        Exception e = assertThrows(Exception.class, () -> {
            try (Model model = Model.newInstance("openvino_tiny", "ExecuTorch")) {
                model.load(DIR, "openvino_tiny");
            }
        });
        String message = String.valueOf(e.getMessage()) + String.valueOf(e.getCause());
        // Without the guard this falls through the generic load-failure path and reports a corrupt
        // or version-mismatched model -- which actively misdirects. The .pte is fine; the platform
        // cannot run it.
        assertTrue(message.contains("OpenvinoBackend"), "must name the backend: " + message);
        assertTrue(
                message.toLowerCase().contains("re-export") || message.contains("does not provide"),
                "must say what to do about it: " + message);
    }
}
