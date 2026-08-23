package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.measly.executorch.TestSupport;

/**
 * Reads the inference precision <b>cold</b> — in a JVM where nothing has loaded an OpenVINO model.
 *
 * <p>Alone in its own class deliberately. {@code openvinoTest} forks per class, so this is the only
 * way to get a process in which the delegate has never run: any test that loads a delegated model
 * leaves the bundle's libraries resident, and a later probe in that process then succeeds by
 * bumping a refcount on an already-loaded graph rather than by resolving one. A cold probe is the
 * only shape that measures what the probe's own loader can do.
 *
 * <p>That is what makes this the behavioural gate on the Windows loader flags. Windows has no
 * {@code $ORIGIN}, so a plain {@code LoadLibrary}, or {@code LoadLibraryExW} missing either search
 * flag, cannot resolve the bundle's siblings and reports {@code "unavailable"} here — while still
 * passing warm. See the comment on {@code ovLoadLibrary} in {@code native/core/et_runtime.cpp} for
 * the three measured load modes, and {@code native/tests/openvino_loader_flags.sh} for the source
 * ban that catches the same reversion without a staged bundle.
 */
@Tag("openvino")
class OpenVinoColdProbeTest {

    @Test
    void readsTheInferencePrecisionWithoutTheDelegateHavingLoadedTheGraph() throws Exception {
        TestSupport.assumeOpenVinoBundleAvailable();

        // Extraction only: it lays the bundle down and resolves a path, and loads nothing.
        OpenVinoRuntime.ensureExtracted();

        String precision = EtEngine.openVinoInferencePrecision();
        // The VALUE is not asserted -- it is a property of the CPU this happens to run on, and
        // asserting it would assert the hardware. What is asserted is that the read succeeded from
        // a cold process, i.e. the vendored C API and its whole dependency graph resolved out of
        // one flat directory. "unavailable" means it did not.
        assertNotEquals(
                "unavailable",
                precision,
                "the C API should have loaded and answered from a cold process; on Windows this is"
                        + " what a plain LoadLibrary or a dropped search flag breaks");
        assertTrue(
                precision.equals("f32") || precision.equals("bf16") || precision.equals("f16"),
                "unexpected precision: " + precision);
    }
}
