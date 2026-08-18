package org.measly.example;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class VariantTest {

    @Test
    void enumMetadata() {
        assertEquals(4, Variant.values().length);
        assertEquals("ExecuTorch", Variant.ET_HYBRID.engine);
        assertEquals("PyTorch", Variant.PYTORCH.engine);
        assertEquals("ExecuTorch", Variant.ET_NATIVE.engine);
        assertEquals("ExecuTorch", Variant.ET_OPENVINO.engine);
        assertEquals(Variant.ET_NATIVE, Variant.valueOf("ET_NATIVE"));
    }

    @Test
    void artifactExtensionFollowsTheEngine() {
        // PYTORCH is the only TorchScript arm; every ExecuTorch arm loads a .pte.
        assertEquals("mobilenet_v2.pt", Variant.PYTORCH.artifact());
        assertEquals("mobilenet_v2.pte", Variant.ET_HYBRID.artifact());
        assertEquals("mobilenet_v2.pte", Variant.ET_NATIVE.artifact());
        assertEquals("mobilenet_v2_openvino.pte", Variant.ET_OPENVINO.artifact());
    }

    @Test
    void onlyTheXnnpackArmsHaveAnUnplannedExport() {
        // The unplanned export exists to exercise ExecuTorch's borrowed-input path, and only the
        // XNNPACK arms have one. PYTORCH has no such notion, and the OpenVINO model is exported
        // once. Both must resolve "unplanned" back to their single artifact rather than naming a
        // file that was never generated -- a missing-file failure at benchmark setup is the bug
        // this guards.
        assertEquals("mobilenet_v2_unplanned.pte", Variant.ET_HYBRID.artifact("unplanned"));
        assertEquals("mobilenet_v2_unplanned.pte", Variant.ET_NATIVE.artifact("unplanned"));
        assertEquals("mobilenet_v2.pt", Variant.PYTORCH.artifact("unplanned"));
        assertEquals("mobilenet_v2_openvino.pte", Variant.ET_OPENVINO.artifact("unplanned"));

        for (Variant v : Variant.values()) {
            assertEquals(v.artifact(), v.artifact("planned"), v + " planned == default export");
        }
    }

    @Test
    void modelNameIsTheArtifactWithoutItsExtension() {
        // DJL resolves a model by name, not file name, so the two must not drift apart.
        for (Variant v : Variant.values()) {
            for (String mode : List.of("planned", "unplanned")) {
                assertTrue(
                        v.artifact(mode).startsWith(v.modelName(mode) + "."),
                        v + "/" + mode + ": " + v.artifact(mode) + " vs " + v.modelName(mode));
            }
        }
    }

    @Test
    void etNativeFactoryBuildsPlainJavaTranslatorWithoutPyTorch() {
        // Only ET_NATIVE is exercised here: building ET_HYBRID/PYTORCH/ET_OPENVINO would construct
        // a PyTorch NDManager (loads LibTorch). ET_NATIVE must build purely in-JVM.
        CloseableImageTranslator t = Variant.ET_NATIVE.newTranslator(List.of("a", "b"));
        assertTrue(t instanceof PlainJavaMobilenetTranslator);
        t.close(); // no-op, must not throw
    }
}
