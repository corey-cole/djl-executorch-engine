package org.measly.example;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class VariantTest {

    @Test
    void enumMetadata() {
        assertEquals(3, Variant.values().length);
        assertEquals("ExecuTorch", Variant.ET_HYBRID.engine);
        assertEquals("PyTorch", Variant.PYTORCH.engine);
        assertEquals("ExecuTorch", Variant.ET_NATIVE.engine);
        assertEquals(Variant.ET_NATIVE, Variant.valueOf("ET_NATIVE"));
    }

    @Test
    void etNativeFactoryBuildsPlainJavaTranslatorWithoutPyTorch() {
        // Only ET_NATIVE is exercised here: building ET_HYBRID/PYTORCH would construct a PyTorch
        // NDManager (loads LibTorch). ET_NATIVE must build purely in-JVM.
        CloseableImageTranslator t = Variant.ET_NATIVE.newTranslator(List.of("a", "b"));
        assertTrue(t instanceof PlainJavaMobilenetTranslator);
        t.close(); // no-op, must not throw
    }
}
