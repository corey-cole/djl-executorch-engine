package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

import ai.djl.engine.Engine;
import org.junit.jupiter.api.Test;

class EtEngineTest {
    @Test
    void engineIsDiscoverableViaSpi() {
        Engine engine = Engine.getEngine("ExecuTorch");
        assertNotNull(engine);
        assertEquals("ExecuTorch", engine.getEngineName());
        assertEquals(10, engine.getRank());
        assertNull(engine.getAlternativeEngine()); // no hybrid in Phase 1
    }
}
