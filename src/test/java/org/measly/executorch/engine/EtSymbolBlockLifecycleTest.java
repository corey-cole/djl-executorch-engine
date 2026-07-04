package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.djl.Model;
import java.nio.file.Paths;
import org.measly.executorch.TestSupport;
import org.junit.jupiter.api.Test;

class EtSymbolBlockLifecycleTest {

    @Test
    void closeReleasesHandleAndIsIdempotent() throws Exception {
        TestSupport.assumeNativeAvailable();
        Model model = Model.newInstance("add", "ExecuTorch");
        model.load(Paths.get("native/spike"), "add");
        EtSymbolBlock block = (EtSymbolBlock) model.getBlock();
        assertFalse(block.isClosed());
        model.close();
        assertTrue(block.isClosed());
        block.close(); // idempotent: handle already 0, must not throw or double-destroy
    }

    @Test
    void repeatedLoadCloseDoesNotDegrade() throws Exception {
        TestSupport.assumeNativeAvailable();
        for (int i = 0; i < 100; i++) {
            try (Model model = Model.newInstance("add", "ExecuTorch")) {
                model.load(Paths.get("native/spike"), "add");
            } // close() must destroy the native Module each iteration
        }
    }

    @Test
    void removeLastBlockIsUnsupported() throws Exception {
        TestSupport.assumeNativeAvailable();
        Model model = Model.newInstance("add", "ExecuTorch");
        model.load(Paths.get("native/spike"), "add");
        EtSymbolBlock block = (EtSymbolBlock) model.getBlock();
        assertThrows(UnsupportedOperationException.class, block::removeLastBlock);
    }
}
