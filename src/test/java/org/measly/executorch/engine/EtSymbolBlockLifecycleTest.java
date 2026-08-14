package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
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
    void forwardAfterCloseThrowsIllegalState() throws Exception {
        TestSupport.assumeNativeAvailable();
        Model model = Model.newInstance("add", "ExecuTorch");
        model.load(Paths.get("native/spike"), "add");
        EtSymbolBlock block = (EtSymbolBlock) model.getBlock();
        // A manager of its own: closing the model closes the model's manager with it, and the
        // inputs must outlive that so the call under test is the forward, not the array creation.
        try (NDManager manager = NDManager.newBaseManager("ExecuTorch")) {
            NDArray a = manager.create(new float[] {2f}, new Shape(1));
            NDArray b = manager.create(new float[] {3f}, new Shape(1));
            try (NDList inputs = new NDList(a, b)) {
                model.close();
                assertTrue(block.isClosed());
                // Without the guard this reaches native code with handle 0 and takes down the JVM.
                IllegalStateException ex =
                        assertThrows(
                                IllegalStateException.class,
                                () -> block.forward(null, inputs, false));
                assertTrue(
                        ex.getMessage().contains("closed"),
                        "message should name the cause, was: " + ex.getMessage());
            }
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
