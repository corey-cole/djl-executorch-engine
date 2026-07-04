package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import org.measly.executorch.TestSupport;
import java.io.FileNotFoundException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;

class EtModelTest {

    /** Subclass that pre-populates the inherited {@code block} field to exercise the dynamic-block guard. */
    private static final class StubBlockModel extends EtModel {
        StubBlockModel(NDManager manager) {
            super("stub", manager);
            this.block = mock(Block.class);
        }
    }

    @Test
    void loadRejectsPreexistingBlock() throws Exception {
        try (NDManager manager = NDManager.newBaseManager("ExecuTorch")) {
            EtModel model = new StubBlockModel(manager);
            Path dir = Files.createTempDirectory("etmodel-block");
            assertThrows(
                    UnsupportedOperationException.class, () -> model.load(dir, null, null));
        }
    }

    @Test
    void loadWithoutPteThrowsFileNotFound() throws Exception {
        try (NDManager manager = NDManager.newBaseManager("ExecuTorch")) {
            EtModel model = new EtModel("missing", manager);
            Path emptyDir = Files.createTempDirectory("etmodel-empty");
            assertThrows(FileNotFoundException.class, () -> model.load(emptyDir, null, null));
        }
    }
    @org.junit.jupiter.api.Test
    void wrongArityThrows() throws Exception {
        org.measly.executorch.TestSupport.assumeNativeAvailable();
        try (ai.djl.Model model = ai.djl.Model.newInstance("add", "ExecuTorch")) {
            model.load(java.nio.file.Paths.get("native/spike"), "add");
            try (ai.djl.ndarray.NDManager m = model.getNDManager().newSubManager()) {
                ai.djl.ndarray.NDArray only =
                        m.create(new float[] {2f}, new ai.djl.ndarray.types.Shape(1));
                org.junit.jupiter.api.Assertions.assertThrows(
                        IllegalArgumentException.class,
                        () -> model.getBlock().forward(null, new ai.djl.ndarray.NDList(only), false));
            }
        }
    }

    @Test
    void loadAndForwardAddModel() throws Exception {
        TestSupport.assumeNativeAvailable();
        try (Model model = Model.newInstance("add", "ExecuTorch")) {
            model.load(Paths.get("native/spike"), "add");
            try (NDManager manager = model.getNDManager().newSubManager()) {
                NDArray a = manager.create(new float[] {2f}, new Shape(1));
                NDArray b = manager.create(new float[] {3f}, new Shape(1));
                NDList out =
                        model.getBlock()
                                .forward(null, new NDList(a, b), false);
                assertArrayEquals(new float[] {5f}, out.singletonOrThrow().toFloatArray());
            }
        }
    }
}
