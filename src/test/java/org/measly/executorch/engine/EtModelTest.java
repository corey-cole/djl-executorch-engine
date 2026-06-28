package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import org.measly.executorch.TestSupport;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;

class EtModelTest {
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
