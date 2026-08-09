package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import java.nio.file.Paths;
import org.measly.executorch.TestSupport;
import org.junit.jupiter.api.Test;

class EtSymbolBlockStatsTest {

    @Test
    void countsForwardsAndReportsNativeBytes() throws Exception {
        TestSupport.assumeNativeAvailable();
        try (Model model = Model.newInstance("add", "ExecuTorch")) {
            model.load(Paths.get("native/spike"), "add");
            EtSymbolBlock block = (EtSymbolBlock) model.getBlock();

            EtModelStats before = block.toStats();
            assertNotNull(before, "counters must be attached at load");
            assertEquals(0L, before.getForwardCount());
            assertTrue(before.getPlannedArenaBytes() > 0);
            assertEquals(0L, before.getStagingBytes(), "add.pte is memory-planned");
            assertTrue(before.getLoadNanos() > 0);

            NDManager manager = model.getNDManager();
            for (int i = 0; i < 3; i++) {
                try (NDArray a = manager.create(new float[] {2.0f});
                        NDArray b = manager.create(new float[] {3.0f});
                        NDList in = new NDList(a, b);
                        NDList out = block.forward(null, in, false)) {
                    assertEquals(5.0f, out.head().toFloatArray()[0], 1e-6f);
                }
            }

            EtModelStats after = block.toStats();
            assertEquals(3L, after.getForwardCount());
            assertTrue(after.getForwardTotalNanos() > 0);
            assertTrue(after.getForwardMaxNanos() > 0);
            assertTrue(after.getForwardMaxNanos() <= after.getForwardTotalNanos());
        }
    }

    @Test
    void reportsUnavailableBytesAfterClose() throws Exception {
        TestSupport.assumeNativeAvailable();
        Model model = Model.newInstance("add", "ExecuTorch");
        model.load(Paths.get("native/spike"), "add");
        EtSymbolBlock block = (EtSymbolBlock) model.getBlock();
        model.close();
        // The handle is gone. Querying native staging bytes now would be a use-after-free, so the
        // guard must report -1 ("unavailable") rather than 0 ("genuinely zero") or crashing.
        assertEquals(-1L, block.toStats().getStagingBytes());
    }
}
