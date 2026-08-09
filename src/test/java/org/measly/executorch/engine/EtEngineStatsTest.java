package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.djl.Model;
import java.nio.file.Paths;
import java.util.List;
import org.measly.executorch.TestSupport;
import org.junit.jupiter.api.Test;

class EtEngineStatsTest {

    @Test
    void reportsConfigurationWithoutAnyModelLoaded() {
        TestSupport.assumeNativeLibraryAvailable();
        EtStatsSnapshot s = EtEngineStats.snapshot();
        assertEquals("1.3.1", s.getExecutorchVersion());
        assertNotNull(s.getPlatform());
        assertNotNull(s.getNativeLibraryPath());
        assertTrue(s.getIntraOpThreads() >= 1);
        assertNotNull(s.getDefaultWorkspaceSharingMode());
    }

    @Test
    void tracksALiveModelThenRollsItUpOnClose() throws Exception {
        TestSupport.assumeNativeAvailable();
        long loadedBefore = EtEngineStats.snapshot().getModelsLoaded();

        Model model = Model.newInstance("add", "ExecuTorch");
        model.load(Paths.get("native/spike"), "add");

        EtStatsSnapshot live = EtEngineStats.snapshot();
        assertEquals(loadedBefore + 1, live.getModelsLoaded());
        assertTrue(live.getModelsLive() >= 1);
        assertTrue(live.getTotalPlannedArenaBytes() > 0);
        List<EtModelStats> models = live.getModels();
        assertTrue(
                models.stream().anyMatch(m -> "add".equals(m.getName())),
                "the live model must appear in the per-model list");

        long liveCount = live.getModelsLive();
        model.close();

        EtStatsSnapshot closed = EtEngineStats.snapshot();
        assertEquals(liveCount - 1, closed.getModelsLive());
        assertEquals(
                loadedBefore + 1,
                closed.getModelsLoaded(),
                "cumulative loads must not decrease when a model closes");
        assertTrue(
                closed.getModels().stream().noneMatch(m -> "add".equals(m.getName())),
                "a closed model must leave the per-model list");
    }

    @Test
    void snapshotIsAnIndependentCopy() throws Exception {
        TestSupport.assumeNativeAvailable();
        try (Model model = Model.newInstance("add", "ExecuTorch")) {
            model.load(Paths.get("native/spike"), "add");
            EtStatsSnapshot first = EtEngineStats.snapshot();
            int sizeBefore = first.getModels().size();
            try (Model second = Model.newInstance("add2", "ExecuTorch")) {
                second.load(Paths.get("native/spike"), "add");
                // The earlier snapshot must not observe the later load.
                assertEquals(sizeBefore, first.getModels().size());
                assertEquals(sizeBefore + 1, EtEngineStats.snapshot().getModels().size());
            }
        }
    }
}
