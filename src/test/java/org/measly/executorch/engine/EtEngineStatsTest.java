package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import java.lang.ref.WeakReference;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.TimeUnit;
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

    /**
     * A model dropped without {@code close()} must not be pinned by the registry.
     *
     * <p>Without weak entries the static registry is the model's last strong referrer, so a caller
     * who leaks a model turns a native-only leak into a permanent Java-heap one, and
     * {@code modelsLive} climbs forever. The forwards it completed must still reach the rollup —
     * losing them would defeat the point of having a rollup at all.
     *
     * <p>Note this does <em>not</em> make a leaked model free: DJL's {@code BaseNDManager} attaches
     * every base manager to a static system manager, so the model's {@code EtNDManager} stays
     * reachable regardless. That retention is DJL's and predates this class.
     */
    @Test
    void aModelLeakedWithoutCloseIsPurgedFromTheRegistry() throws Exception {
        TestSupport.assumeNativeAvailable();
        EtStatsSnapshot before = EtEngineStats.snapshot();
        long liveBefore = before.getModelsLive();
        long rolledUpBefore = before.getClosedForwardCount();

        WeakReference<EtSymbolBlock> leaked = loadForwardAndAbandon(3);
        assertEquals(
                liveBefore + 1,
                EtEngineStats.snapshot().getModelsLive(),
                "the abandoned model must be tracked while it is still reachable");

        EtStatsSnapshot after = awaitPurge(liveBefore);
        assertEquals(
                liveBefore,
                after.getModelsLive(),
                "a garbage-collected model must be purged from the registry");
        assertEquals(
                rolledUpBefore + 3,
                after.getClosedForwardCount(),
                "a leaked model's forwards must still reach the rollup");
        assertEquals(null, leaked.get(), "sanity: the block really was collected");
    }

    /**
     * Loads a model, runs {@code forwards} inferences, and abandons it without closing. The model
     * is a local of this method and never escapes, so it is unreachable once this returns — which
     * is the whole point.
     */
    private static WeakReference<EtSymbolBlock> loadForwardAndAbandon(int forwards)
            throws Exception {
        Model model = Model.newInstance("leaked", "ExecuTorch");
        model.load(Paths.get("native/spike"), "add");
        EtSymbolBlock block = (EtSymbolBlock) model.getBlock();
        NDManager manager = model.getNDManager();
        for (int i = 0; i < forwards; i++) {
            try (NDArray a = manager.create(new float[] {2.0f});
                    NDArray b = manager.create(new float[] {3.0f});
                    NDList in = new NDList(a, b);
                    NDList out = block.forward(null, in, false)) {
                assertEquals(5.0f, out.head().toFloatArray()[0], 1e-6f);
            }
        }
        return new WeakReference<>(block); // deliberately NOT closed
    }

    /**
     * Polls until the registry drops back to {@code targetLive}. Both the collection itself and the
     * reference-queue enqueue that follows it are asynchronous, so this cannot be a single
     * {@code System.gc()} — it drives GC and re-snapshots (which is what drains the queue) until
     * the count settles or the budget runs out.
     */
    private static EtStatsSnapshot awaitPurge(long targetLive) throws InterruptedException {
        final long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        EtStatsSnapshot snapshot = EtEngineStats.snapshot();
        while (snapshot.getModelsLive() != targetLive && System.nanoTime() < deadline) {
            System.gc();
            Thread.sleep(20);
            snapshot = EtEngineStats.snapshot();
        }
        return snapshot;
    }
}
