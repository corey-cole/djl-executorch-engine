package org.measly.executorch.stress;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.measly.executorch.TestSupport;
import org.measly.executorch.engine.EtEngineStats;
import org.measly.executorch.engine.EtModelStats;
import org.measly.executorch.engine.EtStatsSnapshot;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Polls the stats snapshot while several threads forward on their own models. Tagged {@code
 * stress} and excluded from CI: it saturates every core for its duration.
 */
@Tag("stress")
class StatsConcurrencyIT {

    private static final int THREADS = 4;
    private static final int FORWARDS_PER_THREAD = 500;

    @Test
    void snapshotIsSafeWhileForwardsRun() throws Exception {
        TestSupport.assumeNativeAvailable();
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(THREADS);
        AtomicBoolean running = new AtomicBoolean(true);
        AtomicReference<Throwable> failure = new AtomicReference<>();
        List<Thread> workers = new ArrayList<>();

        for (int t = 0; t < THREADS; t++) {
            Thread worker =
                    new Thread(
                            () -> {
                                // One Model per thread: forward() is not safe on a shared model.
                                try (Model model = Model.newInstance("add", "ExecuTorch")) {
                                    model.load(Paths.get("native/spike"), "add");
                                    NDManager manager = model.getNDManager();
                                    start.await();
                                    for (int i = 0; i < FORWARDS_PER_THREAD; i++) {
                                        try (NDArray a = manager.create(new float[] {2.0f});
                                                NDArray b = manager.create(new float[] {3.0f});
                                                NDList in = new NDList(a, b);
                                                NDList out =
                                                        model.getBlock()
                                                                .forward(null, in, false)) {
                                            if (out.head().toFloatArray()[0] != 5.0f) {
                                                throw new AssertionError("wrong result");
                                            }
                                        }
                                    }
                                } catch (Throwable e) {
                                    failure.compareAndSet(null, e);
                                } finally {
                                    done.countDown();
                                }
                            });
            worker.start();
            workers.add(worker);
        }

        Thread poller =
                new Thread(
                        () -> {
                            while (running.get()) {
                                try {
                                    EtStatsSnapshot s = EtEngineStats.snapshot();
                                    assertNotNull(s.getModels());
                                    for (EtModelStats m : s.getModels()) {
                                        // Counters are monotonic and internally consistent: a
                                        // torn 64-bit read would surface as a negative or as a
                                        // max exceeding the total.
                                        assertTrue(m.getForwardCount() >= 0);
                                        assertTrue(m.getForwardTotalNanos() >= 0);
                                        assertTrue(
                                                m.getForwardMaxNanos()
                                                        <= m.getForwardTotalNanos());
                                    }
                                } catch (Throwable e) {
                                    failure.compareAndSet(null, e);
                                    return;
                                }
                            }
                        });
        poller.start();

        start.countDown();
        try {
            assertTrue(done.await(5, TimeUnit.MINUTES), "workers did not finish");
        } finally {
            // Always stop the poller, even when the await times out: a poller left running on a
            // failed test is a non-daemon thread that busy-loops and hangs the test JVM.
            running.set(false);
        }
        poller.join(TimeUnit.SECONDS.toMillis(30));
        for (Thread worker : workers) {
            worker.join(TimeUnit.SECONDS.toMillis(30));
        }

        Throwable t = failure.get();
        if (t != null) {
            throw new AssertionError("concurrent snapshot/forward failed", t);
        }

        EtStatsSnapshot end = EtEngineStats.snapshot();
        assertTrue(
                end.getClosedForwardCount() >= (long) THREADS * FORWARDS_PER_THREAD,
                "closed-model rollup must retain the forwards of every closed model");
    }
}
