package org.measly.executorch.stress;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.measly.executorch.TestSupport;

/**
 * Maximum-contention correctness gate: 8 threads, {@code global} sharing, intra-op pool at its
 * default size. Not a benchmark — the point is to make a race likely, not to be fast.
 *
 * <p><b>Local only.</b> This saturates every core for its whole duration. Never wire it to CI, not
 * even a nightly on a free runner.
 *
 * <p>Duration is {@code -Det.stress.seconds} (default 30), surfaced by the {@code stressGate}
 * Gradle task as {@code -PstressSeconds}.
 */
@Tag("stress")
class StressGateIT {

    private static final int THREADS = 8;
    private static final String MODE = "global";

    @Test
    void eightThreadsAgreeBitwiseWithTheGoldensUnderMaximumContention() throws Exception {
        TestSupport.assumeStressModelAvailable();
        StressGolden golden = StressGolden.load(TestSupport.stressGoldenPath());
        List<StressGolden.Case> cases = golden.cases();
        int seconds = Integer.getInteger("et.stress.seconds", 30);

        // Reference pass, single-threaded, before any worker starts. Verified against the goldens
        // (oracle layer 1) so the reference itself is known-good; the workers then compare against
        // it bit for bit (layer 2).
        float[][] reference = new float[cases.size()][];
        PerThreadContext.resetCounters();
        try (PerThreadContext ctx = PerThreadContext.open(MODE)) {
            for (int i = 0; i < cases.size(); i++) {
                reference[i] = ctx.predict(cases.get(i).v1, cases.get(i).v2);
                golden.verify(cases.get(i), reference[i]);
            }
        }

        AtomicBoolean stop = new AtomicBoolean(false);
        AtomicLong forwards = new AtomicLong();
        Queue<Throwable> failures = new ConcurrentLinkedQueue<>();
        CyclicBarrier start = new CyclicBarrier(THREADS);
        List<Thread> workers = new ArrayList<>(THREADS);

        for (int t = 0; t < THREADS; t++) {
            Thread worker =
                    new Thread(
                            () -> {
                                // THE PATTERN: one model+predictor per thread, scoped to the
                                // thread's own lifetime by try-with-resources. The closing thread
                                // is the one that was doing the forwards, so nothing can be in
                                // flight when close() runs.
                                try (PerThreadContext ctx = PerThreadContext.open(MODE)) {
                                    start.await();
                                    while (!stop.get()) {
                                        for (int i = 0; i < cases.size(); i++) {
                                            float[] out =
                                                    ctx.predict(cases.get(i).v1, cases.get(i).v2);
                                            if (!Arrays.equals(reference[i], out)) {
                                                throw new AssertionError(
                                                        "thread "
                                                                + Thread.currentThread().getName()
                                                                + " diverged bitwise on case "
                                                                + cases.get(i).name
                                                                + " after "
                                                                + forwards.get()
                                                                + " total forwards");
                                            }
                                            forwards.incrementAndGet();
                                        }
                                    }
                                } catch (Throwable e) {
                                    failures.add(e);
                                    start.reset(); // release peers parked at the barrier
                                    stop.set(true); // do not let the others spin for the full run
                                }
                            },
                            "stress-" + t);
            workers.add(worker);
            worker.start();
        }

        Thread.sleep(seconds * 1000L);
        stop.set(true);
        for (Thread w : workers) {
            w.join(60_000);
            assertTrue(!w.isAlive(), w.getName() + " did not terminate within 60s of the stop flag");
        }

        if (!failures.isEmpty()) {
            AssertionError e = new AssertionError(failures.size() + " worker(s) failed");
            failures.forEach(e::addSuppressed);
            throw e;
        }

        assertTrue(forwards.get() > 0, "no forwards ran — the stop flag fired too early");
        assertEquals(
                THREADS + 1,
                PerThreadContext.opened(),
                "8 workers plus the reference pass");
        assertEquals(
                PerThreadContext.opened(),
                PerThreadContext.closed(),
                "every context must be closed — a mismatch is a leaked native handle");
        System.out.printf(
                "stressGate: %d forwards across %d threads in %ds (%s)%n",
                forwards.get(), THREADS, seconds, MODE);
    }

    @Test
    void aWorkerExceptionFailsTheRunRatherThanHangingIt() throws Exception {
        TestSupport.assumeStressModelAvailable();
        // Proves the failure-propagation path itself: a worker that throws must surface as a test
        // failure with its cause attached, not as a silently-passing run or a hang.
        Queue<Throwable> failures = new ConcurrentLinkedQueue<>();
        Thread t =
                new Thread(
                        () -> {
                            try {
                                throw new IllegalStateException("synthetic worker failure");
                            } catch (Throwable e) {
                                failures.add(e);
                            }
                        });
        t.start();
        t.join(10_000);
        if (failures.isEmpty()) {
            fail("the failure queue must capture a worker throwable");
        }
        assertEquals("synthetic worker failure", failures.peek().getMessage());
    }
}
