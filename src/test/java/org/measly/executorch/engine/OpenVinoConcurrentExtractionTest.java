package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.measly.executorch.TestSupport;

/**
 * Extraction publishes by atomic directory rename and adopts a concurrent winner.
 *
 * <p>The two tests here cover different halves, because no single one can cover both.
 * {@link #racingExtractionsConvergeOnOneDirectory()} exercises same-JVM entry: {@code
 * ensureExtracted} is {@code static synchronized} with a cached fast path, so the threads
 * <b>serialise</b> and only the first ever publishes. That test therefore proves idempotency and
 * that no staging directory leaks — <b>not</b> the adoption branch, which it cannot reach.
 *
 * <p>{@link #publishAdoptsADirectoryAnotherPublisherAlreadyWon()} covers the adoption branch
 * directly, by publishing onto a target that already exists. Cross-process racing would reach the
 * same code, but only sometimes and only with two JVMs; driving the branch is deterministic and
 * asserts the property that actually matters — the winner's bytes survive untouched.
 */
@Tag("openvino")
class OpenVinoConcurrentExtractionTest {

    @Test
    void racingExtractionsConvergeOnOneDirectory() throws Exception {
        TestSupport.assumeOpenVinoBundleAvailable();

        int threads = 4;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            List<Callable<Path>> tasks = new ArrayList<>();
            for (int i = 0; i < threads; i++) {
                tasks.add(OpenVinoRuntime::ensureExtracted);
            }
            List<Future<Path>> results = pool.invokeAll(tasks);
            Path first = results.get(0).get();
            for (Future<Path> f : results) {
                assertEquals(first, f.get(), "every racer must land on the same published directory");
            }
            assertTrue(Files.isDirectory(first));
            // No staging directory may survive. A leaked one means the loser could not clean up,
            // which is the failure mode that would matter on a platform refusing to delete a
            // loaded library.
            try (var siblings = Files.list(first.getParent())) {
                assertTrue(
                        siblings.noneMatch(p -> p.getFileName().toString().startsWith("staging-")),
                        "a staging directory leaked; the loser could not clean up after itself");
            }
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    void publishAdoptsADirectoryAnotherPublisherAlreadyWon() throws Exception {
        TestSupport.assumeOpenVinoBundleAvailable();

        // Stand in for a publisher that got there first. Non-empty on purpose: POSIX rename() onto
        // an EMPTY directory succeeds, so an empty stand-in would let the move win and quietly skip
        // the branch this test exists for.
        Path parent = Files.createTempDirectory("openvino-adopt");
        Path target = parent.resolve("already-published");
        Files.createDirectories(target);
        Path winnerFile = target.resolve("winner.marker");
        Files.writeString(winnerFile, "written by the publisher that won");

        try {
            // Must not throw: losing the rename is the expected outcome of a race, not a failure.
            OpenVinoRuntime.publish(target);

            assertTrue(
                    Files.exists(winnerFile),
                    "adoption must leave the winner's directory untouched; overwriting it could pull"
                            + " a library out from under a process that already loaded it");
            assertEquals(
                    "written by the publisher that won",
                    Files.readString(winnerFile),
                    "the winner's bytes must survive verbatim");

            try (var siblings = Files.list(parent)) {
                assertTrue(
                        siblings.noneMatch(p -> p.getFileName().toString().startsWith("staging-")),
                        "the loser must delete its own staging directory after adopting");
            }
        } finally {
            deleteRecursively(parent);
        }
    }

    private static void deleteRecursively(Path dir) throws Exception {
        if (!Files.exists(dir)) {
            return;
        }
        List<Path> paths = new ArrayList<>();
        try (var walk = Files.walk(dir)) {
            walk.forEach(paths::add);
        }
        for (int i = paths.size() - 1; i >= 0; i--) {
            Files.deleteIfExists(paths.get(i));
        }
    }
}
