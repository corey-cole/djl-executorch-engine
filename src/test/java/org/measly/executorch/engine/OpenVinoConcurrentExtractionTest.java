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
 * Extraction publishes by atomic directory rename and adopts a concurrent winner. A unit test of
 * the rename would prove nothing — the interesting case is two extractions racing for the same
 * content-addressed path.
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
}
