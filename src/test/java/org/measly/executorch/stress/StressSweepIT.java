package org.measly.executorch.stress;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.measly.executorch.TestSupport;
import org.measly.executorch.engine.EtEngine;

/**
 * The eight intra-op=1 cells. Run via {@code ./gradlew stressSweep} (or {@code stressSweepCore}),
 * which forks a JVM with {@code -Dai.djl.executorch.num_threads=1} — required, because the intra-op
 * pool is process-global and write-once.
 */
@Tag("stress-sweep")
class StressSweepIT {

    @Test
    void sweepThreadCountsAndSharingModes() throws Exception {
        TestSupport.assumeStressModelAvailable();
        StressGolden golden = StressGolden.load(TestSupport.stressGoldenPath());
        int seconds = Integer.getInteger("et.stress.cellSeconds", 10);

        List<SweepRunner.Result> results = new ArrayList<>();
        List<SweepConfig.Cell> cells = SweepConfig.coreCells();
        // The pool is sealed at the first model load, which the first cell already performed. Assert
        // the fork took effect immediately after the FIRST cell — the earliest point the count is
        // both meaningful and cheap — so a dropped fork flag fails fast instead of after ~8 cells
        // of numbers that measure something else.
        results.add(SweepRunner.run(cells.get(0), golden, seconds));
        assertEquals(
                1,
                EtEngine.getIntraOpThreads(),
                "stressSweepCore must fork with -Dai.djl.executorch.num_threads=1");
        for (int i = 1; i < cells.size(); i++) {
            results.add(SweepRunner.run(cells.get(i), golden, seconds));
        }
        SweepRunner.report(results);

        assertEquals(8, results.size());
        assertTrue(
                results.stream().allMatch(r -> r.forwards() > 0),
                "every cell must have completed at least one forward");
    }
}
