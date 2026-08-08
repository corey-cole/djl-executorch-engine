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
        for (SweepConfig.Cell cell : SweepConfig.coreCells()) {
            results.add(SweepRunner.run(cell, golden, seconds));
        }
        SweepRunner.report(results);

        // The pool is sealed at the first model load, which the first cell already did. Assert the
        // fork actually took effect — without it every number above measures something else.
        assertEquals(
                1,
                EtEngine.getIntraOpThreads(),
                "stressSweepCore must fork with -Dai.djl.executorch.num_threads=1");
        assertEquals(8, results.size());
        assertTrue(
                results.stream().allMatch(r -> r.forwards() > 0),
                "every cell must have completed at least one forward");
    }
}
