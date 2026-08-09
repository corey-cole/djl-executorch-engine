package org.measly.executorch.stress;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.measly.executorch.TestSupport;

/**
 * The single confirmation cell at the real-world intra-op default — one caller thread, default pool
 * size. Separate class and separate tag because it needs its own JVM: the intra-op pool is
 * process-global and write-once, so it cannot share a process with the intra-op=1 cells.
 */
@Tag("stress-baseline")
class StressSweepBaselineIT {

    @Test
    void oneThreadAtTheDefaultIntraOpPoolSize() throws Exception {
        TestSupport.assumeStressModelAvailable();
        StressGolden golden = StressGolden.load(TestSupport.stressGoldenPath());
        int seconds = Integer.getInteger("et.stress.cellSeconds", 10);

        List<SweepRunner.Result> results = new ArrayList<>();
        for (SweepConfig.Cell cell : SweepConfig.baselineCells()) {
            results.add(SweepRunner.run(cell, golden, seconds));
        }
        SweepRunner.report(results);

        assertEquals(1, results.size());
        assertTrue(results.get(0).forwards() > 0);
    }
}
