package org.measly.executorch.stress;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

/** The sweep matrix. Pure data; touches no native code. */
class SweepConfigTest {

    @Test
    void coreCellsAreTheFullThreadByModeCrossProduct() {
        List<SweepConfig.Cell> cells = SweepConfig.coreCells();
        assertEquals(8, cells.size(), "{1,2,4,8} threads x {global,disabled}");
        assertEquals(
                List.of(1, 1, 2, 2, 4, 4, 8, 8),
                cells.stream().map(SweepConfig.Cell::threads).collect(Collectors.toList()),
                "ordered by thread count so the report reads as a scaling curve");
    }

    @Test
    void everyCoreCellPinsOneIntraOpThread() {
        // The workspace lock is only legible at intra-op=1; at the default pool size a single
        // forward saturates the box and caller-thread scaling is flat for unrelated reasons.
        assertTrue(SweepConfig.coreCells().stream().allMatch(c -> c.intraOp() == 1));
    }

    @Test
    void perModelIsExcluded() {
        // Degenerate with `global` for a single-model workload: they differ only across distinct
        // models. Including it would add runtime and produce duplicate rows.
        assertFalse(SweepConfig.coreCells().stream().anyMatch(c -> "per_model".equals(c.mode())));
    }

    @Test
    void baselineIsOneCellAtTheDefaultPoolSize() {
        List<SweepConfig.Cell> cells = SweepConfig.baselineCells();
        assertEquals(1, cells.size());
        assertEquals(1, cells.get(0).threads());
        assertEquals("global", cells.get(0).mode());
        assertEquals(0, cells.get(0).intraOp(), "0 means 'leave the pool at its default size'");
    }

    @Test
    void theTwoArmsTogetherAreTheNineCellSweep() {
        assertEquals(9, SweepConfig.coreCells().size() + SweepConfig.baselineCells().size());
    }

    @Test
    void labelIsStableAndReadable() {
        assertEquals("t=4 mode=disabled intraop=1", new SweepConfig.Cell(4, "disabled", 1).label());
        assertEquals("t=1 mode=global intraop=default", new SweepConfig.Cell(1, "global", 0).label());
    }
}
