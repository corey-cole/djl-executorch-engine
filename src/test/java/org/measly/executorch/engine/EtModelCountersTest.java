package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class EtModelCountersTest {

    private static EtModelCounters counters() {
        return new EtModelCounters("add", "global", 4096L, 1_000_000L);
    }

    @Test
    void startsAtZeroAndKeepsLoadTimeMetadata() {
        EtModelCounters c = counters();
        assertEquals("add", c.name());
        assertEquals("global", c.workspaceSharingMode());
        assertEquals(4096L, c.plannedArenaBytes());
        assertEquals(1_000_000L, c.loadNanos());
        assertEquals(0L, c.forwardCount());
        assertEquals(0L, c.forwardTotalNanos());
        assertEquals(0L, c.forwardMaxNanos());
    }

    @Test
    void accumulatesCountAndTotal() {
        EtModelCounters c = counters();
        c.recordForward(100L);
        c.recordForward(250L);
        c.recordForward(50L);
        assertEquals(3L, c.forwardCount());
        assertEquals(400L, c.forwardTotalNanos());
    }

    @Test
    void tracksTheMaximumNotTheLatest() {
        EtModelCounters c = counters();
        c.recordForward(100L);
        c.recordForward(900L);
        c.recordForward(200L); // a later smaller sample must not lower the peak
        assertEquals(900L, c.forwardMaxNanos());
    }

    @Test
    void recordsAZeroDurationSampleAsAnObservation() {
        // A clock with coarse resolution can legitimately report 0. It still counts as a forward.
        EtModelCounters c = counters();
        c.recordForward(0L);
        assertEquals(1L, c.forwardCount());
        assertEquals(0L, c.forwardTotalNanos());
    }
}
