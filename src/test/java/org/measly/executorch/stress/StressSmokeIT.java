package org.measly.executorch.stress;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.measly.executorch.TestSupport;

/**
 * Single-threaded proof that the stress model loads, predicts, and matches its goldens, and that
 * the per-thread context closes what it opens. Everything the concurrent gate assumes, verified
 * without concurrency first — so a failure in StressGateIT means concurrency, not plumbing.
 */
@Tag("stress")
class StressSmokeIT {

    @Test
    void everyGoldenCaseMatchesOnOneThread() throws Exception {
        TestSupport.assumeStressModelAvailable();
        StressGolden golden = StressGolden.load(TestSupport.stressGoldenPath());

        PerThreadContext.resetCounters();
        try (PerThreadContext ctx = PerThreadContext.open("global")) {
            for (StressGolden.Case c : golden.cases()) {
                float[] out = ctx.predict(c.v1, c.v2);
                golden.verify(c, out);
            }
        }
        assertEquals(1, PerThreadContext.opened());
        assertEquals(1, PerThreadContext.closed(), "close() must run via try-with-resources");
    }

    @Test
    void repeatedPredictsAreBitwiseIdentical() throws Exception {
        TestSupport.assumeStressModelAvailable();
        StressGolden golden = StressGolden.load(TestSupport.stressGoldenPath());
        StressGolden.Case c = golden.cases().get(0);

        try (PerThreadContext ctx = PerThreadContext.open("global")) {
            float[] reference = ctx.predict(c.v1, c.v2);
            for (int i = 0; i < 20; i++) {
                assertTrue(
                        Arrays.equals(reference, ctx.predict(c.v1, c.v2)),
                        "iteration " + i + " diverged bitwise from the first forward");
            }
        }
    }

    @Test
    void everySharingModeLoadsAndAgreesWithTheGoldens() throws Exception {
        TestSupport.assumeStressModelAvailable();
        StressGolden golden = StressGolden.load(TestSupport.stressGoldenPath());
        StressGolden.Case c = golden.cases().get(0);

        for (String mode : new String[] {null, "global", "disabled", "per_model"}) {
            try (PerThreadContext ctx = PerThreadContext.open(mode)) {
                golden.verify(c, ctx.predict(c.v1, c.v2));
            }
        }
    }

    @Test
    void buildInputPutsTheSteeringValueAtElementZero() {
        float[] in = StressTranslator.buildInput(0.25f, 32, 256, 1e-5f);
        assertEquals(32 * 256, in.length);
        assertEquals(0.25f, in[0], "element [0,0] steers the bucket and must be exactly v");
        assertEquals(0.25f + 1e-5f, in[1], 0f, "ramp must be computed in float32, in order");
    }
}
