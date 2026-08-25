package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

class EtProfilingTest {

    @Test
    void absentOptionMeansOff() {
        assertFalse(EtProfiling.resolve(null));
        assertFalse(EtProfiling.resolve(Map.of()));
    }

    @Test
    void parsesBothValuesCaseInsensitivelyAndTrimmed() {
        assertTrue(EtProfiling.resolve(Map.of("profiling", "true")));
        assertTrue(EtProfiling.resolve(Map.of("profiling", "  TRUE ")));
        assertFalse(EtProfiling.resolve(Map.of("profiling", "False")));
    }

    @Test
    void rejectsAnythingElse() {
        // Boolean.parseBoolean would silently read "yes" and "1" as false. An option whose typo
        // disables the feature it names is worse than one that fails.
        assertThrows(IllegalArgumentException.class, () -> EtProfiling.resolve(Map.of("profiling", "yes")));
        assertThrows(IllegalArgumentException.class, () -> EtProfiling.resolve(Map.of("profiling", "1")));
        assertThrows(IllegalArgumentException.class, () -> EtProfiling.resolve(Map.of("profiling", "")));
    }
}
