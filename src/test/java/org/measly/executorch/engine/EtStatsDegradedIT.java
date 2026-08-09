package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.ThrowingSupplier;

/**
 * Runs in its own JVM with {@code EXECUTORCH_LIBRARY_PATH} pointing at a non-library file, so
 * {@code System.load} throws inside {@code EtNative}'s class initializer. A monitoring surface
 * must never be the thing that breaks production: {@link EtEngineStats#snapshot()} has to degrade
 * to sentinel values instead of propagating a class-init failure out of every poll.
 *
 * <p>Deliberately does <em>not</em> call {@code TestSupport.assumeNativeAvailable()}: that would
 * itself attempt the load and abort the test. The broken library is the point.
 */
@Tag("stats-degraded")
class EtStatsDegradedIT {

    @Test
    void snapshotDegradesWhenTheNativeLibraryIsBroken() {
        // Explicitly typed so the call binds to the ThrowingSupplier overload (a value-returning
        // method reference is also void-compatible with Executable, which would be ambiguous).
        ThrowingSupplier<EtStatsSnapshot> snapshotCall = EtEngineStats::snapshot;

        // First access poisons EtNative's <clinit> (ExceptionInInitializerError); later accesses
        // surface NoClassDefFoundError. Neither may escape snapshot().
        assertDoesNotThrow(
                snapshotCall, "snapshot() must never throw on a broken native library");

        EtStatsSnapshot snapshot = assertDoesNotThrow(snapshotCall);
        assertEquals(
                -1,
                snapshot.getIntraOpThreads(),
                "intra-op threads must degrade to -1 when the library is unavailable");
        assertEquals(
                "unknown",
                snapshot.getNativeLibraryPath(),
                "the loaded-path field must degrade to the spec's unknown sentinel");
    }
}
