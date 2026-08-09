package org.measly.executorch.engine;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import org.measly.executorch.jni.EtNative;

/**
 * The engine's production monitoring surface: an always-on, fixed-cost view of throughput, native
 * footprint, and effective configuration.
 *
 * <p>{@link #snapshot()} is a cold-path read — walk it from a scheduled poll, an HTTP health
 * endpoint, or a JMX console. It never throws: values that cannot be read degrade to {@code -1}
 * (bytes) or {@code unknown} (strings) rather than propagating a failure out of a monitoring call.
 *
 * <p><b>Relationship to DJL's {@code Metrics}.</b> {@code Predictor.setMetrics(...)} records
 * per-{@code predict} timings, but it is a time-series buffer built for benchmarking, not a
 * production counter: its {@code limit} defaults to 0 (uncapped, so samples are retained forever
 * unless you wire both {@code setLimit} and {@code setOnLimit}), its {@code addMetric} is a
 * check-then-act race at the flush boundary, and {@code percentile()} sorts the whole buffer on
 * every call. Use it for profiling; use this class for production monitoring.
 *
 * <p>Per-model detail covers live models only. A model's totals fold into the closed-model rollup
 * when it closes, so a restart-on-error loop cannot erase throughput history.
 */
public final class EtEngineStats {

    private static final Map<Long, EtSymbolBlock> LIVE = new ConcurrentHashMap<>();
    private static final AtomicLong MODELS_LOADED = new AtomicLong();
    private static final AtomicLong CLOSED_FORWARD_COUNT = new AtomicLong();
    private static final AtomicLong CLOSED_FORWARD_TOTAL_NANOS = new AtomicLong();

    private static final String UNKNOWN = "unknown";

    private EtEngineStats() {}

    /** Records a newly loaded model. Called from {@link EtModel#load}. */
    static void register(long handle, EtSymbolBlock block) {
        LIVE.put(handle, block);
        MODELS_LOADED.incrementAndGet();
    }

    /**
     * Removes a model and folds its totals into the closed-model rollup. Called from {@link
     * EtSymbolBlock#close()} <b>before</b> the native handle is released, so the counters are still
     * readable. Idempotent: a second close finds nothing to remove.
     */
    static void deregister(long handle) {
        EtSymbolBlock block = LIVE.remove(handle);
        if (block == null) {
            return;
        }
        EtModelStats stats = block.toStats();
        if (stats != null) {
            CLOSED_FORWARD_COUNT.addAndGet(stats.getForwardCount());
            CLOSED_FORWARD_TOTAL_NANOS.addAndGet(stats.getForwardTotalNanos());
        }
    }

    /**
     * Captures the engine's current state.
     *
     * @return an immutable snapshot; never {@code null}, never throws
     */
    public static EtStatsSnapshot snapshot() {
        List<EtModelStats> models = new ArrayList<>(LIVE.size());
        long arena = 0;
        long staging = 0;
        for (EtSymbolBlock block : LIVE.values()) {
            EtModelStats stats = block.toStats();
            if (stats == null) {
                continue; // registered but counters not yet attached; nothing to report
            }
            models.add(stats);
            if (stats.getPlannedArenaBytes() > 0) {
                arena += stats.getPlannedArenaBytes();
            }
            if (stats.getStagingBytes() > 0) {
                staging += stats.getStagingBytes(); // skips -1 so "unavailable" never sums in
            }
        }
        return new EtStatsSnapshot(
                EtEngine.EXECUTORCH_VERSION,
                safePlatform(),
                safeString(LibUtils.loadedPath()),
                safeIntraOpThreads(),
                sharingModeDefault(),
                MODELS_LOADED.get(),
                models.size(),
                arena,
                staging,
                CLOSED_FORWARD_COUNT.get(),
                CLOSED_FORWARD_TOTAL_NANOS.get(),
                Collections.unmodifiableList(models));
    }

    private static String safeString(String value) {
        return (value == null || value.isEmpty()) ? UNKNOWN : value;
    }

    private static String safePlatform() {
        try {
            return LibUtils.platform();
        } catch (RuntimeException e) {
            return UNKNOWN; // unsupported os.arch: reportable, not fatal to a monitoring read
        }
    }

    private static String sharingModeDefault() {
        String value = System.getProperty(EtEngine.WORKSPACE_SHARING_MODE_PROPERTY);
        // "unspecified" is meaningful here and distinct from "unknown": it means no spec is sent
        // and the runtime's compiled-in default (global for our pin) applies.
        return (value == null || value.isEmpty()) ? "unspecified" : value;
    }

    private static int safeIntraOpThreads() {
        try {
            return EtNative.intraOpThreads();
        } catch (RuntimeException | UnsatisfiedLinkError e) {
            return -1; // native library unavailable
        }
    }
}
