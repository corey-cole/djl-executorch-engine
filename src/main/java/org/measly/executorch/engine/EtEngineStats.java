package org.measly.executorch.engine;

import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import javax.management.MBeanServer;
import javax.management.ObjectName;
import org.measly.executorch.jni.EtNative;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

    /** The JMX object name this engine registers under. */
    public static final String OBJECT_NAME = "org.measly.executorch:type=EtEngineStats";

    private static final Logger logger = LoggerFactory.getLogger(EtEngineStats.class);
    // Guards the one-shot auto-registration attempt. A failed attempt is not retried: a
    // per-load retry would log on every load and re-run a failure we already reported.
    private static final AtomicBoolean JMX_ATTEMPTED = new AtomicBoolean();

    /** MXBean implementation. Separate from the static facade because an MXBean needs an instance. */
    private static final class Bean implements EtEngineStatsMXBean {
        @Override
        public EtStatsSnapshot getSnapshot() {
            return EtEngineStats.snapshot();
        }
    }

    /**
     * Registers the JMX MBean under {@value #OBJECT_NAME} on the platform MBean server.
     *
     * <p>Idempotent: registering an already-registered name is a no-op. Any JMX failure is logged
     * and swallowed — a monitoring surface must never be the thing that breaks the application.
     */
    public static void registerMBean() {
        try {
            MBeanServer server = ManagementFactory.getPlatformMBeanServer();
            ObjectName objectName = new ObjectName(OBJECT_NAME);
            if (server.isRegistered(objectName)) {
                return;
            }
            server.registerMBean(new Bean(), objectName);
            logger.info("registered JMX MBean {}", OBJECT_NAME);
        } catch (Exception e) {
            logger.warn(
                    "could not register JMX MBean {} ({}); set {}=false to silence this",
                    OBJECT_NAME,
                    e.toString(),
                    EtEngine.JMX_ENABLED_PROPERTY);
        }
    }

    /** Removes the JMX MBean if present. Safe to call when it was never registered. */
    public static void unregisterMBean() {
        try {
            MBeanServer server = ManagementFactory.getPlatformMBeanServer();
            ObjectName objectName = new ObjectName(OBJECT_NAME);
            if (server.isRegistered(objectName)) {
                server.unregisterMBean(objectName);
            }
        } catch (Exception e) {
            logger.warn("could not unregister JMX MBean {} ({})", OBJECT_NAME, e.toString());
        }
    }

    /**
     * One-shot auto-registration, driven by the first model load. Honours {@link
     * EtEngine#JMX_ENABLED_PROPERTY}; only the exact value {@code false} disables it.
     */
    static void registerMBeanOnce() {
        if (!JMX_ATTEMPTED.compareAndSet(false, true)) {
            return;
        }
        if ("false".equalsIgnoreCase(System.getProperty(EtEngine.JMX_ENABLED_PROPERTY))) {
            logger.info("JMX MBean disabled by {}=false", EtEngine.JMX_ENABLED_PROPERTY);
            return;
        }
        registerMBean();
    }

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
        String value;
        try {
            value = System.getProperty(EtEngine.WORKSPACE_SHARING_MODE_PROPERTY);
        } catch (SecurityException e) {
            // Unreadable under a restrictive SecurityManager: report the spec's convention for an
            // unresolvable config field rather than throwing out of a monitoring read.
            return UNKNOWN;
        }
        // "unspecified" is meaningful here and distinct from "unknown": it means no spec is sent
        // and the runtime's compiled-in default (global for our pin) applies.
        return (value == null || value.isEmpty()) ? "unspecified" : value;
    }

    private static int safeIntraOpThreads() {
        try {
            return EtNative.intraOpThreads();
        } catch (RuntimeException | LinkageError e) {
            // LinkageError (not just UnsatisfiedLinkError) because a failed EtNative class init —
            // e.g. System.load throwing inside <clinit> — surfaces as ExceptionInInitializerError
            // on first access and NoClassDefFoundError afterwards. A broken or absent library must
            // degrade this read, never throw out of snapshot().
            return -1; // native library unavailable
        }
    }
}
