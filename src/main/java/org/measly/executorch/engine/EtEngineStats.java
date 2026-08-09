package org.measly.executorch.engine;

import java.lang.management.ManagementFactory;
import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
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
 * <p>Per-model detail covers live models only. A model's totals fold into the rollup when it stops
 * being tracked — whether it was closed properly or dropped without {@code close()} and later
 * garbage-collected — so a restart-on-error loop cannot erase throughput history.
 *
 * <p>The registry holds models weakly, so it never keeps a leaked model alive on its own account.
 * Note the corollary for {@code modelsLive}: a model that was dropped without {@code close()} is
 * still counted until the GC actually reclaims it, so the figure is an upper bound rather than an
 * instantaneous truth. It is not a substitute for closing models.
 */
public final class EtEngineStats {

    private static final ReferenceQueue<EtSymbolBlock> REAPED = new ReferenceQueue<>();
    private static final Map<Long, ModelRef> LIVE = new ConcurrentHashMap<>();
    private static final AtomicLong MODELS_LOADED = new AtomicLong();
    private static final AtomicLong CLOSED_FORWARD_COUNT = new AtomicLong();
    private static final AtomicLong CLOSED_FORWARD_TOTAL_NANOS = new AtomicLong();

    /**
     * A registry entry: the block <b>weakly</b>, its counters <b>strongly</b>.
     *
     * <p>Weak on the block because this map is static and lives for the JVM. A caller who drops a
     * model without closing it already leaks the native handle; a strong entry here would also pin
     * {@link EtSymbolBlock} — and through it the model's {@link EtNDManager} — for the life of the
     * process, turning a native-only leak into a permanent heap one and making {@code modelsLive}
     * climb forever. (It does not make a leaked model free: DJL's {@code BaseNDManager} attaches
     * every base manager to a static system manager, so the manager stays reachable regardless.
     * That retention is DJL's and predates this class.)
     *
     * <p>Strong on the counters because they are the only thing worth keeping: a few longs and two
     * string references, independent of the block's object graph. Holding them lets a collected
     * model's forwards still reach the rollup, which a bare {@code WeakReference} would lose —
     * exactly the history the rollup exists to preserve.
     */
    private static final class ModelRef extends WeakReference<EtSymbolBlock> {
        final long handle;
        final EtModelCounters counters;

        ModelRef(EtSymbolBlock block, long handle, EtModelCounters counters) {
            super(block, REAPED);
            this.handle = handle;
            this.counters = counters;
        }
    }

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
    static void register(long handle, EtSymbolBlock block, EtModelCounters counters) {
        purgeCollected(); // bounded by what the GC has reaped since the last call, usually nothing
        LIVE.put(handle, new ModelRef(block, handle, counters));
        MODELS_LOADED.incrementAndGet();
    }

    /**
     * Removes a model and folds its totals into the rollup. Called from {@link
     * EtSymbolBlock#close()}. Idempotent: a second close finds nothing to remove.
     */
    static void deregister(long handle) {
        ModelRef ref = LIVE.remove(handle);
        if (ref != null) {
            foldIntoRollup(ref.counters);
        }
    }

    /**
     * Folds every model the GC has reclaimed since the last call into the rollup and drops its
     * entry. Called from {@link #snapshot()} and {@link #register}, so the map self-heals on any
     * activity; draining costs O(models collected), not O(models tracked).
     */
    private static void purgeCollected() {
        for (Reference<? extends EtSymbolBlock> reaped = REAPED.poll();
                reaped != null;
                reaped = REAPED.poll()) {
            ModelRef ref = (ModelRef) reaped;
            // Two-argument remove, deliberately. If this handle was already deregistered by
            // close() and the allocator has since handed the same address to a new model, the
            // current mapping is a different ModelRef — removing by key alone would evict the live
            // model and double-count this one. The compare-and-remove makes both impossible.
            if (LIVE.remove(ref.handle, ref)) {
                foldIntoRollup(ref.counters);
            }
        }
    }

    private static void foldIntoRollup(EtModelCounters counters) {
        CLOSED_FORWARD_COUNT.addAndGet(counters.forwardCount());
        CLOSED_FORWARD_TOTAL_NANOS.addAndGet(counters.forwardTotalNanos());
    }

    /**
     * Captures the engine's current state.
     *
     * @return an immutable snapshot; never {@code null}, never throws
     */
    public static EtStatsSnapshot snapshot() {
        purgeCollected();
        List<EtModelStats> models = new ArrayList<>(LIVE.size());
        long arena = 0;
        long staging = 0;
        for (ModelRef ref : LIVE.values()) {
            EtSymbolBlock block = ref.get();
            if (block == null) {
                continue; // collected between the purge above and here; the next poll folds it in
            }
            EtModelStats stats = block.toStats();
            if (stats == null) {
                // Defensive only, and not reachable through EtModel.load: attachCounters() always
                // precedes register(), and register() now requires the counters outright. Kept so
                // a block registered by some future path without counters degrades to "absent from
                // the list" rather than a NullPointerException out of a monitoring poll.
                continue;
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
