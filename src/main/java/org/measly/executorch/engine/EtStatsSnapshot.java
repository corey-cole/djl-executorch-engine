package org.measly.executorch.engine;

import java.util.List;

/**
 * An immutable point-in-time view of the ExecuTorch engine: its effective configuration,
 * process-wide totals, and per-model detail for every live model.
 *
 * <p>Obtained from {@link EtEngineStats#snapshot()}. Getters follow JavaBean naming because this
 * type is exposed through an MXBean.
 */
public final class EtStatsSnapshot {

    private final String executorchVersion;
    private final String platform;
    private final String nativeLibraryPath;
    private final int intraOpThreads;
    private final String defaultWorkspaceSharingMode;
    private final long modelsLoaded;
    private final long modelsLive;
    private final long totalPlannedArenaBytes;
    private final long totalStagingBytes;
    private final long xnnpackWorkspaceBytes;
    private final long closedForwardCount;
    private final long closedForwardTotalNanos;
    private final List<EtModelStats> models;

    EtStatsSnapshot(
            String executorchVersion,
            String platform,
            String nativeLibraryPath,
            int intraOpThreads,
            String defaultWorkspaceSharingMode,
            long modelsLoaded,
            long modelsLive,
            long totalPlannedArenaBytes,
            long totalStagingBytes,
            long xnnpackWorkspaceBytes,
            long closedForwardCount,
            long closedForwardTotalNanos,
            List<EtModelStats> models) {
        this.executorchVersion = executorchVersion;
        this.platform = platform;
        this.nativeLibraryPath = nativeLibraryPath;
        this.intraOpThreads = intraOpThreads;
        this.defaultWorkspaceSharingMode = defaultWorkspaceSharingMode;
        this.modelsLoaded = modelsLoaded;
        this.modelsLive = modelsLive;
        this.totalPlannedArenaBytes = totalPlannedArenaBytes;
        this.totalStagingBytes = totalStagingBytes;
        this.xnnpackWorkspaceBytes = xnnpackWorkspaceBytes;
        this.closedForwardCount = closedForwardCount;
        this.closedForwardTotalNanos = closedForwardTotalNanos;
        this.models = models;
    }

    /** @return the pinned ExecuTorch runtime version */
    public String getExecutorchVersion() {
        return executorchVersion;
    }

    /** @return the resolved platform, e.g. {@code linux-x86_64}, or {@code unknown} */
    public String getPlatform() {
        return platform;
    }

    /**
     * @return the native library file actually loaded, or {@code unknown}. Distinguishes an
     *     {@code EXECUTORCH_LIBRARY_PATH} override from a classpath extraction.
     */
    public String getNativeLibraryPath() {
        return nativeLibraryPath;
    }

    /** @return the effective intra-op pool size as reported by the native pool */
    public int getIntraOpThreads() {
        return intraOpThreads;
    }

    /**
     * @return the JVM-wide default workspace sharing mode, or {@code unspecified} when no default
     *     is set and the runtime's compiled-in default ({@code global} for our pin) applies
     */
    public String getDefaultWorkspaceSharingMode() {
        return defaultWorkspaceSharingMode;
    }

    /** @return models loaded since JVM start, cumulative; never decreases */
    public long getModelsLoaded() {
        return modelsLoaded;
    }

    /** @return models currently loaded and not yet closed */
    public long getModelsLive() {
        return modelsLive;
    }

    /** @return summed planned activation arenas of all live models; excludes delegate workspace */
    public long getTotalPlannedArenaBytes() {
        return totalPlannedArenaBytes;
    }

    /** @return summed staging bytes of all live models; {@code 0} when all inputs are planned */
    public long getTotalStagingBytes() {
        return totalStagingBytes;
    }

    /**
     * Size in bytes of the XNNPACK delegate's memory arena, or {@code -1} when unavailable.
     *
     * <p>Unlike every other byte field here this is <b>not</b> a sum over live models: the pinned
     * runtime sets {@code EXECUTORCH_XNNPACK_SHARED_WORKSPACE=ON}, so one arena backs every
     * delegate in the process and this figure is already the total. There is deliberately no
     * per-model counterpart — the runtime offers no basis for attributing a shared arena, and a
     * per-model field would report a number that looked attributable and was not.
     *
     * <p>It is a <b>high-water mark</b> including allocator alignment padding: the arena grows on
     * demand and is never shrunk, so it is the peak rather than the live footprint, and it
     * slightly over-states actual tensor bytes.
     *
     * <p>{@code 0} is a real answer, not a failure. It stands until an XNNPACK-delegated method
     * <b>executes</b> — loading one does not grow the arena, and a delegated model that allocates
     * nothing never will (an elementwise add and a Linear both measure 0 on the pinned runtime).
     * Genuine unavailability is {@code -1}, following the same convention as
     * {@link EtModelStats#getStagingBytes()}.
     *
     * @return arena bytes, {@code 0} when nothing has allocated yet, or {@code -1} if unreadable
     */
    public long getXnnpackWorkspaceBytes() {
        return xnnpackWorkspaceBytes;
    }

    /** @return forwards completed by models that have since been closed */
    public long getClosedForwardCount() {
        return closedForwardCount;
    }

    /** @return summed forward wall time of models that have since been closed */
    public long getClosedForwardTotalNanos() {
        return closedForwardTotalNanos;
    }

    /** @return per-model detail for live models only; an unmodifiable, independent copy */
    public List<EtModelStats> getModels() {
        return models;
    }
}
