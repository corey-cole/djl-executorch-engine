package org.measly.executorch.engine;

/**
 * An immutable point-in-time view of one loaded model's counters and native footprint.
 *
 * <p>Getters follow JavaBean naming because this type is exposed through an MXBean, which converts
 * it to {@code CompositeData} automatically. Do not add setters.
 *
 * <p><b>Byte fields use {@code -1} for "unavailable" and {@code 0} for "genuinely zero".</b> The
 * distinction matters: {@link #getStagingBytes()} is legitimately {@code 0} for a memory-planned
 * model, which is every model exported with ExecuTorch's defaults.
 */
public final class EtModelStats {

    private final String name;
    private final String workspaceSharingMode;
    private final long plannedArenaBytes;
    private final long stagingBytes;
    private final long loadNanos;
    private final long forwardCount;
    private final long forwardTotalNanos;
    private final long forwardMaxNanos;

    EtModelStats(
            String name,
            String workspaceSharingMode,
            long plannedArenaBytes,
            long stagingBytes,
            long loadNanos,
            long forwardCount,
            long forwardTotalNanos,
            long forwardMaxNanos) {
        this.name = name;
        this.workspaceSharingMode = workspaceSharingMode;
        this.plannedArenaBytes = plannedArenaBytes;
        this.stagingBytes = stagingBytes;
        this.loadNanos = loadNanos;
        this.forwardCount = forwardCount;
        this.forwardTotalNanos = forwardTotalNanos;
        this.forwardMaxNanos = forwardMaxNanos;
    }

    /** @return the DJL model name */
    public String getName() {
        return name;
    }

    /** @return the effective XNNPACK workspace sharing mode, or {@code unspecified} */
    public String getWorkspaceSharingMode() {
        return workspaceSharingMode;
    }

    /**
     * @return ExecuTorch's planned activation arena in bytes. Excludes the XNNPACK delegate
     *     workspace, which cannot be sized from this layer: {@code xnn_workspace_t} is opaque in
     *     the shipped {@code xnnpack.h}. Tracked upstream at
     *     <a href="https://github.com/measly-java-learning/executorch-runtime-dist/issues/17">
     *     executorch-runtime-dist#17</a>.
     */
    public long getPlannedArenaBytes() {
        return plannedArenaBytes;
    }

    /**
     * @return bytes held by the engine's input staging slots; {@code 0} when every input is
     *     memory-planned (the export default), {@code -1} if the model is closed
     */
    public long getStagingBytes() {
        return stagingBytes;
    }

    /** @return wall time spent loading this model, including delegate initialisation */
    public long getLoadNanos() {
        return loadNanos;
    }

    /** @return completed forward calls */
    public long getForwardCount() {
        return forwardCount;
    }

    /** @return summed wall time of all forward calls; divide by count for the mean */
    public long getForwardTotalNanos() {
        return forwardTotalNanos;
    }

    /** @return the slowest single forward observed */
    public long getForwardMaxNanos() {
        return forwardMaxNanos;
    }
}
