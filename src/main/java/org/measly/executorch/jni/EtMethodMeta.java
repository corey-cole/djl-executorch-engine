package org.measly.executorch.jni;

/** Static I/O metadata for a loaded module's "forward" method. */
public final class EtMethodMeta {
    /** Number of inputs the {@code forward} method declares. */
    public final int numInputs;
    /** Per-input ExecuTorch ScalarType code; {@code -1} for a non-tensor input. Treat as read-only. */
    public final int[] inputScalarTypes;
    /**
     * Per-input: true if the input is memory-planned, i.e. ExecuTorch copies it into its arena at
     * set_input; false means the host pointer is borrowed. {@code false} for non-tensor inputs.
     * Treat as read-only.
     */
    public final boolean[] inputMemoryPlanned;

    /**
     * ExecuTorch's planned activation arena for {@code forward}, in bytes, captured at load.
     *
     * <p>Excludes the XNNPACK delegate workspace, which cannot be sized from this layer:
     * {@code xnn_workspace_t} is opaque in the shipped {@code xnnpack.h}. Tracked upstream at
     * <a href="https://github.com/measly-java-learning/executorch-runtime-dist/issues/17">
     * executorch-runtime-dist#17</a>. Treat this as an exact
     * lower bound on native footprint, not a total.
     */
    public final long plannedArenaBytes;

    /**
     * @param numInputs number of declared inputs
     * @param inputScalarTypes per-input ScalarType code, {@code -1} for a non-tensor input
     * @param inputMemoryPlanned per-input memory-planned flag
     * @param plannedArenaBytes ExecuTorch's planned activation arena in bytes
     */
    public EtMethodMeta(
            int numInputs,
            int[] inputScalarTypes,
            boolean[] inputMemoryPlanned,
            long plannedArenaBytes) {
        this.numInputs = numInputs;
        this.inputScalarTypes = inputScalarTypes;
        this.inputMemoryPlanned = inputMemoryPlanned;
        this.plannedArenaBytes = plannedArenaBytes;
    }
}
