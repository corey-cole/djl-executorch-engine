package org.measly.executorch.jni;

/** Static I/O metadata for a loaded module's "forward" method. */
public final class EtMethodMeta {
    public final int numInputs;
    /** Per-input ExecuTorch ScalarType code; {@code -1} for a non-tensor input. Treat as read-only. */
    public final int[] inputScalarTypes;
    /**
     * Per-input: true if the input is memory-planned, i.e. ExecuTorch copies it into its arena at
     * set_input; false means the host pointer is borrowed. {@code false} for non-tensor inputs.
     * Treat as read-only.
     */
    public final boolean[] inputMemoryPlanned;

    public EtMethodMeta(int numInputs, int[] inputScalarTypes, boolean[] inputMemoryPlanned) {
        this.numInputs = numInputs;
        this.inputScalarTypes = inputScalarTypes;
        this.inputMemoryPlanned = inputMemoryPlanned;
    }
}
