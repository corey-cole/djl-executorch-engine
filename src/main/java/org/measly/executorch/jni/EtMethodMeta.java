package org.measly.executorch.jni;

/** Static I/O metadata for a loaded module's "forward" method. */
public final class EtMethodMeta {
    public final int numInputs;
    /** Per-input ExecuTorch ScalarType code; {@code -1} for a non-tensor input. Treat as read-only. */
    public final int[] inputScalarTypes;

    public EtMethodMeta(int numInputs, int[] inputScalarTypes) {
        this.numInputs = numInputs;
        this.inputScalarTypes = inputScalarTypes;
    }
}
