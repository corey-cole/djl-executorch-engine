package org.measly.executorch.translate;

/**
 * One named scalar input: name, input position, dtype, and (scalar) shape.
 *
 * <p>Note: {@code shape} is an array, so the record's generated {@code equals}/{@code hashCode}
 * compare it by reference, not structurally. Compare {@code ParamSpec}s field-by-field rather than
 * with {@code equals} unless that is later changed.
 */
public record ParamSpec(String name, int position, DType dtype, long[] shape) {}
