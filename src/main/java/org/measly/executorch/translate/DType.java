package org.measly.executorch.translate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/** Scalar parameter dtype: maps a Number to a typed scalar NDArray with strict coercion. */
public enum DType {
    /** IEEE-754 single precision. */
    FLOAT32 {
        @Override
        NDArray createScalar(NDManager m, Number v, long[] shape) {
            return m.create(new float[] {v.floatValue()}, new Shape(shape));
        }
    },
    /** IEEE-754 double precision. */
    FLOAT64 {
        @Override
        NDArray createScalar(NDManager m, Number v, long[] shape) {
            return m.create(new double[] {v.doubleValue()}, new Shape(shape));
        }
    },
    /** 32-bit signed integer; a value outside its range is rejected rather than truncated. */
    INT32 {
        @Override
        NDArray createScalar(NDManager m, Number v, long[] shape) {
            long l = requireIntegral(v);
            if (l < Integer.MIN_VALUE || l > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Value " + v + " does not fit in int32");
            }
            return m.create(new int[] {(int) l}, new Shape(shape));
        }
    },
    /** 64-bit signed integer. */
    INT64 {
        @Override
        NDArray createScalar(NDManager m, Number v, long[] shape) {
            return m.create(new long[] {requireIntegral(v)}, new Shape(shape));
        }
    };

    /** Builds a scalar NDArray of {@code shape} from {@code v}. Package-private. */
    abstract NDArray createScalar(NDManager m, Number v, long[] shape);

    private static final double INT64_LIMIT = 0x1p63;

    /** @throws IllegalArgumentException if v has a fractional part or is NaN/Inf. */
    static long requireIntegral(Number v) {
        if (v instanceof Float || v instanceof Double) {
            double d = v.doubleValue();
            if (Double.isNaN(d) || Double.isInfinite(d) || d != Math.rint(d)) {
                throw new IllegalArgumentException("Value " + v + " is not an integer");
            }
            // Long.MAX_VALUE rounds up to 2^63 as a double (out of long range); Long.MIN_VALUE (-2^63) is exact.
            if (d >= INT64_LIMIT || d < -INT64_LIMIT) {
                throw new IllegalArgumentException("Value " + v + " does not fit in int64");
            }
        }
        return v.longValue();
    }

    /**
     * Maps a dtype name to its constant.
     *
     * @param name one of {@code float32}, {@code float64}, {@code int32}, {@code int64}, with or
     *     without a {@code torch.} prefix
     * @return the matching constant
     * @throws IllegalArgumentException if the name is not recognised
     */
    public static DType from(String name) {
        switch (name) {
            case "float32":
            case "torch.float32":
                return FLOAT32;
            case "float64":
            case "torch.float64":
                return FLOAT64;
            case "int32":
            case "torch.int32":
                return INT32;
            case "int64":
            case "torch.int64":
                return INT64;
            default:
                throw new IllegalArgumentException("Unsupported dtype: " + name);
        }
    }
}
