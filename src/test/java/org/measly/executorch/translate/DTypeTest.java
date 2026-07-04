package org.measly.executorch.translate;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import org.junit.jupiter.api.Test;

class DTypeTest {
    @Test
    void fromMapsNamesAndAliases() {
        assertEquals(DType.FLOAT32, DType.from("float32"));
        assertEquals(DType.FLOAT32, DType.from("torch.float32"));
        assertEquals(DType.FLOAT64, DType.from("float64"));
        assertEquals(DType.FLOAT64, DType.from("torch.float64"));
        assertEquals(DType.INT32, DType.from("int32"));
        assertEquals(DType.INT32, DType.from("torch.int32"));
        assertEquals(DType.INT64, DType.from("int64"));
        assertEquals(DType.INT64, DType.from("torch.int64"));
        assertThrows(IllegalArgumentException.class, () -> DType.from("bfloat16"));
    }

    @Test
    void intTargetsRejectFractionalAndOverflow() {
        try (NDManager m = NDManager.newBaseManager("ExecuTorch")) {
            assertThrows(IllegalArgumentException.class,
                    () -> DType.INT32.createScalar(m, 0.7, new long[] {1}));
            assertThrows(IllegalArgumentException.class,
                    () -> DType.INT32.createScalar(m, 3_000_000_000L, new long[] {1}));
        }
    }

    @Test
    void intTargetsAcceptIntegral() {
        try (NDManager m = NDManager.newBaseManager("ExecuTorch")) {
            NDArray a = DType.INT64.createScalar(m, 12.0, new long[] {1});
            assertEquals(DataType.INT64, a.getDataType());
            assertArrayEquals(new long[] {12L}, a.toLongArray());
        }
    }

    @Test
    void floatTargetsWidenAnyNumber() {
        try (NDManager m = NDManager.newBaseManager("ExecuTorch")) {
            NDArray a = DType.FLOAT32.createScalar(m, 2, new long[] {1}); // int -> float
            assertEquals(DataType.FLOAT32, a.getDataType());
            assertArrayEquals(new float[] {2f}, a.toFloatArray());
        }
    }

    @Test
    void float64CreatesDoubleScalar() {
        try (NDManager m = NDManager.newBaseManager("ExecuTorch")) {
            NDArray a = DType.FLOAT64.createScalar(m, 1.5, new long[] {1});
            assertEquals(DataType.FLOAT64, a.getDataType());
            assertArrayEquals(new double[] {1.5}, a.toDoubleArray());
        }
    }

    @Test
    void int32CreatesIntScalar() {
        try (NDManager m = NDManager.newBaseManager("ExecuTorch")) {
            NDArray a = DType.INT32.createScalar(m, 42, new long[] {1});
            assertEquals(DataType.INT32, a.getDataType());
            assertArrayEquals(new int[] {42}, a.toIntArray());
        }
    }

    @Test
    void integralNumbersSkipFractionalCheck() {
        // Integer/Long are not Float/Double, so requireIntegral bypasses the NaN/rint branch entirely.
        try (NDManager m = NDManager.newBaseManager("ExecuTorch")) {
            NDArray a = DType.INT64.createScalar(m, Long.valueOf(9L), new long[] {1});
            assertArrayEquals(new long[] {9L}, a.toLongArray());
        }
    }

    @Test
    void int64RejectsOutOfRangeDouble() {
        try (ai.djl.ndarray.NDManager m = ai.djl.ndarray.NDManager.newBaseManager("ExecuTorch")) {
            assertThrows(IllegalArgumentException.class,
                    () -> DType.INT64.createScalar(m, 1e19, new long[] {1}));
        }
    }
}
