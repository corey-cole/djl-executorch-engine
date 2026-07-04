package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.Test;

class EtNDArrayTest {
    @Test
    void floatRoundTrip() {
        try (NDManager manager = NDManager.newBaseManager("ExecuTorch")) {
            NDArray arr = manager.create(new float[] {1f, 2f, 3f}, new Shape(3));
            assertEquals(new Shape(3), arr.getShape());
            assertEquals(DataType.FLOAT32, arr.getDataType());
            assertArrayEquals(new float[] {1f, 2f, 3f}, arr.toFloatArray());
        }
    }

    @Test
    void int64RoundTrip() {
        try (NDManager manager = NDManager.newBaseManager("ExecuTorch")) {
            NDArray arr = manager.create(new long[] {1L, 2L, 3L}, new Shape(3));
            assertEquals(DataType.INT64, arr.getDataType());
            assertArrayEquals(new long[] {1L, 2L, 3L}, arr.toLongArray());
        }
    }

    @Test
    void int32RoundTrip() {
        try (NDManager manager = NDManager.newBaseManager("ExecuTorch")) {
            NDArray arr = manager.create(new int[] {4, 5}, new Shape(2));
            assertEquals(DataType.INT32, arr.getDataType());
            assertArrayEquals(new int[] {4, 5}, arr.toIntArray());
        }
    }

    @Test
    void float64RoundTrip() {
        try (NDManager manager = NDManager.newBaseManager("ExecuTorch")) {
            NDArray arr = manager.create(new double[] {1.5, 2.5}, new Shape(2));
            assertEquals(DataType.FLOAT64, arr.getDataType());
            assertArrayEquals(new double[] {1.5, 2.5}, arr.toDoubleArray());
        }
    }

    @Test
    void useAfterCloseThrows() {
        try (NDManager manager = NDManager.newBaseManager("ExecuTorch")) {
            NDArray arr = manager.create(new float[] {1f}, new Shape(1));
            arr.close();
            assertThrows(IllegalStateException.class, arr::getDataType);
            assertThrows(IllegalStateException.class, arr::getShape);
            assertThrows(IllegalStateException.class, () -> arr.toByteBuffer(true));
        }
    }

    @Test
    void internIsNotSupported() {
        try (NDManager manager = NDManager.newBaseManager("ExecuTorch")) {
            NDArray arr = manager.create(new float[] {1f}, new Shape(1));
            assertThrows(UnsupportedOperationException.class, () -> arr.intern(arr));
        }
    }

    @Test
    void detachRehomesToSystemManager() {
        try (NDManager manager = NDManager.newBaseManager("ExecuTorch")) {
            NDArray arr = manager.create(new float[] {1f}, new Shape(1));
            arr.detach();
            assertEquals(EtNDManager.getSystemManager(), arr.getManager());
        }
    }
}
