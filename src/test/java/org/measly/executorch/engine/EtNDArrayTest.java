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
    void useAfterCloseThrows() {
        try (NDManager manager = NDManager.newBaseManager("ExecuTorch")) {
            NDArray arr = manager.create(new float[] {1f}, new Shape(1));
            arr.close();
            assertThrows(IllegalStateException.class, arr::getDataType);
            assertThrows(IllegalStateException.class, arr::getShape);
            assertThrows(IllegalStateException.class, () -> arr.toByteBuffer(true));
        }
    }
}
