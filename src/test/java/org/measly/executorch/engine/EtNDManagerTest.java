package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.Test;


public class EtNDManagerTest {
    
    @Test
    void createInvalidDateTypeThrows() {
        try (NDManager manager = NDManager.newBaseManager("ExecuTorch")) {
            ByteBuffer buffer = ByteBuffer.wrap("Hello".getBytes());
            assertThrows(IllegalArgumentException.class, () -> manager.create(buffer, new Shape(5), DataType.STRING));
        }
    }

    // Obviously this would need to be updated if we ever add support for INT16.
    @Test
    void createUnsupportedNumericTypeThrows() {
        try (NDManager manager = NDManager.newBaseManager("ExecuTorch")) {
            IntBuffer buffer = IntBuffer.wrap(new int[] {1, 2});
            assertThrows(UnsupportedOperationException.class, () -> manager.create(buffer, new Shape(2), DataType.INT16));
        }
    }

    @Test
    void fromReturnsNullForNull() {
        try (NDManager manager = NDManager.newBaseManager("ExecuTorch")) {
            assertNull(manager.from(null));
        }
    }

    @Test
    void fromReturnsSameInstanceForEtNDArray() {
        try (NDManager manager = NDManager.newBaseManager("ExecuTorch")) {
            NDArray arr = manager.create(new float[] {1f}, new Shape(1));
            assertSame(arr, manager.from(arr));
        }
    }

    @Test
    void fromCopiesForeignNDArray() {
        try (NDManager manager = NDManager.newBaseManager("ExecuTorch")) {
            ByteBuffer bytes = ByteBuffer.allocate(8).order(ByteOrder.nativeOrder());
            bytes.asFloatBuffer().put(new float[] {1f, 2f});

            NDArray foreign = mock(NDArray.class);
            when(foreign.toByteBuffer()).thenReturn(bytes);
            when(foreign.getShape()).thenReturn(new Shape(2));
            when(foreign.getDataType()).thenReturn(DataType.FLOAT32);
            when(foreign.getName()).thenReturn("foreign");

            NDArray copy = manager.from(foreign);
            assertNotSame(foreign, copy);
            assertInstanceOf(EtNDArray.class, copy);
            assertEquals(DataType.FLOAT32, copy.getDataType());
            assertEquals("foreign", copy.getName());
            assertArrayEquals(new float[] {1f, 2f}, copy.toFloatArray());
        }
    }
}
