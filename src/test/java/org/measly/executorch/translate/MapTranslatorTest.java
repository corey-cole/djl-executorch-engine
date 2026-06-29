package org.measly.executorch.translate;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class MapTranslatorTest {

    private static MapTranslator translator() {
        return new MapTranslator(List.of(
                new ParamSpec("price", 0, DType.FLOAT32, new long[] {1}),
                new ParamSpec("qty", 1, DType.INT64, new long[] {1})));
    }

    @Test
    void buildsInputsInPositionOrderRegardlessOfMapOrder() {
        try (NDManager m = NDManager.newBaseManager("ExecuTorch")) {
            NDList in = translator().buildInputs(m, Map.of("qty", 4, "price", 2.5));
            assertEquals(2, in.size());
            assertArrayEquals(new float[] {2.5f}, in.get(0).toFloatArray()); // price @ pos 0
            assertArrayEquals(new long[] {4L}, in.get(1).toLongArray());     // qty   @ pos 1
        }
    }

    @Test
    void rejectsMissingUnexpectedAndNull() {
        try (NDManager m = NDManager.newBaseManager("ExecuTorch")) {
            assertTrue(assertThrows(IllegalArgumentException.class,
                    () -> translator().buildInputs(m, Map.of("price", 1.0)))
                    .getMessage().contains("qty"));
            assertTrue(assertThrows(IllegalArgumentException.class,
                    () -> translator().buildInputs(m, Map.of("price", 1.0, "qty", 1, "foo", 9)))
                    .getMessage().contains("foo"));
        }
    }

    @Test
    void outputWidensFloat32AndInt64() {
        try (NDManager m = NDManager.newBaseManager("ExecuTorch")) {
            NDArray f = m.create(new float[] {10f}, new Shape(1));
            assertArrayEquals(new double[] {10.0}, MapTranslator.toDoubleArray(f), 0.0);
            NDArray l = m.create(new long[] {7L}, new Shape(1));
            assertArrayEquals(new double[] {7.0}, MapTranslator.toDoubleArray(l), 0.0);
        }
    }

    @Test
    void getBatchifierIsNull() {
        assertNull(translator().getBatchifier());
    }
}
