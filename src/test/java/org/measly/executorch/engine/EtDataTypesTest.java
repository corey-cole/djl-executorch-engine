package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import ai.djl.ndarray.types.DataType;
import org.junit.jupiter.api.Test;

class EtDataTypesTest {
    @Test
    void roundTripsSupportedTypes() {
        DataType[] types = {
            DataType.FLOAT32, DataType.FLOAT64, DataType.INT32,
            DataType.INT64, DataType.INT8, DataType.UINT8, DataType.BOOLEAN
        };
        for (DataType dt : types) {
            int code = EtDataTypes.toScalarType(dt);
            assertEquals(dt, EtDataTypes.fromScalarType(code), "round trip " + dt);
        }
    }

    @Test
    void knownCodes() {
        assertEquals(6, EtDataTypes.toScalarType(DataType.FLOAT32));
        assertEquals(4, EtDataTypes.toScalarType(DataType.INT64));
        assertEquals(0, EtDataTypes.toScalarType(DataType.UINT8));
        assertEquals(11, EtDataTypes.toScalarType(DataType.BOOLEAN));
    }

    @Test
    void knownCodesReverse() {
        assertEquals(DataType.FLOAT32, EtDataTypes.fromScalarType(6));
        assertEquals(DataType.INT64, EtDataTypes.fromScalarType(4));
        assertEquals(DataType.UINT8, EtDataTypes.fromScalarType(0));
        assertEquals(DataType.BOOLEAN, EtDataTypes.fromScalarType(11));
    }

    @Test
    void rejectsUnsupported() {
        assertThrows(IllegalArgumentException.class,
                () -> EtDataTypes.toScalarType(DataType.FLOAT16));
        assertThrows(IllegalArgumentException.class,
                () -> EtDataTypes.fromScalarType(2)); // Short/INT16
    }
}
