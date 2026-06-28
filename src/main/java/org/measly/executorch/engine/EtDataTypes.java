package org.measly.executorch.engine;

import ai.djl.ndarray.types.DataType;

/** Maps DJL {@link DataType} to/from ExecuTorch ScalarType integer codes (c10 canonical). */
public final class EtDataTypes {

    private EtDataTypes() {}

    /** @return the ExecuTorch ScalarType code for {@code dataType}. */
    public static int toScalarType(DataType dataType) {
        switch (dataType) {
            case UINT8:   return 0;  // Byte
            case INT8:    return 1;  // Char
            case INT32:   return 3;  // Int
            case INT64:   return 4;  // Long
            case FLOAT32: return 6;  // Float
            case FLOAT64: return 7;  // Double
            case BOOLEAN: return 11; // Bool
            // INT16, FLOAT16, and others are intentionally unsupported in Phase 2a; fail fast.
            default:
                throw new IllegalArgumentException("Unsupported dtype for ExecuTorch: " + dataType);
        }
    }

    /** @return the DJL {@link DataType} for an ExecuTorch ScalarType code. */
    public static DataType fromScalarType(int scalarType) {
        switch (scalarType) {
            case 0:  return DataType.UINT8;
            case 1:  return DataType.INT8;
            case 3:  return DataType.INT32;
            case 4:  return DataType.INT64;
            case 6:  return DataType.FLOAT32;
            case 7:  return DataType.FLOAT64;
            case 11: return DataType.BOOLEAN;
            // Codes 2 (Short/INT16), 5 (Half/FLOAT16), etc. are intentionally unsupported in Phase 2a.
            default:
                throw new IllegalArgumentException("Unsupported ExecuTorch ScalarType code: " + scalarType);
        }
    }
}
