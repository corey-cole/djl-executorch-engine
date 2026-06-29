package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;

class MultiDtypeIT {
    @Test
    void int64AndFloat32ThroughPredictor() throws Exception {
        TestSupport.assumeDtypesModelAvailable();
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optEngine("ExecuTorch")
                        .optModelPath(Paths.get("native/spike"))
                        .optModelName("dtypes")
                        .optTranslator(new PassthroughTranslator())
                        .build();
        try (ZooModel<NDList, NDList> model = criteria.loadModel();
                Predictor<NDList, NDList> predictor = model.newPredictor();
                NDManager m = model.getNDManager().newSubManager()) {
            NDArray a = m.create(new long[] {7L}, new Shape(1));   // int64
            NDArray b = m.create(new float[] {2.5f}, new Shape(1)); // float32
            NDList out = predictor.predict(new NDList(a, b));
            assertEquals(2, out.size());
            assertEquals(DataType.INT64, out.get(0).getDataType());
            assertArrayEquals(new long[] {14L}, out.get(0).toLongArray());   // a + a
            assertEquals(DataType.FLOAT32, out.get(1).getDataType());
            assertArrayEquals(new float[] {5.0f}, out.get(1).toFloatArray()); // b + b
        }
    }
}
