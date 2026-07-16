package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;

/**
 * End-to-end golden-vector test for the etnp::lstm custom op through the real DJL path.
 * Mirrors executorch-runtime-dist extras/lstm/test/lstm_runner.cpp: read the flat LE-float32
 * blob, slice into (x, h0, c0), forward, compare output to the eager-nn.LSTM golden within
 * allclose(rtol=1e-4, atol=1e-4). Loading lstm.pte is itself the proof the op is linked:
 * if it were not whole-archived into the shim, program load/execution throws "kernel
 * 'etnp::lstm.out' not found".
 */
class LstmModelIT {

    @Test
    void lstmGoldenVectorThroughPredictor() throws Exception {
        TestSupport.assumeLstmModelAvailable();

        Map<String, Integer> dims = readShape("/lstm/shape");
        int t = dims.get("T");
        int b = dims.get("B");
        int i = dims.get("I");
        int h = dims.get("H");

        float[] in = readFloats("/lstm/in.bin");
        float[] expected = readFloats("/lstm/out.bin");

        int nx = t * b * i;
        int nh = b * h;
        float[] x = Arrays.copyOfRange(in, 0, nx);
        float[] h0 = Arrays.copyOfRange(in, nx, nx + nh);
        float[] c0 = Arrays.copyOfRange(in, nx + nh, nx + 2 * nh);

        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optEngine("ExecuTorch")
                        .optModelPath(Paths.get("src/test/resources/lstm"))
                        .optModelName("lstm")
                        .optTranslator(new PassthroughTranslator())
                        .build();

        try (ZooModel<NDList, NDList> model = criteria.loadModel();
                Predictor<NDList, NDList> predictor = model.newPredictor();
                NDManager m = model.getNDManager().newSubManager()) {
            NDArray ax = m.create(x, new Shape(t, b, i));
            NDArray ah0 = m.create(h0, new Shape(b, h));
            NDArray ac0 = m.create(c0, new Shape(b, h));

            // etnp::lstm follows the nn.LSTM contract and returns (y, h_n, c_n); the golden vector
            // (out.bin) is the y sequence, so compare out.get(0) — mirroring the upstream
            // lstm_runner.cpp, which takes res.get()[0]. Feeding exactly 3 inputs also asserts arity:
            // EtSymbolBlock throws if the input count != the model's numInputs.
            // try-with-resources on the NDLists deterministically closes every NDArray they hold
            // (the inputs and the predictor's outputs) at the end of the test.
            try (NDList inputs = new NDList(ax, ah0, ac0);
                    NDList out = predictor.predict(inputs)) {
                NDArray y = out.get(0);
                assertEquals(new Shape(t, b, h), y.getShape(), "first output y has shape [T,B,H]");

                float[] got = y.toFloatArray();
                assertEquals(expected.length, got.length);
                for (int k = 0; k < expected.length; k++) {
                    double tol = 1e-4 + 1e-4 * Math.abs(expected[k]);
                    assertTrue(
                            Math.abs(got[k] - expected[k]) <= tol,
                            "element " + k + ": got=" + got[k] + " expected=" + expected[k]
                                    + " tol=" + tol);
                }
            }
        }
    }

    private static float[] readFloats(String resource) throws IOException {
        try (InputStream is = LstmModelIT.class.getResourceAsStream(resource)) {
            if (is == null) {
                throw new IllegalStateException("fixture not on classpath: " + resource);
            }
            byte[] bytes = is.readAllBytes();
            FloatBuffer fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            float[] out = new float[fb.remaining()];
            fb.get(out);
            return out;
        }
    }

    private static Map<String, Integer> readShape(String resource) throws IOException {
        try (InputStream is = LstmModelIT.class.getResourceAsStream(resource)) {
            if (is == null) {
                throw new IllegalStateException("fixture not on classpath: " + resource);
            }
            String text = new String(is.readAllBytes(), StandardCharsets.UTF_8);
            Map<String, Integer> dims = new HashMap<>();
            for (String line : text.split("\\R")) {
                if (line.isBlank()) {
                    continue;
                }
                String[] kv = line.split("=", 2); // e.g. LSTM_T=5
                dims.put(kv[0].substring("LSTM_".length()).trim(), Integer.parseInt(kv[1].trim()));
            }
            return dims;
        }
    }
}
