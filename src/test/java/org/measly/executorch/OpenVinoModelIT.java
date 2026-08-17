package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.measly.executorch.engine.EtEngine;

@Tag("openvino")
class OpenVinoModelIT {

    private static final Path DIR = Paths.get("src/test/resources/models/openvino");

    private static float[] readFloats(Path p) throws Exception {
        ByteBuffer bb = ByteBuffer.wrap(Files.readAllBytes(p)).order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[bb.remaining() / Float.BYTES];
        bb.asFloatBuffer().get(out);
        return out;
    }

    @Test
    void delegatedModelMatchesTheEagerGolden() throws Exception {
        TestSupport.assumeOpenVinoBundleAvailable();

        float[] in = readFloats(DIR.resolve("in.bin"));
        float[] golden = readFloats(DIR.resolve("out.bin"));

        try (Model model = Model.newInstance("openvino_tiny", "ExecuTorch")) {
            model.load(DIR, "openvino_tiny");
            // Reported, not asserted: atol=1e-2 alone cannot distinguish "correct in bf16" from
            // "quietly degraded", so the run records which precision it actually got. Printed AFTER
            // load: before it, nothing has extracted the bundle, resolvedLibPath() is null, and the
            // accessor would report "unavailable" -- recording nothing about the run that follows.
            System.out.println("OpenVINO inference precision: " + EtEngine.openVinoInferencePrecision());
            try (NDManager manager = NDManager.newBaseManager("ExecuTorch");
                    NDList inputs = new NDList(manager.create(in, new Shape(1, in.length)));
                    NDList outputs = model.getBlock().forward(null, inputs, false)) {
                float[] actual = outputs.singletonOrThrow().toFloatArray();
                assertEquals(golden.length, actual.length);
                for (int i = 0; i < golden.length; i++) {
                    // atol=1e-2 and DO NOT TIGHTEN. OpenVINO picks its inference precision from the
                    // CPU it lands on, at import time rather than blob-compile time: on
                    // avx512_bf16/AMX hardware it computes in bf16 and lands ~2.5e-3 from this f32
                    // eager golden; elsewhere ~6e-8. Both are correct OpenVINO results. A tolerance
                    // drawn between them asserts which machine CI allocated -- a property this
                    // project does not own -- and fails at random. Upstream hit exactly this
                    // (ea393da in executorch-runtime-dist) after a green and a red run on identical
                    // artifacts. The loose bound still catches zeros, garbage, or the wrong model,
                    // which are orders of magnitude out.
                    assertEquals(golden[i], actual[i], 1e-2, "element " + i);
                }
            }
        }
    }
}
