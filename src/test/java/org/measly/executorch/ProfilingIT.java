package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import org.measly.executorch.engine.EtEngine;
import org.measly.executorch.engine.EtModel;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * End-to-end profiling. Both capability arms are real coverage: where devtools is provisioned the
 * dump must be well formed and must drain on pull; where it is not, the load must fail loudly
 * rather than record nothing.
 */
class ProfilingIT {

    private static final Logger logger = LoggerFactory.getLogger(ProfilingIT.class);

    private static Criteria<float[], Float> criteria(String profiling) {
        Criteria.Builder<float[], Float> b =
                Criteria.builder()
                        .setTypes(float[].class, Float.class)
                        .optEngine("ExecuTorch")
                        .optModelPath(Paths.get("native/spike"))
                        .optModelName("add")
                        .optTranslator(new AddTranslator());
        if (profiling != null) {
            b.optOption(EtEngine.PROFILING_OPTION, profiling);
        }
        return b.build();
    }

    /** The ETDump is a size-prefixed flatbuffer; its identifier sits within the first 16 bytes. */
    private static boolean carriesEtDumpIdentifier(byte[] dump) {
        int n = Math.min(dump.length, 16);
        return new String(dump, 0, n, StandardCharsets.ISO_8859_1).contains("ED00");
    }

    @Test
    void profiledModelYieldsAWellFormedDumpThatDrainsOnPull() throws Exception {
        TestSupport.assumeNativeAvailable();
        assumeTrue(
                EtEngine.devtoolsAvailable(),
                "devtools not provisioned on this platform; skipping the devtools-present arm");
        logger.info("ProfilingIT: running the devtools-present arm");
        try (ZooModel<float[], Float> model = criteria("true").loadModel();
                Predictor<float[], Float> predictor = model.newPredictor()) {
            EtModel etModel = (EtModel) model.getWrappedModel();

            assertEquals(0, etModel.etDump().length, "no forward has run yet");

            for (int i = 0; i < 4; i++) {
                assertEquals(5f, predictor.predict(new float[] {2f, 3f}), 1e-6);
            }
            byte[] four = etModel.etDump();
            assertTrue(four.length > 8, "four forwards must produce a dump");
            assertTrue(carriesEtDumpIdentifier(four), "dump must carry the ED00 identifier");

            // Pulling again with no forward in between must not corrupt the builder. Upstream's
            // get_etdump_data() matches none of its guard branches once finalized, so the cached
            // copy is what makes this safe.
            assertEquals(four.length, etModel.etDump().length, "second pull must repeat the first");

            // The forward after a pull starts a fresh dump: one Execute block, not five.
            assertEquals(5f, predictor.predict(new float[] {2f, 3f}), 1e-6);
            byte[] one = etModel.etDump();
            assertTrue(one.length > 8, "the fresh dump must still be well formed");
            assertTrue(one.length < four.length, "pulling drains; the dump must not accumulate");
        }
    }

    @Test
    void unprofiledModelYieldsAnEmptyDump() throws Exception {
        // Runs on every platform: having no dump is an answer, not an error.
        TestSupport.assumeNativeAvailable();
        try (ZooModel<float[], Float> model = criteria(null).loadModel();
                Predictor<float[], Float> predictor = model.newPredictor()) {
            assertEquals(5f, predictor.predict(new float[] {2f, 3f}), 1e-6);
            assertEquals(0, ((EtModel) model.getWrappedModel()).etDump().length);
        }
    }

    @Test
    void requestingProfilingWithoutDevtoolsFailsTheLoad() throws Exception {
        TestSupport.assumeNativeAvailable();
        assumeFalse(
                EtEngine.devtoolsAvailable(),
                "devtools is provisioned here; skipping the devtools-absent arm");
        logger.info("ProfilingIT: running the devtools-absent arm");
        Throwable t = assertThrows(Throwable.class, () -> criteria("true").loadModel());
        // DJL wraps load failures; the message must still name the provisioning, not fail generically.
        String messages = TestSupport.messageChain(t);
        assertTrue(
                messages.contains("not provisioned"),
                "load failure must explain the platform has no event tracer, got: " + messages);
    }

    @Test
    void unrecognizedOptionValueFailsTheLoad() throws Exception {
        // Runs on every platform: "yes" must fail rather than silently disable the feature it names.
        TestSupport.assumeNativeAvailable();
        Throwable t = assertThrows(Throwable.class, () -> criteria("yes").loadModel());
        String messages = TestSupport.messageChain(t);
        assertTrue(
                messages.contains(EtEngine.PROFILING_OPTION),
                "the failure must name the option key, got: " + messages);
    }
}
