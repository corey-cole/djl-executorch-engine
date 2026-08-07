package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import java.nio.file.Paths;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.measly.executorch.engine.EtEngine;

/**
 * The only test proving the property reaches the native pool end to end. Runs in a dedicated
 * forked JVM (intraOpTest task, -Dai.djl.executorch.num_threads=2): the pool is process-global,
 * so this cannot share a JVM with any other test.
 */
@Tag("intraop")
class IntraOpThreadsIT {

    @Test
    void propertySealsTheNativePoolAtRequestedSize() throws Exception {
        TestSupport.assumeNativeAvailable();
        Criteria<float[], Float> criteria =
                Criteria.builder()
                        .setTypes(float[].class, Float.class)
                        .optEngine("ExecuTorch")
                        .optModelPath(Paths.get("native/spike"))
                        .optModelName("add")
                        .optTranslator(new AddTranslator())
                        .build();
        try (ZooModel<float[], Float> model = criteria.loadModel()) {
            // The load sealed the pool: the native pool must report the property's value, not
            // the performance-core default (4 or 8 on this host -- never 2).
            assertEquals(2, EtEngine.getIntraOpThreads());
            // And the gate is closed: a setter after a real load throws, naming the sealed value.
            IllegalStateException e = assertThrows(
                    IllegalStateException.class, () -> EtEngine.setIntraOpThreads(4));
            assertEquals("2", e.getMessage().substring(
                    e.getMessage().indexOf("at ") + 3,
                    e.getMessage().indexOf("; set")));
        }
    }
}
