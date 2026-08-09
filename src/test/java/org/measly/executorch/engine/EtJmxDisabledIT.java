package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertFalse;

import ai.djl.Model;
import java.lang.management.ManagementFactory;
import java.nio.file.Paths;
import javax.management.ObjectName;
import org.measly.executorch.TestSupport;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Runs in its own JVM with jmx_enabled=false, before auto-registration can fire. */
@Tag("jmx-disabled")
class EtJmxDisabledIT {

    @Test
    void loadDoesNotRegisterTheMBean() throws Exception {
        TestSupport.assumeNativeAvailable();
        try (Model model = Model.newInstance("add", "ExecuTorch")) {
            model.load(Paths.get("native/spike"), "add");
        }
        assertFalse(
                ManagementFactory.getPlatformMBeanServer()
                        .isRegistered(new ObjectName(EtEngineStats.OBJECT_NAME)),
                "jmx_enabled=false must suppress auto-registration");
    }
}
