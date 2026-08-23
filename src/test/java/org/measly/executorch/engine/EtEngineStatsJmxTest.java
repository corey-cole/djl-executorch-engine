package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.management.ManagementFactory;
import javax.management.MBeanServer;
import javax.management.ObjectName;
import javax.management.openmbean.CompositeData;
import org.measly.executorch.TestSupport;
import org.junit.jupiter.api.Test;

class EtEngineStatsJmxTest {

    private static ObjectName name() throws Exception {
        return new ObjectName(EtEngineStats.OBJECT_NAME);
    }

    @Test
    void registersAndExposesTheSnapshotAsCompositeData() throws Exception {
        TestSupport.assumeNativeLibraryAvailable();
        EtEngineStats.registerMBean();
        try {
            MBeanServer server = ManagementFactory.getPlatformMBeanServer();
            assertTrue(server.isRegistered(name()));
            // An MXBean converts the value type automatically; if EtStatsSnapshot is not a
            // conforming bean this throws instead of returning CompositeData.
            Object value = server.getAttribute(name(), "Snapshot");
            assertNotNull(value);
            CompositeData data = (CompositeData) value;
            assertEquals("1.4.1", data.get("executorchVersion"));
            assertNotNull(data.get("models"));
            // The MXBean surface is derived from the getters by reflection, so a new field reaches
            // JMX only if its getter is bean-conforming. Asserting the key is present catches a
            // getter that snapshot() populates but JMX silently drops.
            assertNotNull(
                    data.get("xnnpackWorkspaceBytes"),
                    "the workspace figure must survive the CompositeData conversion");
        } finally {
            EtEngineStats.unregisterMBean();
        }
    }

    @Test
    void registrationIsIdempotent() throws Exception {
        TestSupport.assumeNativeLibraryAvailable();
        EtEngineStats.registerMBean();
        try {
            // A second call must not throw InstanceAlreadyExistsException out to the caller.
            assertDoesNotThrow(EtEngineStats::registerMBean);
            assertTrue(ManagementFactory.getPlatformMBeanServer().isRegistered(name()));
        } finally {
            EtEngineStats.unregisterMBean();
        }
    }

    @Test
    void unregisterIsSafeWhenNotRegistered() {
        assertDoesNotThrow(EtEngineStats::unregisterMBean);
        assertDoesNotThrow(EtEngineStats::unregisterMBean);
    }
}
