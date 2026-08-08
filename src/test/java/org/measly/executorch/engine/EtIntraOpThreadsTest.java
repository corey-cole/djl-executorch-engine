package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** Pure state machine for the intra-op thread pool gate; never touches the native library. */
class EtIntraOpThreadsTest {

    @BeforeEach
    void resetGate() {
        EtEngine.resetIntraOpForTest();
    }

    @Test
    void setIntraOpThreadsRejectsNonPositive() {
        assertThrows(IllegalArgumentException.class, () -> EtEngine.setIntraOpThreads(0));
        assertThrows(IllegalArgumentException.class, () -> EtEngine.setIntraOpThreads(-1));
        assertThrows(IllegalArgumentException.class, () -> EtEngine.setIntraOpThreads(-4));
    }

    @Test
    void setterAfterSealThrowsNamingSealedValue() {
        EtEngine.sealIntraOpThreadsForTest(); // idempotent: no-op if a test already sealed this JVM
        IllegalStateException e = assertThrows(
                IllegalStateException.class, () -> EtEngine.setIntraOpThreads(2));
        assertTrue(e.getMessage().contains("sealed"));
        // The ISE must name the sealed value: a number when a count was sealed, "the runtime
        // default" when the seam sealed the default (-1). Order-independent in the shared JVM.
        int sealed = EtEngine.intraOpThreadCount();
        if (sealed >= 1) {
            assertTrue(e.getMessage().contains(String.valueOf(sealed)));
        } else {
            assertTrue(e.getMessage().contains("the runtime default"));
        }
    }

    @Test
    void setterBeatsPropertyAndPropertyParsing() {
        assertEquals(3, EtEngine.resolveIntraOpThreads(3, "5"));     // setter wins
        assertEquals(5, EtEngine.resolveIntraOpThreads(-1, "5"));     // property alone
        assertEquals(-1, EtEngine.resolveIntraOpThreads(-1, "0"));    // < 1 -> ignore
        assertEquals(-1, EtEngine.resolveIntraOpThreads(-1, "abc"));  // unparseable -> ignore
        assertEquals(-1, EtEngine.resolveIntraOpThreads(-1, null));   // absent -> default
        assertEquals(7, EtEngine.resolveIntraOpThreads(7, "abc"));    // setter beats broken property
    }
}
