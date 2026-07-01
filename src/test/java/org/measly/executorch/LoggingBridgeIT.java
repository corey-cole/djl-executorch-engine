package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.measly.executorch.jni.EtNative;
import org.slf4j.LoggerFactory;

/**
 * End-to-end proof that the PAL-bridge callback routes ET_LOG messages through to the native
 * slf4j logger captured by a logback appender.
 *
 * <p>Loads a corrupt .pte fixture, which causes the ExecuTorch runtime to emit ERROR-level log
 * messages via ET_LOG. Those messages travel: native ET_LOG → PAL sink (etDjlEmitLog) →
 * EtNative.nativeLog (Java) → slf4j logger "org.measly.executorch.native" → logback appender.
 */
class LoggingBridgeIT {

    private Logger logger;
    private ListAppender<ILoggingEvent> appender;

    @BeforeEach
    void attach() {
        logger = (Logger) LoggerFactory.getLogger("org.measly.executorch.native");
        logger.setLevel(Level.TRACE);
        appender = new ListAppender<>();
        appender.start();
        logger.addAppender(appender);
    }

    @AfterEach
    void detach() {
        logger.detachAppender(appender);
    }

    @Test
    void corruptModelLoadEmitsNativeErrorLogThroughSlf4j() throws Exception {
        Path corrupt = Paths.get(getClass().getResource("/models/corrupt.pte").toURI());

        assertThrows(RuntimeException.class,
                () -> EtNative.loadModule(corrupt.toAbsolutePath().toString()));

        assertTrue(
                appender.list.stream().anyMatch(e -> e.getLevel() == Level.ERROR),
                "corrupt .pte load must produce at least one ERROR event through the native "
                        + "ET_LOG → PAL sink → slf4j bridge; captured events: " + appender.list);
    }
}
