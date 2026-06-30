package org.measly.executorch.jni;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

class EtNativeLogTest {

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
    void routesEachLevel() {
        EtNative.nativeLog(0, "dbg");
        EtNative.nativeLog(1, "inf");
        EtNative.nativeLog(2, "wrn");
        EtNative.nativeLog(3, "err");
        assertEquals(4, appender.list.size());
        assertEquals(Level.DEBUG, appender.list.get(0).getLevel());
        assertEquals("dbg", appender.list.get(0).getMessage());
        assertEquals(Level.INFO, appender.list.get(1).getLevel());
        assertEquals(Level.WARN, appender.list.get(2).getLevel());
        assertEquals(Level.ERROR, appender.list.get(3).getLevel());
    }

    @Test
    void unknownLevelDefaultsToInfo() {
        EtNative.nativeLog(99, "huh");
        assertEquals(1, appender.list.size());
        assertEquals(Level.INFO, appender.list.get(0).getLevel());
    }
}
