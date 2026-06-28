package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;

import org.junit.jupiter.api.Test;

class ClasspathTest {
    @Test
    void djlApiIsOnClasspath() {
        // Verifies ai.djl:api is resolvable on the test runtime classpath.
        assertDoesNotThrow(() -> Class.forName("ai.djl.engine.Engine"));
    }
}
