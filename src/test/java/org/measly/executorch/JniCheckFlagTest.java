package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.management.ManagementFactory;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Asserts the JVM under test runs with {@code -Xcheck:jni}, the JNI-contract checker: it catches
 * JNI calls made with an exception already pending, and null array arguments to the
 * {@code Set*ArrayRegion} family.
 *
 * <p>The flag is attached to the {@code Test} task umbrella in {@code build.gradle.kts}, so this
 * assertion must hold for every test task rather than {@code test} alone. {@link
 * JniCheckFlagTaggedTest} inherits it into the eight tag-filtered tasks.
 *
 * <p>This asserts the checker is <em>active</em>, not that it fires. The null-check branches need
 * heap exhaustion mid-loop to reach, so a fire-on-demand probe would not be reproducible and cannot
 * serve as the gate.
 */
class JniCheckFlagTest {

    @Test
    void jvmRunsWithXcheckJni() {
        List<String> args = ManagementFactory.getRuntimeMXBean().getInputArguments();
        assertTrue(
                args.contains("-Xcheck:jni"),
                "test JVM must run with -Xcheck:jni; actual JVM arguments: " + args);
    }
}
