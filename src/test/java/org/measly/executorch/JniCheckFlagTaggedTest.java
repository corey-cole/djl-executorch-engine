package org.measly.executorch;

import org.junit.jupiter.api.Tag;

/**
 * Carries {@link JniCheckFlagTest}'s inherited assertion into the tag-filtered test tasks.
 *
 * <p>{@code tasks.test} excludes all eight of these tags and each filtered task includes exactly
 * one, so no single class can run under every task. This subclass is how the umbrella attachment
 * gets proven where it matters most — above all in {@code oomTest}, which drives the
 * allocation-failure paths {@code -Xcheck:jni} exists to police.
 */
@Tag("leak")
@Tag("oom")
@Tag("intraop")
@Tag("jmx-disabled")
@Tag("stats-degraded")
@Tag("stress")
@Tag("stress-sweep")
@Tag("stress-baseline")
class JniCheckFlagTaggedTest extends JniCheckFlagTest {}
