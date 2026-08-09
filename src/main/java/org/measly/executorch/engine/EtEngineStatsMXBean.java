package org.measly.executorch.engine;

/**
 * JMX view of {@link EtEngineStats}, registered as {@value EtEngineStats#OBJECT_NAME}.
 *
 * <p>An <b>MX</b>Bean rather than a plain MBean: the JMX runtime converts {@link EtStatsSnapshot}
 * and its nested {@code List<EtModelStats>} to {@code CompositeData}/{@code TabularData}
 * automatically, so no hand-written {@code OpenType} mapping is needed. Keeping that conversion
 * working is why both value types are getter-only JavaBeans.
 */
public interface EtEngineStatsMXBean {

    /** @return a fresh snapshot of engine configuration, totals, and live models */
    EtStatsSnapshot getSnapshot();
}
