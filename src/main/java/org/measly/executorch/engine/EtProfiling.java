package org.measly.executorch.engine;

import java.util.Locale;
import java.util.Map;

/**
 * Resolves the per-model profiling opt-in from a DJL model option.
 *
 * <p>There is deliberately <b>no</b> JVM-wide property counterpart, unlike {@link
 * EtWorkspaceSharing}. A property would let one JVM flag attach an event tracer to every model in
 * the process, including models whose owner never pulls the dump — and an ETDump grows across every
 * forward until it is pulled. Profiling is a diagnostic with a real memory cost, so enabling it is
 * a decision at the load site and nowhere else. This absence is the design, not an omission.
 */
final class EtProfiling {

    /** DJL per-model option key, e.g. {@code Criteria.optOption("profiling", "true")}. */
    static final String OPTION_KEY = "profiling";

    private EtProfiling() {}

    /**
     * Resolves the option.
     *
     * @param options the model's DJL options; may be null
     * @return whether to attach an event tracer to this model
     * @throws IllegalArgumentException if the value is neither {@code true} nor {@code false};
     *     case-insensitive and trimmed. Bare truthy spellings are rejected rather than coerced.
     */
    static boolean resolve(Map<String, ?> options) {
        Object raw = options == null ? null : options.get(OPTION_KEY);
        if (raw == null) {
            return false;
        }
        String v = String.valueOf(raw).trim().toLowerCase(Locale.ROOT);
        switch (v) {
            case "true":
                return true;
            case "false":
                return false;
            default:
                throw new IllegalArgumentException(
                        OPTION_KEY + ": unrecognized value '" + raw + "'; expected true|false");
        }
    }
}
