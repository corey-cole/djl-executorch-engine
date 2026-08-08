package org.measly.executorch.engine;

import java.util.Locale;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Resolves the per-model XNNPACK workspace sharing mode from a DJL model option or a JVM-wide
 * default property.
 *
 * <p>Unlike {@code ai.djl.executorch.num_threads}, this is <b>not</b> process-global and is
 * <b>not</b> write-once. ExecuTorch resolves the mode per delegate at method-load time, preferring
 * the per-load runtime spec we supply over the backend's process global, so every model may choose
 * independently and the modes compose: a model electing {@code disabled} is isolated regardless of
 * what any other loaded model chose. Nothing is sealed and load order does not matter.
 *
 * <p>Values map to {@code executorch::backends::xnnpack::WorkspaceSharingMode}. That header is not
 * installed by the runtime tarball, so the ints are hardcoded here and again in
 * {@code native/core/et_runtime.cpp}; keep the two in sync.
 */
final class EtWorkspaceSharing {

    /** DJL per-model option key, e.g. {@code Criteria.optOption("workspaceSharingMode", ...)}. */
    static final String OPTION_KEY = "workspaceSharingMode";

    /** JVM-wide default for models that do not carry {@link #OPTION_KEY}. */
    static final String PROPERTY = "ai.djl.executorch.workspace_sharing_mode";

    /** Send no spec at all, leaving the runtime's compiled-in default. NOT a synonym for GLOBAL. */
    static final int UNSPECIFIED = -1;

    /** Every delegate instance gets its own workspace: maximum parallelism, maximum arena memory. */
    static final int DISABLED = 0;

    /** All delegate instances in one program share a workspace: one method at a time per model. */
    static final int PER_MODEL = 1;

    /** All delegate instances across all loaded methods share one workspace. The shipped default. */
    static final int GLOBAL = 2;

    private static final Logger logger = LoggerFactory.getLogger(EtWorkspaceSharing.class);

    private EtWorkspaceSharing() {}

    /**
     * Maps a mode name to its native int.
     *
     * @param value one of {@code disabled}, {@code per_model}, {@code global}; case-insensitive and
     *     trimmed
     * @return the native mode int
     * @throws IllegalArgumentException if the value is not one of the three names. Bare integers
     *     are rejected too: they are opaque at a call site and would let an out-of-range value
     *     through to a native load failure.
     */
    static int parse(String value) {
        String v = value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
        switch (v) {
            case "disabled":
                return DISABLED;
            case "per_model":
                return PER_MODEL;
            case "global":
                return GLOBAL;
            default:
                throw new IllegalArgumentException(
                        OPTION_KEY
                                + ": unrecognized value '"
                                + value
                                + "'; expected disabled|per_model|global");
        }
    }

    /**
     * Applies the precedence chain: per-model option, then JVM property, then unspecified.
     *
     * <p>A key present with a null value counts as absent. A bad option throws (explicit per-model
     * intent must not degrade silently); a bad property WARNs and is ignored (a typo in a
     * process-wide flag must not fail startup), matching the {@code num_threads} precedent.
     *
     * @param options the DJL model options map; may be null
     * @param propertyValue the value of {@link #PROPERTY}, passed in so this stays pure and
     *     testable; may be null
     * @return the native mode int, or {@link #UNSPECIFIED}
     * @throws IllegalArgumentException if the per-model option carries an unrecognized value
     */
    /**
     * Renders a resolved mode int as a human-readable name, for logging.
     *
     * @param mode one of {@link #UNSPECIFIED}, {@link #DISABLED}, {@link #PER_MODEL}, {@link
     *     #GLOBAL}
     * @return {@code "unspecified"}, {@code "disabled"}, {@code "per_model"}, {@code "global"}, or
     *     {@code "unknown(<mode>)"} for any other int
     */
    static String name(int mode) {
        switch (mode) {
            case UNSPECIFIED:
                return "unspecified";
            case DISABLED:
                return "disabled";
            case PER_MODEL:
                return "per_model";
            case GLOBAL:
                return "global";
            default:
                return "unknown(" + mode + ")";
        }
    }

    static int resolve(Map<String, ?> options, String propertyValue) {
        Object raw = options == null ? null : options.get(OPTION_KEY);
        if (raw != null) {
            return parse(String.valueOf(raw));
        }
        if (propertyValue == null) {
            return UNSPECIFIED;
        }
        try {
            return parse(propertyValue);
        } catch (IllegalArgumentException e) {
            logger.warn(
                    "{}='{}' is not a recognized mode; ignoring and using the runtime default",
                    PROPERTY,
                    propertyValue);
            return UNSPECIFIED;
        }
    }
}
