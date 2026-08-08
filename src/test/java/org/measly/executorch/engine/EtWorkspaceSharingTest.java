package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** Pure resolution logic for the per-model workspace sharing mode; never touches the native lib. */
class EtWorkspaceSharingTest {

    @Test
    void parseAcceptsTheThreeModesCaseInsensitivelyAndTrimmed() {
        assertEquals(EtWorkspaceSharing.DISABLED, EtWorkspaceSharing.parse("disabled"));
        assertEquals(EtWorkspaceSharing.PER_MODEL, EtWorkspaceSharing.parse("per_model"));
        assertEquals(EtWorkspaceSharing.GLOBAL, EtWorkspaceSharing.parse("global"));
        assertEquals(EtWorkspaceSharing.DISABLED, EtWorkspaceSharing.parse("DISABLED"));
        assertEquals(EtWorkspaceSharing.GLOBAL, EtWorkspaceSharing.parse("  Global  "));
    }

    @Test
    void parseRejectsUnrecognisedValuesAndNamesTheLegalOnes() {
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> EtWorkspaceSharing.parse("disabeld"));
        assertTrue(e.getMessage().contains("disabeld"), "message must quote the bad value");
        assertTrue(e.getMessage().contains("per_model"), "message must list the legal values");
    }

    @Test
    void parseRejectsBareIntegers() {
        // Ints are opaque at a call site and would let an out-of-range value reach the runtime.
        assertThrows(IllegalArgumentException.class, () -> EtWorkspaceSharing.parse("0"));
        assertThrows(IllegalArgumentException.class, () -> EtWorkspaceSharing.parse("99"));
    }

    @Test
    void optionWinsOverProperty() {
        Map<String, String> options =
                Collections.singletonMap(EtWorkspaceSharing.OPTION_KEY, "disabled");
        assertEquals(EtWorkspaceSharing.DISABLED, EtWorkspaceSharing.resolve(options, "global"));
    }

    @Test
    void propertyAppliesWhenNoOptionIsPresent() {
        assertEquals(
                EtWorkspaceSharing.GLOBAL,
                EtWorkspaceSharing.resolve(Collections.emptyMap(), "global"));
        assertEquals(EtWorkspaceSharing.GLOBAL, EtWorkspaceSharing.resolve(null, "global"));
    }

    @Test
    void nothingSpecifiedOmitsTheSpec() {
        assertEquals(
                EtWorkspaceSharing.UNSPECIFIED,
                EtWorkspaceSharing.resolve(Collections.emptyMap(), null));
        assertEquals(EtWorkspaceSharing.UNSPECIFIED, EtWorkspaceSharing.resolve(null, null));
    }

    @Test
    void aNullOptionValueCountsAsAbsent() {
        Map<String, String> options = new HashMap<>();
        options.put(EtWorkspaceSharing.OPTION_KEY, null);
        assertEquals(EtWorkspaceSharing.GLOBAL, EtWorkspaceSharing.resolve(options, "global"));
        assertEquals(EtWorkspaceSharing.UNSPECIFIED, EtWorkspaceSharing.resolve(options, null));
    }

    @Test
    void aBadOptionThrowsButABadPropertyIsIgnored() {
        // Asymmetric by design (spec section 6): a per-model option is explicit intent about one
        // model, so a silent fallback would be an invisible latency regression. The property is an
        // ambient default and a typo in a process-wide flag must not fail startup.
        Map<String, String> options =
                Collections.singletonMap(EtWorkspaceSharing.OPTION_KEY, "disabeld");
        assertThrows(
                IllegalArgumentException.class, () -> EtWorkspaceSharing.resolve(options, null));
        assertEquals(
                EtWorkspaceSharing.UNSPECIFIED,
                EtWorkspaceSharing.resolve(Collections.emptyMap(), "disabeld"));
    }

    @Test
    void nameRendersEachModeHumanReadably() {
        assertEquals("unspecified", EtWorkspaceSharing.name(EtWorkspaceSharing.UNSPECIFIED));
        assertEquals("disabled", EtWorkspaceSharing.name(EtWorkspaceSharing.DISABLED));
        assertEquals("per_model", EtWorkspaceSharing.name(EtWorkspaceSharing.PER_MODEL));
        assertEquals("global", EtWorkspaceSharing.name(EtWorkspaceSharing.GLOBAL));
        assertEquals("unknown(99)", EtWorkspaceSharing.name(99));
    }

    @Test
    void nonStringOptionValuesAreCoerced() {
        // DJL's options map is Map<String, ?>; callers can put anything in it.
        Map<String, Object> options = new HashMap<>();
        options.put(EtWorkspaceSharing.OPTION_KEY, new StringBuilder("disabled"));
        assertEquals(EtWorkspaceSharing.DISABLED, EtWorkspaceSharing.resolve(options, null));
    }
}
