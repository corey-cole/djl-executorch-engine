package org.measly.executorch.stress;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Parser + verifier for the committed golden file. Touches no native code. */
class StressGoldenTest {

    private static final String MINIMAL =
            "{\"executorchVersion\":\"1.3.1\",\"seed\":1,"
                    + "\"config\":{\"batch\":2,\"hidden\":2,\"depth\":1,\"nBuckets\":4,\"ramp\":1.0E-5},"
                    + "\"sampleStride\":2,\"measuredUsPerForward\":400.0,"
                    + "\"cases\":[{\"name\":\"c0\",\"v1\":0.25,\"v2\":0.5,"
                    + "\"sum\":10.0,\"absSum\":10.0,\"maxAbs\":4.0,\"samples\":[1.0,3.0]}]}";

    private static Path write(Path dir, String json) throws IOException {
        Path p = dir.resolve("stress_golden.json");
        Files.writeString(p, json);
        return p;
    }

    @Test
    void parsesTheCommittedGoldenFile() {
        StressGolden g = StressGolden.load(Paths.get("src/test/resources/models/stress/stress_golden.json"));
        assertEquals(8, g.cases().size(), "export script writes 8 cases");
        assertEquals(32, g.config().batch);
        assertEquals(256, g.config().hidden);
        assertEquals(512, g.sampleStride());
        assertEquals(16, g.cases().get(0).samples.length);
    }

    @Test
    void parsesAMinimalFile(@TempDir Path dir) throws IOException {
        StressGolden g = StressGolden.load(write(dir, MINIMAL));
        assertEquals(1, g.cases().size());
        assertEquals("c0", g.cases().get(0).name);
        assertEquals(0.25f, g.cases().get(0).v1);
        assertEquals(1e-5f, g.config().ramp);
    }

    @Test
    void aMissingFileNamesThePathAndTheFix() {
        IllegalStateException e =
                assertThrows(
                        IllegalStateException.class,
                        () -> StressGolden.load(Paths.get("no/such/stress_golden.json")));
        assertTrue(e.getMessage().contains("no/such"), "message must quote the path");
        assertTrue(e.getMessage().contains("export_stress_model.py"), "message must name the fix");
    }

    @Test
    void malformedJsonIsRejected(@TempDir Path dir) throws IOException {
        Path p = write(dir, "{ this is not json");
        assertThrows(IllegalStateException.class, () -> StressGolden.load(p));
    }

    @Test
    void anEmptyCaseListIsRejected(@TempDir Path dir) throws IOException {
        Path p = write(dir, MINIMAL.replace("\"cases\":[{", "\"cases2\":[{"));
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> StressGolden.load(p));
        assertTrue(e.getMessage().contains("cases"), "message must say what is missing");
    }

    @Test
    void aSampleCountThatDisagreesWithTheStrideIsRejected(@TempDir Path dir) throws IOException {
        // batch*hidden = 4, stride 2 => 2 samples expected; give it 3.
        Path p = write(dir, MINIMAL.replace("[1.0,3.0]", "[1.0,3.0,5.0]"));
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> StressGolden.load(p));
        assertTrue(e.getMessage().contains("c0"), "message must name the offending case");
    }

    @Test
    void verifyAcceptsAnExactMatch(@TempDir Path dir) throws IOException {
        StressGolden g = StressGolden.load(write(dir, MINIMAL));
        // batch*hidden = 4; stride 2 => samples are elements 0 and 2.
        g.verify(g.cases().get(0), new float[] {1.0f, 2.0f, 3.0f, 4.0f});
    }

    @Test
    void verifyRejectsAWrongSample(@TempDir Path dir) throws IOException {
        StressGolden g = StressGolden.load(write(dir, MINIMAL));
        AssertionError e =
                assertThrows(
                        AssertionError.class,
                        () -> g.verify(g.cases().get(0), new float[] {1.0f, 2.0f, 99.0f, 4.0f}));
        assertTrue(e.getMessage().contains("sample"), "message must localise the failure");
    }

    @Test
    void verifyRejectsAWrongLength(@TempDir Path dir) throws IOException {
        StressGolden g = StressGolden.load(write(dir, MINIMAL));
        assertThrows(AssertionError.class, () -> g.verify(g.cases().get(0), new float[] {1.0f}));
    }
}
