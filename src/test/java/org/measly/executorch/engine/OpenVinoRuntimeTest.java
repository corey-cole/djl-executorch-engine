package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.measly.executorch.TestSupport;

@Tag("openvino")
class OpenVinoRuntimeTest {

    @Test
    void extractsTheBundleToAFlatDirectoryAndResolvesTheVersionedLibrary() throws Exception {
        TestSupport.assumeOpenVinoBundleAvailable();

        Path dir = OpenVinoRuntime.ensureExtracted();
        assertNotNull(dir);
        assertTrue(Files.isDirectory(dir), "bundle must extract to a directory: " + dir);

        // Flat, not nested: RPATH=$ORIGIN only resolves siblings.
        try (var entries = Files.list(dir)) {
            assertTrue(
                    entries.noneMatch(Files::isDirectory),
                    "the library directory must be flat; $ORIGIN does not search subdirectories");
        }

        String lib = OpenVinoRuntime.resolvedLibPath();
        assertNotNull(lib);
        assertTrue(Files.isRegularFile(Paths.get(lib)), "must name a file, not a directory: " + lib);
        // The versioned file, never an unversioned symlink: jars do not carry symlinks, so the
        // extraction never creates one and the resolved path must not depend on one existing.
        assertTrue(lib.contains(".so."), "must resolve the versioned library: " + lib);
    }

    @Test
    void repeatedExtractionIsIdempotentAndReturnsTheSameDirectory() throws Exception {
        TestSupport.assumeOpenVinoBundleAvailable();
        assertEquals(OpenVinoRuntime.ensureExtracted(), OpenVinoRuntime.ensureExtracted());
    }
}
