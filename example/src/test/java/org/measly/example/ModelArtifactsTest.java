package org.measly.example;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ModelArtifactsTest {

    @Test
    void requireThrowsWithExportPointerWhenMissing(@TempDir Path dir) {
        System.setProperty("example.models.dir", dir.toString());
        try {
            IllegalStateException ex =
                    assertThrows(
                            IllegalStateException.class,
                            () -> ModelArtifacts.require("mobilenet_v2.pte"));
            assertTrue(
                    ex.getMessage().contains("./gradlew :example:exportModels"),
                    "message should point at the export task, was: " + ex.getMessage());
        } finally {
            System.clearProperty("example.models.dir");
        }
    }

    @Test
    void requireReturnsPathWhenPresent(@TempDir Path dir) throws Exception {
        System.setProperty("example.models.dir", dir.toString());
        try {
            Path pte = Files.createFile(dir.resolve("mobilenet_v2.pte"));
            assertEquals(pte, ModelArtifacts.require("mobilenet_v2.pte"));
        } finally {
            System.clearProperty("example.models.dir");
        }
    }
}
