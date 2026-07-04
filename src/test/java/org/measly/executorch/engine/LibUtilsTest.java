package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class LibUtilsTest {

    @Test
    void platformResolvesLinuxX8664() {
        String os = System.getProperty("os.name");
        String arch = System.getProperty("os.arch");
        try {
            System.setProperty("os.name", "Linux");
            System.setProperty("os.arch", "amd64");
            assertEquals("linux-x86_64", LibUtils.platform());
        } finally {
            System.setProperty("os.name", os);
            System.setProperty("os.arch", arch);
        }
    }

    @Test
    void platformRejectsUnsupportedArch() {
        String arch = System.getProperty("os.arch");
        try {
            System.setProperty("os.arch", "sparc");
            assertThrows(UnsupportedOperationException.class, LibUtils::platform);
        } finally {
            System.setProperty("os.arch", arch);
        }
    }

    @Test
    void platformRejectsUnsupportedOs() {
        String os = System.getProperty("os.name");
        String arch = System.getProperty("os.arch");
        try {
            System.setProperty("os.name", "Windows 11");
            System.setProperty("os.arch", "amd64");
            assertThrows(UnsupportedOperationException.class, LibUtils::platform);
        } finally {
            System.setProperty("os.name", os);
            System.setProperty("os.arch", arch);
        }
    }
}
