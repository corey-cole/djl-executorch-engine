package org.measly.executorch.engine;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/** Resolves and loads libexecutorch_djl.so. */
public final class LibUtils {

    private static boolean loaded;

    private LibUtils() {}

    // Not unit-tested: drives System.load, the EXECUTORCH_LIBRARY_PATH env override, and classpath
    // extraction, all of which need the real native library and JVM state; only platform() is unit-tested.
    public static synchronized void loadLibrary() {
        if (loaded) {
            return;
        }
        String override = System.getenv("EXECUTORCH_LIBRARY_PATH");
        if (override != null && !override.isEmpty()) {
            System.load(override);
            loaded = true;
            return;
        }
        String platform = platform();
        String resource = "/native/" + platform + "/" + libName(platform);
        try (InputStream is = LibUtils.class.getResourceAsStream(resource)) {
            if (is == null) {
                throw new IllegalStateException(
                        "Native library not found on classpath: " + resource
                                + " (set EXECUTORCH_LIBRARY_PATH or run native/build_desktop.sh)");
            }
            Path tmp = Files.createTempFile("libexecutorch_djl", ".so");
            tmp.toFile().deleteOnExit();
            Files.copy(is, tmp, StandardCopyOption.REPLACE_EXISTING);
            System.load(tmp.toAbsolutePath().toString());
            loaded = true;
        } catch (IOException e) {
            throw new IllegalStateException("Failed to extract native library", e);
        }
    }

    static String platform() {
        String os = System.getProperty("os.name").toLowerCase();
        String arch = System.getProperty("os.arch").toLowerCase();
        boolean x64 = arch.equals("amd64") || arch.equals("x86_64");
        if (os.contains("linux") && x64) {
            return "linux-x86_64";
        }
        if (os.contains("windows") && x64) {
            return "windows-x86_64";
        }
        throw new UnsupportedOperationException(
                "ExecuTorch engine supports only linux-x86_64 and windows-x86_64, got: " + os + "/" + arch);
    }

    /** MSVC emits no `lib` prefix and a .dll suffix. Keep in sync with nativeLibName in build.gradle.kts. */
    static String libName(String platform) {
        return platform.startsWith("windows-") ? "executorch_djl.dll" : "libexecutorch_djl.so";
    }
}
