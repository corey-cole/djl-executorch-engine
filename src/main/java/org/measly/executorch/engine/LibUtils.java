package org.measly.executorch.engine;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/** Resolves and loads the executorch_djl native shim. */
public final class LibUtils {

    private static final int BUF = 64 * 1024;
    private static boolean loaded;

    private LibUtils() {}

    // Not unit-tested: drives System.load, the EXECUTORCH_LIBRARY_PATH env override, and classpath
    // extraction, all of which need the real native library and JVM state. platform(), libName() and
    // cacheRoot() are the unit-tested seams.
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
        String lib = libName(platform);
        String resource = "/native/" + platform + "/" + lib;
        try {
            // Hash-only first pass. The cache key must be known before the path exists, and a hit must
            // not rewrite 11.5 MB. A miss pays a second read; that happens once per version per host.
            Path target = cacheRoot().resolve(sha256(resource)).resolve(lib);
            if (!Files.isRegularFile(target)) {
                extract(resource, target);
            }
            System.load(target.toAbsolutePath().toString());
            loaded = true;
        } catch (IOException e) {
            throw new IllegalStateException("Failed to extract native library " + resource, e);
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

    /**
     * Cache location. Windows cannot delete a loaded DLL, so a per-JVM temp file would leak 11.5 MB per
     * run; a stable per-content directory is reused instead.
     */
    static Path cacheRoot() {
        String os = System.getProperty("os.name").toLowerCase();
        if (os.contains("windows")) {
            String localAppData = System.getenv("LOCALAPPDATA");
            if (localAppData != null && !localAppData.isEmpty()) {
                return Paths.get(localAppData, "executorch-djl");
            }
            return Paths.get(System.getProperty("user.home"), "AppData", "Local", "executorch-djl");
        }
        String xdg = System.getenv("XDG_CACHE_HOME");
        if (xdg != null && !xdg.isEmpty()) {
            return Paths.get(xdg, "executorch-djl");
        }
        return Paths.get(System.getProperty("user.home"), ".cache", "executorch-djl");
    }

    private static InputStream open(String resource) {
        InputStream is = LibUtils.class.getResourceAsStream(resource);
        if (is == null) {
            throw new IllegalStateException(
                    "Native library not found on classpath: " + resource
                            + " (set EXECUTORCH_LIBRARY_PATH or run native/build_desktop.sh)");
        }
        return is;
    }

    private static String sha256(String resource) throws IOException {
        MessageDigest md;
        try {
            md = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is required of every JRE but is unavailable", e);
        }
        byte[] buf = new byte[BUF];
        try (InputStream is = new DigestInputStream(open(resource), md)) {
            while (is.read(buf) != -1) {
                // DigestInputStream updates the digest as a side effect of reading.
            }
        }
        StringBuilder sb = new StringBuilder(64);
        for (byte b : md.digest()) {
            sb.append(Character.forDigit((b >> 4) & 0xF, 16)).append(Character.forDigit(b & 0xF, 16));
        }
        return sb.toString();
    }

    private static void extract(String resource, Path target) throws IOException {
        Path dir = target.getParent();
        Files.createDirectories(dir);
        Path tmp = Files.createTempFile(dir, target.getFileName().toString(), ".tmp");
        try {
            try (InputStream is = open(resource); OutputStream os = Files.newOutputStream(tmp)) {
                byte[] buf = new byte[BUF];
                int n;
                while ((n = is.read(buf)) != -1) {
                    os.write(buf, 0, n);
                }
            }
            try {
                Files.move(tmp, target, StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException e) {
                // A concurrent JVM published first. The path is content-addressed, so the winner's bytes
                // are ours byte-for-byte: adopt it rather than overwrite a file another process may have
                // already mapped. Windows refuses the replace with AccessDeniedException/FileSystemException
                // (not FileAlreadyExistsException), so catch IOException broadly and re-throw anything that
                // did not result in a published file.
                if (!Files.isRegularFile(target)) {
                    throw e;
                }
            }
        } finally {
            Files.deleteIfExists(tmp);
        }
    }
}
