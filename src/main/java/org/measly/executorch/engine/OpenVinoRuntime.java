package org.measly.executorch.engine;

import ai.djl.engine.EngineException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import org.measly.executorch.jni.EtNative;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Extracts the vendored OpenVINO runtime and points the delegate at it.
 *
 * <p>Exists because a JVM cannot use {@code LD_LIBRARY_PATH}: glibc's loader reads it once at
 * process start, {@code System.getenv} is read-only, and {@code ProcessBuilder} affects only child
 * processes. {@code OPENVINO_LIB_PATH} is read at {@code dlopen} time, which makes it the only
 * mechanism available to us.
 *
 * <p>Everything here happens once per process and must complete before the first OpenVINO
 * inference: the delegate's {@code dlopen} runs under {@code std::call_once} with no retry, so a
 * late or failed configuration cannot be repaired without restarting the JVM.
 */
public final class OpenVinoRuntime {

    private static final Logger logger = LoggerFactory.getLogger(OpenVinoRuntime.class);

    /** Backend id as ExecuTorch spells it — lowercase {@code v}. */
    public static final String BACKEND = "OpenvinoBackend";

    private static final int BUF = 64 * 1024;
    private static final String MANIFEST = "MANIFEST";
    private static final String BUILDINFO = "BUILDINFO";

    private static Path extracted;
    private static String libPath;
    private static boolean configured;

    private OpenVinoRuntime() {}

    /** @return true if the OpenVINO bundle jar is on the classpath for this platform */
    public static boolean bundleAvailable() {
        return OpenVinoRuntime.class.getResource(resourceBase() + MANIFEST) != null;
    }

    /**
     * Ensures the delegate can load, if and only if this model needs it.
     *
     * <p>Called before {@code loadModule}, never after: that call constructs the native runtime,
     * whose constructor runs delegate init.
     *
     * @param ptePath the model about to be loaded
     */
    static synchronized void ensureReady(Path ptePath) {
        if (configured) {
            return; // one-shot per process; the delegate's dlopen is too
        }
        String existing = System.getenv("OPENVINO_LIB_PATH");
        boolean overridden = existing != null && !existing.isEmpty();
        // Probed unconditionally, including when no bundle is present. An earlier revision skipped
        // the probe there to save "one extra .pte open" -- but measured, the probe is 6.5-10 us and
        // essentially FLAT in model size (1.6 KB -> 2.7 MB), because method_meta reads program
        // metadata and never the weights. Against model load it is 35% for a toy model and 1.5% for
        // a 2.7 MB one, paid once per load rather than per inference. That is not worth trading a
        // correct error message for -- and only this layer knows the platform and whether a bundle
        // is on the classpath, so only this layer can produce one.
        //
        // The probe also comes BEFORE the override check. Validating an override for every model
        // would fail a pure-XNNPACK workload that happens to carry a stale OPENVINO_LIB_PATH in its
        // environment -- punishing a caller for a variable their models never touch.
        if (!EtNative.pteUsesBackend(ptePath.toString(), BACKEND)) {
            return; // not an OpenVINO model; extract nothing, validate nothing, report nothing
        }
        if (!EtNative.backendRegistered(BACKEND)) {
            throw noDelegateMessage();
        }
        if (!overridden && !bundleAvailable()) {
            // The delegate IS linked -- it ships in every Linux runtime tarball, including
            // linux-aarch64, not just x86_64 -- so the model is runnable here the moment a runtime
            // is present. What is missing is the OpenVINO runtime, which is published per-platform
            // and may not exist for this one, so the message offers both routes rather than
            // promising an artifact that might not be published.
            throw new EngineException(
                    "This .pte uses the "
                            + BACKEND
                            + " delegate, but no OpenVINO runtime is available for "
                            + LibUtils.platform()
                            + ". Add the djl-executorch-engine "
                            + LibUtils.platform()
                            + "-openvino artifact to the runtime classpath if one is published for"
                            + " this platform, or set OPENVINO_LIB_PATH to the full path of an"
                            + " OpenVINO C library file you supply. The delegate itself is linked in"
                            + " this build, so the model runs once a runtime is resolved.");
        }
        if (overridden) {
            // An operator override always wins -- but a wrong one is worth catching here rather
            // than letting it reach the delegate, whose dlopen is once-only. Checked explicitly
            // rather than with a set-if-absent idiom, whose eager default would extract 72 MB even
            // when the variable is already correct.
            validateOverride(existing);
            configured = true;
            libPath = existing;
            return;
        }
        try {
            Path dir = ensureExtracted();
            String ours = resolvedLibPath();
            // Use what the native side reports as in force, not what we asked for. If something
            // installed a path after JVM start -- invisible to the System.getenv check above --
            // that path wins and this is how we find out.
            String effective = EtNative.setOpenVinoLibPathIfAbsent(ours);
            if (effective != null && !effective.equals(ours)) {
                logger.info(
                        "OpenVINO runtime already configured elsewhere; honouring {} instead of the"
                                + " vendored {}", effective, ours);
            }
            libPath = (effective == null) ? ours : effective;
            configured = true;
            logger.info("OpenVINO runtime resolved: {}", libPath);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to extract the OpenVINO runtime bundle", e);
        }
    }

    /**
     * The error for a build that links no OpenVINO delegate.
     *
     * <p>Every shipped runtime tarball carries the delegate, so in practice this means a runtime
     * tree supplied through the {@code ET_INSTALL} escape hatch that was built without it. No
     * runtime artifact can help — the model cannot execute here at all, so the only remedy is to
     * re-export. Deliberately distinct from the no-runtime error, which looks similar to a user and
     * has the opposite remedy.
     *
     * <p>Package-private and separate from its guard so it can be asserted directly: the guard is
     * false on every shipped platform, so a test gated on it would skip everywhere.
     *
     * @return the exception to throw
     */
    static EngineException noDelegateMessage() {
        return new EngineException(
                "This .pte uses the "
                        + BACKEND
                        + " delegate, which this build does not provide ("
                        + LibUtils.platform()
                        + "). The delegate ships only where the ExecuTorch runtime was built"
                        + " with it. Re-export without the OpenVINO partitioner to run here.");
    }

    /**
     * Rejects an {@code OPENVINO_LIB_PATH} that cannot work, before the delegate sees it.
     *
     * <p>Deliberately does <b>not</b> fall back to the vendored bundle. An operator who set this
     * variable meant to, and quietly substituting our runtime for theirs would turn a typo into a
     * silently different OpenVINO — which, because a {@code .pte} embeds a precompiled blob, could
     * surface much later as an import failure. Failing here names the value they actually set.
     *
     * @param value the environment variable's contents
     * @throws EngineException if it does not name a readable regular file
     */
    static void validateOverride(String value) {
        Path candidate;
        try {
            candidate = Paths.get(value);
        } catch (InvalidPathException e) {
            throw new EngineException(
                    "OPENVINO_LIB_PATH is not a usable path: '" + value + "'. It must be the full "
                            + "path to the OpenVINO C library FILE.", e);
        }
        if (Files.isDirectory(candidate)) {
            // Upstream's documented top mistake, and an easy one to make: the error the delegate
            // would otherwise produce mentions LD_LIBRARY_PATH, which reads like it wants a
            // directory. It does not.
            String example = bundleAvailable() ? bundleCLibrary() : "the OpenVINO C library";
            throw new EngineException(
                    "OPENVINO_LIB_PATH points at a directory: '" + value + "'. It must be the full "
                            + "path to the library FILE itself, e.g. <dir>/" + example + ".");
        }
        if (!Files.isRegularFile(candidate) || !Files.isReadable(candidate)) {
            throw new EngineException(
                    "OPENVINO_LIB_PATH does not name a readable file: '" + value + "'."
                            + (bundleAvailable()
                                    ? " Unset it to use the OpenVINO runtime vendored in this"
                                            + " engine's openvino artifact."
                                    : " Set it to the full path of the OpenVINO C library file."));
        }
    }

    /**
     * Extracts the bundle into the content-addressed cache, once.
     *
     * @return the flat directory holding the libraries
     * @throws IOException if extraction fails
     */
    static synchronized Path ensureExtracted() throws IOException {
        if (extracted != null) {
            return extracted;
        }
        String sha = manifest().getProperty("tarball_sha256");
        if (sha == null || sha.isEmpty()) {
            throw new IOException("OpenVINO bundle MANIFEST carries no tarball_sha256");
        }
        // Keyed on the upstream tarball hash rather than on a digest we compute: LibUtils hashes its
        // own resource, but that costs a full read on every JVM start, which is fine at 12 MB and
        // not at 72 MB on the model-load path. A cache hit here reads nothing.
        Path target = LibUtils.cacheRoot().resolve("openvino").resolve(sha);
        if (!Files.isDirectory(target)) {
            publish(target);
        }
        extracted = target;
        return target;
    }

    /** @return absolute path of the OpenVINO C library the bundle declares, or null before extraction */
    public static synchronized String resolvedLibPath() {
        if (libPath != null) {
            return libPath;
        }
        if (extracted == null) {
            return null;
        }
        libPath = extracted.resolve(bundleCLibrary()).toAbsolutePath().toString();
        return libPath;
    }

    // Extract into a staging directory, then publish by atomic rename. Nothing is ever loaded out
    // of the staging directory, which is what lets a loser in a race delete its own work even on a
    // platform that refuses to delete a loaded library.
    // Package-private, not private, so OpenVinoConcurrentExtractionTest can drive the adoption
    // branch below directly. ensureExtracted() cannot reach it: it is static synchronized with a
    // cached fast path, so within one JVM only the first caller ever publishes.
    static void publish(Path target) throws IOException {
        Files.createDirectories(target.getParent());
        Path staging = Files.createTempDirectory(target.getParent(), "staging-");
        try {
            // The bundle names its own files, so nothing here reconstructs a versioned filename.
            // That is what removes the ABI concept from this path: the Windows bundle ships six
            // unversioned DLLs and carries no ov_abi key at all.
            for (String lib : bundleLibs()) {
                copy(resourceBase() + "lib/" + lib, staging.resolve(lib));
            }
            copy(resourceBase() + BUILDINFO, staging.resolve(BUILDINFO));
            try {
                Files.move(staging, target, StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException e) {
                // A concurrent JVM published first. The path is content-addressed, so its bytes are
                // ours byte-for-byte; adopt rather than overwrite a directory another process may
                // already have loaded from.
                if (!Files.isDirectory(target)) {
                    throw e;
                }
            }
        } finally {
            deleteRecursivelyIfPresent(staging);
        }
    }

    private static void copy(String resource, Path target) throws IOException {
        try (InputStream is = open(resource); OutputStream os = Files.newOutputStream(target)) {
            byte[] buf = new byte[BUF];
            int n;
            while ((n = is.read(buf)) != -1) {
                os.write(buf, 0, n);
            }
        }
    }

    private static void deleteRecursivelyIfPresent(Path dir) throws IOException {
        if (!Files.exists(dir)) {
            return;
        }
        List<Path> paths = new ArrayList<>();
        try (var walk = Files.walk(dir)) {
            walk.forEach(paths::add);
        }
        for (int i = paths.size() - 1; i >= 0; i--) {
            Files.deleteIfExists(paths.get(i));
        }
    }

    private static Properties manifest() {
        return readProperties(resourceBase() + MANIFEST);
    }

    /**
     * @return the trimmed value the staged bundle's MANIFEST declares for {@code key}
     * @throws IllegalStateException if the MANIFEST carries no such key
     */
    private static String bundleManifestValue(String key) {
        String value = manifest().getProperty(key);
        if (value == null || value.isBlank()) {
            throw new IllegalStateException("OpenVINO bundle MANIFEST carries no " + key);
        }
        return value.trim();
    }

    /** @return the library filenames the staged bundle declares, in the order it listed them */
    private static List<String> bundleLibs() {
        return List.of(bundleManifestValue("libs").split("\\s+"));
    }

    /** @return the filename of the OpenVINO C API library the delegate must dlopen */
    private static String bundleCLibrary() {
        return bundleManifestValue("c_library");
    }

    private static Properties readProperties(String resource) {
        Properties props = new Properties();
        try (InputStream is = open(resource)) {
            props.load(new java.io.InputStreamReader(is, StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new IllegalStateException("Cannot read " + resource, e);
        }
        return props;
    }

    private static InputStream open(String resource) {
        InputStream is = OpenVinoRuntime.class.getResourceAsStream(resource);
        if (is == null) {
            throw new IllegalStateException("OpenVINO bundle resource missing: " + resource);
        }
        return is;
    }

    private static String resourceBase() {
        return "/native/" + LibUtils.platform() + "/openvino/";
    }
}
