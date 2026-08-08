package org.measly.executorch.stress;

import ai.djl.util.JsonUtils;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * The committed golden digests for the stress model, and the check against them.
 *
 * <p>This is oracle layer 1 of 2 — "is this the right model, wired right?". It is deliberately not
 * tight: the goldens were produced by ExecuTorch's Python runtime, a different build of the same
 * runtime, so a float tolerance is the honest comparison. Layer 2 (bitwise self-reference, see
 * {@link StressGateIT}) is the sharp instrument for concurrency corruption.
 *
 * <p>Full tensors are not stored — eight cases of 8192-float outputs would be megabytes of JSON.
 * Inputs are reconstructed from two scalars by {@link StressTranslator#buildInput}, and the output
 * is captured as three reductions plus 16 strided samples.
 */
public final class StressGolden {

    /** Relative tolerance for reductions and samples. */
    private static final double RTOL = 1e-4;

    /** Absolute tolerance floor, so values near zero do not demand impossible relative accuracy. */
    private static final double ATOL = 1e-5;

    /** Model geometry, as exported. */
    public static final class Config {
        public final int batch;
        public final int hidden;
        public final int depth;
        public final int nBuckets;
        public final float ramp;

        Config(int batch, int hidden, int depth, int nBuckets, float ramp) {
            this.batch = batch;
            this.hidden = hidden;
            this.depth = depth;
            this.nBuckets = nBuckets;
            this.ramp = ramp;
        }
    }

    /** One recorded case: the two steering scalars and the digest of the expected output. */
    public static final class Case {
        public final String name;
        public final float v1;
        public final float v2;
        public final double sum;
        public final double absSum;
        public final double maxAbs;
        public final float[] samples;

        Case(String name, float v1, float v2, double sum, double absSum, double maxAbs, float[] samples) {
            this.name = name;
            this.v1 = v1;
            this.v2 = v2;
            this.sum = sum;
            this.absSum = absSum;
            this.maxAbs = maxAbs;
            this.samples = samples;
        }
    }

    private final Config config;
    private final int sampleStride;
    private final List<Case> cases;

    private StressGolden(Config config, int sampleStride, List<Case> cases) {
        this.config = config;
        this.sampleStride = sampleStride;
        this.cases = Collections.unmodifiableList(cases);
    }

    public Config config() {
        return config;
    }

    public int sampleStride() {
        return sampleStride;
    }

    public List<Case> cases() {
        return cases;
    }

    /** Number of elements in the model's output tensor. */
    public int outputLength() {
        return config.batch * config.hidden;
    }

    public static StressGolden load(Path path) {
        JsonObject root;
        try (Reader r = Files.newBufferedReader(path)) {
            root = JsonUtils.GSON.fromJson(r, JsonObject.class);
        } catch (IOException | JsonParseException e) {
            throw new IllegalStateException(
                    "Cannot read golden file "
                            + path
                            + " — regenerate it with tools/scripts/export_stress_model.py ("
                            + e.getMessage()
                            + ")",
                    e);
        }
        if (root == null) {
            throw new IllegalStateException("Golden file " + path + " is empty");
        }
        JsonObject cfg = require(root, "config", path).getAsJsonObject();
        Config config =
                new Config(
                        cfg.get("batch").getAsInt(),
                        cfg.get("hidden").getAsInt(),
                        cfg.get("depth").getAsInt(),
                        cfg.get("nBuckets").getAsInt(),
                        cfg.get("ramp").getAsFloat());
        int stride = require(root, "sampleStride", path).getAsInt();
        JsonArray arr = require(root, "cases", path).getAsJsonArray();
        if (arr.size() == 0) {
            throw new IllegalStateException("Golden file " + path + " has no cases");
        }

        int expectedSamples = (config.batch * config.hidden) / stride;
        List<Case> cases = new ArrayList<>(arr.size());
        for (int i = 0; i < arr.size(); i++) {
            JsonObject c = arr.get(i).getAsJsonObject();
            String name = c.get("name").getAsString();
            JsonArray s = c.get("samples").getAsJsonArray();
            if (s.size() != expectedSamples) {
                throw new IllegalStateException(
                        "Golden case "
                                + name
                                + " has "
                                + s.size()
                                + " samples but batch*hidden/stride implies "
                                + expectedSamples
                                + " — the golden file and its .pte have drifted; regenerate BOTH"
                                + " with tools/scripts/export_stress_model.py");
            }
            float[] samples = new float[s.size()];
            for (int j = 0; j < samples.length; j++) {
                samples[j] = s.get(j).getAsFloat();
            }
            cases.add(
                    new Case(
                            name,
                            c.get("v1").getAsFloat(),
                            c.get("v2").getAsFloat(),
                            c.get("sum").getAsDouble(),
                            c.get("absSum").getAsDouble(),
                            c.get("maxAbs").getAsDouble(),
                            samples));
        }
        return new StressGolden(config, stride, cases);
    }

    private static com.google.gson.JsonElement require(JsonObject o, String key, Path path) {
        com.google.gson.JsonElement e = o.get(key);
        if (e == null) {
            throw new IllegalStateException(
                    "Golden file " + path + " is missing required key '" + key + "'");
        }
        return e;
    }

    /** Throws {@link AssertionError} if {@code output} does not match the recorded digest. */
    public void verify(Case c, float[] output) {
        int expectedLen = outputLength();
        if (output.length != expectedLen) {
            throw new AssertionError(
                    "case " + c.name + ": output length " + output.length + " != " + expectedLen);
        }
        // Samples first: a wrong element is localised to sample[i] rather than reported as a
        // reduction mismatch, so the diagnostic names the actual failure.
        for (int i = 0; i < c.samples.length; i++) {
            close(c.name, "sample[" + i + "]", c.samples[i], output[i * sampleStride]);
        }
        double sum = 0;
        double absSum = 0;
        double maxAbs = 0;
        for (float v : output) {
            sum += v;
            absSum += Math.abs(v);
            maxAbs = Math.max(maxAbs, Math.abs(v));
        }
        close(c.name, "sum", c.sum, sum);
        close(c.name, "absSum", c.absSum, absSum);
        close(c.name, "maxAbs", c.maxAbs, maxAbs);
    }

    private static void close(String caseName, String what, double expected, double actual) {
        double tol = ATOL + RTOL * Math.abs(expected);
        if (!(Math.abs(expected - actual) <= tol)) {
            throw new AssertionError(
                    "case "
                            + caseName
                            + ": "
                            + what
                            + " expected "
                            + expected
                            + " but was "
                            + actual
                            + " (tolerance "
                            + tol
                            + ")");
        }
    }
}
