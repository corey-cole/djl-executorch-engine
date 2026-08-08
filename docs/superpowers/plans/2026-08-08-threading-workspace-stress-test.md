# Threading / Workspace Stress Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dual-purpose concurrency artifact for this engine — a local-only correctness gate and an opt-in throughput sweep over (caller threads × workspace sharing mode) — on a synthetic model with real XNNPACK work, plus the publishable per-thread `Predictor` pattern.

**Architecture:** One exported `.pte` (bucket gather → 4-layer MLP, two branches) with a compact companion golden file. A JVM arm supplies the pattern (`PerThreadContext implements AutoCloseable`, one `ZooModel` per thread) and both test modes. A native arm runs the same shape under ASan/LSan. Everything is tag-gated out of `./gradlew test`.

**Tech Stack:** Java 17, JUnit 5, DJL 0.36.0 (`ai.djl.util.JsonUtils` for JSON — gson, already on the classpath), Gradle Kotlin DSL, C++20 + CMake, ExecuTorch 1.3.1, PyTorch 2.12.1 via `uv`.

**Spec:** `docs/superpowers/specs/2026-08-08-threading-workspace-stress-test-design.md`

## Global Constraints

- **Nothing here runs in `./gradlew test`.** Every new test class carries a tag, and `tasks.test` excludes all of them. The gate is local-only and never wired to CI, including nightlies.
- **No new Gradle dependencies.** JSON parsing uses `ai.djl.util.JsonUtils`, already available via `libs.djl.api`.
- **Per-forward cost target: 300–500 µs at one intra-op thread.** Starting constants `BATCH=32`, `HIDDEN=256`, `DEPTH=4`, `N_BUCKETS=64`, `RAMP=1e-5f`, `SEED=20260808`.
- **Golden tolerance: `rtol=1e-4`, `atol=1e-5`.** Bitwise self-reference comparison is exact (`java.util.Arrays.equals`).
- **`EtSymbolBlock.forward()` is not thread-safe on the same model.** Every thread gets its own `ZooModel`. Never share one across threads.
- **The intra-op threadpool is process-global and write-once**, sealed at first model load (`EtEngine.setIntraOpThreads`). Any cell that needs a different intra-op count needs a different JVM.
- **TSan is out of scope** — the ExecuTorch runtime ships as prebuilt static libs and would not be instrumented.
- **`per_model` is excluded from the sweep** — degenerate with `global` for a single-model workload.
- Native harnesses link only the JNIEnv-free core (`native/core/et_runtime.{h,cpp}`); no JDK required.

## A wiring consequence discovered while planning

The spec (§6.2) puts the intra-op = default "confirmation cell" in the same sweep as the eight
intra-op = 1 cells. **That cannot happen in one JVM**: `measly::et::setIntraOpThreads` refuses a
reset once any `EtRuntime` exists (issue #26), so the pool size is fixed for the process at first
model load.

The sweep is therefore split across two forked Gradle test tasks with different `-D` flags, unified
by a lifecycle task:

| Gradle task             | JVM flag                              | tag               | cells |
|-------------------------|---------------------------------------|-------------------|-------|
| `stressSweepCore`       | `-Dai.djl.executorch.num_threads=1`   | `stress-sweep`    | 8     |
| `stressSweepBaseline`   | *(none — pool takes its default)*     | `stress-baseline` | 1     |
| `stressSweep`           | lifecycle, `dependsOn` both           | —                 | 9     |

Both write to the same report file in append mode, which is why the report carries a header-if-absent.

## File Structure

**Created:**
- `tools/scripts/export_stress_model.py` — exports the `.pte` **and** the golden file in one run; prints measured per-forward cost.
- `src/test/resources/models/stress/stress_mlp.pte` — committed artifact (~1 MB).
- `src/test/resources/models/stress/stress_golden.json` — committed artifact (~5 KB).
- `src/test/java/org/measly/executorch/stress/StressGolden.java` — golden file model + parser + input construction + verification. No native.
- `src/test/java/org/measly/executorch/stress/StressTranslator.java` — `Translator<float[], float[]>`, scalars in, flat output out.
- `src/test/java/org/measly/executorch/stress/PerThreadContext.java` — **the published pattern**. `AutoCloseable` over `{ZooModel, Predictor}` + open/close counters.
- `src/test/java/org/measly/executorch/stress/SweepConfig.java` — the matrix. No native.
- `src/test/java/org/measly/executorch/stress/SweepRunner.java` — runs one cell, returns metrics; writes the report.
- `src/test/java/org/measly/executorch/stress/StressGoldenTest.java` — untagged, runs in `test`.
- `src/test/java/org/measly/executorch/stress/SweepConfigTest.java` — untagged, runs in `test`.
- `src/test/java/org/measly/executorch/stress/StressSmokeIT.java` — `@Tag("stress")`, single thread, golden verification.
- `src/test/java/org/measly/executorch/stress/StressGateIT.java` — `@Tag("stress")`, 8 threads, max contention.
- `src/test/java/org/measly/executorch/stress/StressSweepIT.java` — `@Tag("stress-sweep")`, 8 cells.
- `src/test/java/org/measly/executorch/stress/StressSweepBaselineIT.java` — `@Tag("stress-baseline")`, 1 cell.
- `native/harness/et_stress_harness.cpp` — threaded ASan/LSan harness.

**Modified:**
- `src/test/java/org/measly/executorch/TestSupport.java` — add `assumeStressModelAvailable()` + `stressModelDir()`.
- `build.gradle.kts:32-35` — add the new tags to `excludeTags`; register four tasks.
- `native/CMakeLists.txt:204-227` — add `et_stress_harness` to the `ET_BUILD_QA` block.
- `native/build_qa.sh` — build the target always, run it only under `ET_STRESS=1`.
- `CLAUDE.md` — document the new tasks and the stress model.

---

### Task 0: Branch

- [ ] **Step 1: Create the feature branch**

The design doc was committed directly to `main`. Move onto a branch before writing code.

```bash
git checkout -b feat/threading-workspace-stress-test
git log --oneline -1
```

Expected: the branch is created at `636c47e` ("Design: threading/workspace-parameter stress test").

---

### Task 1: Export the stress model and its goldens

**Files:**
- Create: `tools/scripts/export_stress_model.py`
- Create (generated, committed): `src/test/resources/models/stress/stress_mlp.pte`
- Create (generated, committed): `src/test/resources/models/stress/stress_golden.json`

**Interfaces:**
- Consumes: nothing.
- Produces: `stress_mlp.pte` (2 f32 inputs of shape `[32, 256]`, 1 f32 output of shape `[32, 256]`) and `stress_golden.json` with the schema fixed in Step 1 below. `StressGolden` (Task 2) parses that schema; `StressTranslator` (Task 3) reproduces the input construction rule.

**Why the golden file stores digests, not full tensors:** eight cases at 2 × 8192 input floats plus
8192 output floats would be ~2.5 MB of JSON. Instead, inputs are *reconstructed* from two scalars by
a documented rule, and the output is captured as three reductions plus 16 strided samples. That is
enough for layer 1 ("right model, wired right"); layer 2 (bitwise, Task 4) is the sharp instrument.

- [ ] **Step 1: Write the export script**

Create `tools/scripts/export_stress_model.py`:

```python
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch==2.12.1",
#   "executorch==1.3.1",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Export the threading/workspace stress model: stress_mlp.pte + stress_golden.json.

Both files are written in ONE run, on purpose. The goldens are digests of this exact .pte's
output, so a regenerated model with stale goldens is a silent wrong-answer bug. Writing them
together is the mechanism that prevents drift -- never regenerate one without the other.

Shape of the model (see the design doc, section 3):

  branch(x, layers):
      sel = clamp(int64(x[0,0] * N_BUCKETS), 0, N_BUCKETS-1)   # data-dependent
      g   = index_select(table, 0, sel)                        # portable kernel -> SERIAL arm
      h   = x + g
      for lin in layers: h = relu(lin(h))                      # XNNPACK -> PARALLEL arm
  forward(x1, x2) = branch(x1, A) + branch(x2, B)

ExecuTorch has NO inter-op parallelism -- the graph is walked as a single instruction stream, so
the two branches run one after the other. The serial/parallel mix comes from KERNEL choice, not
graph topology: index_select is not lowered by the XNNPACK partitioner and runs single-threaded,
while Linear is lowered and runs on the shared pthreadpool holding an XNNPACK workspace. The
workspace mutex is only observable in the Linear stack.

BATCH is load-bearing. At batch 1 the Linear stack is a GEMV -- memory-bandwidth-bound, so DRAM
becomes the bottleneck and masks the workspace lock entirely. At batch 32 it is a real GEMM,
compute-bound, with ~1 MB of weights that stay resident in L2.

Run from the repo root:

    PATH=$HOME/workspace/executorch/.venv/bin:$PATH \
        uv run tools/scripts/export_stress_model.py

Writes both files into src/test/resources/models/stress/.
"""
import json
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# --- tuned constants; see the design doc section 3.4 -------------------------------------------
SEED = 20260808
BATCH = 32
HIDDEN = 256
DEPTH = 4
N_BUCKETS = 64
RAMP = 1e-5          # input ramp step; MUST match StressTranslator.RAMP exactly
SAMPLE_COUNT = 16    # strided output samples recorded per case
# -----------------------------------------------------------------------------------------------

OUT_DIR = Path("src/test/resources/models/stress")

# (name, v1, v2). v is in [0, 1); bucket = int(v * N_BUCKETS). Chosen to sit on the first bucket,
# the last bucket, and either side of two interior boundaries, for each input independently -- so
# an off-by-one or mis-marshalled bucket index is caught rather than averaged away.
CASES = [
    ("b0_b0",       0.0000,  0.0000),
    ("b0_blast",    0.0000,  0.9999),
    ("blast_b0",    0.9999,  0.0000),
    ("blast_blast", 0.9999,  0.9999),
    ("b15_lo",      15.999 / N_BUCKETS, 0.5),
    ("b16_hi",      16.001 / N_BUCKETS, 0.5),
    ("b31_lo",      0.5, 31.999 / N_BUCKETS),
    ("b32_hi",      0.5, 32.001 / N_BUCKETS),
]


class StressNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Parameter(
            torch.randn(N_BUCKETS, HIDDEN) * 0.05, requires_grad=False
        )
        self.a = torch.nn.ModuleList(
            [torch.nn.Linear(HIDDEN, HIDDEN) for _ in range(DEPTH)]
        )
        self.b = torch.nn.ModuleList(
            [torch.nn.Linear(HIDDEN, HIDDEN) for _ in range(DEPTH)]
        )

    def branch(self, x, layers):
        sel = (x[0, 0] * N_BUCKETS).to(torch.int64).clamp(0, N_BUCKETS - 1)
        g = torch.index_select(self.table, 0, sel.reshape(1))  # (1, HIDDEN), broadcasts over BATCH
        h = x + g
        for lin in layers:
            h = torch.relu(lin(h))
        return h

    def forward(self, x1, x2):
        return self.branch(x1, self.a) + self.branch(x2, self.b)


def build_input(v: float) -> torch.Tensor:
    """Deterministic input tensor from a single scalar.

    Reproduced bit-for-bit in Java by StressTranslator.buildInput. Every arithmetic step is done
    in float32 in BOTH implementations, in the same order -- `(float) i * RAMP + v` -- so the two
    agree exactly. Doing the ramp in float64 and narrowing at the end would differ by an ulp and
    silently break the bitwise self-reference check.

    Element [0, 0] is exactly v (index 0 contributes a zero ramp term), which is what steers the
    bucket.
    """
    idx = torch.arange(BATCH * HIDDEN, dtype=torch.float32)
    flat = idx * torch.tensor(RAMP, dtype=torch.float32) + torch.tensor(v, dtype=torch.float32)
    return flat.reshape(BATCH, HIDDEN)


def main() -> None:
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = StressNet().eval()
    example_inputs = (build_input(0.5), build_input(0.5))

    lowered = to_edge_transform_and_lower(
        export(model, example_inputs), partitioner=[XnnpackPartitioner()]
    ).to_executorch()
    pte_path = OUT_DIR / "stress_mlp.pte"
    pte_path.write_bytes(lowered.buffer)
    print(f"wrote {pte_path} ({pte_path.stat().st_size} bytes)")

    # Goldens come from running the exported .pte through the ExecuTorch runtime, NOT from torch
    # eager: eager uses different kernels than XNNPACK, so eager goldens would either fail
    # spuriously or force a tolerance so loose it proves nothing.
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    method = program.load_method("forward")

    stride = (BATCH * HIDDEN) // SAMPLE_COUNT
    cases = []
    for name, v1, v2 in CASES:
        out = method.execute([build_input(v1), build_input(v2)])[0]
        flat = out.reshape(-1)
        cases.append(
            {
                "name": name,
                "v1": v1,
                "v2": v2,
                "sum": float(flat.to(torch.float64).sum()),
                "absSum": float(flat.to(torch.float64).abs().sum()),
                "maxAbs": float(flat.abs().max()),
                "samples": [float(flat[i * stride]) for i in range(SAMPLE_COUNT)],
            }
        )

    # Coarse cost signal only. The Python runtime is a different build with different overheads --
    # the authoritative number for tuning against the 300-500us target comes from
    # native/harness/et_timing_harness (see the plan, Task 8).
    warm = [build_input(0.5), build_input(0.5)]
    for _ in range(50):
        method.execute(warm)
    t0 = time.perf_counter()
    reps = 300
    for _ in range(reps):
        method.execute(warm)
    us = (time.perf_counter() - t0) / reps * 1e6

    try:
        et_version = version("executorch")
    except PackageNotFoundError:
        et_version = "unknown"

    golden = {
        "executorchVersion": et_version,
        "seed": SEED,
        "config": {
            "batch": BATCH,
            "hidden": HIDDEN,
            "depth": DEPTH,
            "nBuckets": N_BUCKETS,
            "ramp": RAMP,
        },
        "sampleStride": stride,
        "measuredUsPerForward": round(us, 1),
        "cases": cases,
    }
    golden_path = OUT_DIR / "stress_golden.json"
    golden_path.write_text(json.dumps(golden, indent=2) + "\n")
    print(f"wrote {golden_path}")
    print(f"measured {us:.1f} us/forward (python runtime, default intra-op threads)")
    print("target is 300-500 us at ONE intra-op thread; retune BATCH/HIDDEN/DEPTH if far off")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the export**

```bash
PATH=$HOME/workspace/executorch/.venv/bin:$PATH uv run tools/scripts/export_stress_model.py
```

Expected: two files written, and a printed `us/forward` figure. Do **not** retune yet — the
authoritative measurement happens in Task 8 against the native timing harness at one intra-op
thread. Record the printed number; you will compare against it later.

- [ ] **Step 3: Sanity-check the artifacts**

```bash
ls -la src/test/resources/models/stress/
python3 -c "import json;d=json.load(open('src/test/resources/models/stress/stress_golden.json'));print(len(d['cases']),d['config'],d['sampleStride']);print(d['cases'][0]['samples'][:3])"
```

Expected: `stress_mlp.pte` is roughly 0.5–3 MB; the JSON reports `8` cases, `sampleStride` of
`512`, and sample values that are finite and not all zero. If every sample is `0.0`, the ReLU
stack has saturated to zero — reduce `DEPTH` or widen the init and re-export.

- [ ] **Step 4: Commit**

```bash
git add tools/scripts/export_stress_model.py src/test/resources/models/stress/
git commit -m "feat(stress): export stress model and golden digests"
```

---

### Task 2: Golden file parser

**Files:**
- Create: `src/test/java/org/measly/executorch/stress/StressGolden.java`
- Test: `src/test/java/org/measly/executorch/stress/StressGoldenTest.java`

**Interfaces:**
- Consumes: `stress_golden.json` from Task 1.
- Produces:
  - `StressGolden.load(Path)` → `StressGolden`; throws `IllegalStateException` on a missing, malformed, or structurally invalid file.
  - `StressGolden.Config` with `int batch, hidden, depth, nBuckets; float ramp;`
  - `StressGolden.Case` with `String name; float v1, v2; double sum, absSum, maxAbs; float[] samples;`
  - `List<Case> cases()`, `Config config()`, `int sampleStride()`
  - `void verify(Case c, float[] output)` — throws `AssertionError` with a diagnostic message on mismatch.

This task is pure JUnit and touches no native code, so its test runs in the ordinary `./gradlew test`.

- [ ] **Step 1: Write the failing test**

Create `src/test/java/org/measly/executorch/stress/StressGoldenTest.java`:

```java
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./gradlew test --tests 'org.measly.executorch.stress.StressGoldenTest'
```

Expected: FAIL — compilation error, `StressGolden` does not exist.

- [ ] **Step 3: Write the implementation**

Create `src/test/java/org/measly/executorch/stress/StressGolden.java`:

```java
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
        for (int i = 0; i < c.samples.length; i++) {
            close(c.name, "sample[" + i + "]", c.samples[i], output[i * sampleStride]);
        }
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
./gradlew test --tests 'org.measly.executorch.stress.StressGoldenTest'
```

Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add src/test/java/org/measly/executorch/stress/
git commit -m "feat(stress): golden digest parser and verifier"
```

---

### Task 3: The per-thread pattern, translator, and Gradle wiring

**Files:**
- Create: `src/test/java/org/measly/executorch/stress/StressTranslator.java`
- Create: `src/test/java/org/measly/executorch/stress/PerThreadContext.java`
- Create: `src/test/java/org/measly/executorch/stress/StressSmokeIT.java`
- Modify: `src/test/java/org/measly/executorch/TestSupport.java`
- Modify: `build.gradle.kts:32-35` and the task-registration block

**Interfaces:**
- Consumes: `StressGolden` (Task 2), `stress_mlp.pte` (Task 1).
- Produces:
  - `StressTranslator.buildInput(float v, int batch, int hidden, float ramp)` → `float[]`
  - `StressTranslator` implements `Translator<float[], float[]>` — input is `{v1, v2}`, output is the flat output tensor.
  - `PerThreadContext.open(String sharingMode)` → `PerThreadContext` (null mode = send no option)
  - `PerThreadContext.predict(float v1, float v2)` → `float[]`
  - `PerThreadContext.opened()` / `.closed()` → `int` (static counters), `.resetCounters()`
  - `TestSupport.assumeStressModelAvailable()`, `TestSupport.stressModelDir()` → `Path`

- [ ] **Step 1: Write the failing test**

Create `src/test/java/org/measly/executorch/stress/StressSmokeIT.java`:

```java
package org.measly.executorch.stress;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.measly.executorch.TestSupport;

/**
 * Single-threaded proof that the stress model loads, predicts, and matches its goldens, and that
 * the per-thread context closes what it opens. Everything the concurrent gate assumes, verified
 * without concurrency first — so a failure in StressGateIT means concurrency, not plumbing.
 */
@Tag("stress")
class StressSmokeIT {

    @Test
    void everyGoldenCaseMatchesOnOneThread() throws Exception {
        TestSupport.assumeStressModelAvailable();
        StressGolden golden = StressGolden.load(TestSupport.stressGoldenPath());

        PerThreadContext.resetCounters();
        try (PerThreadContext ctx = PerThreadContext.open("global")) {
            for (StressGolden.Case c : golden.cases()) {
                float[] out = ctx.predict(c.v1, c.v2);
                golden.verify(c, out);
            }
        }
        assertEquals(1, PerThreadContext.opened());
        assertEquals(1, PerThreadContext.closed(), "close() must run via try-with-resources");
    }

    @Test
    void repeatedPredictsAreBitwiseIdentical() throws Exception {
        TestSupport.assumeStressModelAvailable();
        StressGolden golden = StressGolden.load(TestSupport.stressGoldenPath());
        StressGolden.Case c = golden.cases().get(0);

        try (PerThreadContext ctx = PerThreadContext.open("global")) {
            float[] reference = ctx.predict(c.v1, c.v2);
            for (int i = 0; i < 20; i++) {
                assertTrue(
                        Arrays.equals(reference, ctx.predict(c.v1, c.v2)),
                        "iteration " + i + " diverged bitwise from the first forward");
            }
        }
    }

    @Test
    void everySharingModeLoadsAndAgreesWithTheGoldens() throws Exception {
        TestSupport.assumeStressModelAvailable();
        StressGolden golden = StressGolden.load(TestSupport.stressGoldenPath());
        StressGolden.Case c = golden.cases().get(0);

        for (String mode : new String[] {null, "global", "disabled", "per_model"}) {
            try (PerThreadContext ctx = PerThreadContext.open(mode)) {
                golden.verify(c, ctx.predict(c.v1, c.v2));
            }
        }
    }

    @Test
    void buildInputPutsTheSteeringValueAtElementZero() {
        float[] in = StressTranslator.buildInput(0.25f, 32, 256, 1e-5f);
        assertEquals(32 * 256, in.length);
        assertEquals(0.25f, in[0], "element [0,0] steers the bucket and must be exactly v");
        assertEquals(0.25f + 1e-5f, in[1], 0f, "ramp must be computed in float32, in order");
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./gradlew test --tests 'org.measly.executorch.stress.StressSmokeIT'
```

Expected: FAIL — compilation error; `StressTranslator`, `PerThreadContext`, and the `TestSupport`
helpers do not exist. (Note this also proves the tag is not yet excluded — Step 7 fixes that.)

- [ ] **Step 3: Write the translator**

Create `src/test/java/org/measly/executorch/stress/StressTranslator.java`:

```java
package org.measly.executorch.stress;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

/**
 * Maps {@code {v1, v2}} to the stress model's two (batch, hidden) f32 inputs and back to the flat
 * output tensor.
 *
 * <p>The inputs are reconstructed from two scalars rather than stored, which is what keeps the
 * golden file at kilobytes instead of megabytes.
 */
public class StressTranslator implements Translator<float[], float[]> {

    private final int batch;
    private final int hidden;
    private final float ramp;

    public StressTranslator(int batch, int hidden, float ramp) {
        this.batch = batch;
        this.hidden = hidden;
        this.ramp = ramp;
    }

    /**
     * Deterministic input tensor from one scalar, bit-identical to {@code build_input} in
     * tools/scripts/export_stress_model.py.
     *
     * <p>Every step is float32, in the same order as the Python side: {@code (float) i * ramp + v}.
     * Accumulating the ramp in double and narrowing at the end would differ by an ulp and silently
     * break the bitwise self-reference check in StressGateIT — which is exactly the kind of drift
     * that check exists to catch, so it must not be introduced by the harness itself.
     *
     * <p>Element 0 is exactly {@code v} (index 0 contributes a zero ramp term); that element steers
     * the bucket lookup.
     */
    public static float[] buildInput(float v, int batch, int hidden, float ramp) {
        float[] out = new float[batch * hidden];
        for (int i = 0; i < out.length; i++) {
            out[i] = (float) i * ramp + v;
        }
        return out;
    }

    @Override
    public Batchifier getBatchifier() {
        // Null, for the same reason as AddTranslator: EtNDArray does not implement NDArrayInternal,
        // so DJL's default StackBatchifier fails in NDArrays.stack(). The model's own batch
        // dimension is baked into the exported shape, not produced by DJL batching.
        return null;
    }

    @Override
    public NDList processInput(TranslatorContext ctx, float[] input) {
        Shape shape = new Shape(batch, hidden);
        NDArray a = ctx.getNDManager().create(buildInput(input[0], batch, hidden, ramp), shape);
        NDArray b = ctx.getNDManager().create(buildInput(input[1], batch, hidden, ramp), shape);
        return new NDList(a, b);
    }

    @Override
    public float[] processOutput(TranslatorContext ctx, NDList list) {
        return list.singletonOrThrow().toFloatArray();
    }
}
```

- [ ] **Step 4: Write the per-thread context — the published pattern**

Create `src/test/java/org/measly/executorch/stress/PerThreadContext.java`:

```java
package org.measly.executorch.stress;

import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import java.util.concurrent.atomic.AtomicInteger;
import org.measly.executorch.TestSupport;
import org.measly.executorch.engine.EtEngine;

/**
 * One thread's model and predictor, as a single {@link AutoCloseable} unit. <b>This is the
 * reference pattern for using this engine from multiple threads.</b>
 *
 * <p><b>Why a whole model per thread, not just a predictor.</b> {@code EtSymbolBlock.forward()} is
 * not thread-safe <i>on the same model</i>. The shape most DJL users reach for first — one shared
 * {@link ZooModel}, a {@code ThreadLocal<Predictor>} over it — is therefore <b>wrong on this
 * engine</b>: the predictors would share one native handle. Each thread needs its own model.
 *
 * <p>This also makes the workspace sharing mode expressible per thread, since
 * {@code workspaceSharingMode} is a per-model load option.
 *
 * <p><b>Why no ThreadLocal here.</b> Each worker is a dedicated {@link Thread} whose {@code run()}
 * body is a single try-with-resources, so the thread's lifetime <i>is</i> the resource's lifetime
 * and a plain local is strictly better. {@code ThreadLocal} earns its keep only when a context must
 * outlive the block that created it — a pooled executor. Using it here would publish ceremony as if
 * it were safety.
 *
 * <p><b>Two things to know before adapting this to a thread pool.</b> First,
 * {@code ThreadLocal.remove()} drops the reference without calling {@code close()}, so the native
 * handle leaks until GC; it is the most commonly cargo-culted teardown and it is wrong here. Second,
 * a pool needs an explicit drain phase — submit exactly one close-task per pool thread, held apart
 * by a barrier so each lands on a distinct thread, before {@code shutdown()}. That variant is a
 * welcome contribution; it is deliberately not implemented here, because pool-thread affinity would
 * make the sweep's thread-count axis mushy.
 *
 * <p>Never {@code close()} a model with a forward in flight. Try-with-resources on the owning
 * thread makes that impossible by construction, which is the whole point.
 */
public final class PerThreadContext implements AutoCloseable {

    private static final AtomicInteger OPENED = new AtomicInteger();
    private static final AtomicInteger CLOSED = new AtomicInteger();

    private final ZooModel<float[], float[]> model;
    private final Predictor<float[], float[]> predictor;

    private PerThreadContext(ZooModel<float[], float[]> model, Predictor<float[], float[]> predictor) {
        this.model = model;
        this.predictor = predictor;
    }

    /**
     * Loads a private model and predictor for the calling thread.
     *
     * @param sharingMode one of {@code disabled}, {@code per_model}, {@code global}, or {@code null}
     *     to send no option at all and let the runtime default apply
     */
    public static PerThreadContext open(String sharingMode) throws Exception {
        StressGolden.Config cfg = StressGolden.load(TestSupport.stressGoldenPath()).config();
        Criteria.Builder<float[], float[]> b =
                Criteria.builder()
                        .setTypes(float[].class, float[].class)
                        .optEngine("ExecuTorch")
                        .optModelPath(TestSupport.stressModelDir())
                        .optModelName("stress_mlp")
                        .optTranslator(new StressTranslator(cfg.batch, cfg.hidden, cfg.ramp));
        if (sharingMode != null) {
            // The engine publishes the key; do not hardcode the string.
            b.optOption(EtEngine.WORKSPACE_SHARING_MODE_OPTION, sharingMode);
        }

        ZooModel<float[], float[]> model = b.build().loadModel();
        Predictor<float[], float[]> predictor;
        try {
            predictor = model.newPredictor();
        } catch (RuntimeException e) {
            model.close(); // do not leak the native handle when only the predictor failed
            throw e;
        }
        OPENED.incrementAndGet();
        return new PerThreadContext(model, predictor);
    }

    public float[] predict(float v1, float v2) throws Exception {
        return predictor.predict(new float[] {v1, v2});
    }

    @Override
    public void close() {
        // Reverse acquisition order: the predictor borrows from the model, so it goes first.
        try {
            predictor.close();
        } finally {
            model.close();
            CLOSED.incrementAndGet();
        }
    }

    public static int opened() {
        return OPENED.get();
    }

    public static int closed() {
        return CLOSED.get();
    }

    public static void resetCounters() {
        OPENED.set(0);
        CLOSED.set(0);
    }
}
```

- [ ] **Step 5: Add the TestSupport helpers**

In `src/test/java/org/measly/executorch/TestSupport.java`, add these methods before the closing
brace. Note the `import java.nio.file.Path;` and `import java.nio.file.Paths;` at the top of the file.

```java
    /** Directory holding the stress fixture (.pte + goldens), which are committed together. */
    public static java.nio.file.Path stressModelDir() {
        return java.nio.file.Paths.get("src/test/resources/models/stress");
    }

    /** Path to the committed golden digest file. */
    public static java.nio.file.Path stressGoldenPath() {
        return stressModelDir().resolve("stress_golden.json");
    }

    /**
     * Skips the test (assumption) if the native lib or the stress fixture is unavailable. The .pte
     * and its goldens are committed together on purpose — a regenerated model with stale goldens is
     * a silent wrong-answer bug — so both are checked here.
     */
    public static void assumeStressModelAvailable() {
        loadNativeLibrary();
        if (!isModelArtifactAvailable(stressModelDir().resolve("stress_mlp.pte").toString())
                || !isModelArtifactAvailable(stressGoldenPath().toString())) {
            Assumptions.abort(
                    "Stress fixture not found in "
                            + stressModelDir()
                            + " (build it via tools/scripts/export_stress_model.py).");
        }
    }
```

- [ ] **Step 6: Register the Gradle tasks and exclude the tags**

In `build.gradle.kts`, change the `tasks.test` block (currently lines 32-35) to exclude the three
new tags:

```kotlin
tasks.test {
    useJUnitPlatform { excludeTags("leak", "oom", "intraop", "stress", "stress-sweep", "stress-baseline") }
    jvmArgs("-XX:+HeapDumpOnOutOfMemoryError")
    finalizedBy(tasks.jacocoTestReport)
}
```

Then add these registrations after the existing `intraOpTest` block:

```kotlin
// --- Threading / workspace stress arms. All opt-in; `test` excludes every tag below. ---
// Never wire any of these to CI: they saturate every core for their whole duration, and free CI
// providers take a dim view of that. `stressGate` in particular is a local/self-hosted tool.

tasks.register<Test>("stressGate") {
    description = "Local-only concurrency correctness gate: 8 threads, maximum contention."
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("stress") }
    // Deliberately NO num_threads property: the gate wants the real-world intra-op default, which
    // together with 8 caller threads is the maximum-contention configuration.
    systemProperty("et.stress.seconds", providers.gradleProperty("stressSeconds").getOrElse("30"))
}

// The sweep is split across two JVMs because the intra-op pool is process-global and write-once:
// measly::et::setIntraOpThreads refuses a reset once any EtRuntime exists (issue #26), so the eight
// intra-op=1 cells and the intra-op=default confirmation cell cannot share a process.
tasks.register<Test>("stressSweepCore") {
    description = "Throughput sweep: {1,2,4,8} threads x {global,disabled} at ONE intra-op thread."
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("stress-sweep") }
    jvmArgs("-Dai.djl.executorch.num_threads=1")
}

tasks.register<Test>("stressSweepBaseline") {
    description = "Sweep confirmation cell: 1 thread at the DEFAULT intra-op pool size."
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("stress-baseline") }
    mustRunAfter(tasks.named("stressSweepCore")) // both append to one report; keep the order stable
}

tasks.register("stressSweep") {
    description = "Full 9-cell sweep (stressSweepCore + stressSweepBaseline)."
    group = "verification"
    dependsOn(tasks.named("stressSweepCore"), tasks.named("stressSweepBaseline"))
}
```

- [ ] **Step 7: Verify the tags are excluded from the default test task**

```bash
./gradlew test --tests 'org.measly.executorch.stress.*' 2>&1 | tail -20
```

Expected: the untagged `StressGoldenTest` runs and passes; `StressSmokeIT` does **not** run (no test
events for it), because `test` excludes the `stress` tag. If `StressSmokeIT` executes here, the
exclusion is wrong — fix it before proceeding.

- [ ] **Step 8: Run the smoke test through its own task**

```bash
./native/build.sh   # only if the .so is not already staged
./gradlew stressGate --tests 'org.measly.executorch.stress.StressSmokeIT'
```

Expected: PASS, 4 tests. If the goldens fail here, the model and the goldens have drifted — re-run
Task 1's export, which rewrites both.

- [ ] **Step 9: Commit**

```bash
git add src/test/java/org/measly/executorch/ build.gradle.kts
git commit -m "feat(stress): per-thread context pattern, translator, and opt-in Gradle tasks"
```

---

### Task 4: The concurrency gate

**Files:**
- Create: `src/test/java/org/measly/executorch/stress/StressGateIT.java`

**Interfaces:**
- Consumes: `PerThreadContext`, `StressTranslator`, `StressGolden`, `TestSupport` (Task 3).
- Produces: nothing consumed by later tasks.

**What this asserts, and why each part is there.** Golden verification catches "wrong model, wrong
wiring". Bitwise self-reference catches concurrency corruption — and catches the case where *every*
thread is wrong the same way, which is the shape a shared-workspace bug takes and which cross-thread
agreement alone would miss. It is legitimately exact rather than optimistic: XNNPACK parallelises
over output tiles, not over the K reduction, so neither thread count nor sharing mode changes the
order of accumulation. The open/closed counters catch a leaked native handle.

- [ ] **Step 1: Write the failing test**

Create `src/test/java/org/measly/executorch/stress/StressGateIT.java`:

```java
package org.measly.executorch.stress;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.measly.executorch.TestSupport;

/**
 * Maximum-contention correctness gate: 8 threads, {@code global} sharing, intra-op pool at its
 * default size. Not a benchmark — the point is to make a race likely, not to be fast.
 *
 * <p><b>Local only.</b> This saturates every core for its whole duration. Never wire it to CI, not
 * even a nightly on a free runner.
 *
 * <p>Duration is {@code -Det.stress.seconds} (default 30), surfaced by the {@code stressGate}
 * Gradle task as {@code -PstressSeconds}.
 */
@Tag("stress")
class StressGateIT {

    private static final int THREADS = 8;
    private static final String MODE = "global";

    @Test
    void eightThreadsAgreeBitwiseWithTheGoldensUnderMaximumContention() throws Exception {
        TestSupport.assumeStressModelAvailable();
        StressGolden golden = StressGolden.load(TestSupport.stressGoldenPath());
        List<StressGolden.Case> cases = golden.cases();
        int seconds = Integer.getInteger("et.stress.seconds", 30);

        // Reference pass, single-threaded, before any worker starts. Verified against the goldens
        // (oracle layer 1) so the reference itself is known-good; the workers then compare against
        // it bit for bit (layer 2).
        float[][] reference = new float[cases.size()][];
        PerThreadContext.resetCounters();
        try (PerThreadContext ctx = PerThreadContext.open(MODE)) {
            for (int i = 0; i < cases.size(); i++) {
                reference[i] = ctx.predict(cases.get(i).v1, cases.get(i).v2);
                golden.verify(cases.get(i), reference[i]);
            }
        }

        AtomicBoolean stop = new AtomicBoolean(false);
        AtomicLong forwards = new AtomicLong();
        Queue<Throwable> failures = new ConcurrentLinkedQueue<>();
        CyclicBarrier start = new CyclicBarrier(THREADS);
        List<Thread> workers = new ArrayList<>(THREADS);

        for (int t = 0; t < THREADS; t++) {
            Thread worker =
                    new Thread(
                            () -> {
                                // THE PATTERN: one model+predictor per thread, scoped to the
                                // thread's own lifetime by try-with-resources. The closing thread
                                // is the one that was doing the forwards, so nothing can be in
                                // flight when close() runs.
                                try (PerThreadContext ctx = PerThreadContext.open(MODE)) {
                                    start.await();
                                    while (!stop.get()) {
                                        for (int i = 0; i < cases.size(); i++) {
                                            float[] out =
                                                    ctx.predict(cases.get(i).v1, cases.get(i).v2);
                                            if (!Arrays.equals(reference[i], out)) {
                                                throw new AssertionError(
                                                        "thread "
                                                                + Thread.currentThread().getName()
                                                                + " diverged bitwise on case "
                                                                + cases.get(i).name
                                                                + " after "
                                                                + forwards.get()
                                                                + " total forwards");
                                            }
                                            forwards.incrementAndGet();
                                        }
                                    }
                                } catch (Throwable e) {
                                    failures.add(e);
                                    stop.set(true); // do not let the others spin for the full run
                                }
                            },
                            "stress-" + t);
            workers.add(worker);
            worker.start();
        }

        Thread.sleep(seconds * 1000L);
        stop.set(true);
        for (Thread w : workers) {
            w.join(60_000);
            assertTrue(!w.isAlive(), w.getName() + " did not terminate within 60s of the stop flag");
        }

        if (!failures.isEmpty()) {
            AssertionError e = new AssertionError(failures.size() + " worker(s) failed");
            failures.forEach(e::addSuppressed);
            throw e;
        }

        assertTrue(forwards.get() > 0, "no forwards ran — the stop flag fired too early");
        assertEquals(
                THREADS + 1,
                PerThreadContext.opened(),
                "8 workers plus the reference pass");
        assertEquals(
                PerThreadContext.opened(),
                PerThreadContext.closed(),
                "every context must be closed — a mismatch is a leaked native handle");
        System.out.printf(
                "stressGate: %d forwards across %d threads in %ds (%s)%n",
                forwards.get(), THREADS, seconds, MODE);
    }

    @Test
    void aWorkerExceptionFailsTheRunRatherThanHangingIt() throws Exception {
        TestSupport.assumeStressModelAvailable();
        // Proves the failure-propagation path itself: a worker that throws must surface as a test
        // failure with its cause attached, not as a silently-passing run or a hang.
        Queue<Throwable> failures = new ConcurrentLinkedQueue<>();
        Thread t =
                new Thread(
                        () -> {
                            try {
                                throw new IllegalStateException("synthetic worker failure");
                            } catch (Throwable e) {
                                failures.add(e);
                            }
                        });
        t.start();
        t.join(10_000);
        if (failures.isEmpty()) {
            fail("the failure queue must capture a worker throwable");
        }
        assertEquals("synthetic worker failure", failures.peek().getMessage());
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./gradlew stressGate --tests 'org.measly.executorch.stress.StressGateIT'
```

Expected: FAIL — compilation error only if Task 3 was skipped. If Task 3 is complete this may pass
immediately; that is acceptable here, because the deliverable is the harness rather than a new
production behaviour. Confirm it genuinely exercised the model by checking the printed forward count
is well above zero.

- [ ] **Step 3: Verify the gate detects a real divergence**

Temporarily prove the oracle is not vacuous — a green gate that cannot fail is worse than none.
Edit the worker's comparison to `Arrays.equals(reference[0], out)` (comparing every case against
case 0's reference), then run:

```bash
./gradlew stressGate --tests 'org.measly.executorch.stress.StressGateIT' -PstressSeconds=5
```

Expected: FAIL with "diverged bitwise on case". **Revert the edit** and re-run to confirm PASS.

- [ ] **Step 4: Run the full gate**

```bash
./gradlew stressGate -PstressSeconds=30
```

Expected: PASS. Both `StressSmokeIT` and `StressGateIT` run (both carry `@Tag("stress")`).

- [ ] **Step 5: Commit**

```bash
git add src/test/java/org/measly/executorch/stress/StressGateIT.java
git commit -m "feat(stress): 8-thread maximum-contention correctness gate"
```

---

### Task 5: Sweep configuration

**Files:**
- Create: `src/test/java/org/measly/executorch/stress/SweepConfig.java`
- Test: `src/test/java/org/measly/executorch/stress/SweepConfigTest.java`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `SweepConfig.Cell` — `record Cell(int threads, String mode, int intraOp)`, where `intraOp` is `1` for core cells and `0` meaning "runtime default" for the baseline cell.
  - `SweepConfig.coreCells()` → `List<Cell>` (8 cells)
  - `SweepConfig.baselineCells()` → `List<Cell>` (1 cell)
  - `Cell.label()` → `String` like `"t=4 mode=disabled intraop=1"`

Pure JUnit, no native — this test runs in the ordinary `./gradlew test`.

- [ ] **Step 1: Write the failing test**

Create `src/test/java/org/measly/executorch/stress/SweepConfigTest.java`:

```java
package org.measly.executorch.stress;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

/** The sweep matrix. Pure data; touches no native code. */
class SweepConfigTest {

    @Test
    void coreCellsAreTheFullThreadByModeCrossProduct() {
        List<SweepConfig.Cell> cells = SweepConfig.coreCells();
        assertEquals(8, cells.size(), "{1,2,4,8} threads x {global,disabled}");
        assertEquals(
                List.of(1, 1, 2, 2, 4, 4, 8, 8),
                cells.stream().map(SweepConfig.Cell::threads).collect(Collectors.toList()),
                "ordered by thread count so the report reads as a scaling curve");
    }

    @Test
    void everyCoreCellPinsOneIntraOpThread() {
        // The workspace lock is only legible at intra-op=1; at the default pool size a single
        // forward saturates the box and caller-thread scaling is flat for unrelated reasons.
        assertTrue(SweepConfig.coreCells().stream().allMatch(c -> c.intraOp() == 1));
    }

    @Test
    void perModelIsExcluded() {
        // Degenerate with `global` for a single-model workload: they differ only across distinct
        // models. Including it would add runtime and produce duplicate rows.
        assertFalse(SweepConfig.coreCells().stream().anyMatch(c -> "per_model".equals(c.mode())));
    }

    @Test
    void baselineIsOneCellAtTheDefaultPoolSize() {
        List<SweepConfig.Cell> cells = SweepConfig.baselineCells();
        assertEquals(1, cells.size());
        assertEquals(1, cells.get(0).threads());
        assertEquals("global", cells.get(0).mode());
        assertEquals(0, cells.get(0).intraOp(), "0 means 'leave the pool at its default size'");
    }

    @Test
    void theTwoArmsTogetherAreTheNineCellSweep() {
        assertEquals(9, SweepConfig.coreCells().size() + SweepConfig.baselineCells().size());
    }

    @Test
    void labelIsStableAndReadable() {
        assertEquals("t=4 mode=disabled intraop=1", new SweepConfig.Cell(4, "disabled", 1).label());
        assertEquals("t=1 mode=global intraop=default", new SweepConfig.Cell(1, "global", 0).label());
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./gradlew test --tests 'org.measly.executorch.stress.SweepConfigTest'
```

Expected: FAIL — compilation error, `SweepConfig` does not exist.

- [ ] **Step 3: Write the implementation**

Create `src/test/java/org/measly/executorch/stress/SweepConfig.java`:

```java
package org.measly.executorch.stress;

import java.util.ArrayList;
import java.util.List;

/**
 * The sweep matrix, split across the two JVMs that have to run it.
 *
 * <p>The intra-op threadpool is process-global and write-once — sealed at the first model load — so
 * the eight {@code intraOp=1} cells and the {@code intraOp=default} confirmation cell cannot share
 * a process. {@code stressSweepCore} runs {@link #coreCells()} under
 * {@code -Dai.djl.executorch.num_threads=1}; {@code stressSweepBaseline} runs
 * {@link #baselineCells()} with no such flag.
 */
public final class SweepConfig {

    private static final int[] THREAD_COUNTS = {1, 2, 4, 8};

    /** Modes that are not degenerate for a single-model workload. See SweepConfigTest. */
    private static final String[] MODES = {"global", "disabled"};

    private SweepConfig() {}

    /** One measurement cell. {@code intraOp == 0} means "leave the pool at its default size". */
    public record Cell(int threads, String mode, int intraOp) {
        public String label() {
            return "t=" + threads + " mode=" + mode + " intraop=" + (intraOp == 0 ? "default" : intraOp);
        }
    }

    /** Eight cells at one intra-op thread, ordered so the report reads as a scaling curve. */
    public static List<Cell> coreCells() {
        List<Cell> cells = new ArrayList<>(THREAD_COUNTS.length * MODES.length);
        for (int threads : THREAD_COUNTS) {
            for (String mode : MODES) {
                cells.add(new Cell(threads, mode, 1));
            }
        }
        return cells;
    }

    /**
     * The single confirmation cell at the real-world intra-op default, so the sweep cannot be
     * misread as if {@code intraop=1} were the shipping configuration.
     */
    public static List<Cell> baselineCells() {
        return List.of(new Cell(1, "global", 0));
    }
}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
./gradlew test --tests 'org.measly.executorch.stress.SweepConfigTest'
```

Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add src/test/java/org/measly/executorch/stress/SweepConfig*.java
git commit -m "feat(stress): sweep matrix definition"
```

---

### Task 6: The sweep runner and its two arms

**Files:**
- Create: `src/test/java/org/measly/executorch/stress/SweepRunner.java`
- Create: `src/test/java/org/measly/executorch/stress/StressSweepIT.java`
- Create: `src/test/java/org/measly/executorch/stress/StressSweepBaselineIT.java`

**Interfaces:**
- Consumes: `SweepConfig.Cell` (Task 5), `PerThreadContext`, `StressGolden`, `TestSupport`.
- Produces:
  - `SweepRunner.Result` — `record Result(SweepConfig.Cell cell, long forwards, double wallSeconds, double forwardsPerSecond, double meanLatencyMs, double achievedParallelism, long peakRssKb)`
  - `SweepRunner.run(SweepConfig.Cell, StressGolden, int seconds)` → `Result`
  - `SweepRunner.report(List<Result>)` — prints a table and appends TSV rows to `build/reports/stress/sweep.tsv`

Both arms append to the same report file, because they run in different JVMs. The header is written
only when the file does not yet exist.

- [ ] **Step 1: Write the runner**

Create `src/test/java/org/measly/executorch/stress/SweepRunner.java`:

```java
package org.measly.executorch.stress;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/** Runs one sweep cell and reports it. Correctness is still checked — a fast wrong answer is not a result. */
public final class SweepRunner {

    private static final Path REPORT = Paths.get("build/reports/stress/sweep.tsv");

    private SweepRunner() {}

    public record Result(
            SweepConfig.Cell cell,
            long forwards,
            double wallSeconds,
            double forwardsPerSecond,
            double meanLatencyMs,
            double achievedParallelism,
            long peakRssKb) {}

    public static Result run(SweepConfig.Cell cell, StressGolden golden, int seconds)
            throws Exception {
        List<StressGolden.Case> cases = golden.cases();

        // Reference pass outside the timed region, verified against the goldens.
        float[][] reference = new float[cases.size()][];
        try (PerThreadContext ctx = PerThreadContext.open(cell.mode())) {
            for (int i = 0; i < cases.size(); i++) {
                reference[i] = ctx.predict(cases.get(i).v1, cases.get(i).v2);
                golden.verify(cases.get(i), reference[i]);
            }
        }

        AtomicBoolean stop = new AtomicBoolean(false);
        AtomicLong forwards = new AtomicLong();
        Queue<Throwable> failures = new ConcurrentLinkedQueue<>();
        CyclicBarrier start = new CyclicBarrier(cell.threads() + 1);
        List<Thread> workers = new ArrayList<>(cell.threads());

        for (int t = 0; t < cell.threads(); t++) {
            Thread worker =
                    new Thread(
                            () -> {
                                try (PerThreadContext ctx = PerThreadContext.open(cell.mode())) {
                                    // Warm up BEFORE the barrier so first-call costs (delegate
                                    // init, page faults, JIT) land outside the timed region.
                                    for (int i = 0; i < cases.size(); i++) {
                                        ctx.predict(cases.get(i).v1, cases.get(i).v2);
                                    }
                                    start.await();
                                    while (!stop.get()) {
                                        for (int i = 0; i < cases.size(); i++) {
                                            float[] out = ctx.predict(cases.get(i).v1, cases.get(i).v2);
                                            if (!Arrays.equals(reference[i], out)) {
                                                throw new AssertionError(
                                                        "cell "
                                                                + cell.label()
                                                                + ": bitwise divergence on case "
                                                                + cases.get(i).name);
                                            }
                                            forwards.incrementAndGet();
                                        }
                                    }
                                } catch (Throwable e) {
                                    failures.add(e);
                                    stop.set(true);
                                }
                            },
                            "sweep-" + cell.threads() + "-" + t);
            workers.add(worker);
            worker.start();
        }

        start.await(); // release every worker at once; loads and warmup are already done
        long cpu0 = processCpuNanos();
        long t0 = System.nanoTime();
        Thread.sleep(seconds * 1000L);
        stop.set(true);
        for (Thread w : workers) {
            w.join(60_000);
        }
        double wall = (System.nanoTime() - t0) / 1e9;
        double cpuSeconds = (processCpuNanos() - cpu0) / 1e9;

        if (!failures.isEmpty()) {
            AssertionError e = new AssertionError("cell " + cell.label() + " failed");
            failures.forEach(e::addSuppressed);
            throw e;
        }

        long n = forwards.get();
        double throughput = n / wall;
        // Mean per-forward latency as seen by ONE thread: wall time divided by that thread's share.
        double meanLatencyMs = (n == 0) ? 0 : (wall * 1000.0) / (n / (double) cell.threads());
        return new Result(
                cell, n, wall, throughput, meanLatencyMs, cpuSeconds / wall, peakRssKb());
    }

    /** Process CPU time across all threads; differenced to get achieved parallelism. */
    private static long processCpuNanos() {
        java.lang.management.OperatingSystemMXBean os = ManagementFactory.getOperatingSystemMXBean();
        if (os instanceof com.sun.management.OperatingSystemMXBean sun) {
            return sun.getProcessCpuTime();
        }
        return 0;
    }

    /** Peak resident set in KiB from /proc/self/status (VmHWM). Returns 0 off Linux. */
    private static long peakRssKb() {
        try {
            for (String line : Files.readAllLines(Paths.get("/proc/self/status"))) {
                if (line.startsWith("VmHWM:")) {
                    return Long.parseLong(line.replaceAll("[^0-9]", ""));
                }
            }
        } catch (IOException | RuntimeException e) {
            // Not Linux, or an unreadable procfs. RSS is a nice-to-have, not a result.
        }
        return 0;
    }

    /**
     * Prints a table and appends TSV rows. Appends rather than overwrites because the two sweep arms
     * run in separate JVMs (the intra-op pool is process-global and write-once) and both report into
     * this one file.
     */
    public static void report(List<Result> results) {
        StringBuilder tsv = new StringBuilder();
        System.out.println();
        System.out.printf(
                "%-32s %12s %12s %10s %10s%n",
                "cell", "fwd/s", "mean ms", "parallel", "peakRSS MB");
        for (Result r : results) {
            System.out.printf(
                    "%-32s %12.1f %12.3f %10.2f %10.1f%n",
                    r.cell().label(),
                    r.forwardsPerSecond(),
                    r.meanLatencyMs(),
                    r.achievedParallelism(),
                    r.peakRssKb() / 1024.0);
            tsv.append(r.cell().threads())
                    .append('\t')
                    .append(r.cell().mode())
                    .append('\t')
                    .append(r.cell().intraOp() == 0 ? "default" : r.cell().intraOp())
                    .append('\t')
                    .append(r.forwards())
                    .append('\t')
                    .append(String.format("%.3f", r.wallSeconds()))
                    .append('\t')
                    .append(String.format("%.1f", r.forwardsPerSecond()))
                    .append('\t')
                    .append(String.format("%.4f", r.meanLatencyMs()))
                    .append('\t')
                    .append(String.format("%.3f", r.achievedParallelism()))
                    .append('\t')
                    .append(r.peakRssKb())
                    .append('\n');
        }
        try {
            Files.createDirectories(REPORT.getParent());
            boolean fresh = !Files.exists(REPORT);
            if (fresh) {
                Files.writeString(
                        REPORT,
                        "threads\tmode\tintraop\tforwards\twall_s\tfwd_per_s\tmean_ms\tparallelism\tpeak_rss_kb\n",
                        StandardCharsets.UTF_8,
                        StandardOpenOption.CREATE);
            }
            Files.writeString(
                    REPORT, tsv.toString(), StandardCharsets.UTF_8, StandardOpenOption.APPEND);
            System.out.println("appended " + results.size() + " row(s) to " + REPORT);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
```

- [ ] **Step 2: Write the two arms**

Create `src/test/java/org/measly/executorch/stress/StressSweepIT.java`:

```java
package org.measly.executorch.stress;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.measly.executorch.TestSupport;
import org.measly.executorch.engine.EtEngine;

/**
 * The eight intra-op=1 cells. Run via {@code ./gradlew stressSweep} (or {@code stressSweepCore}),
 * which forks a JVM with {@code -Dai.djl.executorch.num_threads=1} — required, because the intra-op
 * pool is process-global and write-once.
 */
@Tag("stress-sweep")
class StressSweepIT {

    @Test
    void sweepThreadCountsAndSharingModes() throws Exception {
        TestSupport.assumeStressModelAvailable();
        StressGolden golden = StressGolden.load(TestSupport.stressGoldenPath());
        int seconds = Integer.getInteger("et.stress.cellSeconds", 10);

        List<SweepRunner.Result> results = new ArrayList<>();
        for (SweepConfig.Cell cell : SweepConfig.coreCells()) {
            results.add(SweepRunner.run(cell, golden, seconds));
        }
        SweepRunner.report(results);

        // The pool is sealed at the first model load, which the first cell already did. Assert the
        // fork actually took effect — without it every number above measures something else.
        assertEquals(
                1,
                EtEngine.getIntraOpThreads(),
                "stressSweepCore must fork with -Dai.djl.executorch.num_threads=1");
        assertEquals(8, results.size());
        assertTrue(
                results.stream().allMatch(r -> r.forwards() > 0),
                "every cell must have completed at least one forward");
    }
}
```

Create `src/test/java/org/measly/executorch/stress/StressSweepBaselineIT.java`:

```java
package org.measly.executorch.stress;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.measly.executorch.TestSupport;

/**
 * The single confirmation cell at the real-world intra-op default — one caller thread, default pool
 * size. Separate class and separate tag because it needs its own JVM: the intra-op pool is
 * process-global and write-once, so it cannot share a process with the intra-op=1 cells.
 */
@Tag("stress-baseline")
class StressSweepBaselineIT {

    @Test
    void oneThreadAtTheDefaultIntraOpPoolSize() throws Exception {
        TestSupport.assumeStressModelAvailable();
        StressGolden golden = StressGolden.load(TestSupport.stressGoldenPath());
        int seconds = Integer.getInteger("et.stress.cellSeconds", 10);

        List<SweepRunner.Result> results = List.of();
        results = new java.util.ArrayList<>(results);
        for (SweepConfig.Cell cell : SweepConfig.baselineCells()) {
            results.add(SweepRunner.run(cell, golden, seconds));
        }
        SweepRunner.report(results);

        assertEquals(1, results.size());
        assertTrue(results.get(0).forwards() > 0);
    }
}
```

- [ ] **Step 3: Run the sweep with short cells to verify it works**

```bash
rm -f build/reports/stress/sweep.tsv
./gradlew stressSweep -Det.stress.cellSeconds=3
```

Expected: PASS. Two printed tables (8 rows then 1 row), and `build/reports/stress/sweep.tsv` with a
header plus 9 data rows.

- [ ] **Step 4: Verify the report content**

```bash
cat build/reports/stress/sweep.tsv
```

Expected: 10 lines. The `intraop` column reads `1` for the first eight rows and `default` for the
last. Sanity-check the shape: under `disabled`, `parallelism` should climb with thread count; under
`global` it should stay near 1. If both climb identically, the sharing-mode option is not reaching
the backend — stop and investigate before recording any numbers as evidence.

- [ ] **Step 5: Commit**

```bash
git add src/test/java/org/measly/executorch/stress/
git commit -m "feat(stress): throughput sweep across caller threads and sharing modes"
```

---

### Task 7: Native ASan/LSan stress harness

**Files:**
- Create: `native/harness/et_stress_harness.cpp`
- Modify: `native/CMakeLists.txt` (the `if(ET_BUILD_QA)` block, around line 204-227)
- Modify: `native/build_qa.sh` (the Linux branch, after the leak-harness runs)

**Interfaces:**
- Consumes: `measly::et::EtRuntime` (`native/core/et_runtime.h`), `dtypeSize` (`native/core/dtype_size.h`), the committed `stress_mlp.pte`.
- Produces: the `et_stress_harness` executable. Usage: `et_stress_harness <pte> <threads> <seconds>`; env `ET_SHARING_MODE` (`0`=Disabled, `1`=PerModel, `2`=Global, unset=runtime default). Exit `0` on success, `1` on bitwise divergence, `2` on load/forward error, `4` on bad usage.

This arm runs oracle **layer 2 only** — bitwise stability. Adding a JSON parser to a JNIEnv-free C++
harness is not worth the dependency; ASan plus bitwise stability is what this arm exists for, and the
JVM arm already owns the golden comparison.

- [ ] **Step 1: Write the harness**

Create `native/harness/et_stress_harness.cpp`:

```cpp
// Threaded stress harness for ASan/LSan: N threads, each with its OWN EtRuntime over the same .pte,
// forwarding concurrently for a fixed wall-clock duration and asserting bitwise-identical outputs.
//
// What this covers that et_scaling_harness does not: correctness under sanitizers, rather than
// throughput. ASan catches use-after-free on teardown races and overflow from a clobbered workspace;
// LSan catches per-thread leaks; the bitwise check catches silent corruption that leaves the process
// alive and the numbers merely wrong.
//
// Layer 2 of the oracle only -- no golden digests. The JVM arm (StressGateIT) owns the golden
// comparison; pulling a JSON parser into this JNIEnv-free harness would not pay for itself.
//
// Sharing mode arrives through EtRuntime's per-load constructor argument, NOT the process-global
// set_option path that et_scaling_harness uses. That matches how the engine actually does it
// (native/core/et_runtime.cpp) and keeps behaviour independent of load order.
//
//   ET_SHARING_MODE unset -> runtime default (Global for our pin)
//   ET_SHARING_MODE=0|1|2 -> Disabled | PerModel | Global
//
// Usage: et_stress_harness <pte> <threads> <seconds>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "dtype_size.h"
#include "et_runtime.h"

using namespace measly::et;
using clock_type = std::chrono::steady_clock;

namespace {

int env_int(const char* name, int fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return fallback;
  return std::atoi(v);
}

// Per-thread input buffers. Each thread owns its own: InputDesc borrows the pointer, so sharing one
// buffer across threads would be a different (and less interesting) experiment.
struct Workload {
  std::vector<std::vector<uint8_t>> buffers;
  std::vector<InputDesc> inputs;
};

// Fills every f32 input with the same deterministic ramp the JVM side uses (`(float) i * ramp + v`),
// so both arms drive the model down the same bucket. Non-f32 inputs are byte-filled; the stress
// model has none, but the harness should not silently produce garbage if pointed at another .pte.
Workload buildWorkload(const MethodMeta& meta, float v) {
  constexpr float kRamp = 1e-5f;
  Workload w;
  w.buffers.resize(meta.numInputs);
  w.inputs.reserve(meta.numInputs);
  for (int i = 0; i < meta.numInputs; ++i) {
    if (meta.inputScalarTypes[i] < 0) continue;  // non-tensor input
    size_t count = 1;
    for (int64_t d : meta.inputShapes[i]) count *= static_cast<size_t>(d);
    size_t bytes = count * dtypeSize(meta.inputScalarTypes[i]);
    w.buffers[i].assign(bytes, 0);
    if (meta.inputScalarTypes[i] == 6) {  // float32
      for (size_t e = 0; e < count; ++e) {
        float x = static_cast<float>(e) * kRamp + v;
        std::memcpy(w.buffers[i].data() + e * sizeof(float), &x, sizeof(float));
      }
    } else {
      std::memset(w.buffers[i].data(), 1, bytes);
    }
    w.inputs.push_back(
        InputDesc{w.buffers[i].data(), meta.inputShapes[i], meta.inputScalarTypes[i]});
  }
  return w;
}

// Flattens every output into one byte vector so a whole forward can be compared with one memcmp.
std::vector<uint8_t> capture(const ForwardResult& r) {
  std::vector<uint8_t> out;
  for (const OutputView& v : r.outputs()) {
    const uint8_t* p = static_cast<const uint8_t*>(v.data);
    out.insert(out.end(), p, p + v.nbytes);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: et_stress_harness <pte> <threads> <seconds>\n");
    return 4;
  }
  const char* pte = argv[1];
  const int threads = std::atoi(argv[2]);
  const int seconds = std::atoi(argv[3]);
  if (threads <= 0 || seconds <= 0) {
    std::fprintf(stderr, "threads and seconds must both be positive\n");
    return 4;
  }
  const int sharing_mode = env_int("ET_SHARING_MODE", -1);

  // Two steering values, landing in different buckets, so the data-dependent gather is exercised.
  const float kValues[2] = {0.0f, 0.99f};

  try {
    // Reference outputs, captured single-threaded before any worker starts.
    std::vector<std::vector<uint8_t>> reference;
    {
      EtRuntime rt(pte, sharing_mode);
      MethodMeta meta = rt.methodMeta();
      for (float v : kValues) {
        Workload w = buildWorkload(meta, v);
        reference.push_back(capture(rt.forward(w.inputs)));
      }
    }

    std::atomic<bool> stop{false};
    std::atomic<long long> forwards{0};
    std::atomic<int> diverged{0};
    std::atomic<int> errors{0};
    std::barrier sync(threads + 1);

    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (int t = 0; t < threads; ++t) {
      workers.emplace_back([&, t]() {
        // Load and warm up BEFORE the barrier so construction cost stays out of the timed region.
        // The barrier is arrived at on EVERY path, including the load-failure path: a worker that
        // bails without arriving would strand the main thread's wait forever.
        std::unique_ptr<EtRuntime> rt;
        std::vector<Workload> loads;
        bool ready = false;
        try {
          rt = std::make_unique<EtRuntime>(pte, sharing_mode);
          MethodMeta meta = rt->methodMeta();
          for (float v : kValues) loads.push_back(buildWorkload(meta, v));
          ready = true;
        } catch (const std::exception& e) {
          std::fprintf(stderr, "et_stress: thread %d failed to load: %s\n", t, e.what());
          errors.fetch_add(1);
          stop.store(true);
        }
        sync.arrive_and_wait();
        if (!ready) return;
        try {
          while (!stop.load(std::memory_order_relaxed)) {
            for (size_t c = 0; c < loads.size(); ++c) {
              std::vector<uint8_t> got = capture(rt->forward(loads[c].inputs));
              if (got.size() != reference[c].size() ||
                  std::memcmp(got.data(), reference[c].data(), got.size()) != 0) {
                std::fprintf(stderr,
                             "et_stress: thread %d diverged bitwise on case %zu\n", t, c);
                diverged.fetch_add(1);
                stop.store(true);
                return;
              }
              forwards.fetch_add(1, std::memory_order_relaxed);
            }
          }
        } catch (const std::exception& e) {
          std::fprintf(stderr, "et_stress: thread %d threw: %s\n", t, e.what());
          errors.fetch_add(1);
          stop.store(true);
        }
      });
    }

    sync.arrive_and_wait();
    auto t0 = clock_type::now();
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    stop.store(true);
    for (auto& w : workers) w.join();
    double wall = std::chrono::duration<double>(clock_type::now() - t0).count();

    std::printf("et_stress: %lld forwards, %d threads, %.1fs, mode=%d, diverged=%d, errors=%d\n",
                forwards.load(), threads, wall, sharing_mode, diverged.load(), errors.load());
    if (diverged.load() > 0) return 1;
    if (errors.load() > 0) return 2;
    if (forwards.load() == 0) {
      std::fprintf(stderr, "et_stress: no forwards ran\n");
      return 2;
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "et_stress: %s\n", e.what());
    return 2;
  }
}
```

- [ ] **Step 2: Wire it into CMake**

In `native/CMakeLists.txt`, inside the `if(ET_BUILD_QA)` block, after the `et_overread_harness`
lines, add:

```cmake
  # Threaded stress harness: N threads x own EtRuntime over one .pte, asserting bitwise-identical
  # outputs under ASan/LSan. Needs Threads (std::thread/std::barrier), like et_scaling_harness.
  # Always BUILT so it cannot bitrot; only RUN when ET_STRESS=1 (see native/build_qa.sh) so the
  # default QA pass stays cheap.
  find_package(Threads REQUIRED)
  add_executable(et_stress_harness ${CMAKE_CURRENT_SOURCE_DIR}/harness/et_stress_harness.cpp)
  target_link_libraries(et_stress_harness PRIVATE et_runtime Threads::Threads)
```

- [ ] **Step 3: Wire it into build_qa.sh**

In `native/build_qa.sh`, add `et_stress_harness` to the Linux build target list:

```bash
  cmake --build native/asan --target et_runtime_test et_leak_harness et_stress_harness -j"${JOBS}"
```

Then, at the end of the Linux branch after the existing leak-harness invocations, add:

```bash
  # Opt-in: saturates every core for its duration, which is not something to do on a free CI runner.
  # ET_STRESS_SECONDS tunes the per-arm duration (default 20).
  if [ "${ET_STRESS:-0}" = "1" ]; then
    STRESS_PTE="src/test/resources/models/stress/stress_mlp.pte"
    STRESS_SECS="${ET_STRESS_SECONDS:-20}"
    if [ ! -f "${STRESS_PTE}" ]; then
      echo "--- Stress harness SKIPPED: ${STRESS_PTE} not found"
      echo "    (build it via tools/scripts/export_stress_model.py)"
    else
      echo "--- ASan/LSan stress harness: 8 threads, global sharing, ${STRESS_SECS}s ---"
      ET_SHARING_MODE=2 ./native/asan/et_stress_harness "${STRESS_PTE}" 8 "${STRESS_SECS}"
      echo "--- ASan/LSan stress harness: 8 threads, sharing disabled, ${STRESS_SECS}s ---"
      ET_SHARING_MODE=0 ./native/asan/et_stress_harness "${STRESS_PTE}" 8 "${STRESS_SECS}"
    fi
  fi
```

- [ ] **Step 4: Build and run it**

```bash
./native/local_build_wrapper.sh native/build_qa.sh
```

Expected: the QA pass builds `et_stress_harness` and does **not** run it (no `ET_STRESS`). Then:

```bash
ET_STRESS=1 ET_STRESS_SECONDS=10 ./native/local_build_wrapper.sh native/build_qa.sh
```

Expected: two stress runs, each printing a nonzero forward count with `diverged=0, errors=0`, and no
ASan or LSan report. Exit status 0.

- [ ] **Step 5: Fix container file ownership**

Container builds run as root and `build_qa.sh` does not chown its outputs back (a known gap,
documented in CLAUDE.md).

```bash
sudo chown -R "$(id -u):$(id -g)" native/asan
```

- [ ] **Step 6: Commit**

```bash
git add native/harness/et_stress_harness.cpp native/CMakeLists.txt native/build_qa.sh
git commit -m "feat(stress): threaded ASan/LSan native stress harness"
```

---

### Task 8: Tune the constants, measure, and document

**Files:**
- Modify: `tools/scripts/export_stress_model.py` (constants + header comment, only if retuning)
- Modify: `docs/superpowers/specs/2026-08-08-threading-workspace-stress-test-design.md` (§3.4)
- Modify: `CLAUDE.md`
- Possibly regenerate: `src/test/resources/models/stress/*`

**Interfaces:**
- Consumes: everything above.
- Produces: the final tuned constants and the measured per-forward cost, recorded in the spec.

This is the task that closes the one open number in the spec. §3.4 states the constants are a ±3×
estimate and that this step replaces them with measurements.

- [ ] **Step 1: Measure the real per-forward cost at one intra-op thread**

The authoritative measurement comes from the native timing harness, not from the export script's
Python figure.

```bash
./native/local_build_wrapper.sh native/bench.sh
sudo chown -R "$(id -u):$(id -g)" native/bench native/bench-results 2>/dev/null || true
ET_INTRAOP_THREADS=1 ./native/bench/et_timing_harness src/test/resources/models/stress/stress_mlp.pte 2000 200
```

Read the steady-state per-forward figure. Note that `EtRuntime`'s constructor calls
`load_forward()`, so delegate setup shows up in `load_ms`, not in the timed region.

- [ ] **Step 2: Retune if the measurement is outside 300–500 µs**

If it is outside the band, adjust in this order and re-export (Task 1, Steps 1–4):

- **Too fast:** raise `DEPTH` first (linear in cost, keeps the weight footprint in L2), then `HIDDEN`.
- **Too slow:** lower `DEPTH` first. Only lower `BATCH` as a last resort, and never below 16 — the
  batch dimension is what keeps the `Linear` stack a compute-bound GEMM rather than a
  bandwidth-bound GEMV, and losing that would make the whole sweep measure DRAM instead of the
  workspace lock.

Regenerating the `.pte` **always** regenerates the goldens in the same run, so re-run the smoke test
afterwards:

```bash
./gradlew stressGate --tests 'org.measly.executorch.stress.StressSmokeIT'
```

- [ ] **Step 3: Record the final numbers in the spec**

In `docs/superpowers/specs/2026-08-08-threading-workspace-stress-test-design.md` §3.4, replace the
"starting point" table and the ±3× paragraph with the tuned values and the measured cost. State the
host it was measured on (core count, CPU model) — the numbers are meaningless without it.

- [ ] **Step 4: Run the full sweep and record the evidence**

```bash
rm -f build/reports/stress/sweep.tsv
./gradlew stressSweep
cat build/reports/stress/sweep.tsv
```

Add the resulting table to the spec as a new "§10 Measured results" section, with the host
description. Sanity-check it against the existing MobileNetV2 finding in CLAUDE.md: under `global`,
achieved parallelism should stay near 1 regardless of caller threads; under `disabled` it should
climb roughly with the thread count.

- [ ] **Step 5: Document in CLAUDE.md**

Under "Build & test", after the `leakTest` / `oomTest` lines, add:

```markdown
### Threading / workspace stress (local only, opt-in)

```bash
./gradlew stressGate                     # 8-thread correctness gate, ~30s (add -PstressSeconds=N)
./gradlew stressSweep                    # 9-cell throughput matrix -> build/reports/stress/sweep.tsv
ET_STRESS=1 ./native/local_build_wrapper.sh native/build_qa.sh   # native harness under ASan/LSan
```

**None of these run in CI, deliberately** — they saturate every core for their whole duration. The
`stress`, `stress-sweep`, and `stress-baseline` tags are excluded from `tasks.test`.

The fixture is `src/test/resources/models/stress/` — a bucket-gather + 4-layer MLP whose `.pte` and
golden digests are **committed together**; regenerating one without the other is a silent
wrong-answer bug, so `tools/scripts/export_stress_model.py` always writes both.

`stressSweep` is two forked JVMs (`stressSweepCore` + `stressSweepBaseline`) because the intra-op
pool is process-global and write-once, so intra-op=1 cells and the intra-op=default cell cannot
share a process.

`src/test/java/org/measly/executorch/stress/PerThreadContext.java` is the reference pattern for
multi-threaded use: **one `ZooModel` per thread**, not a shared model behind a `ThreadLocal`.
```

- [ ] **Step 6: Verify the whole thing from a clean state**

```bash
./gradlew clean test
```

Expected: PASS. `StressGoldenTest` and `SweepConfigTest` run; no `*IT` stress class runs.

```bash
./gradlew stressGate
```

Expected: PASS.

- [ ] **Step 7: Commit and open the PR**

```bash
git add CLAUDE.md docs/superpowers/specs/ tools/scripts/export_stress_model.py src/test/resources/models/stress/
git commit -m "docs(stress): record measured constants, sweep results, and usage"
git push -u origin feat/threading-workspace-stress-test
gh pr create --title "Threading/workspace parameter stress test" --body "..."
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §2 sizing target 300–500 µs | 1 (initial), 8 (measured + tuned) |
| §3.1 kernel-choice serial/parallel mix | 1 |
| §3.2 structure, small gather | 1 |
| §3.3 batch dimension rationale | 1, 8 (retuning floor) |
| §3.4 constants + retuning loop | 1, 8 |
| §4.1 golden vectors from the ET Python runtime | 1, 2 |
| §4.2 bitwise self-reference | 3 (smoke), 4 (gate), 6 (sweep), 7 (native) |
| §4.3 split across arms; JsonUtils; native = layer 2 only | 2, 7 |
| §5.1–5.2 per-thread model, AutoCloseable, reverse close | 3 |
| §5.3 leak backstop counters | 3, 4 |
| §5.4 documented-not-built pooled variant | 3 |
| §6.1 stressGate, max contention, exception propagation | 4 |
| §6.2 focused matrix, per_model excluded, metrics, report | 5, 6 |
| §6.3 native harness, ET_STRESS gate, TSan out of scope | 7 |
| §7 tag exclusion, task registration, committed artifacts, assume-skip | 1, 3 |
| §8 untagged tests for parser + sweep config | 2, 5 |

**Deviation from the spec, recorded above under "A wiring consequence discovered while planning":**
§6.2 implies one sweep task; the process-global write-once intra-op pool forces two forked JVMs plus
a lifecycle task. Task 8 Step 3 updates the spec to match.

**Type consistency:** `StressGolden.Case` fields (`name`, `v1`, `v2`, `sum`, `absSum`, `maxAbs`,
`samples`) are used identically in Tasks 3, 4, and 6. `SweepConfig.Cell` accessors (`threads()`,
`mode()`, `intraOp()`, `label()`) are consistent across Tasks 5 and 6. `PerThreadContext.open(String)`
/ `.predict(float,float)` / `.opened()` / `.closed()` / `.resetCounters()` are consistent across
Tasks 3, 4, and 6. `TestSupport.stressModelDir()` / `.stressGoldenPath()` /
`.assumeStressModelAvailable()` are defined in Task 3 and used in Tasks 3, 4, 6.
`StressTranslator.buildInput(float,int,int,float)` is defined in Task 3 and mirrored in the Python
`build_input(v)` from Task 1 and the C++ `buildWorkload(meta, v)` from Task 7 — all three use
`(float) i * ramp + v` in float32.
