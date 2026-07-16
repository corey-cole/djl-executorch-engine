# LSTM Op Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Link the runtime's `etnp::lstm` custom op into the shipped `executorch_djl` shim, prove it end-to-end with a golden-vector JVM integration test, and bundle the runtime's third-party notices into the native classifier jars.

**Architecture:** The runtime tarball self-describes its custom ops via `lib/cmake/ETNPExtras/ETNPExtras.cmake`. `native/CMakeLists.txt` auto-detects that file and whole-archives the op into the shim (so it self-registers at load). A committed golden fixture drives a platform-gated JVM IT through the real DJL path. `native/build.sh` stages the tarball's `LICENSE` + `THIRD-PARTY-NOTICES/` next to the `.so`; the CI artifact upload and the Gradle classifier jar carry them through to `META-INF/licenses/executorch-runtime/`.

**Tech Stack:** CMake + Ninja (native shim), C++20 ExecuTorch runtime (downloaded, not built), Java 17 + DJL 0.36.0, JUnit 5, Gradle Kotlin DSL, GitHub Actions.

## Global Constraints

- Supported platforms: `linux-x86_64` and `windows-x86_64` only. The LSTM op ships **linux-only** (Windows tarball has no `ETNPExtras.cmake`).
- The op is a static-init registrar: it **must** be whole-archived, or the linker GCs the registration out.
- Runtime pin: `v1.3.1-6`, ExecuTorch `1.3.1` (`native/cmake/EtRuntimePin.cmake`). Do not hand-edit that generated file.
- Fixture golden tolerance (verbatim from `executorch-runtime-dist` `test_lstm_roundtrip.py`): `np.allclose(got, ref, rtol=1e-4, atol=1e-4)`, i.e. `|got − ref| ≤ 1e-4 + 1e-4·|ref|`.
- Fixture signature: 3 inputs `(x[5,2,4], h0[2,3], c0[2,3])`, one output `[5,2,3]`, all float32; `in.bin`/`out.bin` are row-major **little-endian** float32.
- README lists only the **linked subset**; the jar bundles the **complete** `THIRD-PARTY-NOTICES/` set verbatim.
- Commit style: end messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Commit only the paths a task names (a pre-existing staged `.gitignore` edit must not be swept in — use explicit pathspecs).

## File Structure

| File | Responsibility |
|---|---|
| `src/test/resources/lstm/{lstm.pte,in.bin,out.bin,shape,PROVENANCE.txt}` | Committed golden fixture + provenance record |
| `src/test/java/org/measly/executorch/TestSupport.java` | Add `assumeLstmModelAvailable()` guard |
| `src/test/java/org/measly/executorch/LstmModelIT.java` | Golden-vector integration test |
| `native/CMakeLists.txt` | Auto-detect + whole-archive the LSTM op into the shim |
| `native/build.sh` | Stage `LICENSE` + `THIRD-PARTY-NOTICES/` next to the `.so` |
| `native/build_desktop.sh` | Same notice staging for local (host) builds |
| `.gitignore` | Ignore the locally-staged `licenses/` subtree |
| `.github/workflows/native-build-job.yml` | Broaden upload globs so notices reach the publish job |
| `build.gradle.kts` | Map notices into the classifier jar + empty-notices release guard |
| `README.md`, `CLAUDE.md` | Third-party licenses summary; note the op is now linked |

---

### Task 1: Commit the golden fixture

**Files:**
- Create: `src/test/resources/lstm/lstm.pte`, `in.bin`, `out.bin`, `shape` (extracted from the release tarball)
- Create: `src/test/resources/lstm/PROVENANCE.txt`

**Interfaces:**
- Produces: classpath resources `/lstm/lstm.pte`, `/lstm/in.bin`, `/lstm/out.bin`, `/lstm/shape` — consumed by Task 3's IT.

- [ ] **Step 1: Download, verify, and extract the fixture**

Run:
```bash
mkdir -p src/test/resources/lstm
cd src/test/resources/lstm
BASE="https://github.com/measly-java-learning/executorch-runtime-dist/releases/download/v1.3.1-6"
curl -fsSL -o etnp-lstm-fixtures.tar.gz "$BASE/etnp-lstm-fixtures-1.3.1.tar.gz"
curl -fsSL -o etnp-lstm-fixtures.tar.gz.sha256 "$BASE/etnp-lstm-fixtures-1.3.1.tar.gz.sha256"
# The .sha256 names the original filename; verify by value.
echo "7bd2d3e822959de268daea3de0bffbe442aeebfd9a63bf86249ce2ad0e893c04  etnp-lstm-fixtures.tar.gz" | sha256sum -c -
tar xzf etnp-lstm-fixtures.tar.gz
rm etnp-lstm-fixtures.tar.gz etnp-lstm-fixtures.tar.gz.sha256
ls -l
cd -
```
Expected: `sha256sum -c` prints `etnp-lstm-fixtures.tar.gz: OK`; `ls` shows `in.bin lstm.pte out.bin shape`.

- [ ] **Step 2: Verify the extracted contents match the known signature**

Run:
```bash
cd src/test/resources/lstm
cat shape
echo "in floats: $(( $(wc -c < in.bin) / 4 ))  out floats: $(( $(wc -c < out.bin) / 4 ))"
strings lstm.pte | grep -i etnp
cd -
```
Expected: `shape` prints `LSTM_T=5 / LSTM_B=2 / LSTM_I=4 / LSTM_H=3`; `in floats: 52  out floats: 30`; `strings` shows `etnp::lstm`.

- [ ] **Step 3: Write the provenance record**

Create `src/test/resources/lstm/PROVENANCE.txt`:
```
LSTM golden fixture for LstmModelIT.

Source : https://github.com/measly-java-learning/executorch-runtime-dist/releases/download/v1.3.1-6/etnp-lstm-fixtures-1.3.1.tar.gz
sha256 : 7bd2d3e822959de268daea3de0bffbe442aeebfd9a63bf86249ce2ad0e893c04
         (adjacent .sha256 in the same release; attested by executorch-runtime-dist CI)
Minted : executorch-runtime-dist extras/lstm/aot/emit_fixtures.py (lstm_case.build_case())

Contents:
  lstm.pte  ExecuTorch program using etnp::lstm (weights baked as constants)
  in.bin    52 float32 LE = x[5,2,4] ++ h0[2,3] ++ c0[2,3]   (row-major)
  out.bin   30 float32 LE = golden eager nn.LSTM output [5,2,3]
  shape     LSTM_{T,B,I,H} = 5,2,4,3

Refresh when native/cmake/EtRuntimePin.cmake bumps to a new runtime release.
```

- [ ] **Step 4: Confirm the fixtures are not gitignored**

Run: `git check-ignore -v src/test/resources/lstm/lstm.pte src/test/resources/lstm/in.bin || echo "not ignored"`
Expected: `not ignored`.

- [ ] **Step 5: Commit**

```bash
git add src/test/resources/lstm/lstm.pte src/test/resources/lstm/in.bin \
        src/test/resources/lstm/out.bin src/test/resources/lstm/shape \
        src/test/resources/lstm/PROVENANCE.txt
git commit -m "test: add committed etnp::lstm golden fixture (v1.3.1-6)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Add the `assumeLstmModelAvailable()` test guard

**Files:**
- Modify: `src/test/java/org/measly/executorch/TestSupport.java`

**Interfaces:**
- Consumes: `org.measly.executorch.engine.LibUtils.platform()` (package-private static, returns e.g. `"linux-x86_64"`; `TestSupport` is in a different package `org.measly.executorch`, so use a `System.getProperty`-based OS check instead of calling `platform()` directly).
- Produces: `TestSupport.assumeLstmModelAvailable()` — aborts (JUnit assumption, test skipped) when the native lib is unavailable or the platform is not linux-x86_64.

- [ ] **Step 1: Add the helper**

Add this method to `TestSupport` (after the existing `assumeDtypesModelAvailable()`):
```java
    /**
     * Skips the test (assumption) unless the native lib is loadable AND we are on linux-x86_64.
     * The etnp::lstm custom op ships linux-only (no ETNPExtras.cmake in the Windows tarball), so
     * on any other platform the shim legitimately lacks the op — a skip, not a failure.
     */
    public static void assumeLstmModelAvailable() {
        try {
            Class.forName("org.measly.executorch.jni.EtNative");
        } catch (Throwable t) { // UnsatisfiedLinkError, ExceptionInInitializerError, etc.
            Assumptions.abort("Native library not available: " + t.getMessage());
        }
        String os = System.getProperty("os.name").toLowerCase();
        String arch = System.getProperty("os.arch").toLowerCase();
        boolean linuxX64 = os.contains("linux") && (arch.equals("amd64") || arch.equals("x86_64"));
        if (!linuxX64) {
            Assumptions.abort("etnp::lstm op is linux-x86_64 only; skipping on " + os + "/" + arch);
        }
    }
```

- [ ] **Step 2: Compile the test sources**

Run: `./gradlew compileTestJava`
Expected: `BUILD SUCCESSFUL`.

- [ ] **Step 3: Commit**

```bash
git add src/test/java/org/measly/executorch/TestSupport.java
git commit -m "test: add assumeLstmModelAvailable guard (linux-x86_64 + native lib)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Golden-vector IT + wire the op into the shim (TDD: failing test → wiring → green)

This task pairs the failing integration test with the CMake change that makes it pass — a reviewer would reject one without the other.

**Files:**
- Create: `src/test/java/org/measly/executorch/LstmModelIT.java`
- Modify: `native/CMakeLists.txt` (insert after line 105, `target_link_libraries(executorch_djl PRIVATE et_runtime)`)

**Interfaces:**
- Consumes: `TestSupport.assumeLstmModelAvailable()` (Task 2); classpath fixtures `/lstm/*` (Task 1); `PassthroughTranslator` (existing).
- Produces: none (leaf test) + a shim that whole-archives `etnp_ops_lstm`.

- [ ] **Step 1: Write the failing test**

Create `src/test/java/org/measly/executorch/LstmModelIT.java`:
```java
package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;

/**
 * End-to-end golden-vector test for the etnp::lstm custom op through the real DJL path.
 * Mirrors executorch-runtime-dist extras/lstm/test/lstm_runner.cpp: read the flat LE-float32
 * blob, slice into (x, h0, c0), forward, compare output to the eager-nn.LSTM golden within
 * allclose(rtol=1e-4, atol=1e-4). Loading lstm.pte is itself the proof the op is linked:
 * if it were not whole-archived into the shim, execution throws "kernel 'etnp::lstm.out' not found".
 */
class LstmModelIT {

    @Test
    void lstmGoldenVectorThroughPredictor() throws Exception {
        TestSupport.assumeLstmModelAvailable();

        Map<String, Integer> dims = readShape("/lstm/shape");
        int t = dims.get("T");
        int b = dims.get("B");
        int i = dims.get("I");
        int h = dims.get("H");

        float[] in = readFloats("/lstm/in.bin");
        float[] expected = readFloats("/lstm/out.bin");

        int nx = t * b * i;
        int nh = b * h;
        float[] x = Arrays.copyOfRange(in, 0, nx);
        float[] h0 = Arrays.copyOfRange(in, nx, nx + nh);
        float[] c0 = Arrays.copyOfRange(in, nx + nh, nx + 2 * nh);

        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optEngine("ExecuTorch")
                        .optModelPath(Paths.get("src/test/resources/lstm"))
                        .optModelName("lstm")
                        .optTranslator(new PassthroughTranslator())
                        .build();

        try (ZooModel<NDList, NDList> model = criteria.loadModel();
                Predictor<NDList, NDList> predictor = model.newPredictor();
                NDManager m = model.getNDManager().newSubManager()) {
            NDArray ax = m.create(x, new Shape(t, b, i));
            NDArray ah0 = m.create(h0, new Shape(b, h));
            NDArray ac0 = m.create(c0, new Shape(b, h));

            // etnp::lstm follows the nn.LSTM contract and returns (y, h_n, c_n); the golden vector
            // (out.bin) is the y sequence, so compare out.get(0) — mirroring the upstream
            // lstm_runner.cpp, which takes res.get()[0]. Feeding exactly 3 inputs also asserts arity:
            // EtSymbolBlock throws if the input count != the model's numInputs.
            // try-with-resources on the NDLists deterministically closes every NDArray they hold
            // (the inputs and the predictor's outputs) at the end of the test.
            try (NDList inputs = new NDList(ax, ah0, ac0);
                    NDList out = predictor.predict(inputs)) {
                NDArray y = out.get(0);
                assertEquals(new Shape(t, b, h), y.getShape(), "first output y has shape [T,B,H]");

                float[] got = y.toFloatArray();
                assertEquals(expected.length, got.length);
                for (int k = 0; k < expected.length; k++) {
                    double tol = 1e-4 + 1e-4 * Math.abs(expected[k]);
                    assertTrue(
                            Math.abs(got[k] - expected[k]) <= tol,
                            "element " + k + ": got=" + got[k] + " expected=" + expected[k]
                                    + " tol=" + tol);
                }
            }
        }
    }

    private static float[] readFloats(String resource) throws IOException {
        try (InputStream is = LstmModelIT.class.getResourceAsStream(resource)) {
            if (is == null) {
                throw new IllegalStateException("fixture not on classpath: " + resource);
            }
            byte[] bytes = is.readAllBytes();
            FloatBuffer fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            float[] out = new float[fb.remaining()];
            fb.get(out);
            return out;
        }
    }

    private static Map<String, Integer> readShape(String resource) throws IOException {
        try (InputStream is = LstmModelIT.class.getResourceAsStream(resource)) {
            if (is == null) {
                throw new IllegalStateException("fixture not on classpath: " + resource);
            }
            String text = new String(is.readAllBytes(), StandardCharsets.UTF_8);
            Map<String, Integer> dims = new HashMap<>();
            for (String line : text.split("\\R")) {
                if (line.isBlank()) {
                    continue;
                }
                String[] kv = line.split("=", 2); // e.g. LSTM_T=5
                dims.put(kv[0].substring("LSTM_".length()).trim(), Integer.parseInt(kv[1].trim()));
            }
            return dims;
        }
    }
}
```

- [ ] **Step 2: Run the IT against the current (un-wired) shim to verify it fails**

Prerequisite: a shim must be staged. If none is staged yet, build one first with the blessed container path: `./native/local_build_wrapper.sh` (produces `src/main/resources/native/linux-x86_64/libexecutorch_djl.so` **without** the op, since CMakeLists isn't wired yet).

Run: `./gradlew test --tests 'org.measly.executorch.LstmModelIT'`
Expected: **FAIL** — an exception referencing the unregistered/missing operator `etnp::lstm` is thrown while loading the program or running forward (ExecuTorch resolves operators during method load/execution; the exact stage is runtime-dependent, but the message names the missing op). (If instead the test is *skipped*, the native lib wasn't staged — build it first per the prerequisite.)

- [ ] **Step 3: Wire the op into `native/CMakeLists.txt`**

Insert immediately after `target_link_libraries(executorch_djl PRIVATE et_runtime)` (line 105), inside the `if(NOT ET_BUILD_QA AND NOT ET_BUILD_BENCH)` block:
```cmake
  # First-party custom ops (etnp LSTM) ship only in some runtime tarballs (Linux logging). The
  # tarball's self-describing ETNPExtras.cmake declares the imported targets + the whole-archive
  # helper. Auto-detect by file presence: the Windows tarball omits it, so the op is skipped there
  # with no platform branch of our own. Whole-archive is required — the op is a static-init
  # registrar; a plain link GCs the registration out (same failure mode the XNNPACK guard defends).
  if(EXISTS "${ET_INSTALL}/lib/cmake/ETNPExtras/ETNPExtras.cmake")
    include("${ET_INSTALL}/lib/cmake/ETNPExtras/ETNPExtras.cmake")
    etnp_extras_whole_archive(executorch_djl)
    message(STATUS "etnp extras: LSTM op whole-archived into executorch_djl")
  else()
    message(STATUS "etnp extras: none in runtime (platform=${ET_PLATFORM}); LSTM op not linked")
  endif()
```

- [ ] **Step 4: Rebuild the shim and confirm the op is linked**

Run: `./native/local_build_wrapper.sh 2>&1 | grep -i 'etnp extras'`
Expected: `etnp extras: LSTM op whole-archived into executorch_djl`.

Confirm the registration survived the link:
```bash
nm -C src/main/resources/native/linux-x86_64/libexecutorch_djl.so 2>/dev/null | grep -i 'etnp::lstm' | head
```
Expected: at least one `etnp::lstm`-related symbol is present.

- [ ] **Step 5: Run the IT to verify it passes**

Run: `./gradlew test --tests 'org.measly.executorch.LstmModelIT'`
Expected: **PASS** (1 test).

- [ ] **Step 6: Commit**

```bash
git add src/test/java/org/measly/executorch/LstmModelIT.java native/CMakeLists.txt
git commit -m "feat(native): wire etnp::lstm op into the shim + golden-vector IT

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Stage third-party notices next to the `.so`

**Files:**
- Modify: `native/build.sh` (the `if [ "${STAGE_SO}" = "1" ]` block, lines 113-118)
- Modify: `native/build_desktop.sh` (the staging tail, after `cp native/build/libexecutorch_djl.so`)
- Modify: `.gitignore`

**Interfaces:**
- Produces: `src/main/resources/native/<platform>/licenses/{LICENSE, THIRD-PARTY-NOTICES/*}` — consumed by Tasks 5 (CI upload) and 6 (jar mapping).

- [ ] **Step 1: Stage notices in `native/build.sh`**

Replace the `STAGE_SO` block (lines 113-118) with:
```bash
if [ "${STAGE_SO}" = "1" ]; then
  OUT="src/main/resources/native/${OUT_PLATFORM}"
  mkdir -p "${OUT}"
  cp "${NATIVE_BUILD_DIR}/${OUT_LIB}" "${OUT}/"
  echo "Artifact: ${OUT}/${OUT_LIB}"
  ls -lh "${OUT}/${OUT_LIB}"

  # Third-party notices from the resolved runtime tree: escape-hatch (ET_INSTALL set) or the
  # FetchContent extraction under the build dir. Required — never ship a binary without them.
  ET_RUNTIME_ROOT="${ET_INSTALL:-${NATIVE_BUILD_DIR}/_deps/et_runtime-src}"
  test -f "${ET_RUNTIME_ROOT}/LICENSE" && test -d "${ET_RUNTIME_ROOT}/THIRD-PARTY-NOTICES" \
    || { echo "runtime notices missing under ${ET_RUNTIME_ROOT} (LICENSE + THIRD-PARTY-NOTICES/)"; exit 1; }
  LIC_OUT="${OUT}/licenses"
  rm -rf "${LIC_OUT}"
  mkdir -p "${LIC_OUT}"
  cp "${ET_RUNTIME_ROOT}/LICENSE" "${LIC_OUT}/"
  cp -r "${ET_RUNTIME_ROOT}/THIRD-PARTY-NOTICES" "${LIC_OUT}/"
  echo "Notices: ${LIC_OUT} ($(find "${LIC_OUT}" -type f | wc -l) files)"
else
```

- [ ] **Step 2: Stage notices in `native/build_desktop.sh`**

Append after the existing `ls -lh "${OUT}/libexecutorch_djl.so"` line:
```bash

# Third-party notices from the prebuilt runtime ($ET_INSTALL) — parity with native/build.sh.
test -f "${ET_INSTALL}/LICENSE" && test -d "${ET_INSTALL}/THIRD-PARTY-NOTICES" \
  || { echo "runtime notices missing under ${ET_INSTALL} (LICENSE + THIRD-PARTY-NOTICES/)"; exit 1; }
LIC_OUT="${OUT}/licenses"
rm -rf "${LIC_OUT}"
mkdir -p "${LIC_OUT}"
cp "${ET_INSTALL}/LICENSE" "${LIC_OUT}/"
cp -r "${ET_INSTALL}/THIRD-PARTY-NOTICES" "${LIC_OUT}/"
echo "Notices: ${LIC_OUT} ($(find "${LIC_OUT}" -type f | wc -l) files)"
```

- [ ] **Step 3: Ignore the staged notices subtree**

In `.gitignore`, directly below the existing `src/main/resources/native/**/*.dll` line, add:
```
src/main/resources/native/**/licenses/
```

- [ ] **Step 4: Rebuild and verify notices are staged**

Run: `./native/local_build_wrapper.sh`
Then:
```bash
ls src/main/resources/native/linux-x86_64/licenses/
ls src/main/resources/native/linux-x86_64/licenses/THIRD-PARTY-NOTICES/ | grep -i highway
git check-ignore src/main/resources/native/linux-x86_64/licenses/LICENSE && echo "ignored (good)"
```
Expected: `LICENSE  THIRD-PARTY-NOTICES`; `highway_LICENSE` present; `ignored (good)`.

- [ ] **Step 5: Commit**

```bash
git add native/build.sh native/build_desktop.sh .gitignore
git commit -m "build(native): stage runtime third-party notices next to the shim

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Carry notices through the CI artifact upload

**Files:**
- Modify: `.github/workflows/native-build-job.yml` (the two `Store … shim` upload steps, `path:` fields at ~line 75 and ~line 141)

**Interfaces:**
- Consumes: staged `src/main/resources/native/<platform>/licenses/**` (Task 4).
- Produces: `executorch-libs-<platform>` artifacts that include `<platform>/licenses/**`, so `publish.yml` unpacks them into `build/native-staging/<platform>/licenses/**` for Task 6's jar.

- [ ] **Step 1: Broaden the Linux upload glob**

In the `Store libexecutorch_djl shim` step, replace the single-line `path:` with:
```yaml
          path: |
            ${{ github.workspace }}/src/main/resources/native/**/*.so
            ${{ github.workspace }}/src/main/resources/native/**/licenses/**
```
(Rationale: upload-artifact roots the archive at the path segment before the first wildcard — `.../native/` — so `<platform>/…` is preserved for both patterns; the download into `build/native-staging/` therefore recreates `<platform>/{lib, licenses/}`.)

- [ ] **Step 2: Broaden the Windows upload glob**

In the `Store executorch_djl shim` step, replace the single-line `path:` with:
```yaml
          path: |
            ${{ github.workspace }}/src/main/resources/native/**/*.dll
            ${{ github.workspace }}/src/main/resources/native/**/licenses/**
```

- [ ] **Step 3: Lint the workflow YAML**

Run: `python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/native-build-job.yml')); print('yaml ok')"`
Expected: `yaml ok`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/native-build-job.yml
git commit -m "ci(native): include staged licenses in the shim upload artifacts

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Bundle notices into the native classifier jars

**Files:**
- Modify: `build.gradle.kts` (the `nativeJarTasks` block, lines 63-77)

**Interfaces:**
- Consumes: `build/native-staging/<platform>/licenses/**` (from Task 5's artifacts, or a local copy for verification).
- Produces: each `djl-executorch-engine-<platform>.jar` contains `META-INF/licenses/executorch-runtime/{LICENSE, THIRD-PARTY-NOTICES/*}`; the jar task fails if notices are absent.

- [ ] **Step 1: Map notices into the jar and add the empty-notices guard**

Replace the `nativeJarTasks` registration (lines 64-77) with:
```kotlin
val nativeJarTasks = nativePlatforms.map { platform ->
  tasks.register<Jar>("nativeJar-${platform}") {
    archiveClassifier.set(platform)
    // The native library, excluding the bundled licenses subtree (mapped to META-INF below).
    from(nativeStaging.map { it.dir(platform) }) {
        exclude("licenses/**")
        into("native/${platform}")
    }
    // Third-party notices from the runtime tarball, staged next to the .so by native/build.sh.
    from(nativeStaging.map { it.dir("${platform}/licenses") }) {
        into("META-INF/licenses/executorch-runtime")
    }
    // Resolve to plain Files at configuration time so doFirst captures only File + String
    // (config-cache safe) rather than the enclosing script.
    val resolvedLib = nativeStaging.get().dir(platform).file(nativeLibName(platform)).asFile
    val licensesDir = nativeStaging.get().dir(platform).dir("licenses").asFile
    doFirst { // Fail a release rather than ship an empty native jar or a binary with no notices
        require(resolvedLib.exists()) { "Missing native library for ${platform}: ${resolvedLib}" }
        require(licensesDir.isDirectory && (licensesDir.list()?.isNotEmpty() ?: false)) {
            "Missing third-party notices for ${platform}: ${licensesDir}" +
                " (native/build.sh must stage LICENSE + THIRD-PARTY-NOTICES/)"
        }
    }
  }
}
```

- [ ] **Step 2: Stage a local copy so the jar task can run outside CI**

Run (copies the Task-4 staged tree into the jar's staging root):
```bash
mkdir -p build/native-staging/linux-x86_64
cp -r src/main/resources/native/linux-x86_64/. build/native-staging/linux-x86_64/
ls build/native-staging/linux-x86_64/ build/native-staging/linux-x86_64/licenses/
```
Expected: the platform dir holds `libexecutorch_djl.so` and `licenses/`; `licenses/` holds `LICENSE  THIRD-PARTY-NOTICES`.

- [ ] **Step 3: Build the classifier jar and verify its contents**

Run:
```bash
./gradlew nativeJar-linux-x86_64
JAR=$(ls build/libs/*linux-x86_64*.jar | head -1)
unzip -l "$JAR" | grep -E 'native/linux-x86_64/libexecutorch_djl.so|META-INF/licenses/executorch-runtime/'
```
Expected: `BUILD SUCCESSFUL`; the listing shows the `.so` under `native/linux-x86_64/` AND notices under `META-INF/licenses/executorch-runtime/` (incl. `THIRD-PARTY-NOTICES/highway_LICENSE`), with **no** `native/linux-x86_64/licenses/` entries.

- [ ] **Step 4: Verify the guard fails when notices are absent**

Run:
```bash
mv build/native-staging/linux-x86_64/licenses /tmp/lic-bak
./gradlew nativeJar-linux-x86_64 2>&1 | grep -i 'Missing third-party notices' && echo "guard fired"
mv /tmp/lic-bak build/native-staging/linux-x86_64/licenses
```
Expected: build fails with `Missing third-party notices …`; `guard fired` prints.

- [ ] **Step 5: Commit**

```bash
git add build.gradle.kts
git commit -m "build: bundle runtime third-party notices into native classifier jars

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Document the op and the third-party licenses

**Files:**
- Modify: `README.md` (add a "Third-party licenses" section; note the op is linked when present)
- Modify: `CLAUDE.md` (note the LSTM op is now wired)

**Interfaces:** none (docs only).

- [ ] **Step 1: Confirm the linked-subset license list against the staged notices**

Run:
```bash
ls src/main/resources/native/linux-x86_64/licenses/THIRD-PARTY-NOTICES/
```
Cross-reference the entries that correspond to code actually linked into the shipped `logging` `.so` (ExecuTorch core/kernels/XNNPACK stack, tokenizers stack, and the LSTM op's Highway). Backends we do not ship (arm/cadence/mlx/nxp/vulkan/qualcomm) stay out of the README list but remain in the jar's complete set.

- [ ] **Step 2: Add the README "Third-party licenses" section**

Append to `README.md`:
```markdown
## Third-party licenses

The native library (`libexecutorch_djl.so` / `executorch_djl.dll`) statically links
third-party components from the pinned ExecuTorch runtime. The components linked into the
shipped library are:

| Component | License |
|---|---|
| ExecuTorch (core, portable kernels, extensions) | BSD-3-Clause |
| XNNPACK, cpuinfo, clog, pthreadpool | BSD-3-Clause |
| FP16, FXdiv | MIT |
| FlatBuffers | Apache-2.0 |
| Highway (SIMD support for the `etnp::lstm` op, linux-x86_64 only) | Apache-2.0 |

(Linked subset verified against the shipped `.so` with `nm`/`strings`: abseil,
re2, pcre2, tokenizers, sentencepiece, flatcc are NOT linked — tokenizer/LLM
extension deps the shim does not pull — so they are omitted from the README
subset but remain in the jar's complete `THIRD-PARTY-NOTICES/` set.)

Full license texts for these **and** every other component the runtime distribution tracks
are bundled in each native classifier jar under `META-INF/licenses/executorch-runtime/`
(`LICENSE` + `THIRD-PARTY-NOTICES/`), sourced verbatim from the runtime tarball. This list is
tied to the runtime pin (`native/cmake/EtRuntimePin.cmake`); refresh it when the pin bumps.
```

- [ ] **Step 3: Note the op in the runtime section of `README.md`**

In the "Build the native library" area (where the runtime download is described), add:
```markdown
When the pinned runtime provides the first-party custom ops (the `logging` linux-x86_64
tarball ships an `etnp::lstm` op), the shim auto-detects the tarball's `ETNPExtras.cmake` and
whole-archives the op in. The Windows tarball has no such extras, so the op is simply absent there.
```

- [ ] **Step 4: Note the op in `CLAUDE.md`**

In `CLAUDE.md`, under the "The ExecuTorch runtime is NOT built here" section, add a bullet:
```markdown
- The runtime's first-party custom op `etnp::lstm` (linux-x86_64 `logging` tarball only) is
  whole-archived into the shim when the tarball ships `lib/cmake/ETNPExtras/ETNPExtras.cmake`
  (auto-detected in `native/CMakeLists.txt`). Exercised end-to-end by `LstmModelIT`.
```

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document the etnp::lstm op and bundled third-party licenses

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Final verification

- [ ] **Full native rebuild + JVM suite green**

Run:
```bash
./native/local_build_wrapper.sh
./gradlew clean test
```
Expected: shim log shows `etnp extras: LSTM op whole-archived into executorch_djl`; `BUILD SUCCESSFUL`; `LstmModelIT` passes (not skipped) on linux-x86_64.

- [ ] **Leak suite unaffected**

Run: `./gradlew leakTest`
Expected: `BUILD SUCCESSFUL`.
