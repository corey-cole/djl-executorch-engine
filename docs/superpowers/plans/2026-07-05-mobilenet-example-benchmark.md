# MobileNetV2 Example Subproject + ET-vs-TorchScript Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `:example` Gradle subproject that runs MobileNetV2 through the ExecuTorch engine (didactic app) and hosts a JMH benchmark racing it against the DJL PyTorch engine on the same weights, and consolidate dependency versions into a real `gradle/libs.versions.toml` catalog.

**Architecture:** The engine stays at the Gradle root, untouched. A single new `:example` subproject depends on `project(":")` for the ExecuTorch engine plus `ai.djl.pytorch:pytorch-engine` for the baseline, both loaded in one JVM via DJL `Criteria`. Artifacts are generated on demand by a `uv`-run PEP 723 Python script; `run`/`jmh` fail fast if they're absent. A version catalog replaces the dead `gradle init` stub and both modules consume it.

**Tech Stack:** Gradle 9.6.1 (Kotlin DSL, version catalog, config-cache), Java 17, DJL 0.36.0 (`api` + `pytorch-engine`), `me.champeau.jmh` plugin, `uv` + PEP 723 (torch/torchvision/executorch), JUnit 5.

## Global Constraints

- **Java toolchain:** 17 (`JavaLanguageVersion.of(17)`), matching root.
- **DJL version:** `0.36.0`, single source of truth via the catalog. Engine name is the literal string `"ExecuTorch"` (`EtEngine.ENGINE_NAME`); baseline engine is `"PyTorch"`.
- **Gradle:** 9.6.1; `org.gradle.configuration-cache=true`, `parallel=true`, `caching=true` are on — every change must stay config-cache compatible (no capturing script/project references in task actions).
- **`:example` is NOT published** — the root `publishing {}` and `nativeJar-*` blocks must not apply to it.
- **Spec-less path:** MobileNetV2 uses the general NDList path; the `.pte` needs **no** `model_spec.json`.
- **Native scope:** linux-x86_64 only; the ET `.so` reaches `:example` via the `project(":")` classpath resource `/native/linux-x86_64/libexecutorch_djl.so`.
- **Nothing large in git:** generated model artifacts live under `example/build/` (gitignored by Gradle's `build/` convention); only the small sample image + `synset.txt` are committed.
- **`exportModels` host requirement:** `uv` on `PATH`; the export script's PEP 723 block pins torch/torchvision/executorch.
- **Version catalog limitation:** the `foojay-resolver` settings plugin stays a literal version in `settings.gradle.kts` (Gradle version catalogs are not consumable from a settings `plugins{}` block).

---

### Task 1: Version catalog + migrate the root build onto it

Replace the dead `gradle init` stub catalog with the real dependency set, then migrate the root `build.gradle.kts` to consume it. The proof this refactor is behavior-preserving is that the existing engine test suite stays green with no version changes.

**Files:**
- Modify: `gradle/libs.versions.toml` (replace entire contents)
- Modify: `build.gradle.kts` (plugins block + `dependencies` block; remove `val djlVersion`)

**Interfaces:**
- Produces (catalog accessors consumed by later tasks):
  - Libraries: `libs.djl.api`, `libs.djl.pytorch.engine`, `libs.slf4j.api`, `libs.logback.classic`, `libs.junit.bom`, `libs.junit.jupiter`, `libs.junit.platform.launcher`, `libs.mockito.core`
  - Plugins: `libs.plugins.maven.publish`, `libs.plugins.jacoco.to.cobertura`, `libs.plugins.jmh`

- [ ] **Step 1: Capture the green baseline**

Run: `./gradlew test`
Expected: BUILD SUCCESSFUL (this is the pre-refactor baseline the migration must preserve).

- [ ] **Step 2: Replace the catalog stub**

Overwrite `gradle/libs.versions.toml` entirely with:

```toml
[versions]
djl = "0.36.0"
slf4j = "2.0.17"
logback = "1.5.33"
junit = "5.14.4"
mockito = "5.18.0"

[libraries]
djl-api                 = { module = "ai.djl:api",                    version.ref = "djl" }
djl-pytorch-engine      = { module = "ai.djl.pytorch:pytorch-engine", version.ref = "djl" }
slf4j-api               = { module = "org.slf4j:slf4j-api",           version.ref = "slf4j" }
logback-classic         = { module = "ch.qos.logback:logback-classic", version.ref = "logback" }
junit-bom               = { module = "org.junit:junit-bom",           version.ref = "junit" }
junit-jupiter           = { module = "org.junit.jupiter:junit-jupiter" }
junit-platform-launcher = { module = "org.junit.platform:junit-platform-launcher" }
mockito-core            = { module = "org.mockito:mockito-core",      version.ref = "mockito" }

[plugins]
maven-publish       = { id = "com.vanniktech.maven.publish",   version = "0.37.0" }
jacoco-to-cobertura = { id = "name.remal.jacoco-to-cobertura", version = "2.0.4" }
jmh                 = { id = "me.champeau.jmh",                version = "0.7.2" }
```

- [ ] **Step 3: Migrate the root `plugins` block**

In `build.gradle.kts`, replace the two `id(...) version "..."` lines with catalog aliases (leave core plugins `java-library` and `jacoco` literal):

```kotlin
plugins {
    `java-library`
    jacoco
    alias(libs.plugins.maven.publish)
    alias(libs.plugins.jacoco.to.cobertura)
}
```

- [ ] **Step 4: Migrate the root `dependencies` block and remove `djlVersion`**

Delete the line `val djlVersion = "0.36.0"` and rewrite the `dependencies` block to:

```kotlin
dependencies {
    compileOnly(libs.djl.api)
    compileOnly(libs.slf4j.api)

    testImplementation(libs.djl.api)
    testImplementation(libs.slf4j.api)
    testImplementation(libs.logback.classic)
    testImplementation(platform(libs.junit.bom))
    testImplementation(libs.junit.jupiter)
    testImplementation(libs.mockito.core)
    testRuntimeOnly(libs.junit.platform.launcher)
}
```

- [ ] **Step 5: Verify configuration + resolved versions unchanged**

Run: `./gradlew test`
Expected: BUILD SUCCESSFUL, same tests pass. If Gradle reports an unknown catalog accessor (e.g. `libs.djl.api`), re-check the TOML alias names — dashes map to dots in the accessor.

- [ ] **Step 6: Commit**

```bash
git add gradle/libs.versions.toml build.gradle.kts
git commit -m "build: introduce real version catalog and migrate root build onto it"
```

---

### Task 2: `:example` subproject skeleton (application + jmh plugins, not published)

Stand up the module with both plugins and dependency wiring, a trivial `main` so `run` exists, and prove it compiles and produces no Maven publication.

**Files:**
- Modify: `settings.gradle.kts` (add `include(":example")`)
- Create: `example/build.gradle.kts`
- Create: `example/src/main/java/org/measly/example/MobilenetExample.java` (trivial placeholder `main`, fleshed out in Task 4)
- Create: `example/.gitignore` (ignore generated model artifacts)

**Interfaces:**
- Consumes: catalog accessors from Task 1 (`libs.djl.api`, `libs.djl.pytorch.engine`, `libs.plugins.jmh`).
- Produces: the `:example` project with `run` (application) and `jmh` tasks; package `org.measly.example`.

- [ ] **Step 1: Add the module to settings**

Append to `settings.gradle.kts` (below the `rootProject.name` line):

```kotlin
include(":example")
```

- [ ] **Step 2: Write the example build script**

Create `example/build.gradle.kts`:

```kotlin
plugins {
    application
    alias(libs.plugins.jmh)
}

java {
    toolchain { languageVersion = JavaLanguageVersion.of(17) }
}

repositories { mavenCentral() }

dependencies {
    implementation(project(":"))            // this ExecuTorch engine (brings its native .so via resources)
    implementation(libs.djl.pytorch.engine) // LibTorch baseline (auto-fetches native at runtime)
    implementation(libs.djl.api)            // Image, ImageClassificationTranslator
    runtimeOnly(libs.slf4j.api)
    runtimeOnly(libs.logback.classic)
}

application {
    mainClass = "org.measly.example.MobilenetExample"
}

// Model artifacts are generated on demand into this directory (see the exportModels task, Task 3).
val modelsDir = layout.buildDirectory.dir("models")

// Pass the models directory to the JVM so ModelArtifacts can resolve it at runtime.
tasks.named<JavaExec>("run") {
    systemProperty("example.models.dir", modelsDir.get().asFile.absolutePath)
}
```

- [ ] **Step 3: Write a trivial placeholder main**

Create `example/src/main/java/org/measly/example/MobilenetExample.java`:

```java
package org.measly.example;

/** MobileNetV2 example entry point. Fleshed out in a later task. */
public final class MobilenetExample {
    private MobilenetExample() {}

    public static void main(String[] args) {
        System.out.println("mobilenet example placeholder");
    }
}
```

- [ ] **Step 4: Ignore generated artifacts**

Create `example/.gitignore`:

```
/build/
```

- [ ] **Step 5: Verify the module compiles and configures**

Run: `./gradlew :example:compileJava`
Expected: BUILD SUCCESSFUL.

- [ ] **Step 6: Verify `:example` publishes nothing**

Run: `./gradlew :example:tasks --group publishing`
Expected: no `publish`/`publishToMavenLocal` tasks listed for `:example` (the root publishing config does not leak). If any appear, add to `example/build.gradle.kts` a guard confirming no `MavenPublication` is registered on `:example`.

- [ ] **Step 7: Verify `run` works end to end (placeholder)**

Run: `./gradlew :example:run`
Expected: prints `mobilenet example placeholder`.

- [ ] **Step 8: Commit**

```bash
git add settings.gradle.kts example/build.gradle.kts example/src example/.gitignore
git commit -m "build: add :example subproject skeleton (application + jmh, unpublished)"
```

---

### Task 3: Export script (`uv` + PEP 723) and `exportModels` task

Produce both artifacts from the same weights via a self-provisioning `uv`-run script, wired to a Gradle task that declares them as outputs.

**Files:**
- Create: `tools/scripts/export_mobilenet.py`
- Modify: `example/build.gradle.kts` (add the `exportModels` task)

**Interfaces:**
- Consumes: `uv` on `PATH`; writes into cwd.
- Produces (into `example/build/models/`): `mobilenet_v2.pte`, `mobilenet_v2.pt` (TorchScript, `.pt` so the DJL PyTorch engine resolves it by model name), `versions.json`. The Gradle task name is `exportModels`.

> **Note on the TorchScript extension:** the spec wrote `.torchscript`; DJL's PyTorch engine resolves a model by `optModelName("mobilenet_v2")` and searches for `mobilenet_v2.pt`. Use `.pt` so both engines share `optModelName("mobilenet_v2")` against the same directory. Confirm this against DJL 0.36 during Task 5; if `.pt` is not picked up, pass the explicit file via `optModelPath(dir.resolve("mobilenet_v2.pt"))`.

- [ ] **Step 1: Write the export script with PEP 723 pins**

Create `tools/scripts/export_mobilenet.py`:

```python
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch==2.8.0",
#   "torchvision==0.23.0",
#   "executorch==1.3.1",
# ]
# ///
"""Export MobileNetV2 to both ExecuTorch (.pte) and TorchScript (.pt) from the SAME weights.

Run with uv so the pinned deps are provisioned automatically:

    uv run tools/scripts/export_mobilenet.py

Writes into the current working directory:
  - mobilenet_v2.pte  (torch.export -> XNNPACK to_edge_transform_and_lower -> to_executorch)
  - mobilenet_v2.pt   (torch.jit.trace -> torch.jit.save)  [.pt: DJL PyTorch resolves by model name]
  - versions.json     ({torch, torchvision, executorch} for reproducibility)

The .pte uses the general single-tensor path, so NO model_spec.json is emitted.
"""
import json
from importlib.metadata import PackageNotFoundError, version

import torch
import torchvision
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


def _v(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "unknown"


def main() -> None:
    weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
    model = torchvision.models.mobilenet_v2(weights=weights).eval()
    example = (torch.randn(1, 3, 224, 224),)

    # ExecuTorch .pte, XNNPACK-lowered.
    lowered = to_edge_transform_and_lower(
        export(model, example),
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()
    with open("mobilenet_v2.pte", "wb") as f:
        f.write(lowered.buffer)

    # TorchScript .pt from the SAME weights.
    traced = torch.jit.trace(model, example)
    torch.jit.save(traced, "mobilenet_v2.pt")

    with open("versions.json", "w") as f:
        json.dump(
            {
                "torch": _v("torch"),
                "torchvision": _v("torchvision"),
                "executorch": _v("executorch"),
            },
            f,
            indent=2,
        )

    print("wrote mobilenet_v2.pte, mobilenet_v2.pt, versions.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add the `exportModels` Gradle task**

In `example/build.gradle.kts`, after the `modelsDir` val, add:

```kotlin
val exportModels by tasks.registering(Exec::class) {
    group = "build"
    description = "Generate MobileNetV2 .pte + .pt via uv (heavy; needs uv on PATH)."
    val out = modelsDir.get().asFile
    val script = rootProject.file("tools/scripts/export_mobilenet.py")
    inputs.file(script)
    outputs.files(
        out.resolve("mobilenet_v2.pte"),
        out.resolve("mobilenet_v2.pt"),
        out.resolve("versions.json"),
    )
    doFirst { out.mkdirs() }
    workingDir = out
    commandLine("uv", "run", script.absolutePath)
}
```

- [ ] **Step 3: Verify the task is wired and fails clearly without uv (dry check)**

Run: `./gradlew :example:tasks --group build`
Expected: `exportModels` appears with its description.

- [ ] **Step 4: Run the export (heavy, one-time, network-bound)**

Run: `./gradlew :example:exportModels`
Expected: uv resolves torch/torchvision/executorch (first run downloads them), script prints `wrote mobilenet_v2.pte, mobilenet_v2.pt, versions.json`.

- [ ] **Step 5: Verify the artifacts exist**

Run: `ls -la example/build/models/`
Expected: `mobilenet_v2.pte`, `mobilenet_v2.pt`, `versions.json` present (the two model files each a few MB).

- [ ] **Step 6: Verify up-to-date behavior**

Run: `./gradlew :example:exportModels`
Expected: `Task :example:exportModels UP-TO-DATE` (outputs already present, script unchanged).

- [ ] **Step 7: Commit**

```bash
git add tools/scripts/export_mobilenet.py example/build.gradle.kts
git commit -m "feat(example): add uv/PEP723 mobilenet export script and exportModels task"
```

---

### Task 4: Example `run` app + fail-fast artifact resolution

Turn the placeholder into a real MobileNetV2 classifier over the ExecuTorch engine, with a small testable helper that fails fast (pointing at `exportModels`) when artifacts are missing.

**Files:**
- Create: `example/src/main/java/org/measly/example/ModelArtifacts.java`
- Modify: `example/src/main/java/org/measly/example/MobilenetExample.java`
- Create: `example/src/main/resources/kitten.jpg` (small sample image, committed)
- Create: `example/src/main/resources/synset.txt` (1000 ImageNet class labels, one per line, committed)
- Create: `example/src/test/java/org/measly/example/ModelArtifactsTest.java`
- Modify: `example/build.gradle.kts` (add JUnit test deps + `useJUnitPlatform`)

**Interfaces:**
- Consumes: system property `example.models.dir` (set by the `run` task in Task 2 and the jmh task in Task 5).
- Produces:
  - `ModelArtifacts.dir(): java.nio.file.Path` — reads `example.models.dir` (default `build/models`).
  - `ModelArtifacts.require(String name): java.nio.file.Path` — returns `dir().resolve(name)` if it exists, else throws `IllegalStateException` whose message contains `./gradlew :example:exportModels`.
  - `MobilenetExample.main` — loads `mobilenet_v2.pte` via the ExecuTorch engine and prints top-5.

- [ ] **Step 1: Add test dependencies to the example**

In `example/build.gradle.kts` `dependencies { … }`, add:

```kotlin
    testImplementation(platform(libs.junit.bom))
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
```

And after the `application { … }` block add:

```kotlin
tasks.test { useJUnitPlatform() }
```

- [ ] **Step 2: Write the failing test for `ModelArtifacts`**

Create `example/src/test/java/org/measly/example/ModelArtifactsTest.java`:

```java
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
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `./gradlew :example:test --tests 'org.measly.example.ModelArtifactsTest'`
Expected: FAIL — `ModelArtifacts` does not exist / cannot resolve symbol.

- [ ] **Step 4: Implement `ModelArtifacts`**

Create `example/src/main/java/org/measly/example/ModelArtifacts.java`:

```java
package org.measly.example;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/** Resolves generated model artifacts, failing fast toward the export task when they're absent. */
public final class ModelArtifacts {
    private ModelArtifacts() {}

    /** Directory holding generated artifacts; overridable via -Dexample.models.dir. */
    public static Path dir() {
        return Paths.get(System.getProperty("example.models.dir", "build/models"));
    }

    /** Returns the artifact path if present, else throws pointing at the export task. */
    public static Path require(String name) {
        Path p = dir().resolve(name);
        if (!Files.exists(p)) {
            throw new IllegalStateException(
                    "Missing model artifact: "
                            + p
                            + "\nGenerate it first with: ./gradlew :example:exportModels"
                            + " (requires uv on PATH).");
        }
        return p;
    }
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `./gradlew :example:test --tests 'org.measly.example.ModelArtifactsTest'`
Expected: PASS (both cases).

- [ ] **Step 6: Add the sample image and labels**

- Place a small JPEG at `example/src/main/resources/kitten.jpg` (any ~50–200 KB natural image; a kitten is the DJL convention).
- Place the 1000-line ImageNet synset at `example/src/main/resources/synset.txt` (one label per line, index-aligned to MobileNetV2's 1000 outputs — the standard torchvision/ImageNet ordering).

Run: `test -f example/src/main/resources/kitten.jpg && wc -l example/src/main/resources/synset.txt`
Expected: file exists; `1000 example/src/main/resources/synset.txt`.

- [ ] **Step 7: Implement the real example app**

Replace `example/src/main/java/org/measly/example/MobilenetExample.java` with:

```java
package org.measly.example;

import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/** Classifies a bundled image with MobileNetV2 through the ExecuTorch engine and prints top-5. */
public final class MobilenetExample {
    private MobilenetExample() {}

    public static void main(String[] args) throws Exception {
        Path models = ModelArtifacts.require("mobilenet_v2.pte").getParent();

        List<String> synset = loadSynset();
        ImageClassificationTranslator translator =
                ImageClassificationTranslator.builder()
                        .addTransform(new Resize(224, 224))
                        .addTransform(new ToTensor())
                        .addTransform(
                                new Normalize(
                                        new float[] {0.485f, 0.456f, 0.406f},
                                        new float[] {0.229f, 0.224f, 0.225f}))
                        .optSynset(synset)
                        .optApplySoftmax(true)
                        .build();

        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optEngine("ExecuTorch")
                        .optModelPath(models)
                        .optModelName("mobilenet_v2")
                        .optTranslator(translator)
                        .build();

        try (InputStream imageStream =
                        MobilenetExample.class.getResourceAsStream("/kitten.jpg");
                ZooModel<Image, Classifications> model = criteria.loadModel();
                Predictor<Image, Classifications> predictor = model.newPredictor()) {
            Image image = ImageFactory.getInstance().fromInputStream(imageStream);
            Classifications result = predictor.predict(image);
            System.out.println("Top-5 (ExecuTorch / MobileNetV2):");
            for (Classifications.Classification c : result.topK(5)) {
                System.out.printf("  %-30s %.4f%n", c.getClassName(), c.getProbability());
            }
        }
    }

    private static List<String> loadSynset() throws Exception {
        try (InputStream in = MobilenetExample.class.getResourceAsStream("/synset.txt")) {
            if (in == null) {
                throw new IllegalStateException("synset.txt not found on classpath");
            }
            return Arrays.asList(new String(in.readAllBytes()).trim().split("\\R"));
        }
    }
}
```

- [ ] **Step 8: Run the example end to end**

Run: `./gradlew :example:run`
Expected: prints `Top-5 (ExecuTorch / MobileNetV2):` followed by 5 labelled probabilities. If it throws the missing-artifact message, run `./gradlew :example:exportModels` first (Task 3).

> If DJL cannot resolve `mobilenet_v2.pte` by `optModelName`, pass the explicit file: `.optModelPath(ModelArtifacts.require("mobilenet_v2.pte"))` without `optModelName`.

- [ ] **Step 9: Commit**

```bash
git add example/src/main example/src/test example/build.gradle.kts
git commit -m "feat(example): MobileNetV2 classifier app + fail-fast artifact resolution"
```

---

### Task 5: JMH benchmark — ET vs TorchScript, steady-state + cold-start

Add the comparative benchmark parameterized over both engines, reusing `ModelArtifacts` for fail-fast, plus a smoke run to prove it executes.

**Files:**
- Create: `example/src/jmh/java/org/measly/example/MobilenetBenchmark.java`
- Modify: `example/build.gradle.kts` (jmh config + pass `example.models.dir` to the jmh JVM)

**Interfaces:**
- Consumes: `ModelArtifacts.require(...)` (Task 4); engines `"ExecuTorch"` (`.pte`) and `"PyTorch"` (`.pt`); the `me.champeau.jmh` `jmh` task (Task 2).
- Produces: JMH results for two engines × {steady-state `AverageTime`, cold-start `SingleShotTime`}.

- [ ] **Step 1: Configure jmh and pass the models dir to its JVM**

In `example/build.gradle.kts`, add near the other task config:

```kotlin
jmh {
    warmupIterations = 3
    iterations = 5
    fork = 1
    jvmArgs = listOf("-Dexample.models.dir=" + modelsDir.get().asFile.absolutePath)
}
```

- [ ] **Step 2: Write the benchmark**

Create `example/src/jmh/java/org/measly/example/MobilenetBenchmark.java`:

```java
package org.measly.example;

import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;

public class MobilenetBenchmark {

    /** Engine choice, shared by both benchmark states. ExecuTorch -> .pte, PyTorch -> .pt. */
    @State(Scope.Benchmark)
    public static class Config {
        @Param({"ExecuTorch", "PyTorch"})
        public String engine;

        Path modelsDir;
        Image image;

        @Setup(Level.Trial)
        public void setup() throws Exception {
            modelsDir = ModelArtifacts.require("mobilenet_v2.pte").getParent();
            try (InputStream in = MobilenetBenchmark.class.getResourceAsStream("/kitten.jpg")) {
                image = ImageFactory.getInstance().fromInputStream(in);
            }
        }
    }

    static Criteria<Image, Classifications> criteria(Config cfg) throws Exception {
        List<String> synset;
        try (InputStream in = MobilenetBenchmark.class.getResourceAsStream("/synset.txt")) {
            synset = Arrays.asList(new String(in.readAllBytes()).trim().split("\\R"));
        }
        ImageClassificationTranslator translator =
                ImageClassificationTranslator.builder()
                        .addTransform(new Resize(224, 224))
                        .addTransform(new ToTensor())
                        .addTransform(
                                new Normalize(
                                        new float[] {0.485f, 0.456f, 0.406f},
                                        new float[] {0.229f, 0.224f, 0.225f}))
                        .optSynset(synset)
                        .optApplySoftmax(true)
                        .build();
        return Criteria.builder()
                .setTypes(Image.class, Classifications.class)
                .optEngine(cfg.engine)
                .optModelPath(cfg.modelsDir)
                .optModelName("mobilenet_v2")
                .optTranslator(translator)
                .build();
    }

    /** Warm predictor held across invocations: measures steady-state inference. */
    @State(Scope.Benchmark)
    public static class Warm {
        ZooModel<Image, Classifications> model;
        Predictor<Image, Classifications> predictor;

        @Setup(Level.Trial)
        public void setup(Config cfg) throws Exception {
            model = criteria(cfg).loadModel();
            predictor = model.newPredictor();
            predictor.predict(cfg.image); // warm once so first measured op is steady-state
        }

        @TearDown(Level.Trial)
        public void tearDown() {
            if (predictor != null) predictor.close();
            if (model != null) model.close();
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public Classifications steadyState(Config cfg, Warm warm) throws Exception {
        return warm.predictor.predict(cfg.image);
    }

    @Benchmark
    @BenchmarkMode(Mode.SingleShotTime)
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public Classifications coldStart(Config cfg) throws Exception {
        try (ZooModel<Image, Classifications> model = criteria(cfg).loadModel();
                Predictor<Image, Classifications> predictor = model.newPredictor()) {
            return predictor.predict(cfg.image); // load + first forward, per invocation
        }
    }
}
```

- [ ] **Step 3: Ensure artifacts exist, then compile the benchmark**

Run: `./gradlew :example:exportModels :example:jmhClasses`
Expected: BUILD SUCCESSFUL (benchmark compiles against jmh + DJL). If `jmhClasses` is not the task name in this plugin version, use `./gradlew :example:jmhCompileGeneratedClasses` or inspect `./gradlew :example:tasks --group jmh`.

- [ ] **Step 4: Smoke-run the benchmark at tiny settings**

Run: `./gradlew :example:jmh -Pjmh.iterations=1 -Pjmh.warmupIterations=0 -Pjmh.fork=1`
Expected: JMH runs 4 benchmark records (2 engines × 2 modes) and prints a results table without error. (If `-P` overrides aren't honored by this plugin version, temporarily lower the values in the `jmh { }` block for the smoke run, then restore.)

> The PyTorch arm downloads LibTorch natives on first run — needs network. If offline, the `PyTorch` param will fail at `loadModel`; document this in Task 6's README.

- [ ] **Step 5: Commit**

```bash
git add example/src/jmh example/build.gradle.kts
git commit -m "feat(example): JMH ET-vs-TorchScript benchmark (steady-state + cold-start)"
```

---

### Task 6: Example README (run/export/benchmark + out-of-band footprint/RSS)

Document the module: how to export, run, and benchmark; the `uv` requirement and PEP 723 fallback; and the footprint/RSS measurements JMH does not cover.

**Files:**
- Create: `example/README.md`
- Modify: `docs/benchmarking.md` (close the "Harness docs for user `.pte` artifacts" open item by pointing at the example)

**Interfaces:**
- Consumes: task names `:example:exportModels`, `:example:run`, `:example:jmh`.
- Produces: user-facing docs; no code.

- [ ] **Step 1: Write the example README**

Create `example/README.md`:

```markdown
# MobileNetV2 Example & Benchmark

Runs MobileNetV2 `[1,3,224,224] → [1,1000]` through this ExecuTorch engine, and benchmarks it
head-to-head against the DJL PyTorch engine (LibTorch) on the same weights.

## Prerequisites

- **`uv`** on `PATH` (used to export the models; it self-provisions the pinned
  torch/torchvision/executorch via PEP 723 inline metadata in `tools/scripts/export_mobilenet.py`).
- Network on first run: `uv` downloads the export deps, and the PyTorch benchmark arm downloads
  LibTorch natives.

> **`uv` fallback:** if torch wheels misbehave under inline script metadata (index URLs / CPU-only
> variants), create a `uv` project or venv from the same pins in the script header and run
> `python tools/scripts/export_mobilenet.py` inside it. The pins in the script stay the source of truth.

## Generate the model artifacts (once)

    ./gradlew :example:exportModels

Writes `mobilenet_v2.pte`, `mobilenet_v2.pt`, and `versions.json` into `example/build/models/`.
Nothing large is committed to git.

## Run the example

    ./gradlew :example:run

Classifies a bundled image through the ExecuTorch engine and prints the top-5 labels.

## Run the benchmark

    ./gradlew :example:jmh

Races `ExecuTorch` (`.pte`) vs `PyTorch` (`.pt`) over two modes:
- **steady-state** (`AverageTime`) — warm inference loop, the fair race;
- **cold-start** (`SingleShotTime`) — load + first forward, where AOT compilation helps.

Both arms fail fast pointing back at `exportModels` if the artifacts are missing.

## Out-of-band metrics (not measured by JMH)

JMH covers latency only. Capture the other two axes from `docs/benchmarking.md` manually:

- **Runtime / binary size** — compare the shipped native footprint:
  `ls -la $(find ~/.djl.ai -name '*.so' -path '*pytorch*')` (LibTorch) vs the engine's
  `libexecutorch_djl.so` (~11.5 MB).
- **Resident memory (RSS)** — run each arm as its own process and sample RSS during the steady-state
  loop (e.g. `/usr/bin/time -v` or `ps -o rss=`), reported per engine.
```

- [ ] **Step 2: Close the benchmarking-doc open item**

In `docs/benchmarking.md`, update the "Harness docs for user `.pte` artifacts" open item to note that the `:example` subproject now provides the MobileNetV2 run + JMH benchmark, and reference `example/README.md`. (Replace the *deferred* framing with a pointer; keep the rest of the item's history intact.)

- [ ] **Step 3: Verify the docs render and links resolve**

Run: `test -f example/README.md && grep -q 'exportModels' example/README.md && grep -q 'example/README.md' docs/benchmarking.md`
Expected: exit 0 (both files updated, cross-reference present).

- [ ] **Step 4: Commit**

```bash
git add example/README.md docs/benchmarking.md
git commit -m "docs(example): README for run/export/benchmark; close benchmarking open item"
```

---

## Final verification

- [ ] **Full engine suite still green (catalog migration safe):** `./gradlew test` → BUILD SUCCESSFUL.
- [ ] **Example compiles and its unit test passes:** `./gradlew :example:build` → BUILD SUCCESSFUL.
- [ ] **End-to-end example runs:** `./gradlew :example:exportModels :example:run` → prints top-5.
- [ ] **Benchmark executes (smoke):** `./gradlew :example:jmh` at reduced iterations → results table for 2 engines × 2 modes.
- [ ] **No large binaries staged:** `git status` shows no files under `example/build/`.
