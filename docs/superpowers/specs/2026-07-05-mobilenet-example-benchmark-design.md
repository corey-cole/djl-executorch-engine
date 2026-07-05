# MobileNetV2 Example Subproject + ET-vs-TorchScript Benchmark — Design

> **Status:** approved design (2026-07-05). Backs the "why ExecuTorch in DJL" benchmark narrative
> from [`docs/benchmarking.md`](../../benchmarking.md) by giving MobileNetV2 a real home in the repo:
> a runnable example that doubles as the head-to-head benchmark vehicle.

## Goal

Turn the deferred MobileNetV2 benchmark into a first-class, in-repo `:example` Gradle subproject that:

1. **Runs** — a didactic "how to use this engine" app that classifies an image through the
   ExecuTorch engine via the standard DJL `Criteria`/`Predictor` surface.
2. **Benchmarks** — a JMH task that races **this ExecuTorch engine** against the **DJL PyTorch
   engine (LibTorch)** on the *same* MobileNetV2 weights, both loaded in one JVM via DJL `Criteria`.
3. **Consolidates dependency management** — introduce a real Gradle version catalog
   (`gradle/libs.versions.toml`) so the engine and the example share one source of truth for
   `djl = "0.36.0"` et al. (folded in now because going multi-module is exactly when duplication
   would otherwise start).

Non-goals (deliberately deferred, consistent with `benchmarking.md`): ResNet-18, quantization, and
profiling/devtools event tracing. First pass is **MobileNetV2, two engines, two metrics**.

## Key facts this design rests on

- The engine's DJL name is `"ExecuTorch"` (`EtEngine.ENGINE_NAME`). Benchmark arms select engines
  with `Criteria.optEngine("ExecuTorch")` vs `Criteria.optEngine("PyTorch")`.
- The ET native library lives at `src/main/resources/native/linux-x86_64/libexecutorch_djl.so` on
  the classpath. A `project(":")` dependency therefore hands the example the ET native for free;
  the DJL PyTorch engine auto-fetches LibTorch at runtime.
- The **general Phase 2a NDList path is spec-less** (zero-copy, arbitrary shape). The MobileNetV2
  `.pte` (single tensor `[1,3,224,224] → [1,1000]`) needs **no** `model_spec.json` — unlike the
  named-parameter `MapTranslator` fixtures produced by `tools/scripts/export_with_spec.py`.
- The existing `gradle/libs.versions.toml` is a dead `gradle init` stub (`commons-math3`, `guava`)
  referenced nowhere. It is replaced, not extended.

## Architecture

### Module structure

The engine **stays at the Gradle root**, untouched in its coordinates, native-staging paths, and
publishing wiring (minimally invasive, chosen over a `:engine`/`:example` split to avoid a large
restructure right after 1.0). A single new subproject depends on the root engine.

```
settings.gradle.kts        → include(":example")            # engine stays at root
gradle/libs.versions.toml  → real catalog (replaces the gradle-init stub)

build.gradle.kts (root)    → engine; migrated onto libs.* ; publishing/native-jar scoped away from :example

example/
  build.gradle.kts         → application + me.champeau.jmh plugins
                             implementation(project(":"))                 # this ExecuTorch engine
                             implementation(libs.djl.pytorch.engine)      # LibTorch baseline
                             implementation(libs.djl.api)                 # Image, ImageClassificationTranslator
  src/main/java/…           → MobileNetV2 example app (the `run` target)
  src/main/resources/       → sample image + ImageNet labels (small, committed)
  src/jmh/java/…            → the ET-vs-TorchScript benchmark
  README.md                 → how to run; out-of-band footprint/RSS notes
```

The `:example` module is **not published**. The root's `publishing {}` and `nativeJar-*` blocks must
apply only to the engine, not leak to `:example` — verified by confirming `:example` exposes no
Maven publication.

### Unit boundaries

- **Export script** (`tools/scripts/export_mobilenet.py`) — pure producer of artifacts. Input:
  torchvision pretrained weights. Output: two files + a version sidecar. No Gradle/Java knowledge.
- **`exportModels` Gradle task** (`:example`) — the only bridge between the Python world and the JVM
  build. Shells to the script; writes into `example/build/models/`.
- **Example app** — single-engine, single responsibility: load → classify → print. Depends only on
  DJL `api` + the ExecuTorch engine on the classpath.
- **JMH benchmark** — the comparative harness. Depends on both engines + the generated artifacts.
- **Version catalog** — shared dependency declarations; every module consumes, none duplicates.

## Components

### 1. Version catalog (`gradle/libs.versions.toml`)

Replace the stub. Capture the versions currently hardcoded in the root `build.gradle.kts` plus the
new example/JMH dependencies.

```toml
[versions]
djl = "0.36.0"
slf4j = "2.0.17"
logback = "1.5.33"
junit = "5.14.4"
mockito = "5.18.0"
jmh = "1.37"                 # runtime; me.champeau.jmh plugin version tracked in [plugins]

[libraries]
djl-api                   = { module = "ai.djl:api",                     version.ref = "djl" }
djl-pytorch-engine        = { module = "ai.djl.pytorch:pytorch-engine",  version.ref = "djl" }
slf4j-api                 = { module = "org.slf4j:slf4j-api",            version.ref = "slf4j" }
logback-classic           = { module = "ch.qos.logback:logback-classic", version.ref = "logback" }
junit-bom                 = { module = "org.junit:junit-bom",            version.ref = "junit" }
junit-jupiter             = { module = "org.junit.jupiter:junit-jupiter" }
junit-platform-launcher   = { module = "org.junit.platform:junit-platform-launcher" }
mockito-core              = { module = "org.mockito:mockito-core",       version.ref = "mockito" }

[plugins]
maven-publish       = { id = "com.vanniktech.maven.publish",              version = "0.37.0" }
jacoco-to-cobertura = { id = "name.remal.jacoco-to-cobertura",            version = "2.0.4" }
foojay-resolver     = { id = "org.gradle.toolchains.foojay-resolver-convention", version = "1.0.0" }
jmh                 = { id = "me.champeau.jmh",                           version = "0.7.2" }
```

Root `build.gradle.kts` loses its `val djlVersion = "0.36.0"` and inline coordinates in favor of
`libs.*` / `alias(libs.plugins.*)`. `settings.gradle.kts` references `alias(libs.plugins.foojay.resolver)`.
Exact plugin/runtime versions (the `me.champeau.jmh` plugin and jmh-core) are pinned during
implementation to the set compatible with the Gradle version in use; the values above are the
current known-good starting point.

### 2. Export script (`tools/scripts/export_mobilenet.py`)

Mirrors the structure of `export_with_spec.py`, but for a single-tensor image model and emitting
**both** targets from the **same** weights:

- Load `torchvision.models.mobilenet_v2(weights=…DEFAULT)`, `.eval()`.
- Example input `torch.randn(1, 3, 224, 224)`.
- **`.pte`**: `to_edge_transform_and_lower(export(model, example))` with the **XNNPACK** partitioner
  → `.to_executorch()` → write `mobilenet_v2.pte`.
- **`.torchscript`**: `torch.jit.trace(model, example)` → `torch.jit.save` → `mobilenet_v2.torchscript`.
- **`versions.json`** sidecar: `{torch, torchvision, executorch}` versions (reproducibility, the same
  spirit as `model_spec.json`'s `executorch_version`).
- No `model_spec.json` — the general NDList path is spec-less.
- Writes into the cwd; the Gradle task sets cwd to `example/build/models/`.

### 3. `exportModels` Gradle task (`:example`)

- A task that shells to `python3 tools/scripts/export_mobilenet.py` with cwd `example/build/models/`.
- Declares `mobilenet_v2.pte`, `mobilenet_v2.torchscript`, `versions.json` as outputs (up-to-date
  checks skip re-export when present).
- **Opt-in / heavy**: requires a Python env with the pinned `torch`/`torchvision`/`executorch`. Not
  wired as an automatic dependency of `run`/`jmh`.
- `run` and `jmh` **fail fast** if the artifacts are absent, with a message pointing at
  `./gradlew :example:exportModels` (and the Python requirements). Nothing large enters git.

### 4. Example app — the `run` target

- `application` plugin, one `main` class.
- Loads `mobilenet_v2.pte` through the **ExecuTorch** engine:
  `Criteria.builder().optEngine("ExecuTorch").setTypes(Image.class, Classifications.class)
  .optTranslator(ImageClassificationTranslator…).optModelPath(build/models/mobilenet_v2.pte)…`.
- Classifies a **bundled sample image** (committed under `src/main/resources`), prints top-5 with
  ImageNet labels (labels committed alongside).
- Single engine, no benchmark scaffolding — this is the readable "how do I use this engine" sample.

### 5. JMH benchmark — the ET-vs-TorchScript race

- `me.champeau.jmh` plugin; benchmarks in `src/jmh/java`.
- A `@State(Scope.Benchmark)` holds a loaded `Predictor` (and its `Model`) for the arm under test,
  built in `@Setup` and closed in `@TearDown`.
- `@Param({"ExecuTorch", "PyTorch"})` drives the engine choice symmetrically; `ExecuTorch` loads the
  `.pte`, `PyTorch` loads the `.torchscript`. Same input tensor, same warmup/iteration counts across
  arms (per `benchmarking.md` harness notes).
- Two benchmark methods:
  - **Steady-state** — `Mode.AverageTime` (and/or `Throughput`): warm `predict` loop. The fair race;
    reported even if a wash (candor per the doc).
  - **Cold-start** — `Mode.SingleShotTime`: fresh model load + first `forward` per invocation, where
    AOT compilation helps. Its own `@State` lifecycle so each shot pays the load cost.
- **Footprint** (`.so`/LibTorch size) and **RSS** are out of JMH's domain — captured as out-of-band
  measurements documented in `example/README.md`, not as benchmark methods.

## Data flow

```
torchvision weights
  └─ export_mobilenet.py ──▶ mobilenet_v2.pte  ─────────────┐
                        └──▶ mobilenet_v2.torchscript ───┐  │
                        └──▶ versions.json               │  │
                                                         │  │
:example run  ── Criteria(optEngine "ExecuTorch") ───────┼──┴─▶ Predictor ─▶ top-5
:example jmh  ── @Param ExecuTorch → .pte ───────────────┘        (steady + cold)
             ── @Param PyTorch    → .torchscript ─▶ LibTorch (auto-fetched)
```

## Error handling

- **Missing artifacts** (`run`/`jmh`): fail fast at task start with a message naming
  `./gradlew :example:exportModels` and the Python requirement. No silent fallback, no auto-invoke of
  the heavy Python path.
- **`exportModels` without a Python env**: the script/task surfaces the missing-dependency error from
  `python3` directly; the task documents the pinned `torch`/`torchvision`/`executorch` requirement.
- **Engine not on classpath** (e.g. LibTorch fetch fails offline): DJL's own `Criteria` load error
  propagates; the README notes the PyTorch engine needs network on first run to fetch LibTorch.

## Testing

- **Build wiring**: `./gradlew :example:build` configures and compiles (no artifacts needed to
  compile). A cheap check that the module resolves both engines and the catalog.
- **Publication isolation**: assert `:example` produces no Maven publication (the root publishing
  blocks must not leak).
- **Catalog migration**: existing engine tests (`./gradlew test`, `leakTest`) stay green after the
  root build is migrated onto `libs.*` — proves the refactor changed no resolved versions.
- **Example smoke** (optional, guarded): a JUnit test that runs the example end-to-end is gated on the
  artifacts being present (skipped by default in CI, like the export path itself), so CI needs no
  Python/torch.
- The JMH benchmark itself is a manual task, not a CI gate (consistent with `benchmarking.md`).

## Open items carried forward

- Exact plugin/runtime versions (`me.champeau.jmh`, jmh-core) pinned to the Gradle-compatible set
  during implementation.
- Whether steady-state reports `AverageTime`, `Throughput`, or both — decided when first numbers land.
- `benchmarking.md`'s own open items (artifact storage beyond on-demand, quantization recipe,
  profiling-overhead spike) remain out of scope here.
