# PyTorch-Free MobileNetV2 Preprocessing (ET_NATIVE) — Design

> **Status:** approved design (2026-07-05). Adds a genuinely LibTorch-free image-classification path
> to the MobileNetV2 `:example` subproject: a plain-Java `Translator` that runs preprocessing and
> post-processing without PyTorch or `NDArrayEx`, exposed both as a runnable example variant and as a
> third JMH benchmark leg. Builds on the example/benchmark from
> [`2026-07-05-mobilenet-example-benchmark-design.md`](2026-07-05-mobilenet-example-benchmark-design.md).

## Goal

The existing example preprocesses images on a PyTorch-backed `NDManager` (`MobilenetTranslator`),
because ExecuTorch's `EtNDArray` implements no `NDArrayEx` ops and `EtEngine.getAlternativeEngine()`
returns `null` ("Phase 1: no hybrid mode"). That means even the ExecuTorch arm still pulls in
LibTorch for preprocessing — qualifying the "ExecuTorch stands alone" story.

This change adds a **third path** that needs no PyTorch at all: preprocessing and post-processing in
plain Java, feeding the ExecuTorch engine directly through its zero-copy tensor factory. It is:

1. **Runnable** — selectable in `MobilenetExample` so a user can see a LibTorch-free classification.
2. **Benchmarked** — a third JMH leg (`ET_NATIVE`) alongside `ET_HYBRID` and `PYTORCH`.

The prize: in its own JMH fork, the `ET_NATIVE` path loads no LibTorch — an honest cold-start and
(out-of-band) footprint number for ExecuTorch on its own.

Non-goals (YAGNI): hybrid mode (`getAlternativeEngine`→PyTorch — a separate concern), a full
footprint/RSS harness (deferred; the README keeps documenting the out-of-band methodology), a
programmatic no-LibTorch assertion (deferred), and extracting shared `MEAN`/`STD` constants (trivial
duplication; the two translators stay independent units).

## Key facts this design rests on (verified against the codebase)

- `EtNDManager.create(Buffer, Shape, DataType)` is public (`EtNDManager.java:53-64`) and copies any
  `FloatBuffer`/`float[]` into an ExecuTorch-owned array — no PyTorch, no `NDArrayEx`.
- `EtNDArray.toByteBuffer` (`EtNDArray.java:45`) returns the forward output's raw bytes, so logits
  come back as a `float[]` with no op support needed.
- DJL's `Image` exposes engine-free `resize(int,int,boolean)` (`Image.java:62`) and
  `getWrappedImage()` (`Image.java:52`). The example depends only on `ai.djl:api`, so the active
  `ImageFactory` is `BufferedImageFactory` and `getWrappedImage()` returns a
  `java.awt.image.BufferedImage`.
- `Classifications` has a `(List<String>, List<Double>)` constructor, so post-processing builds
  results from plain Java lists with no `NDArray`.
- LibTorch loads only when a `PyTorch` engine/manager is actually constructed
  (`NDManager.newBaseManager("PyTorch")` or `Criteria.optEngine("PyTorch")`). Referencing
  `MobilenetTranslator` from a lambda does not load it; invoking that lambda does.

## Architecture

Four units, each with one responsibility and a well-defined interface:

```
example/src/main/java/org/measly/example/
  Variant.java                     (NEW) enum: names the 3 arms + how to build each
  PlainJavaMobilenetTranslator.java (NEW) PyTorch-free Translator<Image,Classifications>
  MobilenetTranslator.java         (MODIFY) — unchanged behavior; already AutoCloseable
  MobilenetExample.java            (MODIFY) — select variant from args, default preserves today
example/src/jmh/java/org/measly/example/
  MobilenetBenchmark.java          (MODIFY) — @Param over Variant instead of engine string
example/src/test/java/org/measly/example/
  PlainJavaMobilenetTranslatorTest.java (NEW) — unit tests for the pure functions
```

### Uniform translator contract

Both translators implement `Translator<Image, Classifications>, AutoCloseable`.
`MobilenetTranslator` is already `AutoCloseable` (owns a native `preManager`).
`PlainJavaMobilenetTranslator.close()` is a **no-op** (it owns no native manager). Making both
`AutoCloseable` keeps the example and benchmark lifecycle code **branch-free** — every arm's
translator is closed the same way (in try-with-resources / `@TearDown`), regardless of whether the
close actually does anything.

## Components

### 1. `Variant` enum (shared selection seam)

The single source of truth for "which arm," used by both the example and the benchmark.

A small combined interface gives callers a single closeable translator type (cleaner than an
intersection-type generic, which the enum field cannot infer):

```java
// New: the uniform translator type both implementations satisfy.
interface CloseableImageTranslator extends Translator<Image, Classifications>, AutoCloseable {}

enum Variant {
    ET_HYBRID("ExecuTorch", MobilenetTranslator::new),
    PYTORCH  ("PyTorch",    MobilenetTranslator::new),
    ET_NATIVE("ExecuTorch", PlainJavaMobilenetTranslator::new);

    final String engine;
    private final Function<List<String>, ? extends CloseableImageTranslator> factory;
    Variant(String engine, Function<List<String>, ? extends CloseableImageTranslator> factory) { ... }
    CloseableImageTranslator newTranslator(List<String> synset) { return factory.apply(synset); }
}
```

- Both `MobilenetTranslator` and `PlainJavaMobilenetTranslator` are declared to implement
  `CloseableImageTranslator` (each already implements both super-interfaces, so this is a marker-only
  change on `MobilenetTranslator`).
- `engine` feeds `Criteria.optEngine(...)`.
- `newTranslator(synset)` builds the arm's translator. The factory is a method reference, so the
  `MobilenetTranslator` constructor (which allocates a `PyTorch` base manager) runs **only** when
  `ET_HYBRID`/`PYTORCH` `newTranslator` is called — never at enum init and never for `ET_NATIVE`.
  This is what keeps the `ET_NATIVE` fork LibTorch-free. Callers get a single `AutoCloseable`
  translator, so try-with-resources / `@TearDown` stay uniform across arms.

### 2. `PlainJavaMobilenetTranslator` (the core new unit)

`final class PlainJavaMobilenetTranslator implements Translator<Image,Classifications>, AutoCloseable`.

- Constructor: `PlainJavaMobilenetTranslator(List<String> synset)`. Stores the synset. No native
  resources.
- `MEAN = {0.485f,0.456f,0.406f}`, `STD = {0.229f,0.224f,0.225f}` (local constants; intentionally
  duplicated from `MobilenetTranslator`, per YAGNI).

**processInput(ctx, image):**
1. `Image resized = image.resize(224, 224, false);`
2. `BufferedImage bi = (BufferedImage) resized.getWrappedImage();`
3. `float[] chw = preprocess(bi);` — the pure function below.
4. `NDArray input = ctx.getNDManager().create(FloatBuffer.wrap(chw), new Shape(1,3,224,224), DataType.FLOAT32);`
5. `return new NDList(input);`

**processOutput(ctx, list):**
1. `float[] logits = list.singletonOrThrow().toFloatArray();`
2. `List<Double> probs = softmax(logits);` — the pure function below.
3. `return new Classifications(synset, probs);`
   (`Classifications` sorts and exposes `topK` itself, so no manual top-k is required here — passing
   the full probability list is sufficient and matches how `MobilenetTranslator` returns results.)

**Pure, engine-free static functions (the unit-test seams):**
- `static float[] preprocess(BufferedImage img)` — reads each of the 224×224 pixels via
  `img.getRGB(x,y)` (packed ARGB), extracts r/g/b, computes `((channel/255f) - MEAN[c]) / STD[c]`,
  and writes channel-major CHW: `chw[c*224*224 + y*224 + x]`. Assumes a 224×224 input (the caller
  resized); may assert dimensions.
- `static List<Double> softmax(float[] logits)` — numerically-stable softmax (subtract max) returning
  probabilities as `Double`s aligned to the logit indices.

- `getBatchifier()` returns the same no-op, batch-size-1 `Batchifier` pattern used by
  `MobilenetTranslator` (input already carries the batch dim; avoids `NDArrays.stack`, which needs
  `NDArrayEx`).
- `close()` — no-op.

### 3. `MobilenetExample` (additive; default behavior preserved)

- Parse the variant: `Variant variant = args.length > 0 ? Variant.valueOf(args[0]) : Variant.ET_HYBRID;`
  On an unknown name, catch `IllegalArgumentException` and print the valid variant names + usage.
- Build the translator from the variant, then `Criteria` with `optEngine(variant.engine)` and that
  translator. The existing try-with-resources is unchanged in shape (translator declared first →
  closed last) because every variant's translator is `AutoCloseable`.
- `./gradlew :example:run` (no args) runs `ET_HYBRID` — **byte-for-byte the current behavior**.
  `./gradlew :example:run --args="ET_NATIVE"` runs the LibTorch-free path.
- Print which variant ran, then the top-5 (unchanged formatting).

### 4. `MobilenetBenchmark` (refactor engine-param → variant-param)

- Replace `@Param({"ExecuTorch","PyTorch"}) String engine` with `@Param Variant variant` (JMH
  populates all enum values automatically; three arms).
- `criteria(cfg, translator)` reads `cfg.variant.engine` for `optEngine(...)` (was `cfg.engine`).
- `Warm.setup`: `translator = cfg.variant.newTranslator(cfg.synset)`; load model via the variant's
  engine; build predictor; warm once. `@TearDown` closes predictor → model → translator (uniform;
  all `AutoCloseable`). Keep the existing `@Setup`-throw close-on-failure guard added in the prior
  cycle.
- `coldStart`: try-with-resources over translator/model/predictor (translator first → closed last).
- **Fold in the carried Minor:** `Config.setup` fail-fast requires the **active variant's** artifact
  — `mobilenet_v2.pte` for `ET_HYBRID`/`ET_NATIVE`, `mobilenet_v2.pt` for `PYTORCH` — instead of
  always `.pte`. Resolve `modelsDir` from whichever artifact the variant needs.
- The `-Dai.djl.pytorch.num_interop_threads=1` jvmArg stays; it is harmless in the `ET_NATIVE` fork
  (no PyTorch engine is created there).

## Data flow

```
ET_NATIVE (no LibTorch in this JMH fork):
  Image ──resize(224,224)──▶ BufferedImage ──preprocess()──▶ float[1,3,224,224]
        ──EtNDManager.create()──▶ NDList ──ExecuTorch forward──▶ EtNDArray logits
        ──toFloatArray()──▶ softmax() ──▶ Classifications

ET_HYBRID / PYTORCH (unchanged): image preprocessing on a PyTorch NDManager
  (MobilenetTranslator), forward on the arm's engine.
```

## Error handling

- **Unknown variant arg** (example): catch `IllegalArgumentException` from `Variant.valueOf`, print
  the valid names + a usage line, exit non-zero. No stack trace dump.
- **Missing artifact** (example + benchmark): `ModelArtifacts.require(<variant's artifact>)` throws
  the existing fail-fast message pointing at `./gradlew :example:exportModels`.
- **Non-`BufferedImage` wrapped image** (defensive): if `getWrappedImage()` is not a `BufferedImage`
  (would require a non-default `ImageFactory` we don't ship), fail with a clear message naming the
  assumption rather than a raw `ClassCastException`.
- **Benchmark `@Setup` failure** for a LibTorch-loading arm keeps the prior cycle's close-on-throw
  guard so a partially-constructed state does not leak its native manager.

## Testing

- **TDD the pure functions** (`PlainJavaMobilenetTranslatorTest`, no engine, no network):
  - `preprocess`: a small synthetic `BufferedImage` (e.g. a solid known color, or a handful of known
    pixels) → assert exact CHW-normalized `float[]` values and length `3*224*224`, and channel-major
    layout (a red pixel lands in the R plane at the right offset).
  - `softmax`: a known logit vector → assert probabilities sum to 1.0 (within epsilon), the argmax
    index matches, and known relative magnitudes hold; include a large-magnitude vector to prove
    numerical stability (max-subtraction).
- **Integration (verified during implementation):** `./gradlew :example:run --args="ET_NATIVE"`
  prints a correct top-5 (cat classes), and `./gradlew :example:run` (no args) still prints the
  ET_HYBRID top-5.
- **Benchmark:** compiles; a reduced-settings smoke run is driven manually by the user (per the
  established "user drives benchmark runs" practice), not by CI. No programmatic no-LibTorch
  assertion this cycle (scope choice).
- Existing engine/example suites stay green (`./gradlew test :example:test`).

## Caveats to document (README)

- **Resize differs by arm:** `ET_HYBRID`/`PYTORCH` use DJL's tensor `Resize` (bilinear on the
  tensor); `ET_NATIVE` uses `Image.resize` (Graphics2D). Both are valid preprocessing and both yield
  correct classes, but outputs are not bit-identical — the comparison is latency + dependency
  footprint, not identical pixels.
- **Isolation invariant:** `ET_NATIVE` must touch no PyTorch class so its JMH fork stays
  LibTorch-free. The lambda-factory `Variant` design and never calling `newBaseManager("PyTorch")` on
  that path are what guarantee it; JMH's per-`@Param` forking (fork ≥ 1, which the benchmark uses)
  keeps each arm in its own JVM. This invariant is called out for reviewers.

## Documentation updates

- `example/README.md`: add the `ET_NATIVE` arm to the benchmark description; document
  `./gradlew :example:run --args="ET_NATIVE"` for the runnable LibTorch-free demo; refine the
  existing preprocessing caveat now that a genuinely PyTorch-free arm exists (the "still pulls in
  LibTorch" note now applies only to `ET_HYBRID`).
- No `benchmarking.md` change required beyond what the prior cycle already added; a one-line pointer
  to the `ET_NATIVE` arm is optional.
