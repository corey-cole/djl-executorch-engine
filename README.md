# ExecuTorch Engine for DJL

As of its most recent version (0.36.0), [DJL](https://djl.ai/) can only load PyTorch models exported
through the TorchScript API, which PyTorch deprecated several point releases ago. The successor is
the [ExecuTorch](https://executorch.ai/) backend — a lightweight, cross-language runtime built for
edge deployment, producing `.pte` files. This project supplies ExecuTorch as a DJL engine, so models
exported with the current API run under DJL's familiar `Criteria`/`Predictor`/`Translator`
programming model.

It is registered as a *separate* engine (`optEngine("ExecuTorch")`) rather than a replacement for
DJL's PyTorch engine, which means both can be present in one process and a codebase can migrate off
TorchScript model by model instead of in one cut. The engine is **CPU-only** — inference runs on
the portable kernels plus the XNNPACK delegate — and implements a **deliberately small slice of
DJL's `NDArray` surface**: enough to marshal inputs and outputs across the JNI boundary, not a
general tensor library. Arithmetic, broadcasting and stacking on `EtNDArray` are unsupported; do
that work before you hand data to the translator.

## Supported platforms

| Platform | Artifact | Runtime variant | QA |
|---|---|---|---|
| `linux-x86_64` | `libexecutorch_djl.so` | `logging` | Catch2 + ASan/LSan leak harness |
| `linux-aarch64` | `libexecutorch_djl.so` | `logging` | Catch2 + ASan/LSan leak harness |
| `windows-x86_64` | `executorch_djl.dll` | `logging` | Catch2 (MSVC has no LeakSanitizer) |

The native library ships in a per-platform classifier jar (`<artifact>-<platform>.jar`) and is extracted
on first load to a content-addressed cache — `%LOCALAPPDATA%\executorch-djl\<sha256>\` on Windows,
`$XDG_CACHE_HOME` (or `~/.cache`) `/executorch-djl/<sha256>/` elsewhere. Set `EXECUTORCH_LIBRARY_PATH` to
load a specific library instead and bypass extraction entirely.

Windows is built with MSVC 2022 against the `logging` runtime variant; `bare` and `devtools` are
Linux-only benchmarking builds (see `native/build_variants.sh`).

## Add the dependency

The native jars are published as Gradle variants with per-platform capabilities, so Gradle consumers
should request the platform by capability rather than by classifier:

```kotlin
dependencies {
    implementation("org.measly:djl-executorch-engine:<version>")
    runtimeOnly("org.measly:djl-executorch-engine:<version>") {
        capabilities { requireCapability("org.measly:djl-executorch-engine-linux-x86_64") }
    }
}
```

Swap `linux-x86_64` for `linux-aarch64` or `windows-x86_64` (or add several) as needed. Maven consumers add the
classifier form alongside the main (classifier-less) dependency:

```xml
<dependency>
    <groupId>org.measly</groupId>
    <artifactId>djl-executorch-engine</artifactId>
    <version>&lt;version&gt;</version>
    <classifier>linux-x86_64</classifier>
    <scope>runtime</scope>
</dependency>
```

Arm64 hosts use the `linux-aarch64` classifier (and capability) instead.

## Quickstart

Loading a `.pte` is ordinary DJL: name the engine, point `Criteria` at the model directory, supply a
`Translator`, and predict.

```java
/** Turns a {@code float[]} into the model's input list and its output back into a float. */
private static final class AddTranslator implements Translator<float[], Float> {
    // EtNDArray does not support NDArrays.stack (used by the default STACK batchifier),
    // so we provide a no-op batchifier. This example always predicts one input at a time.
    private static final Batchifier BATCHIFIER =
            new Batchifier() {
                @Override
                public NDList batchify(NDList[] inputs) {
                    if (inputs.length != 1) {
                        throw new UnsupportedOperationException("Batch size 1 only");
                    }
                    return inputs[0];
                }

                @Override
                public NDList[] unbatchify(NDList inputs) {
                    return new NDList[] {inputs};
                }
            };

    @Override
    public NDList processInput(TranslatorContext ctx, float[] input) {
        // One NDArray per model input. The add model takes two 1-element float32 tensors.
        NDArray a = ctx.getNDManager().create(new float[] {input[0]});
        NDArray b = ctx.getNDManager().create(new float[] {input[1]});
        return new NDList(a, b);
    }

    @Override
    public Float processOutput(TranslatorContext ctx, NDList list) {
        return list.singletonOrThrow().toFloatArray()[0];
    }

    @Override
    public Batchifier getBatchifier() {
        return BATCHIFIER;
    }
}

public static void main(String[] args) throws Exception {
    Path modelDir = Paths.get(args.length > 0 ? args[0] : "native/spike");
    String modelName = args.length > 1 ? args[1] : "add";

    Criteria<float[], Float> criteria =
            Criteria.builder()
                    .setTypes(float[].class, Float.class)
                    .optEngine("ExecuTorch") // this engine, by name
                    .optModelPath(modelDir)
                    .optModelName(modelName)
                    .optTranslator(new AddTranslator())
                    .build();

    // One ZooModel and one Predictor per thread: forward() is not safe to share.
    try (ZooModel<float[], Float> model = criteria.loadModel();
            Predictor<float[], Float> predictor = model.newPredictor()) {
        System.out.println("2 + 3 = " + predictor.predict(new float[] {2f, 3f}));
    }
}
```

The `getBatchifier()` override is not optional decoration. DJL's default batchifier is `STACK`,
which calls `NDArrays.stack()` — an operation `EtNDArray` does not implement — so a translator
without it fails at the first `predict()`.

The full file is [`example/src/main/java/org/measly/example/QuickStart.java`](example/src/main/java/org/measly/example/QuickStart.java);
run it with `./gradlew :example:runQuickStart`.

## Configuration and tuning

### `ai.djl.executorch.num_threads`

Sizes ExecuTorch's intra-op (XNNPACK) threadpool, either as a JVM system property or via
`EtEngine.setIntraOpThreads(n)`. The pool is **process-global and write-once**: the value is applied
and sealed at the first model load, and a later attempt to change it is refused rather than silently
ignored. Read the value the native pool actually adopted with `EtEngine.getIntraOpThreads()` — the
runtime may clamp a request. Absent the setting, ExecuTorch's own default applies: the
performance-core count as derived by cpuinfo, which is not the same as `nproc`. There is
deliberately no environment variable — nothing in the v1.3.1 threadpool, pthreadpool, or XNNPACK
init reads one, and `OMP_NUM_THREADS` is inert.

### `workspaceSharingMode`

Selects how the XNNPACK delegate shares its scratch workspace. Set it per model on the criteria, or
JVM-wide with the `ai.djl.executorch.workspace_sharing_mode` system property, which supplies the
default for models that do not name one:

```java
Criteria.builder()
        .optEngine("ExecuTorch")
        .optOption("workspaceSharingMode", "disabled")   // or "per_model", "global"
        // ...
        .build();
```

- `disabled` — a private workspace per delegate instance. The most memory, and the only mode under
  which independent caller threads scale.
- `per_model` — one workspace shared by all delegates within a model.
- `global` — one workspace for the whole process, guarded by a process-global mutex. This is the
  ExecuTorch default for our pin, and therefore the effective default if you set neither the option
  nor the property.

Unlike `num_threads` this is neither process-global nor write-once: ExecuTorch resolves it per
delegate at load time, so modes compose freely and load order is irrelevant. An unrecognised
*option* fails the model load; an unrecognised *property* logs a warning and is ignored.

> **Note:** Under the default `global` mode, adding caller threads usually makes things *slower*,
> not faster. An XNNPACK-delegated model already parallelises inside a single `forward()` on the
> shared intra-op pool, and concurrent delegate calls then serialise on the process-global workspace
> mutex — so you pay for N threads' memory and get one thread's throughput. Tune
> `ai.djl.executorch.num_threads` before you add caller threads. Measured on a 4-core/8-thread host
> with MobileNetV2:
>
> | Caller threads | Throughput (`global`) |
> |---|---|
> | 1 | 462 forwards/s |
> | 4 | 305 forwards/s |
> | 8 | 147 forwards/s |
>
> Peak RSS over that sweep went from 33 MB to 224 MB — so the eight-thread configuration costs
> roughly seven times the memory to deliver under a third of the throughput.
>
> Those figures are conditional on that mutex. With `workspaceSharingMode=disabled` each model gets
> a private workspace and caller threads do scale. Achieved parallelism at one intra-op thread, at
> 1/2/4/8 caller threads:
>
> | Sharing mode | 1 | 2 | 4 | 8 |
> |---|---|---|---|---|
> | `global` | 1.12 | 1.12 | 1.12 | 1.17 |
> | `disabled` | 1.12 | 2.23 | 4.35 | 7.13 |
>
> Ratios on larger hosts are unmeasured.

## Monitoring

`EtEngineStats.snapshot()` returns an immutable `EtStatsSnapshot`: effective configuration, process
totals, and per-model detail for every live model. It is a cold-path read designed for a scheduled
poll or a health endpoint, and it never throws — a value that cannot be read degrades rather than
propagating a failure out of a monitoring call.

```java
EtStatsSnapshot stats = EtEngineStats.snapshot();
System.out.println(stats.getModelsLive() + " live, " + stats.getIntraOpThreads() + " threads");
stats.getModels().forEach(m -> System.out.println(m.getName() + ": " + m.getForwardCount()));
```

The same snapshot is exposed over JMX as an MXBean under the object name
`org.measly.executorch:type=EtEngineStats`, auto-registered at the first model load. Set
`ai.djl.executorch.jmx_enabled=false` to opt out; registration failures are logged and swallowed
rather than breaking the application.

Byte-valued fields follow one convention throughout: **`-1` means unavailable** (the model is
closed, or the native library could not be reached) and **`0` means genuinely zero**. The
distinction matters most for `stagingBytes`, which is legitimately `0` whenever every model input is
memory-planned — the ExecuTorch export default, and true of very nearly every `.pte` in practice.
Unavailable values are excluded from the rollup totals rather than summed as negatives.

## Limitations

- **One `Model`/`Predictor` per thread.** `EtSymbolBlock.forward()` is not thread-safe on the same
  model. Share nothing; give each thread its own `ZooModel`. In particular do not put a shared model
  behind a `ThreadLocal` `Predictor` — that shares the model. And never `close()` a model with a
  `forward()` still in flight: the native handle goes away underneath the running call.
- **The XNNPACK weight cache is deliberately not exposed.** Enabling it makes
  `XnnpackBackend::execute()` hold a second process-global mutex for the whole delegate call, which
  would undo everything `workspaceSharingMode=disabled` buys. It is off by default in the pinned
  runtime.
- **The XNNPACK delegate workspace is not counted in the reported native footprint.**
  `plannedArenaBytes` is ExecuTorch's planned activation arena only, so a delegated model's real
  native usage is higher than the reported figure by the size of its workspace — which under
  `disabled` scales with the number of live models.
- **`NDArray` support is minimal**, as described above; and the engine is CPU-only, with no CUDA,
  Metal, or NPU delegates.

## Building from source

The Java side is an ordinary Gradle build on JDK 17, but the engine also needs a native JNI shim
that is **built from source, not committed** — the JVM integration tests will not run until it
exists. The ExecuTorch runtime the shim links against is *not* compiled here either: CMake downloads
a hash-pinned, build-attested tarball, so no ExecuTorch checkout is required, only network access.
On Linux the shim is built inside a `manylinux_2_28` container to hold the glibc 2.28 floor that
ExecuTorch's `torch` dependency imposes; on Windows it is built directly with MSVC 2022.

See [docs/building.md](docs/building.md).

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

Full license texts for these **and** every other component the runtime distribution tracks
are bundled in each native classifier jar under `META-INF/licenses/executorch-runtime/`
(`LICENSE` + `THIRD-PARTY-NOTICES/`), sourced verbatim from the runtime tarball. This list is
tied to the runtime pin (`native/cmake/EtRuntimePin.cmake`); refresh it when the pin bumps.
