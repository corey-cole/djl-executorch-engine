# Profiling ExecuTorch models (opt-in ETDump)

ExecuTorch's event tracer is available in this engine as a **per-model, load-time opt-in**. A
profiled model records one event block per forward into an ETDump buffer; the caller pulls the
bytes whenever it wants and analyzes them offline with ExecuTorch's Python **Inspector** — the same
tool upstream uses. This page is the "how"; the measurement that decided the shipping architecture
(one artifact, profiling as a runtime opt-in) is in [benchmarking.md](benchmarking.md), and the
build/capability mechanics are in the design spec
[`2026-08-24-executorch-devtools-profiling-design.md`](superpowers/specs/2026-08-24-executorch-devtools-profiling-design.md).

## Enabling

Attach the tracer to one model at load time with the per-model DJL option:

```java
Criteria<..., ...> criteria =
        Criteria.builder()
                .setTypes(...)
                .optEngine("ExecuTorch")
                .optModelPath(...)
                .optOption(EtEngine.PROFILING_OPTION, "true")   // or "false" (the default)
                .build();
```

The option key is published as `EtEngine.PROFILING_OPTION`; accepted values are case-insensitive
`true`/`false`, and anything else fails the load.

**There is deliberately no JVM property counterpart**, unlike `workspaceSharingMode`. A property
would let one JVM flag attach an event tracer to every model in the process — including models
whose owner never pulls the dump — and an ETDump grows across every forward until it is pulled.
Profiling is a diagnostic with a real memory cost, so enabling it is a decision at the load site
and nowhere else. The absence is the design, not an omission (stated the same way in `EtProfiling`).

## Pulling the dump

```java
byte[] dump = ((EtModel) zooModel.getWrappedModel()).etDump();
```

`EtModel.etDump()` returns the finalized ETDump covering **every forward since the last call**, as a
fresh `byte[]` (never null):

- The buffer **grows across forwards until pulled** — a long-running profiled model should be
  drained periodically.
- **Pulling is the drain**: the forward after a pull starts a fresh dump, so two pulls with no
  forward between them return equal bytes and the buffer never accumulates unboundedly.
- Empty when the model was not loaded with `PROFILING_OPTION`, or when no forward has run since the
  last call.
- **Threading:** pull only from the thread that owns the model, or with no forward in flight. A
  pull concurrent with a forward races on the native dump state (`dumpFinalized`,
  `everForwarded`, `lastDump`) and on `ETDumpGen`, which upstream does not document as
  thread-safe; the lock `EtSymbolBlock.etDump()` takes serializes a pull against `close()` only,
  and the forward path is deliberately lock-free. The repo's one-model-per-thread `forward()` rule
  already implies it; the dump API makes it explicit.
- There is no `writeEtDump(Path)` convenience — it is `Files.write(path, dump)` at the call site.

The pulled bytes are a size-prefixed flatbuffer (root built with `start_as_root_with_size`, so the
`ED00` identifier sits within the first 16 bytes).

## Platforms

**`EtEngine.devtoolsAvailable()` is the contract — never a platform name.** It reports whether the
runtime this build links was compiled with the event tracer, and it is a static query you can make
before loading anything:

```java
if (!EtEngine.devtoolsAvailable()) {
    // profiling is not provisioned on this platform
}
```

Today `linux-x86_64` ships a `devtools` runtime and answers `true`. `linux-aarch64` is **ready to
be provisioned**: its devtools tarball ships the same layout as the x86_64 one —
`lib/cmake/ETNPExtras/`, `lib/libetnp_ops_lstm.a`, and `lib/libopenvino_backend.a` (parity verified
at pin v1.4.1-3) — so it joins by adding `linux-aarch64` to `ET_DEVTOOLS_SUPPORTED_PLATFORMS` plus a
test run; the 2026-08-25 radxa run on the logging runtime, where `ProfilingIT`'s devtools-absent arm
executed and passed, is that test baseline. `windows-x86_64` was verified on the 2026-08-25 winbox
run: the MSVC build and the full JVM suite pass on the logging variant, `ProfilingIT`'s
devtools-absent arm executes and passes there, and the static-CRT gate holds. The pin publishes
Windows devtools rows as of v1.4.1-3 and the CMake variant guard already accepts `devtools` there
(only the Linux-only `bare` benchmarking build is refused), so provisioning is now truly a list
edit plus a test — but a Windows devtools build must use `flatcc_builder_aligned_free` for the
ETDump buffer: the current `etDump()` uses `free()`, which is correct for POSIX only (the code
comment already records this).

Requesting profiling where the capability is absent fails the load with a message identifying the
platform's runtime as lacking the event tracer; a model loaded without the option returns an empty
dump rather than throwing. An unrecognized option value fails the load regardless of platform.

## Exporting an ETRecord

The Inspector maps runtime events back to **graph ops** only when an ETRecord was emitted at export
time. `tools/scripts/export_mobilenet.py` emits the ETRecord with `--etrecord`, default off (an
ETRecord embeds the program buffer and the graph modules, and the script's common case is producing
a demo model):

```bash
cd example/build/models
uv run ../../../tools/scripts/export_mobilenet.py --etrecord
```

This writes `mobilenet_v2.etrecord` next to `mobilenet_v2.pte` (and the other artifacts). The same
pinned torch/torchvision/executorch environment backs `./gradlew :example:exportModels`; the
ETRecord is the only addition.

## The manual Inspector procedure

The full export → run → inspect loop is **manual** — it cannot run in CI: the Inspector is Python,
the artifacts are gitignored build outputs that need `uv`, torch, and network, and the JVM tests
deliberately use committed fixtures. Run it **once per pin bump** (see below). The steps:

**1. Export the model with its ETRecord** (from the repo root):

```bash
cd example/build/models
uv run ../../../tools/scripts/export_mobilenet.py --etrecord
```

**2. Run several forwards with profiling on and write the dump.** Add the option to the model's
`Criteria` and pull the bytes (a pass-through translator like the repo's `AddOutputTranslator` is
all the engine's general path needs; your app's own translator works unchanged — profiling is
orthogonal to it):

```java
// DumpProfiledModel.java — run a few forwards, then write mobilenet_v2.etdump.
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.nio.file.Files;
import java.nio.file.Path;
import org.measly.executorch.engine.EtEngine;
import org.measly.executorch.engine.EtModel;

public class DumpProfiledModel {
    public static void main(String[] args) throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray input = manager.zeros(new Shape(1, 3, 224, 224));
            try (ZooModel<NDList, NDArray> model =
                            Criteria.builder()
                                    .setTypes(NDList.class, NDArray.class)
                                    .optEngine("ExecuTorch")
                                    .optModelPath(Path.of("mobilenet_v2.pte"))
                                    .optOption(EtEngine.PROFILING_OPTION, "true")
                                    .optTranslator(
                                            new Translator<NDList, NDArray>() {
                                                @Override
                                                public NDList processInput(
                                                        TranslatorContext ctx, NDList input) {
                                                    return input;
                                                }

                                                @Override
                                                public NDArray processOutput(
                                                        TranslatorContext ctx, NDList list) {
                                                    return list.singletonOrThrow();
                                                }

                                                @Override
                                                public Batchifier getBatchifier() {
                                                    return null;
                                                }
                                            })
                                    .build()
                                    .loadModel();
                    Predictor<NDList, NDArray> predictor = model.newPredictor()) {
                for (int i = 0; i < 10; i++) {
                    predictor.predict(new NDList(input));
                }
                byte[] dump = ((EtModel) model.getWrappedModel()).etDump();
                Files.write(Path.of("mobilenet_v2.etdump"), dump);
            }
        }
    }
}
```

**3. Inspect.** The Inspector ships in the `executorch` PyPI package (`executorch.devtools`); run
the same executorch version as the pinned runtime (`EtEngine.EXECUTORCH_VERSION`, currently
`1.4.1`), because the ETDump schema is a compatibility surface across versions:

```bash
uv run --python 3.11 --with executorch==1.4.1 python inspect_etdump.py
```

with `inspect_etdump.py`:

```python
from executorch.devtools import Inspector

inspector = Inspector(
    etdump_path="mobilenet_v2.etdump",
    etrecord_path="mobilenet_v2.etrecord",
)
for block in inspector.event_blocks:
    print(block.name)
    for event in block.events:
        print(f"  {event.name}: {event.duration_ms:.3f} ms")
```

Events attribute to graph ops only because the ETRecord was emitted; without it the Inspector still
reads the dump but cannot correlate events to the model graph.

**Why it is a per-pin-bump check:** the ETRecord↔ETDump correlation cannot run in CI, and the JVM
tests deliberately use committed fixtures. After every `native/cmake/EtRuntimePin.cmake` bump
(`v<etver>-<pkgrev>` release, which can change the ETDump schema or the Inspector's expectations),
re-run this procedure and confirm events still attribute to graph ops. It is a documented procedure,
not an automated gate — a `tools/scripts/` correlation script would be exercised approximately never.

## The cost

Profiling rides on the shipped `linux-x86_64` `.so` — there is no second artifact. The measured
cost of shipping a devtools runtime with **no tracer attached** (the steady state of every
unprofiled model):

- **Binary size:** the design-time measurement in [benchmarking.md](benchmarking.md) recorded
  `libexecutorch_djl.so` growing from 12,440,632 to 12,578,440 bytes, **+137,808 bytes (+1.11%)**.
  The linux-x86_64 `.so` staged on this branch is 12,710,016 bytes (~12.7 MB) — the delta sits on a
  slightly later baseline, so treat the percentage, not the absolute figure, as the stable number.
- **Steady-state latency, no tracer attached:** bounded, not just undetected — devtools − logging
  measured +0.066% (0.0038 ms on a 5.72 ms MobileNetV2 forward) against a standard error of 0.0085
  ms, with a 95% confidence interval of about ±0.35% on the difference. At the shipped thread
  setting (`intraop=16`) the variants measured tied (1.0162 ms vs 1.0244 ms). Peak RSS grows ~0.25
  MB (under 1%).
- **A tracer attached is a different regime:** recording an event block per forward costs real
  per-forward time that this measurement deliberately does not cover — which is exactly why
  profiling is a per-model opt-in rather than a default.
