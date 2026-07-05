# MobileNetV2 Example & Benchmark

Runs MobileNetV2 `[1,3,224,224] → [1,1000]` through this ExecuTorch engine, and benchmarks it
head-to-head against the DJL PyTorch engine (LibTorch) on the same weights.

## Prerequisites

- **`uv`** on `PATH` (used to export the models; it self-provisions the pinned
  torch/torchvision/executorch via PEP 723 inline metadata in `tools/scripts/export_mobilenet.py`).
- Network on first run: `uv` downloads the export deps, and the PyTorch benchmark arm downloads
  LibTorch natives on its first invocation.

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

    ./gradlew :example:jmh --no-configuration-cache

Races `ExecuTorch` (`.pte`) vs `PyTorch` (`.pt`) over two modes:
- **steady-state** (`AverageTime`) — warm inference loop, the fair race;
- **cold-start** (`SingleShotTime`) — load + first forward, where AOT compilation helps.

Both arms fail fast pointing back at `exportModels` if the artifacts are missing.

> **`--no-configuration-cache` is required.** This repo runs with Gradle's configuration cache on
> globally, but the `me.champeau.jmh` plugin's `jmhJar` task (which builds the benchmark's shaded
> jar) is not configuration-cache compatible. Without the flag you'll hit a configuration-cache
> error rather than a benchmark run.

## Sample benchmark results

Test results on i7-1185G7 w/ 32GB of memory, Zulu17.66+19-CA

```
Benchmark                         (engine)  Mode  Cnt    Score    Error  Units
MobilenetBenchmark.steadyState  ExecuTorch  avgt    5   19.401 ±  1.164  ms/op
MobilenetBenchmark.steadyState     PyTorch  avgt    5   28.995 ±  1.574  ms/op
MobilenetBenchmark.coldStart    ExecuTorch    ss    5   35.252 ± 22.887  ms/op
MobilenetBenchmark.coldStart       PyTorch    ss    5  286.984 ± 50.077  ms/op
```

ExecuTorch (with XNNPACK) shows a modest improvement over PyTorch at steady-state.  The real win from
the benchmarks above is the reduction in cold-start, something that will matter if models are frequently
loaded from disk.

## Caveats

- **Preprocessing still uses LibTorch, even on the ExecuTorch arm.** The ExecuTorch engine's
  `NDArray` is a minimal data holder with no `NDArrayEx` support, so DJL's built-in image transforms
  (resize/to-tensor/normalize) can't run on it directly. `MobilenetTranslator` works around this by
  doing those transforms on a PyTorch-backed `NDManager` for *both* arms, and only hands the
  ExecuTorch manager a plain tensor for the forward pass itself. That makes the steady-state
  forward-pass comparison fair (identical preprocessing on both sides), but it means this example's
  "no LibTorch dependency" story is qualified — it's the image-preprocessing surface that currently
  needs LibTorch, not inference, and this is a "Phase 1: no hybrid mode" limitation, not a
  fundamental one.
- **Reported numbers are illustrative, not authoritative.** A single-iteration smoke run of this
  benchmark is not a rigorous measurement (no meaningful warmup, no repeat forks, tiny sample size,
  no CPU pinning). Treat any numbers you see quoted elsewhere as a sanity check that both arms run,
  not as a performance verdict — run the benchmark yourself (with adequate warmup/iterations) for
  real figures, and see `docs/benchmarking.md` for the full methodology.

## Out-of-band metrics (not measured by JMH)

JMH covers latency only. Capture the other two axes from `docs/benchmarking.md` manually:

- **Runtime / binary size** — compare the shipped native footprint:
  `ls -la $(find ~/.djl.ai -name '*.so' -path '*pytorch*')` (LibTorch) vs the engine's
  `libexecutorch_djl.so` (~11.5 MB). Note the caveat above: this example's ExecuTorch arm still
  links LibTorch in-process for preprocessing, so a footprint comparison taken *from this example's
  process* won't reflect a LibTorch-free deployment — measure the libraries directly instead.
- **Resident memory (RSS)** — run each arm as its own process and sample RSS during the steady-state
  loop (e.g. `/usr/bin/time -v` or `ps -o rss=`), reported per engine.
