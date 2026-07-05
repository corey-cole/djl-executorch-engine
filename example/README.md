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

    ./gradlew :example:run                      # ET_HYBRID (default)
    ./gradlew :example:run --args="ET_NATIVE"   # LibTorch-free preprocessing
    ./gradlew :example:run --args="PYTORCH"     # DJL PyTorch engine

Classifies a bundled image and prints the top-5 labels. The variant selects engine + preprocessing:
`ET_HYBRID` and `PYTORCH` preprocess on a PyTorch-backed manager; `ET_NATIVE` preprocesses in plain
Java and runs the ExecuTorch forward with **no LibTorch loaded** (see Caveats).

## Run the benchmark

    ./gradlew :example:jmh --no-configuration-cache --rerun-tasks

Races three arms over two modes:
- **steady-state** (`AverageTime`) — warm inference loop, the fair race;
- **cold-start** (`SingleShotTime`) — load + first forward, where AOT compilation helps.

The `(variant)` column is:
- `ET_HYBRID` — ExecuTorch forward, PyTorch-backed preprocessing;
- `PYTORCH` — DJL PyTorch engine (LibTorch);
- `ET_NATIVE` — ExecuTorch forward, plain-Java preprocessing; its JMH fork loads no LibTorch.

Each arm fails fast pointing back at `exportModels` if its artifact (`.pte`/`.pt`) is missing.

> **`--no-configuration-cache` is required.** This repo runs with Gradle's configuration cache on
> globally, but the `me.champeau.jmh` plugin's `jmhJar` task (which builds the benchmark's shaded
> jar) is not configuration-cache compatible. Without the flag you'll hit a configuration-cache
> error rather than a benchmark run.

## Sample benchmark results

Test results on i7-1185G7 w/ 32GB of memory, Zulu17.66+19-CA

```
Benchmark                       (variant)  Mode  Cnt    Score    Error  Units
MobilenetBenchmark.steadyState  ET_HYBRID  avgt    5   12.777 ±  0.590  ms/op
MobilenetBenchmark.steadyState    PYTORCH  avgt    5   20.554 ±  0.491  ms/op
MobilenetBenchmark.steadyState  ET_NATIVE  avgt    5   12.991 ±  0.187  ms/op
MobilenetBenchmark.coldStart    ET_HYBRID    ss    5   18.211 ±  4.637  ms/op
MobilenetBenchmark.coldStart      PYTORCH    ss    5  140.853 ± 20.606  ms/op
MobilenetBenchmark.coldStart    ET_NATIVE    ss    5   19.453 ± 17.756  ms/op
```

ExecuTorch (with XNNPACK) shows an improvement over PyTorch at steady-state.  The `ET_NATIVE` variant
is roughly the same speed.  It's primary benefit is that it shows how to use the engine for complex
tasks without requiring any PyTorch dependencies.  Both ExecuTorch arms show a massive reduction in cold-start,
something that will matter if there are multiple models that are frequently loaded/unloaded.

## Caveats

- **`ET_HYBRID` preprocessing uses LibTorch; `ET_NATIVE` does not.** ExecuTorch's `NDArray` is a
  minimal data holder with no `NDArrayEx` support, so DJL's built-in image transforms can't run on
  it. `ET_HYBRID` (and `PYTORCH`) work around this by preprocessing on a PyTorch-backed `NDManager`
  — so `ET_HYBRID`'s "no LibTorch" story is qualified (it's the preprocessing surface, not
  inference, and this is a "Phase 1: no hybrid mode" limitation, not a fundamental one).
  `ET_NATIVE` instead preprocesses in plain Java (`Image.resize` + a hand-written normalize) and
  builds the input tensor straight in the ExecuTorch manager, so on that path LibTorch never loads.
  Note the two do **not** produce bit-identical tensors — the resize algorithms differ (DJL's tensor
  `Resize` vs `Image.resize`/Graphics2D) — but both yield correct classifications; the comparison is
  latency + dependency footprint, not identical pixels.
- **Reported numbers are illustrative, not authoritative.** A single-iteration smoke run of this
  benchmark is not a rigorous measurement (no meaningful warmup, no repeat forks, tiny sample size,
  no CPU pinning). Treat any numbers you see quoted elsewhere as a sanity check that both arms run,
  not as a performance verdict — run the benchmark yourself (with adequate warmup/iterations) for
  real figures, and see `docs/benchmarking.md` for the full methodology.

## Out-of-band metrics (not measured by JMH)

JMH covers latency only. Capture the other two axes from `docs/benchmarking.md` manually:

- **Runtime / binary size** — compare the shipped native footprint:
  `ls -la $(find ~/.djl.ai -name '*.so' -path '*pytorch*')` (LibTorch) vs the engine's
  `libexecutorch_djl.so` (~11.5 MB). Note: `ET_HYBRID` and `PYTORCH` link LibTorch in-process (for
  preprocessing / inference respectively), so a footprint taken from those runs won't reflect a
  LibTorch-free deployment — measure the libraries directly, or sample the `ET_NATIVE` run, whose
  process loads no LibTorch.
- **Resident memory (RSS)** — run each arm as its own process and sample RSS during the steady-state
  loop (e.g. `/usr/bin/time -v` or `ps -o rss=`), reported per engine.
