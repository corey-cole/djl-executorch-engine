# Benchmarking ExecuTorch vs. TorchScript

> **Status:** living document. A starting plan for the manual benchmark that backs the "why
> ExecuTorch in DJL" argument for an eventual upstream PR. Expect to add models and metrics as we go.

**This is a manual task, not a CI gate.** It runs ad hoc to produce evidence for the PR narrative,
not on every commit.

## Goal

Produce a defensible answer to *"why add an ExecuTorch engine when DJL already has a PyTorch
engine?"* by running the **same representative model** through both:

- **DJL PyTorch engine** (LibTorch) loading a **TorchScript** artifact, and
- **this ExecuTorch engine** loading the equivalent **`.pte`** artifact,

and comparing them on the axes where ExecuTorch's case actually lives.

## Which engine surface we benchmark

The engine has two input/output surfaces; the benchmark uses the **general** one:

- **General Phase 2a path** — zero-copy, arbitrary-shape, multi-dtype tensors via `NDList` /
  `Predictor`. **Benchmark through this.** It has been capable since Phase 2a and imposes no shape
  restriction, so any standard single-tensor-in/out model runs without engine changes.
- **Phase 2b `MapTranslator`** — named *scalar* params → `double[]`. A tabular/feature surface; **no
  well-known DL model fits it**, so it is not the benchmark vehicle. (The models that fit it — a
  housing-regression MLP, a DLRM-style scorer — aren't recognizable enough to make the PR case.)

Practically: load via `Criteria`/`Predictor` with an `NDList`-producing translator (or a thin image
translator), not the scalar `MapTranslator`.

## Model selection

### Primary: MobileNetV2

The pick, chosen because it *strengthens* the "why" rather than just being convenient:

- **ExecuTorch's home turf.** ExecuTorch is Meta's on-device successor to PyTorch Mobile/Lite, and
  MobileNetV2 appears in ExecuTorch's own examples/tutorials. Benchmarking the runtime on the class
  of model it exists to serve is the honest framing.
- **Single tensor in/out:** `[1,3,224,224] → [1,1000]`. Clean for both export paths and the engine.
- **Exports cleanly to both targets** (see below); torchvision ships pretrained weights, so it is
  reproducible.
- **Recognizable** to any DJL reviewer.

### Secondary: ResNet-18

Same `[1,3,224,224] → [1,1000]` shape, heavier, a classic baseline — a useful second data point.

### Deliberately deferred

- **DistilBERT / MobileBERT** — token + attention-mask multi-input adds `torch.export` friction and
  a multi-input translator we don't need for a first pass. Revisit once the engine grows a
  multi-input tensor translator.

## Export paths (both scripts live under `tools/scripts/`)

For each model, produce two artifacts from the **same weights**:

- **TorchScript** (DJL PyTorch engine baseline): `torch.jit.trace(model, example_input)` →
  `*.pt`/`*.torchscript`.
- **ExecuTorch** (`.pte`): `torch.export(...)` → XNNPACK-lowered `to_edge_transform_and_lower(...)`
  → `.to_executorch()`. Mirror the existing `tools/scripts/export_with_spec.py` structure.

Pin the torch / torchvision / executorch versions used, and record them alongside the artifacts
(the same way `model_spec.json` records `executorch_version`) so the numbers are reproducible.

## Metrics — measure where the argument actually lives

A pure **steady-state desktop-CPU latency** race against LibTorch may not flatter ExecuTorch;
LibTorch is heavily tuned for exactly that. ExecuTorch's real case is footprint, cold-start,
AOT-compiled portability, no Python runtime, and quantization/XNNPACK on edge. So record:

1. **Runtime / binary size** — shipped native footprint of each engine's libraries.
2. **Resident memory** — RSS during steady-state inference.
3. **Cold-start / first-inference latency** — load + first `forward` (where AOT compilation helps).
4. **Steady-state latency / throughput** — warm loop. **Report this even if it's a wash** — candor
   strengthens the PR more than a cherry-picked win.
5. **(Stretch) XNNPACK-lowered + quantized `.pte` vs. TorchScript CPU** — closest to the intended
   deployment, and the most flattering *and* most honest comparison.

Lead the eventual write-up with footprint and cold-start; be transparent about steady-state.

## Profiling (ExecuTorch devtools) — measuring the overhead that gates its architecture

ExecuTorch profiling is **build-time, not a runtime switch on a normal build**: the runtime must be
compiled with `EXECUTORCH_BUILD_DEVTOOLS=ON` + `EXECUTORCH_ENABLE_EVENT_TRACER=ON`, the `Module`
constructed with an `ETDumpGen` event tracer, the ETDump buffer pulled after a run, and analyzed
offline with the **Inspector** — which only maps events back to graph ops when an **ETRecord** was
emitted at export time. Building profiling support is its own spec; what belongs *here* is the one
measurement that decides its shipping architecture.

**The gating question:** what does a devtools-enabled-but-not-tracing build cost the default path, in
**binary size** and **steady-state latency**, versus the plain XNNPACK build we ship today? Two
outcomes:

- **Negligible →** ship **one** devtools-enabled `.so` with profiling as a runtime opt-in (attach the
  tracer per model, or not).
- **Material →** ship a **separate** profile-capable `.so` (an extra build-matrix row) and keep the
  default artifact lean.

Run this as a small spike alongside the size/latency metrics above: build the runtime twice (plain
vs. devtools+tracer-enabled), link both shims, and record the **size delta** and the **steady-state
delta** over `add.pte` + MobileNetV2 with **no tracer attached**. That single number picks
one-artifact-vs-two before any profiling code is written.

**Cross-cutting touch points** (for the eventual profiling spec, noted so the measurement is read in
context):

- **Build:** a devtools-enabled runtime (one shipped artifact, or a separate profiling one).
- **Python:** `tools/scripts/export_*.py` gain an ETRecord option alongside the `.pte` (the Inspector
  needs it to correlate runtime events to graph ops).
- **Core + shell + Java:** `EtRuntime` grows an event-tracer-aware `forward` + an ETDump-buffer
  accessor (owned / valid-until-next-run, the same lifetime discipline as the output views); the JNI
  shell and Java engine expose a profiling toggle + dump retrieval. This is the first feature to
  extend the core's C++ surface.

## Harness notes (to fill in)

- Same input tensor, same warmup count, same iteration count for both engines.
- Fixed thread count / pin to avoid scheduler noise; record CPU + OS + glibc.
- Run both engines in the same JVM process via DJL `Criteria` so the comparison is apples-to-apples
  at the DJL API boundary.

## Open items

- Decide artifact storage (committed small fixtures vs. generated on demand by the export scripts).
- Pick the measurement tool for steady-state (JMH vs. a simple timed loop) — JMH if we want
  defensible JVM-side numbers.
- Quantization recipe for the stretch comparison.
- **Profiling-overhead spike** — measure the devtools-enabled-but-not-tracing size + steady-state
  latency delta vs. the plain build; gates the one-artifact-vs-two decision for profiling (see the
  Profiling section).
- **Ship logging or not (`EXECUTORCH_ENABLE_LOGGING`)** — the ET_LOG→slf4j bridge needs the runtime
  built with `-DEXECUTORCH_ENABLE_LOGGING=ON` (off by default in Release). ExecuTorch's own note:
  *"the logging strings can be large."* Measure the shipped `.so` size delta (and any steady-state
  cost) of logging-on vs. logging-off, then decide whether the default artifact ships with logging
  enabled. Currently selected via the `logging` runtime variant (`ET_RUNTIME_VARIANT` in
  `native/cmake/EtRuntimePin.cmake`), the default the engine downloads; this is the decision that
  confirms or reverts it.
  **Coupling:** `LoggingBridgeIT` runs in the default `./gradlew test` and asserts a real ET log —
  it only passes with logging on. If this decision reverts the flag, tag/guard that test in the same
  change (e.g. `@Tag("logging")` excluded by default, or a conditional), or it fails with an empty
  appender and no obvious cause. **Measured so far (linux-x86_64):** the shim `.so` grew **8.5 MB → 11.5 MB (~35%)** when the
  runtime was rebuilt with logging on — a first data point for this decision.

  > **Note (2026-07-03):** `native/build_variants.sh` no longer builds the three ExecuTorch variants
  > from source — it downloads the prebuilt `bare`/`logging`/`devtools` tarballs from Repo A (selected
  > via `ET_RUNTIME_VARIANT`) and times each. Same table, faster, no torch/ET build.
- **Logging level-gating** — the bridge maps ET levels to slf4j but does not gate on
  `logger.isXxxEnabled()` before the JNI hop. Measure the per-message JNI-hop cost to decide whether
  gating is worth adding (deferred from the logging spec; ET_LOG volume is low).
