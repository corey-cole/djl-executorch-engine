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

- **General Phase 2a path** — direct-buffer, arbitrary-shape, multi-dtype tensors via `NDList` /
  `Predictor`. **Benchmark through this.** It has been capable since Phase 2a and imposes no shape
  restriction, so any standard single-tensor-in/out model runs without engine changes. (Direct
  buffers are what the engine hands to the runtime; they are not zero-copy end to end — ExecuTorch
  copies memory-planned inputs into its arena, which is the export default. See
  `executorch-host-buffer-contract-brief.md`.)
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

## Profiling (ExecuTorch devtools) — decided: one artifact, profiling is a runtime opt-in

ExecuTorch profiling is **build-time, not a runtime switch on a normal build** — or it was: the
runtime must be compiled with `EXECUTORCH_BUILD_DEVTOOLS=ON` + `EXECUTORCH_ENABLE_EVENT_TRACER=ON`,
the `Module` constructed with an `ETDumpGen` event tracer, the ETDump buffer pulled after a run, and
analyzed offline with the **Inspector** — which only maps events back to graph ops when an
**ETRecord** was emitted at export time. The engine now ships a devtools runtime on `linux-x86_64`
and exposes profiling as a per-model opt-in (`Criteria.optOption(EtEngine.PROFILING_OPTION,
"true")`); see [profiling.md](profiling.md) for how to use it. What belongs *here* is the
measurement that decided the shipping architecture.

**The gating question was:** what does a devtools-enabled-but-not-tracing build cost the default
path, in **binary size** and **steady-state latency**, versus the plain XNNPACK build we ship? Two
outcomes were on the table:

- **Negligible →** ship **one** devtools-enabled `.so` with profiling as a runtime opt-in (attach
  the tracer per model, or not).
- **Material →** ship a **separate** profile-capable `.so` (an extra build-matrix row) and keep the
  default artifact lean.

**The answer: negligible — one artifact.** Measured on `linux-x86_64`, MobileNetV2, **no tracer
attached**, seven interleaved reps per variant at `intraop=1` on an idle Ryzen 7 5800XT:

| variant | per-forward mean (ms) | sd | min | max |
|---|---|---|---|---|
| bare | 5.7160 | 0.0128 | 5.7014 | 5.7405 |
| logging | 5.7168 | 0.0118 | 5.6974 | 5.7295 |
| devtools | 5.7206 | 0.0192 | 5.6823 | 5.7459 |

devtools − logging is **+0.0038 ms (+0.066%)** against a standard error of 0.0085 ms (t ≈ 0.45),
and the 95% confidence interval on the difference is about **±0.35%** — an upper bound on the cost,
not merely a failure to detect one. At the shipped thread setting (`intraop=16`) devtools measured
1.0162 ms against logging's 1.0244 ms — tied, with the logging arm pulled up by one outlier.
**Binary size:** `libexecutorch_djl.so` grows from 12,440,632 to 12,578,440 bytes,
**+137,808 bytes (+1.11%)** (the current linux-x86_64 `.so` on this branch is 12,710,016 bytes,
~12.7 MB, so the absolute figure has drifted slightly from the design-time baseline). Peak RSS grows
~0.25 MB (under 1%).

An earlier pass on a 4-core laptop showed run-to-run spread of ~8%, which bounds nothing useful; the
desktop's 0.2–0.7% spread is what makes the interval meaningful. Both runs used
`native/build_variants.sh` semantics with the model parameterized; `add.pte` cannot resolve a
build-flag delta at all and prints `warm_mean_ms=0.001` for every variant.

**Decision: one artifact.** 138 KB and a bounded-under-0.35% steady-state cost do not justify a
second build-matrix row, a second staging path, and a `LibUtils` selection rule. Profiling ships as
a per-model runtime opt-in on the shipped artifact, gated on the `devtoolsAvailable()` capability
query rather than on a platform name. (This measurement provisioned `linux-x86_64` only;
`linux-aarch64` and `windows-x86_64` followed once each was verified on real hardware, so all three
now ship a devtools runtime — see [profiling.md](profiling.md) for the per-platform evidence. The
one-artifact decision held for each of them, and the capability query is what made extending it a
list edit rather than a redesign.)
Note the boundary the measurement drew: attaching a tracer costs **real per-forward time** that was
deliberately not measured here — which is exactly why the option is opt-in per model. The
cross-cutting touch points this decision resolved: the build ships one devtools-enabled artifact on
Linux; the export script gained an ETRecord option (`tools/scripts/export_mobilenet.py --etrecord`)
so the Inspector can correlate events to graph ops; and the core/shell/Java surface grew the
event-tracer-aware construction, the ETDump-buffer accessor, and the profiling toggle. The manual
Inspector procedure is documented in [profiling.md](profiling.md).

## Harness notes (to fill in)

- Same input tensor, same warmup count, same iteration count for both engines.
- Fixed thread count / pin to avoid scheduler noise; record CPU + OS + glibc.
- Run both engines in the same JVM process via DJL `Criteria` so the comparison is apples-to-apples
  at the DJL API boundary.

## Decisions

### Ship logging — RESOLVED 2026-07-04: ship the `logging` variant (the downloaded default)

> **Superseded 2026-08-24 for the variant, not for the finding.** The measurement below stands:
> logging's cost is real, repeatable, and confined to startup. What changed is the variant shipped —
> the devtools measurement later in this file resolved to ship `devtools`, which is built *with*
> logging, so the conclusion "ship with logging on" survived and only the tarball changed.

Benchmarked `bare` / `logging` / `devtools` over `add.pte` (4 runs, linux-x86_64 desktop, the Release
timing harness). Logging's cost is real, repeatable, and **confined to startup**:

- **Load:** `logging − bare` ≈ **+7–8 µs** (per-run 2–12 µs; logging > bare in 4/4 runs).
- **First inference (cold):** `logging − bare` ≈ **+100 µs**, one-time — but noisy (per-run 5–189 µs;
  the box was not CPU-pinned). Direction consistent in 4/4 runs.
- **Steady-state (warm):** **below the harness's 1 µs reporting resolution** for every variant — no
  measurable difference (under granularity, not proven-equal).
- **Footprint:** shim `.so` **~8.5 → 11.5 MB (+35%)** with logging on. Irrelevant on a desktop/server
  JVM (the JVM + ExecuTorch/XNNPACK dominate); would only matter for an edge/mobile target, which this
  engine does not serve.

**Why ship it:** the only cost is a one-time, sub-millisecond startup penalty that amortizes to zero
over any served workload, plus MBs that don't matter for this deployment target — in exchange for the
ET_LOG→slf4j bridge working out of the box. Caveat for the write-up: `add.pte` is a 1-op model, so the
~100 µs cold cost is a floor — a large model's first inference logs proportionally more (still one-time).

**Consequence:** `LoggingBridgeIT` (in the default `./gradlew test`) asserts a real ET log and only
passes with logging on — shipping the `logging` default keeps it green. If this is ever reverted, guard
that test in the same change (e.g. `@Tag("logging")` excluded by default) or it fails with an empty
appender and no obvious cause.

**Deferred (not now):** give the `devtools` variant logging too, rather than adding a 4th variant —
the penalties are additive but still one-time and sub-ms, and devtools *is* the observability build, so
logs + event traces belong together. This is a **Repo A** change (the variant flag map lives in its
`build-runtime.sh`) that re-rolls the pin (`EtRuntimePin.cmake` bump), not engine code. **Resolved by
Repo A, not by us:** the devtools variant is built **with** logging (pins `1.4.1-2` and later,
including the current `1.4.1-3`), so it is not a logging-free comparison point in
`native/build_variants.sh` — only `bare` is.

## Open items

- Decide artifact storage (committed small fixtures vs. generated on demand by the export scripts).
- Pick the measurement tool for steady-state (JMH vs. a simple timed loop) — JMH if we want
  defensible JVM-side numbers.
- Quantization recipe for the stretch comparison.
- ~~**Profiling-overhead spike**~~ — **RESOLVED 2026-08-25: one artifact.** The
  devtools-enabled-but-not-tracing delta was negligible (+137,808 bytes of `.so`, +0.066% steady
  state with a ±0.35% 95% CI bound); all three shipped platforms link a devtools runtime with
  profiling as a per-model opt-in. See the [Profiling section](#profiling-executorch-devtools--decided-one-artifact-profiling-is-a-runtime-opt-in).
- ~~**Ship logging or not (`EXECUTORCH_ENABLE_LOGGING`)**~~ — **RESOLVED 2026-07-04: ship logging.**
  The shipped tarball is now `devtools` rather than `logging`, which is built *with* logging, so the
  decision holds and only the variant changed. See [Decisions](#decisions) for the data.
- **Harness docs for user `.pte` artifacts** *(partly addressed)* — the timing-harness binary
  already accepts a model path (`et_timing_harness <pte> [iters] [warmup]`); `bench.sh` just
  hardcodes `native/spike/add.pte`. The **MobileNetV2** benchmark has now landed as the `:example`
  subproject (export + run + JMH benchmark against the DJL PyTorch engine) — see
  `example/README.md` for the user-facing walkthrough (`exportModels`, `run`, `jmh
  --no-configuration-cache`). That covers the JVM/DJL-API-level "representative numbers for a real
  model" need; a direct-invocation doc and/or `MODEL=` override for the raw native `bench.sh`
  harness is still undone and can be revisited separately if a non-JVM entry point is wanted.

  > **Note (2026-07-03):** `native/build_variants.sh` no longer builds the three ExecuTorch variants
  > from source — it downloads the prebuilt `bare`/`logging`/`devtools` tarballs from Repo A (selected
  > via `ET_RUNTIME_VARIANT`) and times each. Same table, faster, no torch/ET build.
- **Logging level-gating** — the bridge maps ET levels to slf4j but does not gate on
  `logger.isXxxEnabled()` before the JNI hop. Measure the per-message JNI-hop cost to decide whether
  gating is worth adding (deferred from the logging spec; ET_LOG volume is low).
