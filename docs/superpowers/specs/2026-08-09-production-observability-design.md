# Production observability surface

**Date:** 2026-08-09
**Status:** Approved, ready for planning
**Scope:** Close the monitoring gaps that block operating this engine in production. Documentation
cleanup is a separate, subsequent cycle.

## Problem

The engine has logging but no metrics. Today an operator can see:

- slf4j output, including native `ET_LOG` forwarded through the PAL bridge (`native/jni/et_logging.cpp`).
- USDT/DTrace probes on the staging path (`native/core/et_probes.h`) — Linux only, requires
  `bpftrace`/`perf`, invisible to any ordinary monitoring stack.
- `EtEngine.getIntraOpThreads()`.

That is not enough to answer the three questions a production operator actually asks:

1. **Is inference slow, and is throughput what I expect?**
2. **Is the native side growing without bound?**
3. **Is this deployment configured the way I think it is?**

### Why DJL's own metrics do not close this

DJL provides `ai.djl.metric.Metrics` and `Predictor.setMetrics(...)`, which records
`Preprocess`/`Inference`/`Postprocess` timings per `predict()`. It does not serve as a production
signal, for reasons verified against the DJL 0.36.0 sources:

- **`Metrics.limit` defaults to 0, meaning uncapped.** Every `predict()` appends three retained
  `Metric` objects. In a long-running server that is unbounded retention unless the caller wires
  both `setLimit` and `setOnLimit`.
- **`addMetric` is check-then-act.** `list.size() >= limit` → `onLimit.accept(...)` → `list.clear()`
  → `list.add(...)` run with no lock held, over a `Collections.synchronizedList`. The natural usage
  — one `Predictor` per thread sharing one `Metrics` — races at the flush boundary: double-fired
  `onLimit` callbacks and dropped samples.
- **`Predictor.timestamp` is a plain non-volatile `long`** instance field, read-modify-written
  across `preprocessEnd`/`predictEnd`/`postProcessEnd`. Correct under our one-`Predictor`-per-thread
  contract, silently wrong if anyone shares one.
- **Aggregation is O(n).** `percentile()` copies the full list and sorts it; `mean()` streams it.
  Cost is proportional to everything buffered since the last flush.

`Metrics` is a time-series buffer suited to benchmarking and tracing, not a counter aggregator
suited to always-on production monitoring. It stays available and unmodified; our documentation
will present it as a profiling tool and name the uncapped default and the flush race so users do
not reach for it as a production signal.

## Design

### Public API

One new public class in `org.measly.executorch.engine`, plus immutable value types.

```java
EtEngineStats.snapshot()          // -> EtStatsSnapshot; never throws; cold path
EtEngineStats.registerMBean()     // explicit escape hatch
EtEngineStats.unregisterMBean()
```

`EtStatsSnapshot` is immutable and carries three groups.

**Configuration** — answers "is this deployment configured the way I think":

| Field | Source |
|---|---|
| `executorchVersion` | `EtEngine.EXECUTORCH_VERSION` |
| `platform` | resolved platform string (`linux-x86_64`, …) |
| `nativeLibraryPath` | the path `LibUtils` actually loaded |
| `intraOpThreads` | `EtNative.intraOpThreads()`, the effective native count |
| `defaultWorkspaceSharingMode` | resolved `ai.djl.executorch.workspace_sharing_mode`, or `unspecified` |

**Process totals**: `modelsLoaded` (cumulative), `modelsLive`, `totalPlannedArenaBytes`,
`totalStagingBytes`, and the closed-model rollup (`closedForwardCount`, `closedForwardTotalNanos`).

**Per-model** — `List<EtModelStats>`, live models only:

`name`, effective `workspaceSharingMode`, `plannedArenaBytes`, `stagingBytes`, `loadNanos`,
`forwardCount`, `forwardTotalNanos`, `forwardMaxNanos`.

### Counters

Per-model counters are plain `volatile long` fields on a holder owned by `EtSymbolBlock`, the class
that already makes the JNI forward call. `forwardInternal()` brackets the `EtNative.forward(...)`
call with `System.nanoTime()` and updates count, total, and max.

`forward()` is single-writer by the engine's existing threading contract (one `Model`/`Predictor`
per thread), so no CAS, no `LongAdder`, and no allocation on the hot path. `volatile` is present
solely so the snapshot reader observes the updates and so 64-bit reads cannot tear; the max update
is a safe read-compare-write given the single writer.

`loadNanos` is measured in `EtModel.load` around `EtNative.loadModule` + `EtNative.methodMeta`.
Because `EtRuntime`'s constructor calls `Module::load_forward()` unconditionally, this figure
includes delegate initialisation — that is intentional and matches where the cost actually lands.

### Registry and lifecycle

`EtEngineStats` holds a `ConcurrentHashMap` of live models keyed by native handle. `EtModel.load`
registers; `EtSymbolBlock.close()` removes.

On removal the model's totals fold into a process-level closed-model bucket rather than
disappearing. A restart-on-error loop would otherwise erase exactly the throughput history an
operator needs. Per-model detail is live-only; aggregates cover the process lifetime.

### JMX

`EtEngineStatsMXBean` is registered as `org.measly.executorch:type=EtEngineStats` on the platform
MBean server at the first model load, opt-out via `ai.djl.executorch.jmx_enabled=false`.

It is an **MXBean**, not a plain MBean, so `List<EtModelStats>` converts to `CompositeData`/
`TabularData` automatically and no hand-written `OpenType` code is needed. `EtModelStats` must
therefore be a conforming bean: public getters, no setters.

Registration is attempted exactly once. Failure — name collision, `SecurityManager`, a restricted
container — produces a single logged warning and is never retried per load and never fails a load.

## Native layer

Two changes. No pin bump and no `executorch-runtime-dist` dependency.

### Planned arena bytes — no new JNI method

`buildMethodMeta()` in `native/core/et_runtime.cpp` sums `meta->memory_planned_buffer_size(i)` over
`meta->num_memory_planned_buffers()` into a new `MethodMeta::plannedArenaBytes`. Both accessors are
verified present in the pinned v1.3.1 (`runtime/executor/method_meta.h:228,236`).

`MethodMeta` is built once at load and cached in `RuntimeState`, so this rides the existing
`methodMeta()` call at zero recurring cost. Java gains `EtMethodMeta.plannedArenaBytes`.

### Staging bytes — one new JNI method

`EtRuntime::stagingBytes()` returns the sum of `StagingSlot::capacity()` across the slot vector.
`EtNative.stagingBytes(long handle)` exposes it. O(numInputs), called only from `snapshot()`; the
forward path is untouched.

**`stagingBytes` is 0 for most real models, and that is correct.** Staging slots are allocated at
construction only for inputs where `inputMemoryPlanned[i] == 0`; planned inputs keep capacity 0 and
never stage. Memory-planned is the ExecuTorch export default and true for every `.pte` in this
repo. The gauge therefore reports the borrow-path footprint specifically. The javadoc must say so,
or it reads like a broken gauge.

### Two footguns the implementation must respect

- **`g_metaCtor` is a cached JNI method ID with a hardcoded signature string** in
  `native/jni/executorch_djl_jni.cpp`. Adding a constructor parameter to `EtMethodMeta` without
  updating that literal fails at class init. The Java field, the constructor, and the signature
  literal change as one atomic step, covered by a test that loads a model.
- **`stagingBytes(handle)` on a closed handle is a use-after-free.** The Java side guards it:
  `EtSymbolBlock` already tracks `handle == 0` behind a `volatile long`, and the registry removes
  entries on `close()`. The snapshot reads the handle once into a local and skips the entry if it is
  zero, closing the close-concurrent-with-snapshot race.

### Build cost

All three platforms — `linux-x86_64`, `linux-aarch64`, `windows-x86_64` — need a rebuild and
restage before the JVM tests pass.

### Upstream gap: XNNPACK delegate workspace

The XNNPACK delegate workspace cannot be sized from this layer. `xnn_workspace_t` is opaque in the
shipped `xnnpack.h` (`xnn_create_workspace` / `xnn_release_workspace` only, no size accessor), and
ExecuTorch's `XNNWorkspace` wrapper exposes none either. Under the default `global` sharing mode it
is not per-model in any case.

The design reports the two components it can measure exactly and documents the exclusion rather
than shipping a proxy number users would over-trust. An RSS-delta proxy was considered and rejected:
noisy under concurrent loads, meaningless under `global` sharing, and OS-specific.

**Deliverable:** file a GitHub issue against `measly-java-learning/executorch-runtime-dist`
requesting a workspace-size accessor on the `XNNWorkspace` wrapper, and reference it from the
javadoc documenting the gap.

## Error handling

A monitoring surface must never be the thing that breaks production.

- `snapshot()` never throws. A model whose native state cannot be read contributes a degraded entry
  rather than propagating the failure.
- Byte gauges use **`-1` for "unavailable"** and **`0` for "genuinely zero"**. The distinction is
  load-bearing for `stagingBytes` and must be stated in the javadoc.
- Unresolvable configuration fields report `unknown`, never null.
- JMX registration failure is one logged warning at first load.

## Testing

- **Catch2 (native):** `plannedArenaBytes > 0` for `add.pte`; `stagingBytes()` is 0 for an
  all-planned model and non-zero for an unplanned-input fixture; both survive a forward.
- **JVM unit:** snapshot contents across load → forward → close, including the closed-model rollup;
  counter accumulation; `forwardMaxNanos` tracks the maximum.
- **JVM integration:** the MBean registers and its attributes read back through the platform MBean
  server; `ai.djl.executorch.jmx_enabled=false` suppresses registration; repeated engine
  initialisation does not double-register.
- **Concurrency (tagged `stress`, excluded from CI):** `snapshot()` called repeatedly while N
  threads forward on their own models; asserts no exception and no torn values.
- **Overhead:** a JMH run in `example/` comparing steady-state MobileNetV2 before and after. The
  counters must not move the number. If they do, the design is wrong and we revisit rather than
  ship a hot-path regression.

## Out of scope

- **Error/failure taxonomy.** Explicitly deselected; exceptions stay as they are.
- **Micrometer, OpenTelemetry, and Prometheus bridges.** The snapshot is the integration point. A
  bridge is roughly thirty lines of user code; we document the shape rather than shipping and
  supporting three of them, and we avoid forcing a metrics-library dependency on every consumer.
- **XNNPACK workspace attribution.** Blocked upstream; issue filed instead.
- **Documentation cleanup.** Its own brainstorm → spec → plan cycle, starting once this lands, so
  the docs describe the final API rather than being written twice.

## Measured overhead

Hot-path verification (Task 9): `MobilenetBenchmark.steadyState` (AverageTime, 5 iterations × 10 s,
1 fork) before and after the counters landed, on the reference host — 11th Gen Intel Core i7-1185G7,
**8 cores** (`nproc`). The pre-change baseline in `scratchpad.txt` predates the benchmark's
variant/export-mode split, so its single `ExecuTorch` arm maps to today's `ET_HYBRID` (engine
`ExecuTorch`, PyTorch-backed preprocessing); both ExecuTorch arms are listed for completeness.

| Date | Source | Arm | Score (ms/op) |
|---|---|---|---|
| 2026-08-09 | baseline (scratchpad.txt, pre-change) | ExecuTorch steadyState | 19.049 ± 1.119 |
| 2026-08-09 | baseline (scratchpad.txt, pre-change) | ExecuTorch steadyState | 19.401 ± 1.164 |
| 2026-08-09 | baseline (scratchpad.txt, pre-change) | PyTorch steadyState | 32.558 ± 3.272 |
| 2026-08-09 | baseline (scratchpad.txt, pre-change) | PyTorch steadyState | 28.995 ± 1.574 |
| 2026-08-09 | post-change `:example:jmh` | ET_HYBRID (planned) steadyState | 18.716 ± 1.164 |
| 2026-08-09 | post-change `:example:jmh` | ET_HYBRID (unplanned) steadyState | 18.822 ± 0.920 |
| 2026-08-09 | post-change `:example:jmh` | ET_NATIVE (planned) steadyState | 19.299 ± 2.434 |
| 2026-08-09 | post-change `:example:jmh` | ET_NATIVE (unplanned) steadyState | 19.064 ± 3.072 |
| 2026-08-09 | post-change `:example:jmh` | PYTORCH (planned) steadyState | 31.370 ± 2.874 |
| 2026-08-09 | post-change `:example:jmh` | PYTORCH (unplanned) steadyState | 31.343 ± 1.500 |

Verdict: every post-change ExecuTorch steady-state center (18.7–19.3 ms/op) sits at or slightly
below the pre-change baseline (19.049 / 19.401 ms/op) and within its reported error bars
(~±1.1 ms/op). The counters do not move steady-state throughput; the hot-path overhead premise of
this design holds. A second post-change run corroborates: 18.525 ± 9.986 / 18.589 ± 6.644
(planned) and 18.816 ± 0.886 / 19.373 ± 0.487 (unplanned), same verdict.
