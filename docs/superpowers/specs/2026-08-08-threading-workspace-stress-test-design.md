# Threading / workspace-parameter stress test — design

Date: 2026-08-08
Status: approved, not yet implemented

## 1. Goal

Two outcomes from one artifact:

1. **A safety gate.** Concurrent use of this engine — many threads, each with its own model —
   either produces the right numbers or fails loudly. Covers wrong answers, crashes, native
   leaks, and teardown races.
2. **A measurement rig.** Throughput and latency across (caller threads × workspace sharing mode)
   on a workload that is not `a + b`, so the performance guidance in `CLAUDE.md` rests on a model
   with real delegate work in it.

A third, deliberate outcome: the JVM arm is the **published, validated Java pattern** for running
this engine from multiple threads — how to hold a `Predictor` per thread and close it safely.

Everything here is **off by default**. Free CI providers take a dim view of sustained full-core
load; `./gradlew test` must never pay for any of it, and the correctness arm is local-only.

## 2. Sizing target

One `forward()` costs **300–500 µs at one intra-op thread**, matching where the author's
production model sits.

The floor matters as much as the ceiling. Too cheap and loop overhead dominates the lock hold
time; too expensive and the box is saturated by a single thread, at which point caller-thread
scaling is flat for reasons that have nothing to do with any mutex — the saturation masks exactly
what the harness is named for. This is the lesson already recorded in
`native/harness/et_scaling_harness.cpp`, and it is why the sweep runs at `ET_INTRAOP_THREADS=1`.

## 3. The model

### 3.1 What the premise got wrong, and why the design still works

The original sketch was: *n* weight buckets, a bucket chosen per input value, several inputs whose
bucket lookups proceed in parallel before a serial combine. The intent — serial and parallel work
inside one `forward()` — is right. The mechanism is not.

**ExecuTorch has no inter-op parallelism.** The runtime walks the graph as a single instruction
stream, one op after another. Two independent branches in the graph execute sequentially, not
concurrently. Graph topology therefore cannot produce the parallel arm.

Kernel choice can, and does:

- `index_select` is **not lowered by the XNNPACK partitioner**. It runs on a portable,
  single-threaded CPU kernel, takes no workspace, and holds no lock. This is the serial arm.
- `Linear` / `addmm` **is** lowered to XNNPACK. It runs on ExecuTorch's shared pthreadpool, takes
  a workspace, and contends on the workspace mutex under `global` sharing. This is the parallel
  arm, and it is the only part where the sharing mode is observable at all.

So the mix is real, it just arrives by kernel selection rather than by graph shape. Multiple inputs
stay in the design — not for parallelism, but for **data dependence**: each input steers its own
bucket, so a concurrency bug that corrupts an intermediate surfaces as a *wrong number* rather
than only as a crash.

### 3.2 Structure

Two input tensors. For each:

```
idx     = clamp((x[0, 0] * n_buckets).to(int64), 0, n_buckets - 1)   # data-dependent
gathered = index_select(bucket_table, 0, idx)                        # portable kernel, serial
h        = Linear(...) x depth                                       # XNNPACK, parallel
```

The two branches then `add` into a single output.

`bucket_table` is `[64, 256]` f32 — about 64 KB. The gather stays deliberately small, per the
decision that the MLP should dominate: the gather exists for data dependence and for the serial
arm, not to be the cost centre.

### 3.3 Why there is a batch dimension

At batch 1 the `Linear` stack is a GEMV: memory-bandwidth-bound, not compute-bound. DRAM becomes
the bottleneck and masks the workspace lock entirely, which would make the whole sweep measure the
wrong thing. With a batch of ~32 it is a genuine GEMM, compute-bound, and the ~1 MB of weights stay
resident in L2 — so contention that shows up is lock contention, not bandwidth starvation.

### 3.4 Final constants, measured

Measured on the tuning host — **11th Gen Intel Core i7-1185G7 @ 3.00 GHz (4P/8T, Linux)** — with
the authoritative native figure, `ET_INTRAOP_THREADS=1 ./native/bench/et_scaling_harness <pte> 1 2000 200`
(`per_thread_mean_ms` at one caller thread *is* the per-forward cost at one intra-op thread).
Consecutive bursts drift as the laptop's turbo budget drains, so each run was taken after ≥4 min
of idle, with the budget fresh.

| constant     | final |
|--------------|-------|
| `BATCH`      | 32    |
| `HIDDEN`     | 256   |
| `DEPTH`      | 5     |
| `N_BUCKETS`  | 64    |

Measured cost at one intra-op thread, three runs: **351 / 354 / 358 µs** — inside the 300–500 µs
target of §2 with headroom on both sides. For reference, the default-pool (8-thread) figure from
`et_timing_harness` is 140 µs, and the export script's Python-runtime figure is ~285 µs at default
threads (a different build with different overheads — not the tuning number).

Tuning history: `DEPTH=4` measured **284 µs** — just under the band floor — so `DEPTH` was raised to
5. Cost scales linearly with `DEPTH` (~1.25× for +25% FLOPs) because the ~1.25 MB of weights per
branch stay resident in the per-core L2; the batch stays at 32 to keep the `Linear` stack a
compute-bound GEMM (§3.3). The final constants are the ones in
`tools/scripts/export_stress_model.py`, whose header records the measured cost, and the `.pte` +
goldens committed together carry them.

## 4. Correctness oracle

Two layers, because they catch different failures.

### 4.1 Golden vectors — "is this the right model, wired right?"

The export script runs the **exported `.pte` through ExecuTorch's Python runtime** and records 8
input/output pairs. Not torch eager: eager uses different kernels than XNNPACK and would not match,
so eager goldens would either fail spuriously or force a tolerance so loose it proves nothing.

The 8 cases span the bucket boundaries — first bucket, last bucket, and values landing either side
of two interior boundaries — so a mis-marshalled or off-by-one bucket index is caught rather than
averaged away.

Compared at `rtol=1e-4, atol=1e-5`. The tolerance is not tight because the Python runtime is a
different build of the same runtime; this layer is not the sharp instrument.

### 4.2 Bitwise self-reference — "did concurrency corrupt anything?"

The first `forward()` in the JVM becomes the reference. Every subsequent forward on every thread,
for the same input, must match it **bit for bit**.

This is legitimately exact rather than optimistic: XNNPACK parallelises over output tiles, not over
the K reduction, so neither the thread count nor the sharing mode changes the order of accumulation.
Identical inputs must give identical bits.

It is the sharp instrument. A clobbered shared workspace shows as a bit difference long before the
error grows large enough to breach a float tolerance — and critically, it catches the case where
*every* thread is wrong the same way, which is precisely the shape of a shared-workspace bug and
which cross-thread agreement alone would miss.

### 4.3 Split across the two arms

The JVM arm runs both layers. Golden parsing uses `ai.djl.util.JsonUtils` (gson, already on the DJL
classpath) — **no new dependency**.

The native arm runs **layer 2 only**. Adding a JSON parser to a JNI-free C++ harness is not worth
the dependency; ASan plus bitwise stability is what that arm exists for, and the JVM arm already
owns the golden comparison.

## 5. The Java pattern

### 5.1 The sharp edge this exists to document

`EtSymbolBlock.forward()` is not thread-safe **on the same model**. A `ThreadLocal<Predictor>`
handing out predictors derived from one shared `ZooModel` — the shape most DJL users reach for
first — is therefore **wrong on this engine**. Each thread needs its own `ZooModel`, and thus its
own native handle.

This also lines up with `workspaceSharingMode` being per-model: each thread's model carries the
mode, which is what makes the sweep's mode axis expressible at all.

### 5.2 Shape

```java
final class PerThreadContext implements AutoCloseable {
    private final ZooModel<...> model;       // own native handle — NOT shared
    private final Predictor<...> predictor;

    static PerThreadContext open(String sharingMode) { ... }

    @Override public void close() {          // reverse acquisition order
        predictor.close();
        model.close();
    }
}
```

Nothing needed adding to the interfaces to make this work. `ai.djl.Model` already extends
`AutoCloseable`, `ZooModel implements Model`, `Predictor` implements `AutoCloseable` directly, and
on our side `EtSymbolBlock` already declares it explicitly (`EtSymbolBlock.java:40`) while
`EtModel` / `EtNDManager` / `EtNDArray` inherit it via `BaseModel` / `BaseNDManager` /
`NDArrayAdapter`.

Each worker is a `Thread` whose `run()` body is:

```java
try (PerThreadContext ctx = PerThreadContext.open(mode)) {
    // iterate
}
```

The thread's lifetime *is* the resource's lifetime, so `AutoCloseable` does all the work and there
is no teardown protocol to get wrong. No `ThreadLocal` appears in the primary pattern — when the
owning thread is dedicated, a plain local is strictly better, and pretending otherwise would
publish ceremony as if it were safety.

The inner loop wraps the input `NDList` and the `predict()` output in try-with-resources, per this
project's test-resource hygiene.

### 5.3 Leak backstop

An `AtomicInteger` pair counts contexts opened and closed; the coordinator asserts they are equal
after every worker has joined. A context that escapes its `finally` fails the test loudly instead
of leaking a native handle silently.

### 5.4 What is documented but not built

A class comment states why the shared-`ZooModel` `ThreadLocal` is wrong here, and records the
pooled-executor variant — `ExecutorService` + `ThreadLocal` cache + an explicit drain phase that
submits exactly one close-task per pool thread behind a barrier before `shutdown()` — as a
**welcome contribution**, not as scope. It is the version real applications need and get wrong, but
pool-thread affinity makes the thread-count axis mushy, so it must not be the thing carrying the
stress matrix.

Also worth a line in that comment: `ThreadLocal.remove()` drops the reference without calling
`close()`, so it leaks the native handle until GC. It is the most commonly cargo-culted teardown
and it is wrong here.

## 6. The two arms

### 6.1 `stressGate` — correctness, local only

- 8 threads, `workspaceSharingMode=global`, intra-op threads left at default. **Maximum
  contention**, not a sweep: the goal is to make a race likely, not to be fast.
- Fixed duration, tuned to 30–60 s.
- Asserts golden vectors, bitwise self-reference, and `opened == closed`.
- Worker exceptions are captured into a concurrent collection and rethrown by the coordinator, so a
  dying thread surfaces as a failure rather than as a hung run.
- **Never wired to CI**, including nightlies. It is a local and self-hosted tool.

### 6.2 `stressSweep` — measurement

Focused matrix, 8 cells plus 1:

| axis          | values                  |
|---------------|-------------------------|
| caller threads| 1, 2, 4, 8              |
| sharing mode  | `global`, `disabled`    |
| intra-op      | 1 (fixed)               |

Plus one confirmation cell at **intra-op = default, 1 caller thread** — the real-world
configuration, so the sweep cannot be read as if intra-op = 1 were the shipping default.

`per_model` is excluded. It differs from `global` only when multiple *distinct* models are loaded;
a single-model stress loop makes those cells degenerate by construction, so including them would
add runtime and produce duplicate rows.

Reports throughput (forwards/s), per-thread mean latency, achieved parallelism, and peak RSS — to
stdout as a table and to `build/reports/stress/`.

### 6.3 Native arm

`native/harness/et_stress_harness.cpp`: N threads, each with its **own** `EtRuntime` over the same
`.pte`, forwarding concurrently under ASan/LSan. Asserts bitwise stability (§4.2) and exits non-zero
on divergence; ASan/LSan cover use-after-free on teardown races, overflow from a clobbered
workspace, and per-thread leaks.

Built into the existing `native/asan` tree under `-DET_BUILD_QA=ON`. **Run only when `ET_STRESS=1`**,
so the default `build_qa.sh` stays as cheap as it is today. Reuses the env knobs already established
by `et_scaling_harness.cpp` (`ET_SHARING_MODE`, `ET_INTRAOP_THREADS`).

TSan is **out of scope**, on purpose. The ExecuTorch runtime arrives as prebuilt static libraries
from the pinned tarball and is never compiled here, so TSan would not instrument XNNPACK's workspace
or the runtime — exactly where the interesting concurrency lives. It would still see our own code
and, via `pthread_mutex_*` interposition, the synchronisation around it; the failure mode is missed
races rather than false alarms, so what it reports would be trustworthy. But a clean TSan run would
be actively misleading about the delegate, and `EtRuntime` holds too little shared mutable state
today to justify a third build tree. Revisit if that changes.

## 7. Wiring and artifacts

- `tasks.test` gains `excludeTags("stress")` alongside the existing `leak` exclusion.
- Two registered `Test` tasks: `stressGate` and `stressSweep`, both selecting `@Tag("stress")`.
  Two verbs, two costs.
- `native/build_qa.sh` grows an `ET_STRESS=1` branch.
- `stress_mlp.pte` (~1 MB) and `stress_golden.json` are **committed together** under
  `src/test/resources/models/stress/`. Small enough to commit, and co-committing is the mechanism
  that keeps them consistent — a regenerated model with stale goldens is the failure this avoids.
  The export script writes both in one run for the same reason.
- Both arms call `assumeNativeAvailable()`, so a checkout without a staged `.so` skips rather
  than fails.

## 8. Testing the test

Pure-JUnit tests, **untagged** so they run in the normal `./gradlew test` and touch no native code:

- Golden-file parsing: well-formed file, missing file, malformed JSON, case count mismatch.
- Sweep-config logic: the matrix expands to the expected 9 cells; mode strings map correctly.

The stress harness is itself code, and the parts of it that can be regression-proofed cheaply
should be.

## 9. Out of scope

- TSan (§6.3).
- The pooled-executor `ThreadLocal` pattern (§5.4).
- `per_model` in the sweep (§6.2).
- `weight_cache_enabled`, which remains deliberately unexposed per `CLAUDE.md`.
- Any CI wiring for either arm.

## 10. Measured results

Host: **11th Gen Intel Core i7-1185G7 @ 3.00 GHz, 4P/8T, Linux** (ThinkPad T14 Gen 2i). All
measurements below were taken on this box; absolute numbers are state-dependent (this laptop's
turbo budget drains under sustained AVX2 load, roughly halving single-thread clocks after ~1 min
of continuous work), so the *ratios* are the evidence, not the wall figures.

`./gradlew stressSweep` (9 cells, 10 s each, two forked JVMs because the intra-op pool is
process-global and write-once), report `build/reports/stress/sweep.tsv`. Rows carry a `run_id`
column (one value across both arms of a single invocation; see `sweep.tsv`), and `peak_rss_kb` is
each cell's **own** peak (the process VmHWM mark is reset before each timed region), so cells may
now differ or decline — the pre-fix column was a cumulative high-water mark and forced adjacent
cells identical.

| run_id | threads | mode | intraop | forwards | wall_s | fwd_per_s | mean_ms | parallelism | peak_rss_kb |
|--------|---------|------|---------|----------|--------|-----------|---------|-------------|-------------|
| 2026-08-09T00:33:08.014052585Z | 1 | global | 1 | 13928 | 10.006 | 1392.0 | 0.7184 | 1.083 | 592220 |
| 2026-08-09T00:33:08.014052585Z | 1 | disabled | 1 | 13648 | 10.005 | 1364.1 | 0.7331 | 1.025 | 593092 |
| 2026-08-09T00:33:08.014052585Z | 2 | global | 1 | 13784 | 10.009 | 1377.1 | 1.4523 | 1.159 | 656616 |
| 2026-08-09T00:33:08.014052585Z | 2 | disabled | 1 | 20976 | 10.003 | 2097.0 | 0.9537 | 2.009 | 657916 |
| 2026-08-09T00:33:08.014052585Z | 4 | global | 1 | 13048 | 10.010 | 1303.5 | 3.0686 | 1.159 | 723268 |
| 2026-08-09T00:33:08.014052585Z | 4 | disabled | 1 | 26760 | 10.008 | 2673.8 | 1.4960 | 4.000 | 721844 |
| 2026-08-09T00:33:08.014052585Z | 8 | global | 1 | 12648 | 10.031 | 1260.9 | 6.3448 | 1.165 | 795724 |
| 2026-08-09T00:33:08.014052585Z | 8 | disabled | 1 | 28648 | 10.013 | 2861.0 | 2.7963 | 6.778 | 793356 |
| 2026-08-09T00:33:08.014052585Z | 1 | global | default | 19536 | 10.003 | 1953.1 | 0.5120 | 5.820 | 587572 |

Shape, as predicted:

- **`global` is flat near 1** — achieved parallelism 1.121/1.146/1.149/1.154 at 1/2/4/8 caller
  threads; total throughput ~1.2–1.3 k forwards/s regardless of thread count. The process-global
  workspace mutex serializes the delegate calls, exactly the §6.2 expectation.
- **`disabled` climbs with caller threads** — 1.029/2.012/3.983/6.666. Sub-linear at 8 threads on
  a 4P/8T host, as expected.
- This corroborates the MobileNetV2 finding recorded in `CLAUDE.md` (global
  1.12/1.12/1.12/1.17 vs disabled 1.12/2.23/4.35/7.13): same shape, same mechanism, a different
  model — so the result is not a MobileNetV2 artifact.
- The intra-op=default confirmation cell is ~1.4× faster per forward than the intra-op=1 cells
  (the baseline cell's parallel GEMM), which is why the sweep's contention cells must pin
  `ai.djl.executorch.num_threads=1`.
