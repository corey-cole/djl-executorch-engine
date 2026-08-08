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

### 3.4 Starting constants, and how they get finalised

Starting point, chosen to land near 8 MFLOP per forward:

| constant     | start |
|--------------|-------|
| `BATCH`      | 32    |
| `HIDDEN`     | 256   |
| `DEPTH`      | 4     |
| `N_BUCKETS`  | 64    |

**This estimate carries roughly a ±3× error bar** — it assumes ~20 GFLOP/s of single-threaded f32
GEMM, which is a guess about the host, not a measurement. The export script therefore takes all
four as named constants and **prints the measured per-forward cost** at export time. Retuning is
mechanical: adjust, re-export, read the printed number. This document is updated with the final
tuned values once measured, and the measured cost is recorded in the script's header comment.

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
