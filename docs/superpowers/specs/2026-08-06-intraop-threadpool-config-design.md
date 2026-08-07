# Intra-op threadpool configuration — design

**Date:** 2026-08-06
**Status:** approved, not yet implemented
**Scope:** expose ExecuTorch's intra-op (XNNPACK) threadpool size through the DJL engine.

## 1. Why

XNNPACK in this engine runs on ExecuTorch's shared pthreadpool — `libxnnpack_backend.a` carries an
undefined reference to `executorch::extension::threadpool::get_pthreadpool`. The pool sizes itself
from `cpuinfo_utils`' performance-core count and there is **no** way to influence it from outside
the process (see §7). That is fine on a laptop and wrong on the hosts this engine is headed for:
production use is smaller, less compute-constrained models on 18–40 core / 36–80 thread machines,
where a pool that sizes itself to the whole machine is not what you want per model.

The measurements that motivated this (2026-08-06, `native/scaling.sh`, MobileNetV2, on a
**4-physical-core / 8-thread** host — SMT siblings 0/4, 1/5, 2/6, 3/7, so the default pool of 8 was
already oversubscribed):

| caller threads | forwards/sec | peak RSS |
|---|---|---|
| 1 | 462 | 33 MB |
| 2 | 457 | 57 MB |
| 4 | 305 | 115 MB |
| 8 | 147 | 224 MB |

One caller thread wins. Adding caller threads costs throughput, latency, and memory. Two mechanisms
combine to produce that: concurrent XNNPACK delegate calls serialize on a process-global workspace
mutex (`XNNWorkspace::acquire()`, mode `Global` in our build), and a single forward already
saturates the machine via intra-op parallelism, so the serialization costs little *throughput* while
buying nothing. Achieved parallelism was 7.1–8.1x in every cell regardless of caller-thread count.

**Caveat, load-bearing:** 4 physical cores is not the 18–40 core production target, and an 8C/16T
host was anecdotally slow in earlier experiments but is currently offline. Ratios above should not
be extrapolated. The knob is justified by the mechanism, not by these specific numbers.

## 2. Decisions

- **Surface:** system property + static setter, mirroring `ai.djl.pytorch.num_threads` /
  `num_interop_threads`. Not a `Criteria` option — the pool is process-global, and a per-model
  option that silently applies process-wide would misrepresent the mechanism.
- **Scope:** intra-op pool size only. `workspace_sharing_mode` is **deferred** (§8).
- **Lifecycle:** write window closes at the first model load.

## 3. Architecture

One process-global setting with a single write window.

```
-Dai.djl.executorch.num_threads=N ─┐
EtEngine.setIntraOpThreads(n) ─────┴─> EtEngine (gate: throws once sealed)
                                          │  sealed by EtModel.load()
                                          v
                              EtNative.setIntraOpThreads(int)   [JNI]
                                          │
                                          v
                       measly::et::setIntraOpThreads(uint32_t)  [core]
                                          │
                                          v
        extension::threadpool::get_threadpool()->_unsafe_reset_threadpool(n)
```

The flush point is `EtModel.load()`: it asks `EtEngine` to seal the setting, which applies the
pending value and marks the pool fixed. Applying lazily at first load rather than eagerly at
class-init keeps the setter from racing `EtNative`'s static initializer and preserves today's lazy
native-library load.

Upstream names the call `_unsafe_reset_threadpool` because it must not race in-flight work, and
delegate init during load already submits to the pool — so "before first load" is the only provably
safe window.

## 4. Java surface

Three additions to `EtEngine`; nothing on `Criteria`.

```java
public static final String NUM_THREADS_PROPERTY = "ai.djl.executorch.num_threads";

/**
 * @throws IllegalArgumentException if n < 1
 * @throws IllegalStateException once any model has been loaded
 */
public static void setIntraOpThreads(int n);

/** Effective pool size as reported by the native pool. Triggers the native load. */
public static int getIntraOpThreads();
```

`getIntraOpThreads()` returns the pool's actual `get_thread_count()`, not the requested value. On a
40-core host the difference between "I asked for 20" and "I got 20" is the entire point, and the
default is derived from performance-core count rather than `nproc`.

Gate state is guarded so concurrent `loadModel()` calls cannot both seal.

## 5. Native surface

In `et_runtime.h` (the JNIEnv-free core, so the shim, Catch2 units, and harnesses all share one
implementation):

```cpp
namespace measly::et {
  // Returns the pool size in effect AFTER the attempt. Upstream's reset cannot fail and
  // silently ignores 0, so the caller must compare rather than trust a status.
  uint32_t setIntraOpThreads(uint32_t n);
  uint32_t intraOpThreads();
}
```

`ThreadPool::_unsafe_reset_threadpool` **always returns `true`** — verified in the v1.3.1 source. It
early-returns for `n == 0` and for `n == get_thread_count()`. A `bool` return would therefore be
meaningless; the core reads the count back instead.

`extension_threadpool` becomes an explicit `target_link_libraries` entry on the `et_runtime` core
rather than arriving transitively through `xnnpack_backend`. `native/harness/et_scaling_harness.cpp`
switches to these functions and drops its direct ExecuTorch threadpool include, so the harness
measures the same code path the engine ships.

## 6. Error handling

| Condition | Behavior |
|---|---|
| `setIntraOpThreads(n)` with `n < 1` | `IllegalArgumentException` in Java, before any JNI call |
| `setIntraOpThreads(n)` after first load | `IllegalStateException`, naming the sealed value |
| Property present but unparseable / `< 1` | WARN, fall back to the runtime default; do not fail startup |
| Applied value ≠ requested value | WARN with both numbers |
| Both property and setter used | Setter wins, logged at WARN |

A typo'd JVM flag should not take down a production process when the fallback is a working default.
To keep the outcome observable, `EtModel.load()` logs the sealed pool size at INFO on first load.

## 7. No environment variable to account for

Verified against the v1.3.1 checkout and the shipped runtime tarball: there is no `getenv` /
`GetEnvironmentVariable` in `extension/threadpool/`, `backends/xnnpack/runtime/`, or `runtime/`, and
none in the vendored `pthreadpool`, XNNPACK init, or `cpuinfo`. The shipped `libpthreadpool.a` has
no OpenMP symbols, so `OMP_NUM_THREADS` is inert. The only `getenv` users in the runtime tree are
absl's flags and time-zone code.

Consequence: our property is the only control surface. No precedence rule is needed, and nothing can
silently override the setting.

**Named hazard:** ExecuTorch's pool and LibTorch's are independent. `ai.djl.pytorch.num_threads`
does not touch ours and ours does not touch theirs. A hybrid JVM — such as `example/`, which
benchmarks both engines — has two pools that will each size themselves to the machine and
oversubscribe it.

## 8. Deferred: workspace sharing mode

`workspace_sharing_mode` is a runtime backend option in ET 1.3.1
(`set_option("XnnpackBackend", …)`), settable with no rebuild and no pin bump. It is **not** exposed
here.

Evidence, measured with `ET_INTRAOP_THREADS=1` so intra-op saturation does not mask it — achieved
parallelism (CPU-seconds ÷ wall-seconds):

| caller threads | `Global` (shipped) | `Disabled` |
|---|---|---|
| 1 | 1.12 | 1.12 |
| 2 | 1.12 | 2.23 |
| 4 | 1.12 | 4.35 |
| 8 | 1.17 | 7.13 |

The lock is a genuine serializer, but it only becomes a *ceiling* once intra-op is turned down — a
configuration nobody has asked for. Revisit if a low-intra-op deployment appears. Note the option
keys must be hardcoded: the runtime tarball installs `runtime/backend/` but not
`backends/xnnpack/runtime/`, so `XNNPACKBackend.h` and its key constants are unavailable to
consumers.

## 9. Testing

The pool is process-global and Gradle shares one test JVM, so a test that changes it contaminates
every test after it. (Same constraint hit previously with the DJL PyTorch engine under JMH: the
global thread setting required a discrete process per arm.) Three tiers, split so nothing does:

- **Native (Catch2, `et_runtime_test.cpp`)** — separate process, free to mutate. Asserts
  `setIntraOpThreads(1)` then `intraOpThreads() == 1`, and pins the upstream quirks as executable
  documentation: `setIntraOpThreads(0)` leaves the count unchanged, and a reset to the current value
  is a no-op. If upstream ever makes these fail instead of silently succeeding, this test says so.
- **Java unit (no native)** — `n < 1` throws; setter after seal throws; setter beats property. Pure
  state machine, safe in the shared JVM.
- **Java integration, dedicated Gradle task** — `intraOpTest`, mirroring the existing `leakTest`
  pattern: forks a JVM with `-Dai.djl.executorch.num_threads=2`, loads a model, asserts
  `getIntraOpThreads() == 2`. The only test proving the property reaches the native pool end to end;
  excluded from `./gradlew test` because it cannot share a JVM. `./gradlew build` includes it.

Any future JMH arm that varies pool size must fork per arm for the same reason.

## 10. Docs changes

- **`CLAUDE.md`** — add the threading note below, adjacent to the existing safety rule; document the
  new property in the conventions section.
- **`EtSymbolBlock.java`** — the same note in javadoc, where an IDE user lands.
- **`EtEngine`** — javadoc on the property and both methods, including the performance-core-count
  default and §7's "no environment variable".

The existing sentence — *"`EtSymbolBlock.forward()` is not thread-safe on the same model — one
`Model`/`Predictor` per thread, and never `close()` a model with a forward in flight"* — is correct
and **stays as written**. It is a safety rule, and nothing measured here contradicts it. The
addition exists because it reads like a scaling recipe:

> **Threading, and why more threads is usually wrong.** The rule above is about *safety*, not
> throughput. XNNPACK-delegated models already parallelize inside a single `forward()` on
> ExecuTorch's shared intra-op pool, and concurrent delegate calls serialize on a process-global
> workspace mutex — so N `Predictor`s on N threads is typically slower than one, not N× faster.
> Tune `ai.djl.executorch.num_threads` before adding caller threads. Measured on a 4-core/8-thread
> host with MobileNetV2: 1 thread 462 forwards/s, 4 threads 305, 8 threads 147 (peak RSS 33 MB →
> 224 MB). Ratios on larger hosts are unmeasured.

## 11. Out of scope

`workspace_sharing_mode`, the XNNPACK weights cache, any change to the forward path, and any attempt
to make `forward()` thread-safe.
