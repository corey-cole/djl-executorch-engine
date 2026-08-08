# Per-model XNNPACK workspace sharing mode

Status: approved, not yet implemented
Supersedes: section 8 ("Deferred: workspace sharing mode") of
`2026-08-06-intraop-threadpool-config-design.md`

## 1. Why this reopens a deferred decision

The intra-op spec deferred `workspace_sharing_mode` on the grounds that it is a process-global
`set_option` knob whose only benefit appears in a low-intra-op configuration nobody had asked for.
Two things changed.

**The deployment shape is now known.** Several distinct models in one JVM, memory and CPU both
uncontended, differing call rates and SLOs per model, and enough application control to set each
model deliberately. That is precisely the shape the shipped `Global` default penalises: every
XNNPACK-delegated `forward()` in the process, across all models, serialises on one workspace mutex.

**It is not only a process-global knob.** ExecuTorch 1.3.1 resolves the mode *per delegate at
method-load time*, preferring a per-load runtime spec over the process global
(`XnnpackBackendOptions::resolve_sharing_mode`). `Module::load(const LoadBackendOptionsMap&,
Verification)` is the public entry point for supplying that spec, and both required headers ship in
our pinned tarball. So the mode can differ per `.pte`, which removes the write-once seal, the
load-order dependence, and the process-global ordering hazards that shaped the intra-op design.

The measured evidence from the earlier spec stands. With `ET_INTRAOP_THREADS=1` so intra-op
saturation does not mask the lock, achieved parallelism (CPU-seconds ÷ wall-seconds):

| caller threads | `Global` (shipped default) | `Disabled` |
|---|---|---|
| 1 | 1.12 | 1.12 |
| 2 | 1.12 | 2.23 |
| 4 | 1.12 | 4.35 |
| 8 | 1.17 | 7.13 |

These figures were obtained via `native/harness/et_scaling_harness.cpp`, which sets the mode through
the process-global `set_option("XnnpackBackend", ...)` backend path — not through the per-load
runtime spec this design ships (section 5). Both paths converge on the same `WorkspaceSharingMode` at
delegate init (`XnnpackBackendOptions::resolve_sharing_mode` does not distinguish how the option
arrived), so the numbers are expected to transfer, but the shipped per-load path has not itself been
measured.

## 2. What the modes mean

`WorkspaceSharingMode` (`backends/xnnpack/runtime/XNNPACKBackend.h`):

- `Disabled = 0` — every `CALL_DELEGATE` instance gets its own workspace. Maximum parallelism,
  maximum activation-arena memory.
- `PerModel = 1` — all delegate instances within one program share a workspace. Only one method of
  that model executes at a time.
- `Global = 2` — all delegate instances across all loaded methods share one workspace. Lowest
  memory; one delegate call at a time process-wide. The workspace does not shrink when a method is
  unloaded, so memory is reclaimed only when every XNNPACK-delegated method is gone.

Our pin's default is `Global`, confirmed from the runtime-dist build (`et-build-logging/CMakeCache.txt`:
`EXECUTORCH_XNNPACK_SHARED_WORKSPACE:BOOL=ON`), not assumed from the header.

**The modes compose.** `XNNWorkspaceManager::get_or_create_workspace(program_id, mode)` branches on
the mode passed for *that* delegate, so a model electing `Disabled` is fully isolated regardless of
what any other loaded model chose. A hot, tight-SLO model can run unserialised while background
models keep sharing a `Global` arena. No cross-model coordination is required.

## 3. Scope

Expose `workspace_sharing_mode` per model, via the per-load runtime spec.

Explicitly **not** exposed: `weight_cache_enabled`. See section 8.

Explicitly **not** used: the process-global `set_option` path. It affects only subsequently loaded
models, making behaviour depend on load order and racy when loads happen on multiple threads — the
same hazard class as the intra-op seal, for no capability the per-load path lacks. One mechanism
only.

## 4. Configuration surface

```java
Criteria.builder().optOption("workspaceSharingMode", "disabled")   // per model
```
```
-Dai.djl.executorch.workspace_sharing_mode=global                  // JVM-wide default
```

Accepted values: `disabled`, `per_model`, `global` — case-insensitive, trimmed. Bare integers are
**not** accepted in the public API: they are opaque at a call site, and accepting them would let an
out-of-range value through to a native failure.

Resolution happens per load, in `EtModel.load(Path, String, Map<String, ?>)` — where DJL already
delivers the options map and where the engine currently ignores it:

1. `workspaceSharingMode` present in the options map → use it.
2. Otherwise `ai.djl.executorch.workspace_sharing_mode` present → use it.
3. Otherwise **omit the spec entirely**.

Step 3 is deliberately not "pass `global`". Omitting follows whatever default the tarball was
compiled with; passing a value pins it, which would silently diverge if a future pin flips
`EXECUTORCH_XNNPACK_SHARED_WORKSPACE`. Both are read at load, not at class init, so the precedence is
testable.

Both paths feed the same per-load spec. The process-global backend option is never written.

## 5. Plumbing

`EtNative.loadModule(String ptePath)` gains an int parameter carrying the resolved mode, with `-1`
meaning "omit". There is one call site, so the existing signature changes rather than gaining an
overload — an overload would mean two JNI entry points to keep in sync. This is a breaking change to
`org.measly.executorch.jni`, which is public by visibility but internal by intent.

`EtRuntime` takes the mode as a constructor parameter defaulting to `-1`, and at the single
`module.load()` site in `native/core/et_runtime.cpp`:

```cpp
executorch::runtime::Error err;
if (mode >= 0) {
  BackendOptions<1> opts;
  opts.set_option("workspace_sharing_mode", mode);
  LoadBackendOptionsMap map;
  map.set_options("XnnpackBackend", opts.view());
  err = state_->module.load(map);
} else {
  err = state_->module.load();
}
// Force the "forward" Method to load too, unconditionally.
if (err == executorch::runtime::Error::Ok) {
  err = state_->module.load_forward();
}
```

`LoadBackendOptionsMap` does not own its option spans, but `Module::load` deep-copies into
Module-owned storage before returning, so the stack-local `BackendOptions` and map are correct here.

This turned out to need one more call than originally believed. The original draft of this section
assumed the lazy `load_method` triggered by the first `forward()` would consume the deep-copied spec,
so `module.load()` (or `module.load(map)`) alone would be enough. That is wrong: `Module::load()` and
`Module::method_meta()` are both **program-level** — `method_meta()` calls `load()` and then
`program_->method_meta()`, and never touches `load_method`. Delegate init, which is the only place
`XnnpackBackendOptions::resolve_sharing_mode` runs, happens inside `load_method`, which is otherwise
triggered lazily by the first `forward()`. Left unforced, the runtime spec built above would sit
unused until first inference — and worse, `EtRuntime`'s "load throws" contract, which is meant to
surface a bad `.pte` or a bad option at construction, would not: an invalid mode would only surface at
`predict()` time.

So the shipped constructor in `native/core/et_runtime.cpp` calls `state_->module.load_forward()`
**unconditionally** — not only when a mode is specified — right after the `module.load()`/
`module.load(map)` call above, and throws if it does not return `Error::Ok`. This is required for
every model this engine loads, whether or not `workspaceSharingMode` is set, because it is also what
makes delegate init (and therefore any other XNNPACK-backend errors) surface at load rather than at
first `forward()`.

**Accepted consequence.** The XNNPACK subgraph compile that used to happen lazily on the first
`forward()` now happens at construction, for every model. In `native/harness/et_timing_harness.cpp`
this moves that cost from `cold_ms` into `load_ms`; the harness discards a warmup call before timing
steady state, so steady-state throughput numbers are unaffected. Model loading is correspondingly
slower and the first inference is correspondingly faster than before this change.

Required headers, both present in the pinned tarball: `runtime/backend/backend_options_map.h` and
`runtime/backend/options.h`. `backends/xnnpack/runtime/XNNPACKBackend.h` is **not** installed, so the
key string and the enum values are hardcoded in `et_runtime.cpp` with a comment pointing at the
upstream header.

### 5.1 The silent-failure trap

The backend id is `"XnnpackBackend"` (`xnnpack_backend_key`) — not `"XNNPACKBackend"`, and not the
`"CoreMLBackend"` pattern the header's own doc comment shows. `Method::load` looks the options up
with `backend_options->get_options(delegate.id())`; a mismatched id, or a mistyped option key,
returns an empty span and the option is **silently ignored**. The load succeeds, the mode stays at
the default, and a hot model quietly misses its SLO.

There is no read-back that would catch this. `get_option` returns the process-global value, not the
per-model resolved one, and `XnnpackBackend::init` does not log the mode it resolved. Detection is
therefore by negative control — see section 6.

## 6. Error handling

| Condition | Behaviour |
|---|---|
| Unrecognised value in the per-model option | `IllegalArgumentException`, load fails |
| Unrecognised value in the JVM property | WARN, ignored, fall through to step 3 |
| Out-of-range int reaching the runtime | `resolve_sharing_mode` returns `InvalidArgument`; `init` propagates it; load fails |

The third row is unreachable through the Java API, which only ever emits `-1`, `0`, `1`, or `2`. It
is the contract at the `EtRuntime` boundary, and section 7 depends on it.

The asymmetry is deliberate. A per-model option is explicit intent about a specific model, and
falling back silently produces exactly the invisible latency regression the knob exists to prevent.
The JVM property is an ambient default, and a typo in a process-wide flag must not fail startup —
matching the `ai.djl.executorch.num_threads` precedent.

## 7. Testing

**No new Gradle task and no forked JVM.** This is the concrete payoff of per-model over
process-global: nothing leaks between tests, so everything runs under `./gradlew test`. Compare the
intra-op work, which needed a dedicated `intraOpTest` task precisely because its state was
process-wide.

- **Native (Catch2, `et_runtime_test.cpp`)** — construct against `add.pte` at modes 0, 1, and 2;
  each succeeds. Then mode `99` throws. That last case is the wiring proof: an out-of-range mode can
  only fail if the spec actually reached the XNNPACK backend under our exact backend-id and
  option-key spellings. Both strings live in `et_runtime.cpp`, so this is the correct layer. It is
  deterministic and involves no timing. Construction alone is sufficient to reach the backend because
  the `EtRuntime` constructor calls `module.load_forward()` unconditionally (section 5) — delegate
  init, where `resolve_sharing_mode` runs, happens inside `load_method`, so there is no need to drive
  an actual `forward()` call to exercise this path.
- **Java unit (no native)** — the string→int resolution table, option-beats-property precedence, and
  the unrecognised-value contract from section 6. Pure, safe in the shared JVM.
- **Java integration** — `optOption("workspaceSharingMode", "disabled")` loads successfully; an
  unrecognised value throws `IllegalArgumentException`.
- **Native scaling harness (`native/scaling.sh`)** — where the actual parallelism numbers get
  measured. Not a pass/fail test; run on demand.

## 8. Why `weight_cache_enabled` is not exposed

`XnnpackBackend::execute()` takes a **second** process-global mutex, `weights_cache_mutex_`, held
for the entire delegate execution, whenever `executor->uses_weight_cache()`. Enabling the weight
cache therefore reintroduces exactly the serialisation that `Disabled` removes: you would trade one
global lock for another and see none of the parallelism in the section 1 table. Models that do not
use the cache are unaffected, but every model that does serialises against every other one.

The cache is off in our pin — runtime-dist builds with
`EXECUTORCH_XNNPACK_ENABLE_WEIGHT_CACHE:BOOL=OFF` — which is why the measured `Disabled` column
reaches 7.13× at 8 caller threads. That is load-bearing luck, not design, and it is worth stating
plainly so a future pin bump that flips the flag is recognised as a performance regression rather
than a mystery.

**If you want it anyway**, no rebuild is required. `ENABLE_XNNPACK_WEIGHTS_CACHE` appears in exactly
two places upstream — the `add_definitions` in `backends/xnnpack/CMakeLists.txt` and the `#ifdef` in
`XnnpackBackendOptions.h` that selects the default. It guards no code, and `XNNWeightsCache` is
compiled into `libxnnpack_backend.a` in the shipped tarball. So:

- Set `weight_cache_enabled` (a **bool**, key `weight_cache_option_key`) in the same
  `LoadBackendOptionsMap` built in section 5, alongside or instead of the sharing mode.
- Accept the global `weights_cache_mutex_` across `execute()` for every model that enables it, and
  keep those models off the hot path.
- Rebuilding runtime-dist with `EXECUTORCH_XNNPACK_ENABLE_WEIGHT_CACHE=ON` only flips the *default*;
  it buys nothing the per-load spec cannot already do.

The trade is real — the cache shares packed weights across delegate instances, which matters when
memory is tight. Our target deployment has memory headroom and latency SLOs, so it is the wrong
trade here, not universally.

## 9. Documentation changes

- `CLAUDE.md` — a conventions bullet for `workspaceSharingMode` and
  `ai.djl.executorch.workspace_sharing_mode`, mirroring the `num_threads` bullet.
- **Correction to the existing threading note** in `EtSymbolBlock` and `CLAUDE.md`. "More threads is
  usually wrong", and the 1/4/8-caller-thread figures behind it (462 / 305 / 147 forwards/s, peak RSS
  33 MB → 224 MB), were measured against the `Global` serialiser that this knob removes. As written
  the claim is unconditional and will actively mislead anyone tuning a hot model. Scope it to the
  default sharing mode and cross-reference this knob.
- A short note that `weight_cache_enabled` is withheld by design, pointing at section 8.
