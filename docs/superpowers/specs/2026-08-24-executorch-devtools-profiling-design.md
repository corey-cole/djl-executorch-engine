# ExecuTorch devtools profiling (opt-in ETDump)

**Date:** 2026-08-24
**Status:** approved design
**Follows:** the v1.4.1-2 pin bump and the OpenVINO Windows bundle, in that order
**Phase 0 status:** landed in `executorch-runtime-dist` `v1.4.1-3` (2026-08-25), all criteria verified

Makes ExecuTorch's event tracer a per-model, load-time opt-in of this engine. A profiled model
accumulates an ETDump across its forwards; the caller pulls the bytes and analyzes them offline with
ExecuTorch's Python `Inspector`. The shipped Linux `.so` links a devtools runtime so no second
artifact and no second build-matrix row exist.

## 1. The shipping decision, and what it rests on

`docs/benchmarking.md` posed one gating question: what does a devtools-enabled-but-not-tracing
runtime cost the default path, in binary size and steady-state latency, against the `logging`
runtime we ship today? Negligible meant one artifact with profiling as a runtime opt-in; material
meant a separate profile-capable `.so`.

Measured on `linux-x86_64`, MobileNetV2, no tracer attached, seven interleaved reps per variant at
`intraop=1` on an idle Ryzen 7 5800XT:

| variant | per-forward mean (ms) | sd | min | max |
|---|---|---|---|---|
| bare | 5.7160 | 0.0128 | 5.7014 | 5.7405 |
| logging | 5.7168 | 0.0118 | 5.6974 | 5.7295 |
| devtools | 5.7206 | 0.0192 | 5.6823 | 5.7459 |

devtools − logging is +0.0038 ms (+0.066%) against a standard error of 0.0085 ms — t ≈ 0.45. The
95% confidence interval on the difference is about ±0.35%, so this is an upper bound on the cost,
not merely a failure to detect one. At the shipped thread setting (`intraop=16`) devtools measured
1.0162 ms against logging's 1.0244 ms — tied, with the logging arm pulled up by one outlier.

Size, measured on the actual shipped shim: `libexecutorch_djl.so` grows from 12,440,632 to
12,578,440 bytes, **+137,808 bytes (+1.11%)**. Peak RSS grows ~0.25 MB (under 1%).

An earlier pass on a 4-core laptop showed run-to-run spread of ~8%, which bounds nothing useful. The
desktop's 0.2–0.7% spread is what makes the interval above meaningful. Both runs used
`native/build_variants.sh` semantics with the model parameterized; `add.pte` cannot resolve a
build-flag delta at all and prints `warm_mean_ms=0.001` for every variant.

**Conclusion: one artifact.** 138 KB and a bounded-under-0.35% steady-state cost do not justify a
second build-matrix row, a second staging path, and a `LibUtils` selection rule.

## 2. Phase 0 — the runtime distribution prerequisite (landed)

**Delivered in `v1.4.1-3`.** All five criteria below were verified against the published assets:
devtools tarballs carry `include/executorch/devtools/etdump/etdump_flatcc.h` and both `data_sinks/`
headers, plus the whole `include/flatcc/` tree; `logging` contains none of it (zero matches for
devtools, flatcc, or `libetdump`); BUILDINFO carries `event_tracer=on` for devtools and `off` for
logging. Criterion 2 was verified by compiling and linking a translation unit that constructs an
`ETDumpGen` and calls `get_etdump_data()` against the shipped archives — it runs and reports size 0,
the correct `Init`-state answer.

Windows devtools rows shipped too, ahead of the request: the release publishes
`devtools_windows-x86_64` and `devtools_windows-x86_64-static`. That changes §5's gate for Windows
from a dist dependency to an engine-side list edit, and does not change the staging order.

The original criteria are kept below as the record of what was asked and verified.


The devtools tarball ships `lib/libetdump.a` and `lib/libflatccrt.a`, and
`lib/cmake/ExecuTorch/ExecuTorchTargets.cmake` exports working `etdump` and `flatccrt` imported
targets. It installs **no devtools headers**: `include/executorch/devtools/` does not exist, so
`ETDumpGen` can be linked but not included.

This is upstream ExecuTorch's own packaging gap, not something the dist repo dropped —
`devtools/etdump/CMakeLists.txt` installs the targets with no header `FILE_SET`, so
`etdump_flatcc.h` never reaches the prefix. Vendoring the header into this repo is rejected: it
pulls `data_sinks/buffer_data_sink.h`, `data_sinks/data_sink_base.h`, and the flatcc include tree
behind it, and that set would have to track every pin bump by hand.

The fix belongs in `executorch-runtime-dist`, which already carries a `patches/` directory and an
`ETNPExtras` precedent for installing extra material. Acceptance criteria:

1. Every devtools tarball contains `include/executorch/devtools/etdump/etdump_flatcc.h` and the
   `data_sinks/` headers it includes.
2. A consumer calling `find_package(executorch)` and linking the `etdump` target compiles a
   translation unit that includes `etdump_flatcc.h` — the header and the imported target agree.
3. The `logging` and `bare` tarballs are unchanged in scope: no devtools headers, no `etdump`
   target, no `libetdump.a`. This must not perturb the variant the engine ships on Windows. The new
   install rule needs an explicit `EXECUTORCH_BUILD_DEVTOOLS` guard — upstream processes the
   `devtools/` subdirectory in *both* branches of that condition, so an unguarded rule would leak
   the headers into every variant.
4. BUILDINFO carries `event_tracer=on|off`, sourced from the same place as the
   `-DEXECUTORCH_ENABLE_EVENT_TRACER` flag. See the capability-gate discussion in §3.
5. Second priority, not a Linux blocker: install flatcc's `flatcc_builder.h` as well. See §3 for
   why a Windows devtools build needs `flatcc_builder_aligned_free`, and why having the declaration
   already installed turns that into a one-line platform arm instead of a hand-written `extern "C"`
   against a vendored third-party symbol.

The full paste-ready brief for that work is
[docs/research/2026-08-24-devtools-header-install-handover.md](../../research/2026-08-24-devtools-header-install-handover.md).

Then the pin bump here: replace `native/cmake/EtRuntimePin.cmake` wholesale with the release asset,
re-apply the comment header, and re-run `./native/gen_clangd_db.sh`. Skipping that last step leaves
clangd resolving against the previous runtime's headers, so the new `devtools/` include fails to
resolve in the editor while the build succeeds.

No engine work starts before this pin lands. The `ET_INSTALL` escape hatch could unblock local
development, but nothing would be reproducible from a clean checkout and CI could not run the new
tests.

## 3. Native core

### Capability gate

`native/CMakeLists.txt` sets `ET_HAVE_DEVTOOLS` when `TARGET etdump` exists after
`find_package(executorch)`, and links `etdump` — which pulls `flatccrt` through its
`INTERFACE_LINK_LIBRARIES`. This is the auto-detect shape already used for `openvino_backend` and
`ETNPExtras`: the build adapts to what the pinned tarball provides, so a `logging` runtime and a
`devtools` runtime both configure and compile with no flag to pass.

**Linking `etdump` does not carry the compile definitions its own headers need.** The exported
target's interface is `flatccrt;$<LINK_ONLY:executorch>`, and `$<LINK_ONLY:>` suppresses usage
requirements, so `C10_USING_CUSTOM_GENERATED_MACROS` — which other ExecuTorch targets do carry —
never propagates through `etdump`. Without it, `etdump_flatcc.h` reaches
`torch/headeronly/macros/Macros.h`, which includes a `cmake_macros.h` that no tarball installs, and
the compile fails. The shim already links the main `executorch` targets, so the definition arrives
in practice; the requirement is simply that it keep doing so. Do not "simplify" the link line to
`etdump` alone.

**`TARGET etdump` alone is not a sufficient capability signal, and the configure must not trust it
on its own.** At pin `1.4.1-2` it discriminates correctly — the `logging` tarball ships no
`libetdump.a` and its `ExecuTorchTargets.cmake` declares no `etdump` target. But that is not
guaranteed by upstream: `add_library(etdump ...)` and its `install(TARGETS ...)` are unguarded, and
the root `CMakeLists.txt` adds `devtools/` in both branches of `if(EXECUTORCH_BUILD_DEVTOOLS)`. The
exclusion is a property of how the distribution builds, not a promise. Were it to lapse, a
`logging` build would link `etdump`, report `devtoolsAvailable() == true`, and produce empty dumps —
because `EXECUTORCH_ENABLE_EVENT_TRACER=OFF` compiles the tracer hooks out of `Method::execute`.
Silent empty output is the worst available failure.

So the gate reads **two** signals and asserts they agree: `TARGET etdump` (can we link it) and
BUILDINFO's `event_tracer` key (was the tracer compiled in). A disagreement fails the configure with
a message naming both. The BUILDINFO key does not exist yet — Phase 0 criterion 4 adds it, following
the existing `usdt=on` key. When BUILDINFO is absent entirely, as under the `ET_INSTALL` escape
hatch, the configure falls back to `TARGET etdump` and warns that the capability is unverified.

### Surface

Three additions to `native/core/et_runtime.h`, all JNIEnv-free so the Catch2 units and the harnesses
reach them:

```cpp
explicit EtRuntime(const std::string& ptePath, int workspaceSharingMode = -1,
                   bool traceEvents = false);

// Finalized ETDump covering the forwards since the last call. Empty when not tracing and when no
// forward has run. Safe to call twice: the second call returns a copy of the same bytes until
// another forward() has run.
std::vector<uint8_t> etDump();

bool devtoolsAvailable();   // free function, alongside xnnpackWorkspaceBytes()
```

`etDump()` is deliberately non-const: it finalizes builder state.

### Construction

`RuntimeState` grows a non-owning tracer pointer; the `Module` owns the tracer, because its
constructor takes `std::unique_ptr<runtime::EventTracer>`:

```cpp
struct RuntimeState {
  Module module;
  ETDumpGen* tracer;   // non-owning; the Module owns it. null when not tracing.
  ...
};
```

Nothing else in the constructor changes. `Module::load_method` uses `this->event_tracer()` when its
own tracer argument is null, so the existing unconditional `load_forward()` picks the tracer up
untouched — the workspace-sharing options path and the OpenVINO precondition check are unaffected.

### Pull semantics, and two hazards the implementation must handle

`ETDumpGen`'s default constructor takes an empty `Span`, which selects malloc-backed growth. Events
accumulate for the runtime's life until pulled. `Method::execute` calls
`event_tracer_create_event_block(event_tracer_, "Execute")` on every execution, so one pulled dump
contains one block per forward — the shape the Inspector expects for aggregate statistics.

Draining is the runtime's own behaviour, not something this engine adds: `get_etdump_data()` sets
`state_ = Done`, and the next execution's `create_event_block()` sees `Done` and calls `reset()`.
Pulling *is* the drain.

Two properties are not handled for us:

- **The pulled buffer is caller-owned.** In the malloc path `get_etdump_data()` returns
  `flatcc_builder_finalize_aligned_buffer(...)`, a freshly allocated buffer. ExecuTorch's own
  reference consumer (`examples/devtools/example_runner/example_runner.cpp`) releases it with plain
  `free()`, which is correct for flatcc's aligned allocator on POSIX. It is **not** portable: under
  MSVC flatcc allocates with `_aligned_malloc`, where `free()` is undefined behaviour. A Windows
  devtools build must call `flatcc_builder_aligned_free` instead — see Phase 0 criterion 4.
- **A double pull corrupts the builder.** With `state_ == Done`, none of `get_etdump_data()`'s three
  guard branches match and it runs `run_data_push_end` / `ETDump_end` against an already-finalized
  builder. `EtRuntime` therefore caches the copy from the last pull behind a `bool dumpFinalized_`,
  set on pull and cleared in `forward()`. While set, `etDump()` returns the cached copy without
  touching the builder.

The cache also makes the Java contract forgiving: pulling twice, or pulling around teardown, cannot
corrupt anything. The `Init` case — tracing enabled, zero forwards — already returns `{nullptr, 0}`
upstream and becomes an empty vector here.

### Without devtools

`devtoolsAvailable()` returns false, `etDump()` returns empty, and constructing with
`traceEvents=true` throws, naming the platform and the runtime variant. Profiling requested where it
cannot work is a load failure, not a silent no-op — the same discipline by which an unrecognized
`workspaceSharingMode` fails the load.

### Threading and lifetime

No new constraint. `ETDumpGen` is not thread-safe and neither is `forward()` on one model; the
existing one-model-per-thread rule covers both. The pulled `std::vector` is caller-owned and
outlives the runtime — the one place this differs from the borrowed-view discipline of
`OutputView`, and worth a comment at the declaration.

## 4. JNI and Java surface

```java
public static native long loadModule(String ptePath, int workspaceSharingMode, boolean profiling);
public static native byte[] etDump(long handle);      // empty array, never null
public static native boolean devtoolsAvailable();
```

`etDump` marshals into a fresh `byte[]`, consistent with how `forward()` already returns outputs
(single-copy out, heap array, not a direct buffer). `devtoolsAvailable` is a static query with no
handle, alongside `backendRegistered` and `xnnpackWorkspaceBytes`.

Option plumbing follows `EtWorkspaceSharing` exactly: a package-private `EtProfiling` with
`OPTION_KEY = "profiling"`, a `parse` accepting case-insensitive `true`/`false` and throwing
`IllegalArgumentException` otherwise, and `resolve(options)` called in `EtModel.load` beside the
existing `EtWorkspaceSharing.resolve`. Published as `EtEngine.PROFILING_OPTION`.

**There is deliberately no JVM property**, unlike `workspaceSharingMode`. A property would let one
JVM flag turn on unbounded event accumulation for every model in the process, including models whose
owner never pulls, without touching any code — the precise failure mode profiling carries. The
per-model option keeps enabling it a decision at the load site. This absence is a design choice and
must be stated as such in `EtProfiling`, so it does not read as an oversight to the next reader.

`EtModel.load` calls `EtNative.devtoolsAvailable()` when profiling is requested and throws before
`loadModule`, with a message naming the platform and runtime variant. The native constructor still
throws on its own: `EtNative` is public and bypasses `EtModel`, so both layers check — the same
deliberate duplication the OpenVINO precondition already documents.

Retrieval is one method:

```java
/** Returns the ETDump for forwards since the last call; empty if profiling is off. */
public byte[] etDump()
```

Callers reach it as they reach any engine-specific surface:
`((EtModel) zooModel.getWrappedModel()).etDump()`, then write the bytes wherever they like. There is
no `writeEtDump(Path)` convenience — it is `Files.write` at the call site, and it would put the
engine into the filesystem business this design keeps it out of.

`EtStatsSnapshot`'s per-model counters gain a `profiling` boolean beside the existing
`workspaceSharingMode` name. A model quietly accumulating an ETDump in production is exactly what the
monitoring surface should show.

## 5. Build, capability, and platforms

Which platforms ship a profiling-capable runtime is an engine-side decision, not a mirror of what the
pin publishes — the same principle as `ET_OPENVINO_SUPPORTED_PLATFORMS`. Today `native/build.sh:87`
defaults `ET_RUNTIME_VARIANT` to `logging` for every platform and no workflow overrides it. That
becomes a per-platform list living beside the OpenVINO list, with `ET_RUNTIME_VARIANT` still
overriding for benchmarking.

As of pin `1.4.1-3` the pin publishes `devtools` for every platform the engine ships:
`linux-x86_64`, `linux-aarch64`, `windows-x86_64`, and `windows-x86_64-static`. Nothing is blocked
on the distribution any more; what remains is engine-side provisioning and proof.

| platform | shipped variant | profiling | gate to change it |
|---|---|---|---|
| `linux-x86_64` | `devtools` | yes | — proven by this spec |
| `linux-aarch64` | `logging` initially | not yet | verify tarball parity (below), then flip the list |
| `windows-x86_64` | `logging` | not yet | flip the list and add a test — both devtools CRT rows ship as of `v1.4.1-3` |

Nothing here says a platform cannot profile. The contract is the `devtoolsAvailable()` query, never
the platform name, so a platform joining is a list edit plus a test — no engine redesign, no API
change, no new Java surface. Documentation must say "not provisioned on this platform yet", never
"unsupported".

**Tarball parity, verified and unverified.** The devtools `linux-x86_64` tarball ships
`lib/cmake/ETNPExtras/`, `lib/libetnp_ops_lstm.a`, and `lib/libopenvino_backend.a` — confirmed by
listing the extracted archive — so switching that platform drops neither the `etnp::lstm` custom op
nor the OpenVINO delegate. The devtools `linux-aarch64` tarball has **not** been inspected; parity
there is a plan step, run on the radxa host, and gates that platform's flip. If parity fails,
aarch64 stays on `logging` and the capability query already covers it.

`native/build_qa.sh` currently defaults to `logging` too. It should follow the shipped variant, or
the ASan/UBSan gate exercises a runtime that is no longer shipped. That change is what puts the
ETDump pull — a manual `free()` of a third-party allocation — under ASan.

Cost: the shipped Linux `.so` grows ~138 KB. No new build-matrix rows, no second staging path, no
`LibUtils` selection rule. CI gains tests but not a build.

## 6. Export-side ETRecord

The Inspector maps runtime events back to graph ops only when an ETRecord was emitted at export.
`tools/scripts/export_mobilenet.py` gains `--etrecord`, default off. The script already holds both
objects `generate_etrecord(et_record, edge_dialect_program, executorch_program, ...)` needs; it
currently discards one:

```python
program = lowered.to_executorch()          # hoisted out of the write below
with open("mobilenet_v2.pte", "wb") as f:
    f.write(program.buffer)
if args.etrecord:
    generate_etrecord("mobilenet_v2.etrecord", lowered, program)
```

Default off because an ETRecord embeds the program buffer and the graph modules — not a small
sidecar — and `./gradlew :example:exportModels` exists to produce a demo model, not a profiling kit.
The `_unplanned` and TorchScript artifacts are untouched. Other export scripts stay as they are
until something needs them.

## 7. Testing and verification

**Java tests live in the main `test` task**, not a new tagged task. The OpenVINO precedent uses a
separate task because its runtime is a separate opt-in jar with a classpath dimension; profiling has
none — it is in the shipped `.so` or it is not. Tests branch on `EtEngine.devtoolsAvailable()`:

- Capability present (`assumeTrue`): the dump is empty before any forward; carries the `ED00` file
  identifier after one (the root is built with `start_as_root_with_size`, so the buffer is
  size-prefixed); grows across several forwards; and after a pull, the next forward yields a fresh
  dump rather than a cumulative one. Two `etDump()` calls with no forward between them return equal
  bytes and corrupt nothing — that guard's failure mode is a native crash, which earns it a test.
- Capability absent (`assumeFalse`): requesting `profiling=true` fails the load with a message
  naming the platform, and `etDump()` on an unprofiled model returns empty rather than throwing.
- Everywhere: an unrecognized option value fails the load, mirroring `workspaceSharingMode`.

The suite must name which arm it took and what it skipped. A capability-gated test that silently
no-ops where the capability is missing is indistinguishable from a passing one.

`-Xcheck:jni` is already attached to every `Test` task, so the new array-allocating return path is
JNI-contract checked without further work.

**Native.** `et_leak_harness` gains a profiling arm — load, forward, pull, repeat, destroy. A leaked
ETDump buffer is the most likely defect in this change and LSan catches it for a few lines of
harness code. Catch2 units cover the semantics; the harness covers the allocation.

**Windows is a human gate.** `EtNative.loadModule` gains a parameter, and Catch2 links the core only,
so it cannot catch a JNI signature mismatch. `gradlew.bat test` on the winbox is a release gate for
this change. That run also executes the capability-absent arm.

**Manual, once per pin bump:** export with `--etrecord`, profile several forwards, run the Inspector,
and confirm events attribute to graph ops. ETRecord↔ETDump correlation cannot run in CI — the
Inspector is Python, the artifacts are gitignored build outputs needing `uv`, torch, and network,
and the JVM tests deliberately use committed fixtures. The procedure belongs in `docs/` with exact
commands, admitting that it is manual. A `tools/scripts/` correlation script is rejected: it would be
exercised approximately never, and an unexercised script that looks like a gate is worse than a
documented procedure that does not.

## 8. Documentation

- `CLAUDE.md`: platforms no longer "all ship the `logging` runtime variant"; `devtools` is a shipping
  variant on Linux, not solely a benchmarking one. The claim that `etnp::lstm` is in the
  `linux-x86_64` `logging` tarball **only** is wrong at pin `1.4.1-2` — it is in devtools too.
- `docs/benchmarking.md`: the gating question is answered. Record the numbers and the decision in
  place of the open question.
- A profiling page under `docs/`: enabling the option, pulling the dump, exporting an ETRecord, the
  manual Inspector procedure, and the growth property.
- `docs/README.md`: index the new page.

## 9. Out of scope

- **In-process per-op timings.** Parsing the ETDump in Java and surfacing durations through
  `EtEngineStats`/JMX would make the ETDump schema a compatibility surface across pin bumps. The
  native and JNI layers make the bytes retrievable, so this stays reachable later without rework.
- **Intermediate output logging.** `ETDumpGen::set_debug_buffer` and the `log_evalue` path capture
  tensor values, not just timings — a much larger buffer and a data-exposure question of its own.
- **A second profile-capable artifact.** Closed by §1.
- **`weight_cache_enabled`**, unchanged and still deliberately unexposed.
