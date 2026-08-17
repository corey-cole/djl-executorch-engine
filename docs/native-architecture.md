# The native layer

Everything under `native/` is C++. It is no longer a thin JNI shim: most of the interesting
behaviour — input staging, the load-time validation, the process-global threadpool guard — lives in
a core that knows nothing about the JVM, and the JNI code above it does translation and nothing
else. This document describes that shape. For how to build any of it, see
[building.md](building.md).

## 1. Three layers

| Directory | Artifact | Knows about the JVM |
|---|---|---|
| `native/core/` | `et_runtime`, a static library | no |
| `native/jni/` | `executorch_djl`, the shipped shared library | yes, and only here |
| `native/harness/`, `native/test/` | standalone executables | no |

**`native/core/`** is `measly::et::EtRuntime`, a C++ wrapper over ExecuTorch's `extension::Module`.
It owns the module, the method metadata snapshot, and the input staging slots; it validates inputs,
performs the staging copy, and exposes results as borrowed views. Its public surface is
`native/core/et_runtime.h` — `InputDesc`, `OutputView`, `MethodMeta`, `ForwardResult`, `EtRuntime`,
plus the two free functions that read and set the intra-op thread count. Supporting headers
alongside it: `staging.h` (the slot allocator), `et_probes.h` (USDT + in-process probes), and
`dtype_size.h`. One more, `native/jni/array_size_limits.h`, lives in `jni/` but is deliberately free
of `<jni.h>` so the Catch2 units can pin the `jsize` output boundary without a JNIEnv — which is why
`native/CMakeLists.txt` adds `native/jni` to `et_runtime_test`'s include path.

**`native/jni/`** is the only part that includes `jni.h`. `executorch_djl_jni.cpp` holds the
`Java_org_measly_executorch_jni_EtNative_*` entry points; `et_logging.cpp` is a PAL bridge that
replaces ExecuTorch's log emitter with one that forwards `ET_LOG` output to slf4j through
`EtNative.nativeLog`. Both are translation layers. The JNI file caches every class, field and method
ID once in `JNI_OnLoad` (per-call `FindClass` is expensive, and unsafe with an exception pending),
converts C++ exceptions to Java ones, and marshals `EtTensor[]` in and out. The `jlong` handle it
returns from `loadModule` is a raw `EtRuntime*`: there is no registry and no validation, so handle
discipline is entirely the Java side's job — see the comment block above `loadModule` for what that
does and does not guarantee.

**`native/harness/` and `native/test/`** are the QA and benchmarking binaries: `et_runtime_test`
(Catch2 units), `et_leak_harness` (ASan/LSan), `et_timing_harness` (Release benchmark),
`et_scaling_harness` (thread scaling), `et_stress_harness` (N threads, N runtimes, bitwise-identical
outputs), and `et_overread_harness` (the manual guard-page experiment). All of them link
`et_runtime` and nothing else of ours.

## 2. Why the core is JNIEnv-free

This is a deliberate constraint, not an accident of layering. Because the core has no JVM
dependency, every binary in the table above links it directly, which means **a QA or bench configure
needs no JDK and no `JAVA_HOME`**. `native/CMakeLists.txt` enforces the split from the other
direction: the `executorch_djl` shared library is declared inside `if(NOT ET_BUILD_QA AND NOT
ET_BUILD_BENCH)`, so the only configure that looks for a JDK is the one that builds the shipping
library. `-DET_BUILD_QA=ON` never resolves `javac` and never compiles a line of JNI.

The consequence worth knowing about is for editors. Because the shim and the test targets are
mutually exclusive, **no single CMake configure covers both `jni/` and `test/`** — a compile
database from either one is missing half the tree, and clangd then reports phantom errors in
whichever half it cannot see. `native/gen_clangd_db.sh` exists for that: it runs two configures
(default and `-DET_BUILD_QA=ON`) and merges their `compile_commands.json` into
`native/build-clangd/`, shim entries winning for `core/et_runtime.cpp` since those are the flags the
shipped library is really built with. Re-run it after a pin bump or a compile-flag change; nothing
refreshes it automatically. See [building.md](building.md) for the editor setup.

## 3. Ownership and data flow

One `forward()` call moves data like this:

1. Java hands the shim an `EtTensor[]`. Each carries a shape, a ScalarType code, and a **direct**
   `ByteBuffer` — the shim calls `GetDirectBufferAddress` and rejects anything else with an
   `IllegalArgumentException`. No copy here.
2. The shim fills an `InputDesc` per input: a borrowed `const void*`, the shape, the dtype. The
   `jobjectArray` parameter keeps the buffers reachable for the whole frame, which is what makes the
   pointers valid until `forward()` returns.
3. `EtRuntime::forward` checks each input's dtype against the model's and its byte count against the
   model's declared bound, stages it if the input is unplanned (§4), builds a non-owning `from_blob`
   tensor, and calls `Module::forward`.
4. Outputs come back as `OutputView`s: shape, dtype, and a pointer **into ExecuTorch's arena**,
   owned by the `ForwardResult` and invalid after the next `forward()` or the runtime's destruction.
   The copy out of the arena happens once, in the shim, into a fresh JVM `byte[]` wrapped by
   `ByteBuffer.wrap`. Outputs are therefore heap arrays, not direct buffers.

**The input borrow is not zero-copy end to end.** This is the claim most worth getting right,
because the engine borrows honestly and ExecuTorch then does whatever the `.pte` tells it to.
`Method::set_input` branches on `TensorInfo::is_memory_planned()`: when it is true — the export
default, `MemoryPlanningPass(alloc_graph_input=True)`, and so the case for **any model a user brings
unless they went out of their way** — ExecuTorch `memcpy`s the input into its own planned arena and
our pointer is borrowed only for the duration of that copy. The borrow is honoured, via
`share_tensor_data`, only for models exported with `alloc_graph_input=False`. Describing the input
path as zero-copy without that qualification is wrong, and was wrong in this repository's own docs
until it was audited.

Both cases occur here, which is why §4 is not describing dead code. The models this repository ships
and exports for real work are planned — `native/spike/add.pte`, the MobileNetV2 export in
`example/`. The unplanned fixtures exist specifically to exercise the borrow path:
`native/spike/add_unplanned.pte` (the same add model, exported by `export_add_unplanned.py` with
`MemoryPlanningPass(alloc_graph_input=False)`, consumed by `et_runtime_test` and — through
`build_qa.sh` — by `et_leak_harness`), `clamp5.pte` and `lin129.pte` from `export_w4_models.py`,
and a `mobilenet_v2_unplanned.pte` that `tools/scripts/export_mobilenet.py` generates alongside
the planned one. Reading which one you have is not guesswork: the flag is plumbed all the way to
Java (`EtMethodMeta.inputMemoryPlanned`) and logged per input at model load.

`native/spike/conv.pte` (from `export_conv.py`) is unrelated to memory planning and exists for a
different reason: it is the only fixture that makes XNNPACK allocate a workspace arena. Delegating
and allocating are separate properties — `add.pte` is a single node with external input and output,
and `lin129.pte` lowers to a GEMM over statically packed weights, so both delegate and both grow the
arena by exactly zero. Any `xnnpackWorkspaceBytes()` assertion built on them would pass vacuously or
fail against a correct build. The arena also grows on the first *execute*, not at delegate init, so
the fixture must be forwarded and not merely loaded.

The full contract, both directions, with the ExecuTorch source references:
[executorch-host-buffer-contract-brief.md](executorch-host-buffer-contract-brief.md).

Two load-time rejections shape everything downstream, both in `EtRuntime`'s constructor. A method
with a non-tensor input (a prim `int`/`double`/`bool`, or `None`) is refused, because `InputDesc`
cannot express a traced prim value — and refusing it is also what makes `is_memory_planned == 0`
unambiguous, since a non-tensor input would carry that same 0 with no `TensorInfo` behind it. A
dtype whose size the engine cannot compute is refused for the same reason: a dtype it cannot size is
a dtype it cannot stage.

## 4. Staging slots

`native/core/staging.h` defines `StagingSlot`, a grow-only aligned buffer, and `EtRuntime` keeps one
per input position. Four properties, all of which have a reason:

- **Allocated at construction, from the model's declared bound.** `TensorInfo::nbytes()` is
  available at load for planned and unplanned inputs alike, so a slot is sized once and "grow-only"
  degenerates to "allocate once". Steady state is allocation-free.
- **Grow-only.** `ensure()` never shrinks, and does not preserve contents across a growing call —
  the only caller overwrites the whole slot immediately afterwards.
- **64-byte aligned.** Not an XNNPACK requirement: it is what makes the size rounding legal, since
  POSIX `aligned_alloc` requires the size be a multiple of the alignment. Hence the round up to a
  multiple of 64 rather than allocating `needed` directly.
- **Padded by `kStagingPadding` (128 bytes).** XNNPACK documents an out-of-bounds *read* of up to
  `XNN_EXTRA_BYTES` past a tensor (16 on x86/ARM, 128 on Hexagon) and expects callers to allocate
  the slack. `xnnpack.h` is delegate-internal and not on our include path, so the maximum is
  hardcoded. `StagingSlot` itself knows nothing about the padding — every caller must ask for tensor
  bytes *plus* the padding, and a caller that forgets is an out-of-bounds read no assertion catches.

Staging exists for safety, not speed. When an input is unplanned we do not get to decline the borrow
— `share_tensor_data` takes whatever pointer it is handed, with no alignment or padding check and no
refusal path — so the only question is *whose* buffer ExecuTorch retains. Staging answers "ours,
padded, aligned, and owned by the same object as the `Method`", which closes the over-read, the
alignment question and the pointer-lifetime hazard together, at the cost of one `memcpy`.

**Planned inputs are never staged, and their slots stay at capacity 0.** That is why
`EtModelStats.getStagingBytes()` — and the `totalStagingBytes` rollup in `EtEngineStats.snapshot()`
— reports 0 for most real models. The 0 is a measurement, not a missing value: it means every input
is memory-planned and ExecuTorch is doing the copying. The distinct "unavailable" value is `-1`,
which the JNI layer returns after scheduling an exception and `EtSymbolBlock.toStats()` returns for
a closed block.

## 5. Process-global state

Two things in the native layer are process-wide rather than per-model, and they behave in opposite
ways. Conflating them is the easy mistake.

**The intra-op (XNNPACK) thread pool is a process singleton, write-once.** It is ExecuTorch's
`extension::threadpool` singleton, which sizes itself to the performance-core count and reads no
environment variable, so `setIntraOpThreads` is the only control surface there is. It is applied and
sealed at the first model load: `EtRuntime`'s constructor sets a global flag — even when that
constructor goes on to throw — and the native `measly::et::setIntraOpThreads` refuses any later
reset, logging and returning the count already in effect. The refusal is not conservatism. XNNPACK
captures the `pthreadpool_t` when it creates a runtime, and `_unsafe_reset_threadpool` destroys the
old pool object, so a late reset is a **use-after-free on the next `forward()`**, not merely a race.
The native function returns the count in effect *after* the attempt, so callers compare rather than
checking a status. **The Java-layer `EtEngine.setIntraOpThreads` does not follow this return-value
convention** — it throws `IllegalStateException` on a late reset instead of returning a value to
compare; see `README.md`'s "Configuration and tuning" section.

**Workspace sharing is the opposite in every respect.** It is a per-load backend option:
`EtRuntime`'s constructor builds a `LoadBackendOptionsMap` with `workspace_sharing_mode` under the
`XnnpackBackend` key and passes it to `Module::load`, and ExecuTorch resolves it per delegate during
delegate init. So it is neither global nor write-once — modes compose across models and load order
is irrelevant. Two details follow from that. The constructor calls `Module::load_forward()`
unconditionally, because delegate init happens in `load_method` and would otherwise be deferred to
the first `forward()`, which would make an invalid mode surface at predict time instead of at load.
And a mode of `-1` omits the spec entirely rather than passing the runtime's current default, so the
pin's default is followed instead of being pinned to a value we would have to keep in sync.

## 6. Probes

`native/core/et_probes.h` instruments the staging path, which is otherwise invisible: it happens
entirely inside the native allocator, below the JNI boundary. Each probe site expands twice — a
USDT/DTrace static probe under provider `measly`, and a call to an in-process dispatcher the Catch2
units and the leak harness install a handler on, so the same event drives both a `bpftrace` session
and an in-repo assertion.

| Probe | Arguments | Fires |
|---|---|---|
| `staging_grow` | `slot`, `old_bytes`, `new_bytes` | when a slot has to reallocate |
| `staging_input` | `slot`, `nbytes`, `planned`, `staged` | once per input, per `forward()` |

`staging_grow` should **never** fire in a healthy process. Slots are sized at load from the declared
bound and `forward()` rejects any input past that bound, so the growth path is unreachable by
construction; the probe is a tripwire on those two invariants, and `et_leak_harness` asserts a count
of exactly zero. A `staging_grow` on the hot path means a slot was under-sized at load, which is a
bug. `staging_input` is the routine observability: it reports, per input, how many bytes crossed and
which path it took.

Only the USDT half compiles out when not on Linux/GCC. The in-process dispatch is compiled in everywhere and
costs a relaxed atomic load and a not-taken branch when no handler is installed, which is what makes
it acceptable on the hot path.

To watch them on a running process:

```bash
# Attach by PID; -p resolves the USDT targets from the process's own mappings.
sudo bpftrace -p "$(pgrep -f et_timing_harness)" -e '
  usdt:*:measly:staging_input { @bytes[arg0, arg3] = sum(arg1); @n[arg0, arg3] = count(); }
  usdt:*:measly:staging_grow  { printf("GROW slot=%d %d -> %d\n", arg0, arg1, arg2); }'
```

`arg3` on `staging_input` is the `staged` flag, so the aggregation separates staged from
pass-through inputs. Two caveats when the target is a JVM rather than a harness. The shim is
`System.load`ed from a content-addressed cache (`~/.cache/executorch-djl/<sha256>/`), so a literal
`usdt:/path/...` target changes with every build — attach with `-p`, or resolve the path from
`/proc/<pid>/maps`. And the library is not mapped until `EtNative`'s static initializer runs, so
attaching before the first model load finds no probes at all. Both are avoided by developing against
`et_timing_harness` or `et_leak_harness`, which link the same core.

## 7. Where the ExecuTorch runtime comes from

It is not built here. CMake downloads a hash-pinned, build-attested tarball published by the
separate `executorch-runtime-dist` repository, via `FetchContent`, and links against that. The pin —
version, per-platform URLs, and SHA256 for every variant row — lives in
`native/cmake/EtRuntimePin.cmake`, which is **generated and must not be hand-edited**; it is
replaced wholesale from the next release's asset, and the SHA256 change is the supply-chain review
gate. Setting `ET_INSTALL` to an existing install tree skips the download entirely. Platform
identity (`ET_PLATFORM`) and pin-row key (`ET_RUNTIME_ROW`) are separate variables because Windows
publishes two rows for one platform. [building.md](building.md) has the rest, including the runtime
variants and the escape hatch.
