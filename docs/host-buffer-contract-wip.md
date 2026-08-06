# ExecuTorch host-buffer contract: W4 + W7 + W8 implementation record

Date: 2026-08-05. Plain record of what was requested, what was delivered, and
how the delivered work deviates from the approved plan
(`local://et-overread-staging-plan.md`). The manual W4 instructions the user
still needs to run are appended at the end.

## Outcome requested

The user approved the plan titled "W4 + W7 + W8: XNNPACK over-read harness,
unplanned-input staging, USDT probes" and asked for it to be executed step by
step, with each step verified before the next. In substance, three work items
against the ExecuTorch host-buffer contract (`docs/executorch-host-buffer-contract-brief.md`):

- **W7** — grow-only, 64-byte-aligned, 128-byte-padded per-slot staging in
  `EtRuntime::forward` for inputs whose `is_memory_planned()` is false, so
  XNNPACK's documented over-read (`XNN_EXTRA_BYTES`) lands in engine-owned
  slack instead of a caller's exact-sized buffer.
- **W8** — two USDT probes (`staging_grow`, `staging_input`) plus exact-count
  leak-harness assertions, so a reallocate-every-call staging bug fails loudly.
- **W4** — a committed-but-not-CI-wired guard-page over-read harness (borrowed
  and arena configurations), the delegated unplanned/planned fixtures it needs,
  and a manual run recipe producing dated evidence.

## Summary of work done

All work committed on top of `a1a8ff1` in five commits:
`104ecbe` (W7+W8 core), `4d50fa2` (W4 fixtures), `6872adb` (W4 harness),
`5cea904` (brief), `e026b21` (Java test).

1. **`native/core/dtype_size.h`** — moved out of `native/harness/` via
   `git mv`, content unchanged. The `et_runtime` target PUBLIC-includes
   `native/core`, so the existing `"dtype_size.h"` quote-includes in both
   harnesses still resolve; no CMake or include edits were needed.

2. **`native/core/staging.h`** — `kStagingPadding = 128` (the maximum of
   `XNN_EXTRA_BYTES` across x86/ARM/Hexagon, hardcoded with a comment because
   `xnnpack.h` is delegate-internal) and `StagingSlot`, a grow-only slot whose
   `ensure()` rounds capacity up to a 64-byte multiple and preserves the first
   `min(old, new)` bytes on growth. Allocation is `_aligned_malloc`/`_aligned_free`
   on Windows and `std::aligned_alloc(64, size)`/`std::free` elsewhere.

3. **`native/core/et_probes.h`** — one header, no new `.cpp`. USDT probes via
   `DTRACE_PROBE*` (provider `measly`, no semaphore, matching the dist's
   `etnp::lstm` probes so one set of bpftrace/perf tooling covers both) plus an
   in-process dispatch (`probe_dispatch` → one shared `inline` atomic handler
   that `et_probe_set_handler`/`et_probe_clear_handler` also use). Untraced
   cost is a single relaxed atomic load per input.

4. **`native/core/et_runtime.cpp`** — `RuntimeState` now owns `MethodMeta meta`
   and `std::vector<std::unique_ptr<StagingSlot>> staging`. The old
   `methodMeta()` body became a file-static `buildMethodMeta(Module&)`, called
   once in the constructor; `methodMeta()` returns the stored copy. In
   `forward()`, each input's byte count is computed (`dtypeSize` × shape
   product; empty shape = 1) and the input is either passed through unchanged
   (planned; fires `staging_input(…, planned=1, staged=0)`) or copied into its
   slot with `ensure(actual + kStagingPadding)` (unplanned; fires
   `staging_grow` on allocation/growth and `staging_input(…, planned=0,
   staged=1)`). The borrowed pointer ExecuTorch retains now always points into
   engine-owned memory that outlives the `Method`.

5. **`native/test/et_runtime_test.cpp`** — seven new Catch2 cases: four
   `StagingSlot` units (aligned + padded at `kStagingPadding`; the
   `100 + kStagingPadding` slack row; no realloc when capacity suffices; grow
   preserves the first 64 bytes), a stage-vs-pass-through pair using an RAII
   probe-counting guard (unplanned: `grow == 2`, `staged == 2`; planned:
   `grow == 0`, `staged == 0`), and the ASan lifetime case (heap-allocated
   inputs freed after the first forward, second forward with stack inputs).

6. **`native/harness/et_leak_harness.cpp`** — registers a counting probe
   handler at start, clears it at exit, reads `forwardsPerLoad` from argv[3]
   (default 4, so existing invocations are unchanged), and asserts exact counts
   after the loop: `grow == numUnplanned × outerIters`,
   `stagedInput == numUnplanned × outerIters × forwardsPerLoad`,
   `totalInput == numTensorInputs × outerIters × forwardsPerLoad`. Mismatch →
   `STAGING ASSERT FAIL` on stderr, exit 1. The planned run doubles as a pin
   guard: a future ExecuTorch that borrows planned inputs instead of copying
   them starts staging and the run fails loudly.

7. **`native/build_qa.sh`** — after the existing `add.pte` run, two new
   invocations: `add_unplanned.pte ${ITERS} 4` (asserts `grow == 2×ITERS`) and
   `add_unplanned.pte 1 10000` (inverted: asserts `grow == 2`, isolating steady
   state). `set -euo pipefail` propagates any assertion failure.

8. **`native/spike/export_w4_models.py`** — PEP 723 uv script (torch 2.12.1,
   executorch 1.3.1, pytorch-cpu index; same `XnnpackPartitioner` +
   `MemoryPlanningPass` pattern as `export_add_unplanned.py`). Exports
   `clamp5.pte` (clamp over 5 floats, `alloc_graph_input=False`, Route A
   N % 4 == 1), `lin129.pte` (`Linear(129, 64)`, `alloc_graph_input=False`,
   Route B K % 4 == 1), and `lin129_planned.pte` (same Linear, default
   memory-planned config, config (b)). All three verified: single
   `executorch_call_delegate` node, `clamp5`/`lin129` report
   `is_memory_planned() == False` / `lin129_planned` `True`, and each loads and
   forwards through the pinned v1.3.1 runtime.

9. **`native/harness/et_overread_harness.cpp`** — standalone JNI-free binary,
   `<borrowed|arena> <pte>`, built only in the QA tree, never executed in CI.
   Prints the detected uarch + ISA flags first (cpuinfo; required for the
   evidence block). `GuardedRegion` mmaps `size + 2×page`, places the data
   pointer so `data + size` lands exactly on a page boundary, and
   `mprotect(PROT_NONE)`s everything after it. `borrowed` mode puts input 0 in
   a `GuardedRegion` sized exactly `nbytes` and runs one forward; `arena` mode
   (config b) replaces the memory-planned arena with a `HierarchicalAllocator`
   over guarded spans sized exactly `memory_planned_buffer_size(i)`, prints the
   per-buffer sizes and whether input 0 is placed last in its buffer, then
   forwards. Outcome contract: exit 0 = clean, SIGSEGV (139) = over-read
   faulted. CMake target added inside `if(ET_BUILD_QA)` in
   `native/CMakeLists.txt`.

10. **`docs/executorch-host-buffer-contract-brief.md`** — §1 status table
    updated (W4 "complete (manual runs pending — see §3/W4)", W7 complete, W8
    complete, fixture-prerequisite row updated), §3/W7's "two gaps" note
    closed, and the full run recipe added to §3/W4's Deliverables.

11. **`src/test/java/org/measly/executorch/LeakStressTest.java`** —
    `inferencePathUnderPressureUnplanned()`, a copy of
    `inferencePathUnderPressure` with `optModelName("add_unplanned")` and a
    leading `assumeUnplannedModelAvailable()`: 20,000 predicts through the
    staging path under `-Xmx256m -XX:MaxDirectMemorySize=64m`. Native staging
    memory is not counted against either cap, so the assertion is the same
    "does not OOM/crash" as the planned variant.

### Verification evidence

- `bash native/build_qa.sh` (final run): **15 test cases, 115 assertions, all
  pass**. Leak harness exact counts: `add.pte` → `grow=0 … total_input=8000`;
  `add_unplanned.pte 1000 4` → `grow=2000 staged_input=8000`;
  `add_unplanned.pte 1 10000` → `grow=2 staged_input=20000`. All three
  expected values matched exactly (a `>`/`>=` slip in `ensure` would have made
  the second `grow=8000` and failed).
- Shim rebuilt with the staging core (`cmake -B native/build -G Ninja
  -DET_INSTALL=/tmp/et-runtime-v1.3.1-logging/root`, JAVA_HOME zulu-17 — the
  container-only `native/build.sh` cannot run on this host). XNNPACK
  post-link registration assertion passed.
- `EXECUTORCH_LIBRARY_PATH=<abs path to native/build/libexecutorch_djl.so>
  ./gradlew test` → BUILD SUCCESSFUL; `./gradlew leakTest` → all 3 leak cases
  ran, 0 skipped, 0 failures (including the new unplanned variant). Note:
  `System.load` requires an **absolute** path; a relative
  `EXECUTORCH_LIBRARY_PATH` fails 3 tests with `UnsatisfiedLinkError`.
- `readelf -n native/asan/et_leak_harness`: `.note.stapsdt` shows provider
  `measly`, names `staging_grow` / `staging_input`, `Semaphore: 0x0`.
- Native smoke (Tiger Lake): `et_overread_harness borrowed lin129.pte` →
  `uarch_id=0x0010020d`, `nbytes=516`, `planned=0`, exit 0 (safe avx512f
  kernel); `arena lin129_planned.pte` → 1 planned buffer of 848 B,
  `input placed last in its buffer: no`, exit 0.
- qemu Route A reproduction (implementer smoke, stronger than the required
  launch check): `ASAN_OPTIONS=detect_leaks=0:handle_segv=0 qemu-x86_64 -cpu
  Nehalem ./native/asan/et_overread_harness borrowed native/spike/clamp5.pte`
  → `uarch_id=0x00100205` (SSE-only), qemu **exit 139, "uncaught target
  signal 11 (Segmentation fault)"** — the SSE2 clamp tail load over-read the
  20-byte buffer onto the guard page.

## Deviations from the plan

1. **The harness's borrowed mode bypasses `EtRuntime::forward`.** The plan's
   step 13 text said `EtRuntime rt(pte); … rt.forward(…)`. With W7 shipped in
   the same tree, that would copy the guarded buffer into the padded slot and
   the over-read would land in the slot's slack — the fault would be
   unreachable and every route would report a meaningless negative. The harness
   instead uses the raw path (`Module` + `from_blob` → `set_input`
   `share_tensor_data`), which is exactly what the brief's config table names
   ("our exact-sized host buffer, borrowed via share_tensor_data"; "caller's
   buffer abuts PROT_NONE page") and what the plan's own expected-SIGSEGV
   outcomes require. Documented in the harness header and the run recipe.

2. **`ET_PROBE_STAGING_INPUT` as written did not compile.** The plan's macro
   passed six arguments to the five-parameter `probe_dispatch` (a trailing
   `0`). The trailing `0` was kept in the `DTRACE_PROBE5` call (the SDT macro
   needs five data args) and dropped from the in-process dispatch call.

3. **`state_->staging` is populated with explicit `make_unique`.** The plan's
   `staging.resize(numInputs)` on a `vector<unique_ptr<StagingSlot>>`
   default-constructs **null** pointers, so `*state_->staging[i]` would have
   been a null dereference on the first unplanned forward. The constructor
   reserves and pushes a fresh slot per input position instead.

4. **The run recipe adds `handle_segv=0` to `ASAN_OPTIONS`.** Without it, ASan
   catches the guard-page SIGSEGV and reports `DEADLYSIGNAL` instead of the
   outcome contract's raw `$? == 139`. Both behaviors were observed; the recipe
   documents the option so the evidence block shows the documented signal.

5. **Harness input tensors need an explicit keepalive.** In exec_aten mode an
   `EValue`'s `Tensor` is a raw `TensorImpl*` into the `Storage` owned by the
   aliasing `shared_ptr` created by `from_blob` — the EValue holds no
   refcount. The first harness draft returned only the `EValue` vector and the
   storage died before `forward()` (use-after-free, caught by the QA ASan run
   in `TensorImpl::internal_resize_contiguous`). The harness now carries a
   `HostTensors { keepalive, evalues }` pair, honoring the same lifetime rule
   `EtRuntime::forward` already obeys (its `tensors` vector outlives
   `module.forward()`).

Minor, behavioral no-ops: the plan's prose said "6 new Catch2 cases" but the
stage-vs-pass-through coverage was written as two `TEST_CASE`s (one per model),
so seven landed; and the shim for the Gradle runs was built with a direct CMake
configure against the pinned tarball because `native/build.sh` is
container-only (root-owned `/opt/corretto`, `/workspace` paths).

## Manual W4 instructions (user-executed)

Build the harness (QA tree, no JDK):

```bash
cmake --build native/asan --target et_overread_harness
```

Fixtures are already committed under `native/spike/`: `clamp5.pte` (Route A),
`lin129.pte` (Route B), `lin129_planned.pte` (config b). Run from the repo
root. Every qemu run needs `ASAN_OPTIONS=detect_leaks=0:handle_segv=0` (LSan's
ptrace scan fatally errors under qemu-user; `handle_segv=0` surfaces the fault
as the raw 139) and a fresh gdb port per run.

- **Route A** (f32, forced SSE):
  `ASAN_OPTIONS=detect_leaks=0:handle_segv=0 qemu-x86_64 -cpu Nehalem
  ./native/asan/et_overread_harness borrowed native/spike/clamp5.pte` →
  **expect SIGSEGV (139)**: Nehalem → sse2 →
  `xnn_f32_vclamp_ukernel__sse2_u8`, whose tail load is `xnn_load_tail_f32`
  (`f32-sse2-base.h:195`, `XNN_OOB_READS`, full `_mm_loadu_ps`) reading 12
  bytes past the 20-byte buffer (`N % 4 == 1`). Already reproduced in the
  implementer smoke (see verification above); record it as the dated evidence.
- **Route B** (plain f32 Linear, AMD Zen):
  `ASAN_OPTIONS=detect_leaks=0:handle_segv=0 qemu-x86_64 -cpu EPYC
  ./native/asan/et_overread_harness borrowed native/spike/lin129.pte` →
  **expect SIGSEGV (139)**. **`-cpu EPYC` (Naples, Zen 1), NOT `-cpu
  EPYC-Rome`** — under EPYC-Rome cpuinfo reports `xnn_uarch_zen2` (0x20010A),
  which misses the gemm-config `case xnn_uarch_zen:`/`dhyana:` branch
  (`gemm-config.c:933-973`) and selects the safe default kernels; a negative
  from EPYC-Rome is void. Under `-cpu EPYC` cpuinfo reports `zen` (0x200109)
  and gdb confirms `xnn_f32_gemm_minmax_ukernel_1x16s4__fma3_broadcast` (the
  annotated `XNN_OOB_READS` kernel) executes — K=129 (`K % 4 == 1`) over-reads
  the activation (borrowed input) by 12 bytes.
- **Config (b)** (stock ExecuTorch arena):
  `ASAN_OPTIONS=detect_leaks=0:handle_segv=0 qemu-x86_64 -cpu EPYC
  ./native/asan/et_overread_harness arena native/spike/lin129_planned.pte` →
  placement line first (input last in its buffer: yes/no), then fault-or-clean.
  If "yes" + SIGSEGV → re-verify against ExecuTorch `main`
  (`~/workspace/executorch`, tag v1.3.1 is not upstream's tip) before filing
  Claim 2 (§3/W4 of the brief). If "no" or clean → record "the planner does
  not place an XNNPACK external input last" and close the concern; do NOT file.
- **Kernel observation** per run (the evidence block is unusable without the
  selected-kernel line): gdb under qemu — terminal 1:
  `ASAN_OPTIONS=detect_leaks=0:handle_segv=0 qemu-x86_64 -g 12345 -cpu EPYC
  ./native/asan/et_overread_harness borrowed native/spike/lin129.pte`;
  terminal 2: `gdb -batch -ex 'target remote :12345' -ex 'break
  xnn_f32_gemm_minmax_ukernel_1x16s4__fma3_broadcast' -ex continue -ex 'bt 4'
  ./native/asan/et_overread_harness` (Route A: break
  `xnn_f32_vclamp_ukernel__sse2_u8`). On real Zen hardware instead:
  `perf record -F 999 -- ./native/asan/et_overread_harness borrowed
  native/spike/lin129.pte && perf report --stdio | grep ukernel` — the top
  symbol is the selected kernel (ukernel symbols are global in the linked dist
  lib). The user's `perf_users` group wrapper (`/usr/bin/perf`,
  root:perf_users) covers `perf_event_paranoid=4`.
- **Manual re-run with staging enabled** is a nice-to-have confirmation only —
  never a regression gate. Remember the harness's borrowed mode deliberately
  bypasses `EtRuntime::forward`, so the "with staging" run means running the
  real engine path (`et_leak_harness`/JNI) rather than this harness.

Record each run as a dated evidence block in the brief's §8 (W4 evidence log),
using the existing template: date, exact command, model, N/K, the kernel
XNNPACK actually selected, and the fault or its absence. A negative from any
route does not license dropping the staging padding (§3/W4 of the brief).

---

## Independent review, 2026-08-05

Review of the five commits above against the requirements in
`docs/executorch-host-buffer-contract-brief.md`. Read the code rather than this
record; built a reproducer; re-ran the harnesses. **Nothing below is fixed as of
this writing.**

### Verified as claimed

Re-derived independently, not taken from the verification section above:

- USDT probes present in both the QA harness and the **shipped shim**
  (`native/build/libexecutorch_djl.so`), provider `measly`, names
  `staging_grow` / `staging_input`, `Semaphore: 0x0` — matching the dist's
  `etnp::lstm` convention as required.
- Leak-harness exact counts reproduce: `add_unplanned.pte 50 4` →
  `grow=100 staged=400 total=400`; `add_unplanned.pte 1 1000` →
  `grow=2 staged=2000 total=2000`; `add.pte 20` → `grow=0 staged=0 total=160`.
- `GuardedRegion`'s page arithmetic is correct, including the interaction with
  mmap's own rounding of `map_len_`: for both `size % page == 0` and `!= 0`,
  `PAGE_ALIGN(guard_start + guard_len)` lands exactly on the end of the rounded
  mapping, so the `mprotect` cannot ENOMEM off the end.
- Config (b)'s plumbing is sound: `load_method(name, HierarchicalAllocator*)` is
  the correct v1.3.1 API (`module.h:281`), and `Module::execute` reuses an
  already-loaded method, so the subsequent `mm.forward()` does run against the
  guarded arena rather than silently re-planning.
- The Java dtype surface does bound `dtypeSize`'s input today: `EtDataTypes`
  fails fast outside the 7 supported codes, and `EtSymbolBlock` rejects a
  dtype/meta mismatch before the JNI call.

### F1 — W7 replaced a validated copy with an unvalidated one (confirmed)

**`native/core/et_runtime.cpp:117`.** The staging memcpy's length `actual` is
computed entirely from the **caller-supplied shape**. It is checked against
neither the source buffer's extent nor the model's declared shape — even though
`MethodMeta.inputShapes` already carries the latter, populated at load.

Before W7 this input never reached a copy. `Method::set_input`
(`method.cpp:1199-1255`) validates `scalar_type` and calls `resize_tensor`
*before* dispatching to `copy_tensor_data` / `share_tensor_data`, so a bad shape
returned a clean error. W7's memcpy now runs before `module.forward()` is
entered at all.

Reproducer — two inputs on `add_unplanned.pte`, input 0 declaring shape `{64}`
over a 1-element heap allocation, built against the ASan QA tree:

```cpp
EtRuntime rt("native/spike/add_unplanned.pte");
auto* a = new float[1]{2.0f};
auto* b = new float[1]{3.0f};
std::vector<InputDesc> inputs = {{a, {64}, 6}, {b, {1}, 6}};  // shape lies
rt.forward(inputs);
```

```
==1292117==ERROR: AddressSanitizer: heap-buffer-overflow
READ of size 256 at 0x5020000000f4 thread T0
    #0 memcpy
    #1 measly::et::EtRuntime::forward(...) native/core/et_runtime.cpp:117
    #2 main probe_shape.cpp:15
allocated by thread T0 here:
    #0 operator new[](unsigned long)
    #1 main probe_shape.cpp:11
```

The identical call against `add.pte` (planned → pass-through) yields
`threw: EtRuntime: forward() failed`. Same lie, clean error on one path and a
256-byte over-read on the other — and the over-read is on the path W7 exists to
make safe.

**Reachability.** Not through `EtSymbolBlock` / `Predictor`: shape and buffer
both derive from one `EtNDArray`, so they cannot diverge. It *is* reachable
through the public `EtNative.forward` with a hand-built `EtTensor` (public
class, public final fields, public constructor), and through direct C++
`EtRuntime` consumers — a consumer class CLAUDE.md names explicitly ("linked by
the shim, the Catch2 unit tests, and the leak harness alike"). So: not a live
DJL exploit, but the JNIEnv-free core's API contract is now weaker than it was
before this change.

### F2 — Unlisted deviation: slots are sized lazily, not at load

The brief's W8 is explicit that `TensorInfo::nbytes()` is available at load for
planned *and* unplanned inputs, so "the slots can be sized once in
`methodMeta()` and never resized — 'grow-only' degenerates to 'allocate once'."
The implementation instead sizes each slot at first `forward()` from the
caller's shape. `buildMethodMeta` reads `info->sizes()` but never captures a
byte count.

**This is not among the five deviations recorded above**, and it has two
consequences beyond being the proximate cause of F1:

- **It inverts `staging_grow`'s designed role.** The brief specifies an
  *anomaly detector*: "fires once per slot, ideally never after — a fire means
  an input exceeded its declared bound." As built it fires once per slot per
  load as normal operation, and `et_leak_harness` now *asserts* that it does
  (`grow == numUnplanned × outerIters`). A genuine dynamic-shape overflow is
  indistinguishable from routine first touch. Sizing at load restores the
  intended semantics and makes the steady-state assertion the stronger
  `grow == 0`.
- With slots sized from declared metadata, the caller's shape becomes something
  to validate against a known bound rather than something to trust — which is
  exactly what F1 needs.

### F3 — Non-tensor inputs are classified as unplanned and staged

`inputMemoryPlanned[i] == 0` conflates "borrowed tensor" with "no `TensorInfo`
exists". The brief's W2 section calls the flag "meaningless rather than false"
for non-tensor inputs; W7 now branches on it as though it were false, staging
such an input at `dtypeSize(-1)` → the `default: return 4` fallback.

There is also a live inconsistency between the core and its own gate:
`et_leak_harness` computes `numUnplanned` only over inputs with
`inputScalarTypes[i] >= 0`, while `forward()` stages regardless of tensor-ness.
On a model with a non-tensor input the exact-count assertion fails spuriously.
No fixture exercises this, so it is latent.

### F4 — Nothing demonstrates the padding is *sufficient*

Deviation 1 (borrowed mode bypasses `EtRuntime::forward`) is correct and
necessary — concur with the reasoning. But the consequence is that every W4
route proves *necessity* and none proves *sufficiency*: no run exercises the
engine path under a hostile uarch. The manual instructions gesture at this
("means running the real engine path") without giving a command.

One is already viable and was verified to run natively during this review:

```bash
qemu-x86_64 -cpu EPYC ./native/asan/et_leak_harness native/spike/lin129.pte 1 2
```

Natively that reports `grow=1 staged_input=2 total_input=2`. Under `-cpu EPYC`
it drives the annotated `1x16s4__fma3_broadcast` kernel through the staging
path, and should be **clean** where `borrowed lin129.pte` faults. That is the
sufficiency arm for the cost of one line in the recipe.

### F5 — Config (b)'s negative is over-generalized

The manual instructions say: if "no" or clean → *"record 'the planner does not
place an XNNPACK external input last' and close the concern; do NOT file."*
That is n=1 — one fixture, one planned buffer of 848 B — written up as a general
claim about the planner. It is the same unqualified negative the brief forbids
for Route A ("record it as such rather than as a clearance").

Cheaper and stronger than more fixtures: the planner packs by lifetime and graph
inputs have the earliest start, so they should structurally tend to be placed
*first*. Establishing that from `exir/passes/memory_planning_pass.py` would
settle "ever" in a way no number of fixtures can.

### Minor

- **`dtypeSize`'s `default: return 4` is now load-bearing for a memcpy length.**
  Bounded today by `EtDataTypes`' fail-fast over 7 codes, but `dtype_size.h`
  still documents itself as sizing harness buffers. Adding FLOAT16 (code 5) to
  `EtDataTypes` later would silently make the staging memcpy read 2× the
  buffer. Worth a hard reject on unknown codes now that it is in the production
  path.
- **`StagingSlot::ensure` ignores allocation failure** — `memcpy(fresh, …)` with
  `fresh == nullptr`. The comment says "never fails except via allocation
  failure"; it does not fail, it segfaults.
- **`ET_PROBE_*` macros lack `do { … } while (0)`** — they expand to two
  statements, so `if (cond) ET_PROBE_STAGING_GROW(...);` would run the dispatch
  unconditionally. Every current call site is braced, so this is latent.
- `ensure()` preserving the old bytes on growth is dead work in the forward path
  (immediately overwritten by the staging memcpy). Harmless.
- The brief's status table marks W4 **complete** with "manual runs pending"
  while deliverable 3 (dated evidence) is unmet and §8's log is empty. It is
  annotated, so it is honest — "harness complete, result pending" would read
  truer.

### Suggested fix order

1. **F2** — size slots at load from `inputShapes` / `nbytes`. This closes F1 as
   a side effect, restores `staging_grow`'s anomaly semantics, and flips the
   harness assertion to the stronger steady-state `grow == 0`.
2. **F1** — reject (or clamp) a caller shape exceeding the declared bound, with
   a regression test built from the reproducer above.
3. **F3** — branch on tensor-ness as well as the planned flag, and align the
   harness's expected-count formula with whatever `forward()` does.
4. **F4 / F5** — two additions to the manual recipe: the sufficiency command,
   and qualifying the config (b) negative as fixture-specific.

---

## F2 fixed, 2026-08-05

Slots are now sized at load from the model's declared bound, and `staging_grow`
is back to being an anomaly detector. Uncommitted; `F1`, `F3`, `F4`, `F5` remain
open.

**Correction to the review above.** It claimed sizing at load "closes F1 as a
side effect." **That is wrong** — verified by re-running the F1 reproducer
against the fixed core:

```
READ of size 256 at 0x502000000114
    #1 measly::et::EtRuntime::forward(...) native/core/et_runtime.cpp:138
```

F2 makes the declared bound *available* at the copy site; the memcpy still
takes its length from the caller's shape. Closing F1 is the separate explicit
check, now a three-line follow-up rather than a design change.

### Changes

- **`et_runtime.h`** — `MethodMeta` gains `std::vector<size_t> inputNbytes`,
  the declared byte count per input (0 for non-tensor). Exact for a static
  shape, an upper bound for a dynamic one.
- **`et_runtime.cpp`** — `buildMethodMeta` captures `info->nbytes()`. The
  constructor sizes each staged slot to `inputNbytes[i] + kStagingPadding`,
  gated on `inputMemoryPlanned[i] == 0 && inputScalarTypes[i] >= 0`. The
  tensor-ness half of that condition is deliberate and commented: only a real
  tensor has a bound to size from, and `forward()` still branches on the
  planned flag alone, so a non-tensor input grows on first use instead of being
  rejected — F3, unchanged by this commit.
- **`et_runtime.cpp` / `forward()`** — the staging branch now uses `slot.data()`
  directly and only calls `ensure()` + fires `staging_grow` when
  `actual + kStagingPadding > slot.capacity()`.
- **`staging.h`** — `ensure()` throws `std::bad_alloc` on allocation failure
  instead of memcpy'ing through null. In scope because load-time sizing makes
  the allocation eager and potentially large; this was a "minor" in the review.
- **`et_leak_harness.cpp`** — `expectGrow` is now the constant `0`, for any
  number of loads or forwards. Strictly stronger than the old
  `numUnplanned × outerIters`: it catches both a realloc-per-forward slip and a
  regression to sizing from the caller's shape.
- **`build_qa.sh`** — comments updated to the new expectation.
- **`et_runtime_test.cpp`** — the stage-vs-pass-through case now asserts
  `grow == 0`; three new cases: declared byte counts captured at load, 100
  forwards with `grow == 0` and `staged == 200`, and the anomaly path firing
  `staging_grow` exactly once.

### A nuance the anomaly test exposed

The probe's threshold is the declared bound **plus the padding**, not the
declared bound. A 1-float slot holds `4 + 128` rounded to 192 bytes, so an input
must exceed 192 — not 4 — before `staging_grow` fires. The first draft of the
anomaly test used 16 floats (`64 + 128 = 192`) and failed with `0 == 1`; it uses
64 floats now. This is correct behavior (the slot genuinely has the room) but it
means the probe cannot detect a *small* overrun of the declared bound. If that
matters later, compare against `inputNbytes[i]` explicitly rather than against
capacity — which is the same check F1 needs.

### Verification

- `et_runtime_test`: **18 cases, 222 assertions, all pass**.
- Leak harness, all `grow=0 (expected 0)`:
  `add_unplanned.pte 50 4` → `staged=400 total=400`;
  `add_unplanned.pte 1 1000` → `staged=2000 total=2000`;
  `add.pte 20` → `staged=0 total=160`;
  `lin129.pte 1 2` → `staged=2 total=2`.
- Shim relinked; XNNPACK post-link assertion passed (2653 `xnn_*` text symbols).
- `./gradlew test --rerun-tasks` → BUILD SUCCESSFUL;
  `./gradlew leakTest --rerun-tasks` → `tests="3" skipped="0" failures="0"`.
