# Research Brief: The ExecuTorch Host-Buffer Contract

**Status:** Proposal for feasibility assessment, opened 2026-08-04. This brief
originated from reading `docs/iree-lessons-learned/2026-08-04-borrowed-host-buffers-findings.md`
— the completed zero-copy spike on the sibling `djl-iree-engine` — and asking
which of its findings transfer here. Several do. The most important one
**inverts**, and that inversion is the reason this brief exists rather than a
one-line "already handled" note.

**Assume the reader has deep context on `djl-executorch-engine` and its
history.** Unlike the IREE brief, this one opens with W1 already complete: the
audit was performed on 2026-08-04 and its answer is recorded below with
evidence. What remains open is what to *do* about it.

**Progress as of 2026-08-05.** W1, W2, and W3 are complete and on `main`. W4 (harness + fixtures + run
recipe), W7 (staging), and W8 (probes) landed in code on 2026-08-05; W4's manual runs are pending
and their evidence goes into §8. The unplanned-input fixture that W7 named as a prerequisite now
exists in both its non-delegated (`add_unplanned.pte`) and delegated (`clamp5.pte`, `lin129.pte`,
`lin129_planned.pte`) forms.

| Item | State |
|---|---|
| W1 — audit | **complete** 2026-08-04 |
| W2 — make the copy observable | **complete** — `is_memory_planned()` plumbed through to Java and asserted (#20) |
| W3 — correct the documented contract | **complete** — live surfaces corrected (`20768a0`), one residual noted in §3/W3 |
| W7 fixture prerequisite | **complete** — `add_unplanned.pte` (`7eed3b8`) plus delegated `clamp5.pte`/`lin129.pte`/`lin129_planned.pte` (`export_w4_models.py`); leak harness wired with exact-count staging assertions |
| W4 — over-read confirmation | **harness complete, result pending** — `native/harness/et_overread_harness.cpp` covers both configurations and builds in the QA tree, never CI; deliverable 3 (dated evidence) is unmet and §8's log is empty, so the *question* W4 asks is still open |
| W7 — grow-only per-slot staging | **complete** — `native/core/staging.h` + `forward()` integration, in-repo coverage (§3/W7) |
| W8 — USDT probes and leak-test coverage | **complete** — `native/core/et_probes.h`, exact-count assertions in `et_leak_harness` + `build_qa.sh` |
| W5 — establish the cost | **harness + recipe complete, runs pending** — edits on `feature/w5-w6-direct-outputs` (W5-1..W5-3); run recipes in §3/W5, evidence log in §8/W5 |
| W6 — direct-buffer outputs | **complete (prototype, decision pending on W5 numbers)** — W6-1..W6-4 on `feature/w5-w6-direct-outputs` |
| W9 — shared aligned-buffer abstraction | open |

---

## 1. Context

### The finding that transfers, and the finding that inverts

The IREE spike asked "does the engine copy host data per inference call, and can
it borrow instead?" It answered: IREE copies unless the host pointer is 64-byte
aligned, the outcome is *observable* (`WRAPPED` vs `STAGED`), and — crucially —
a misaligned pointer is **refused** at import time. IREE's conclusion was that
misalignment is "a zero-copy miss, not a fault," which bounded the entire hazard
class.

ExecuTorch has no such refusal. That single difference reverses the risk
profile, and it is the core of this brief.

### The original claim, and its correction

**Corrected 2026-08-04 by W3; recorded here as the finding, not as live text.**
`native/core/et_runtime.h:13` used to state:

```cpp
// Borrowed input: data is a host pointer the caller keeps valid across forward(). Zero-copy in.
```

CLAUDE.md repeated it ("Zero-copy in (borrowed input pointers), single-copy
out"), as did at least six design docs under `docs/superpowers/`. The claim is
accurate about what *this engine* does and false about what *ExecuTorch* does
with the pointer it is handed. Both of those surfaces now say the true thing —
see W3.

`EtRuntime::forward` (`native/core/et_runtime.cpp:70-76`) builds `from_blob`
tensors over the caller's pointers and calls `module.forward(evalues)`. That
reaches `Module::execute` (`extension/module/module.cpp`), which calls
`Method::set_input` per input, which branches
(`runtime/executor/method.cpp:1244-1255`):

```cpp
auto tensor_meta = this->method_meta().input_tensor_meta(input_idx);
if (tensor_meta->is_memory_planned()) {
  internal::copy_tensor_data(t_dst, t_src);   // memcpy into the arena
} else {
  internal::share_tensor_data(t_dst, t_src);  // a real borrow: set_data(ptr)
}
```

`is_memory_planned()` is a property baked into the `.pte` at export time by
`MemoryPlanningPass(alloc_graph_input=...)`, whose default is `True`
(`exir/passes/memory_planning_pass.py:151`). `tools/scripts/export_mobilenet.py:64`
calls bare `.to_executorch()` and inherits that default.

**So: every `.pte` in this repository causes ExecuTorch to memcpy each input on
every `forward()`.** Our `from_blob` pointer is borrowed only for the duration
of that copy. Nothing logs this and no test asserts it — unlike IREE, the copy
is completely invisible.

Verified 2026-08-04 against a source checkout at `~/workspace/executorch`, tag
`v1.3.1` (exactly our pinned runtime version), using its `.venv`:

```
native/spike/add.pte     num_inputs=2
   input 0 memory_planned=True
   input 1 memory_planned=True

fresh export, default                 -> input0 memory_planned = True
fresh export, alloc_graph_input=False -> input0 memory_planned = False
```

Both directions proven on real artifacts. The export config is the lever.

### What genuinely differs between the two runtimes

| | IREE | ExecuTorch |
|---|---|---|
| Borrow decision | runtime, per allocator compatibility | **export time**, baked into the `.pte` |
| Observable? | yes — `WRAPPED` / `STAGED` | no — nothing surfaces it |
| Misaligned pointer | **refused** (`IREE_STATUS_OUT_OF_RANGE`) → staged copy | **accepted silently** |
| Alignment contract | 64 B, documented (`IREE_HAL_HEAP_BUFFER_ALIGNMENT`) | none at the borrow site; allocator default is `alignof(void*)` = 8 |
| Over-read past buffer end | none | XNNPACK reads up to `XNN_EXTRA_BYTES` = 16 past the data |
| Borrow lifetime | released at end of invoke | pointer **retained** in the `Method` after `execute()` returns |

`share_tensor_data`
(`runtime/core/exec_aten/util/tensor_util_portable.cpp:140-163`) checks only
`nbytes` equality and non-null, then:

```cpp
t_dst.unsafeGetTensorImpl()->set_data(t_src_data_ptr);
```

No alignment check. No padding check. No refusal path at all.

And XNNPACK documents an out-of-bounds **read** (`xnnpack.h:24-32`):

> The number of bytes XNNPACK may read beyond array bounds. The caller must
> allocate at least this many extra bytes after the tensor data passed to
> XNNPACK. `XNN_EXTRA_BYTES 16` (128 on Hexagon). Note: XNNPACK reads, but never
> writes beyond array bounds.

Nothing in ExecuTorch's own `backends/xnnpack/` references `XNN_EXTRA_BYTES` —
only XNNPACK's vendored `bench/` and `test/` files do. Today the memory-planned
copy masks this entirely. It stops being masked the moment anyone acts on the
"we borrow user buffers" claim.

### The IREE usage-style constraint, re-examined

IREE's spike landed on "the engine allocates; the user writes into what the
engine hands back," implemented as a 64-byte-aligned allocator behind
`-Diree.engine.alignedBuffers`. **That prototype does not port.** While inputs
are memory-planned, ExecuTorch copies regardless of how well the source was
aligned — an aligned allocator here buys exactly nothing.

Nor is the lever an export-time config users must apply. An earlier draft of
this brief assumed it was, and treated the resulting "please re-export your
models" as a first-class hazard. That was wrong. The correct framing is
narrower and lands entirely inside the engine: when an input is *not* memory-
planned we have no choice about the borrow — `share_tensor_data` takes whatever
pointer we hand it — so the only question is **whose** buffer ExecuTorch
borrows. Answering "ours, padded and aligned" is W7, needs nothing from the
user, and works for either artifact mode.

---

## 2. What this brief is trying to answer

Five questions, in order:

1. **Does the engine copy, and where?** — *Pre-answered (W1).* It copies, in
   three places, one of which is invisible and none of which are where the docs
   say. See §3/W1.
2. **Can the unplanned-input case be made safe without asking users to change
   their models?** — *Answered by design (W7).* Yes: stage every unplanned
   input into a padded, engine-owned, grow-only per-slot buffer. We do not get
   to decline the borrow, so the only lever is *whose* buffer ExecuTorch
   borrows. What remains is confirming the padding is necessary and sufficient
   (W4), not deciding whether to ship.
3. **What does the current path actually cost?** Both the invisible ExecuTorch
   input copy and the heap `byte[]` output copy need numbers before any user-
   facing constraint is justified.
4. **Is the output path the better target?** It is engine-controlled end to end
   and independent of the input question. It may still be where most of the
   measurable win is.
5. **Is a shared aligned-buffer abstraction worth extracting** across this
   engine and `djl-iree-engine`? IREE's spike already answered "duplicate" for
   its side; this is the confirmation pass, not a reopening.

---

## 3. Work items

### W1 — Audit the host buffer path

Determine what the engine does today on input and output: copy, borrow, or
mixed, and whether ExecuTorch honors the borrow.

*Answers:* whether anything below is worth doing. This gates everything.

**Status: COMPLETE (2026-08-04).** It copies. Inventory:

| Step | Copy? |
|---|---|
| `EtNDManager.create` → `allocateDirect` + `copyInto` (`EtNDManager.java:60-61`) | **copy 1** (user data → direct buffer) |
| `manager.from()` / `toByteBuffer()` (`EtSymbolBlock.java:48-55`) | none for an `EtNDArray`; full copy otherwise |
| JNI `GetDirectBufferAddress` (`executorch_djl_jni.cpp:158`) | none |
| ET `set_input` → `copy_tensor_data` | **copy 2 — invisible, undocumented** |
| JNI out: `allocOutputBuffer` → `NewDirectByteBuffer` (`executorch_djl_jni.cpp`; pre-W6: `NewByteArray` + `ByteBuffer.wrap`) | **copy 3, into a JNI-allocated direct buffer, freed by a Cleaner (pre-W6: onto the JVM heap)** |

Copy 3 has a second-order cost the IREE engine does not have: the returned
buffer is a **heap** `byte[]`, not direct. Chaining model A → model B therefore
re-enters the `!buf.isDirect()` branch at `EtSymbolBlock.java:56` and copies a
fourth time. It also puts every output on the JVM heap — noise for MobileNet's
4 KB logits, sustained GC pressure for a segmentation head (a 1×21×512×512 f32
output is 21 MB per inference).

### W2 — Make the copy observable

Plumb `TensorInfo::is_memory_planned()` (public, `runtime/executor/method_meta.h:63`)
through `MethodMeta` → `EtMethodMeta` and log it at model load. `EtRuntime::methodMeta()`
already holds the `input_tensor_meta(i)` it needs at `native/core/et_runtime.cpp:48`,
so this is a single added field. Assert the expected value in the existing model
tests.

*Answers:* nothing by itself — it is the instrument every other item reads. This
is the ExecuTorch analogue of IREE's `WRAPPED`/`STAGED` signal, whose absence is
why this went unnoticed for the engine's whole life.

**Status: COMPLETE (2026-08-05, PR #20 `3c26ad6`).** As shipped:

- `RuntimeState`/`MethodMeta` carries `std::vector<uint8_t> inputMemoryPlanned`,
  filled from `info->is_memory_planned()` in `EtRuntime::methodMeta()`
  (`native/core/et_runtime.cpp:47,54`). Non-tensor inputs keep `0` — there is no
  `TensorInfo` for them, so the flag is meaningless rather than false.
- Marshalled across JNI as a `boolean[]` (`executorch_djl_jni.cpp:164`) into
  `EtMethodMeta.inputMemoryPlanned` (`EtMethodMeta.java:13`).
- Logged per input at model load: `EtModel.java:60` emits
  `model {} input {} memoryPlanned={}`.
- Asserted at all three layers: `native/test/et_runtime_test.cpp:34-51` (Catch2,
  both directions), `EtMethodMetaTest` (both directions), and
  `EtModelTest.java:76`.

W7 can branch per input on this flag as designed; nothing further is needed from
this item.

### W3 — Correct the documented contract

Fix `native/core/et_runtime.h:13`, CLAUDE.md, and the design docs under
`docs/superpowers/` to say what is true: *borrowed by this engine, subject to
the model's memory plan; copied by ExecuTorch for any memory-planned input,
which is the export default.*

*Answers:* nothing measurable. It stops the false claim from being built on
again, which is how this became load-bearing in six documents.

**Status: COMPLETE (2026-08-04, `20768a0`).** Corrected in the three live
surfaces — `native/core/et_runtime.h` (comment-only; `struct InputDesc`
untouched), `CLAUDE.md`, and `docs/benchmarking.md` — each now stating the
memory-plan branch explicitly and pointing here.

**Residual, deliberately not fixed.** Five *tracked* dated docs under
`docs/superpowers/` still carry "zero-copy in": the phase1 design, the phase2a
design and its plan, the mobilenet-example benchmark design, and the
pytorch-free preprocessing design. These are records of what was believed and
decided on their dates, not live contract statements, so they were left intact
rather than retro-edited. Anyone reading them for the input contract should be
sent here instead. Flagging it so the omission reads as a call rather than a
miss.

### W4 — Over-read confirmation, run manually

Determine whether XNNPACK's documented over-read is reachable through a
borrowed, exact-sized host buffer.

**Demoted 2026-08-04 from "the decisive gate" to a confirmation test.** W7
stages every unplanned input into a padded, engine-owned buffer, so the
over-read cannot reach a caller's buffer on any microarchitecture regardless of
what this experiment finds. W4 does not decide whether anything ships; it
establishes that W7's padding is *necessary* rather than cargo-culted, and it
produces a reproducer that keeps the question closed.

#### Revised 2026-08-05: this is a manual step, not a repo test

**It cannot be a regular test, because the hardware that would fail it is not
the hardware we run tests on.** Every route below needs a microarchitecture that
selects an `XNN_OOB_READS`-annotated kernel for the *first op touching the graph
input*. This dev box (Tiger Lake) selects `avx512f_broadcast` for f32, which is
masked and architecturally safe, and the qs8 paths that do over-read on AVX-512
touch XNNPACK-internal int8 buffers rather than ours (§ the disproven sub-route
below). So reaching an annotated kernel means either **CPUID masking under
`qemu-x86_64`** or **real AMD Zen 1–3 hardware**. Neither belongs in
`./gradlew test` or the Catch2 suite.

What that implies, concretely:

- **The harness source still lands in-repo** — a small standalone binary under
  `native/harness/`, alongside `et_leak_harness`, buildable by the existing QA
  path with no JDK. What does *not* land is an assertion that runs
  unconditionally. Committing it is what makes the result re-derivable later
  instead of a paragraph someone has to take on faith.
- **The run is manual and its output is evidence**, recorded back into this
  brief: the date, the exact command, the model, `N`/`K`, **the kernel XNNPACK
  actually selected** (not the one predicted), and the fault or its absence. A
  result without the selected-kernel line is not usable — see the negative-result
  caveat below.
- **Prefer real Zen hardware to emulation if a Zen instance is available.**
  Route B on native Zen is a stronger result and sidesteps the open question
  qemu introduces: whether XNNPACK's cpuinfo-based uarch detection resolves to
  `xnn_uarch_zen` under qemu-user at all. Until that is confirmed, a negative
  from qemu carries no information.

**Consequence for W7 — the important part.** An earlier draft said "the same
harness re-run with staging enabled is W7's regression test." **That no longer
holds.** If W4 is manual, so is its re-run, and W7 would then ship with no
in-repo guard on the property it exists to provide. W7's coverage must come from
mechanisms that run anywhere, on any uarch:

- a direct assertion that each staging slot is 64-byte aligned and over-allocated
  by the padding constant (a unit test on the allocator, no XNNPACK involved);
- W8's `staging_input` / `staging_grow` probes, which cover the
  realloc-per-call failure mode.

Those are what police the invariant day to day. W4's job shrinks to **justifying
the constant once, with evidence** — and the manual re-run under staging is a
nice-to-have confirmation, not the regression test. Run W4 before W7 so the
justification exists before the code does.

**Revised 2026-08-04 after reading the XNNPACK sources.** The original framing of
this item ("run it under ASan") was wrong and would have produced a false
negative. Three findings change the design:

**The over-read is declared, not accidental.** `XNN_OOB_READS`
(`src/xnnpack/common.h:320`) annotates the affected microkernels — 365 files in
`qs8-qc8w-gemm/gen`, 297 in `f32-dwconv2d-chw/gen`, 160 in `f32-vbinary/gen`,
and more. So this is a contract XNNPACK states and expects callers to honor, not
a latent bug we would be discovering.

**ASan cannot see it.** That macro expands to
`XNN_DISABLE_TSAN XNN_DISABLE_MSAN XNN_DISABLE_HWASAN XNN_DISABLE_ASAN`, each of
which is `__attribute__((__no_sanitize__("address")))` plus
`XNN_NO_INLINE_SANITIZER`. A sanitizer run over these kernels reports nothing by
construction. **The guard page is the only viable detector** — it is a hardware
mechanism the annotation cannot suppress. Do not substitute ASan for it.

**ExecuTorch does not pad.** `backends/xnnpack/runtime/XNNExecutor.cpp`,
`prepare_args`, sets `externals_[i].data = tensor->mutable_data_ptr<float>()` —
the raw pointer, unpadded. XNNPACK's own harness allocates every external value
as `malloc(size + XNN_EXTRA_BYTES)` for the identical role
(`bench/subgraph/benchmark.cc:91`). The violation is *mostly* harmless today
because memory-planned inputs sit inside the arena, a large contiguous block
where a 16-byte over-read lands in a neighbouring planned tensor. A fault
requires the tensor at the end of a mapping — exactly what borrowing an
exact-sized user buffer creates.

**Open question, and it may make this a live upstream bug.** "Mostly" is doing
work in that sentence. `Module` allocates each planned buffer at *exactly*
`meta.memory_planned_buffer_size(i)` with no slack — re-verified 2026-08-05 at
`v1.3.1`: `module.cpp:336` is `planned_buffers.emplace_back(size)`, a
`std::vector<uint8_t>` sized exactly, and `XNNExecutor.cpp:97` is
`externals_[i].data = tensor->mutable_data_ptr<float>()`, raw and unpadded. So a
tensor that memory planning places **last in the arena** is over-read past the
end of a heap allocation — today, on stock ExecuTorch, with no borrowing
involved. It would rarely fault (malloc leaves slack and metadata after the
block) and ASan cannot see it (`XNN_OOB_READS`), which is consistent with nobody
having reported it.

UNVERIFIED, and this is the gate on the upstream report below: whether the
planner ever actually places an XNNPACK external input at the arena end — it
packs by lifetime, so it is plausible but not established. Settling it changes
this from "a hazard we would introduce" to "a defect we found." Cheap to test:
the same guard-page harness, pointed at a memory-planned model, with the arena
allocator swapped for a guarded one — that is configuration (b) below.

**Mechanism and shape rule.** The tail path is ISA-dependent. AVX
(`src/xnnpack/simd/f32-avx-base.h:172`) uses `_mm256_maskload_ps` and is
architecturally safe. SSE2 (`src/xnnpack/simd/f32-sse2-base.h:195`) is not:

```c
xnn_load_tail_f32(const float* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements < xnn_simd_size_f32);
  return _mm_loadu_ps(input);   // full 16 bytes regardless of num_elements
}
```

For f32/SSE the rule is **N % 4 != 0**, worst at **N % 4 == 1** (12 bytes past
the end). Annotated x86 ISAs by kernel family:

| Kernel family | Over-reads on |
|---|---|
| `f32-vbinary`, `f32-vsigmoid` | sse, sse2, sse41 only |
| `f32-gemm`, `f32-igemm` | sse, **fma3** |
| `f32-dwconv` | sse (+ neon/wasm) |
| `qs8-qc8w-gemm` | **avx512vnni, avx512skx, avx2, avx**, sse41, sse2 |

The f32 paths are safe on AVX-512 hardware; the quantized paths are not — int8
has no cheap masked byte-load on x86, so they over-read across the whole ISA
range.

**Construction — two routes, run in this order.**

*Route A (f32, forced ISA).* Single-op model (clamp or add) consuming the graph
input directly, `N % 4 == 1`, exported with `alloc_graph_input=False` so
`share_tensor_data` is taken. `mmap` a region, `mprotect(PROT_NONE)` the
following page, place the tensor so its last byte abuts the guard page. Force
SSE kernel selection with `qemu-x86_64 -cpu Nehalem` — `src/configs/hardware-config.c`
has no environment override, so masking CPUID under qemu-user is the cheap
lever and needs no runtime rebuild. Deterministic; proves the mechanism.

*Route B (plain f32 model, AMD Zen).* **Revised 2026-08-04 — the original
"statically quantized int8 model" version of this route is dead, tested and
disproven; see below.** The live route is a plain f32 `Linear`/`Conv` model
whose first op consumes the graph input, with **K % 4 != 0** (`K` = in_features,
or `C*kh*kw` for conv), exported `alloc_graph_input=False`.

The lever is microarchitecture, not ISA. `src/configs/gemm-config.c:1444-1471`
selects the f32 igemm microkernel by a priority chain (avx512f > fma3 > avx >
sse2), and inside the fma3 branch switches on uarch:

```c
case xnn_uarch_zen:
case xnn_uarch_dhyana:
    ...ukernel_4x16s4__fma3_broadcast    // ANNOTATED XNN_OOB_READS
default:
    ...ukernel_5x16__fma3_broadcast_prfm // not annotated
```

The `s4` ("shuffle 4") kernels read 4 K-elements per iteration and over-read the
A matrix — the activations, i.e. **our borrowed input for the first op** — when
K is not a multiple of 4. So:

| Hardware | f32 first-touch kernel | Over-reads |
|---|---|---|
| Intel AVX-512 (incl. this dev box, Tiger Lake) | `avx512f_broadcast` | no |
| **AMD Zen 1–3** (Ryzen 1000–5000, EPYC Naples/Rome/Milan) | **`4x16s4__fma3_broadcast`** | **yes** |
| Other FMA3 (Haswell–Skylake client/Xeon) | `5x16__fma3_broadcast_prfm` | no |
| AVX only | `avx_broadcast` | no |
| SSE2 only | `sse` | yes |

This is the IREE brief's "passes every test on the developer's machine" hazard
in its sharpest form: our dev box selects the safe kernel and a very large
production population — every Zen 1–3 cloud instance — selects the annotated
one. Emulate with `qemu-x86_64 -cpu EPYC-Rome` (Zen 2 CPUID → `xnn_uarch_zen`).
Verify that XNNPACK's cpuinfo-based uarch detection actually resolves to
`xnn_uarch_zen` under qemu before trusting a negative from this route.

**Disproven sub-route, recorded so it is not retried.** The original Route B
proposed a statically quantized int8 model reaching `qs8-qc8w-gemm` (annotated
across avx512vnni/skx/avx2/avx). Tested 2026-08-04 with the standard PT2E flow
(`XNNPACKQuantizer` + `prepare_pt2e`/`convert_pt2e`, `Linear(129, 64)`,
`alloc_graph_input=False`). Result: the whole quantize/linear/dequantize cluster
is absorbed into the delegate, and the runtime reports

```
input 0: dtype=6 (float32) sizes=(1,129) nbytes=516 memory_planned=False
```

— the graph input stays **f32** and feeds `executorch_call_delegate` directly.
The borrow works (`memory_planned=False`), but the kernel that touches our
buffer is the f32→qs8 convert, and `f32-qs8-vcvt` is annotated only on
sse2/sse41, hence safe on AVX. The qs8 GEMM over-reads an XNNPACK-internal int8
buffer, not ours. Getting an int8 *graph input* would require hand-building a
graph whose placeholder is already quantized; not worth it now that Route B has
a native f32 path.

*Answers:* two things, from one harness in two configurations (below) — whether
W7's padding is necessary, once, by evidence rather than on an ongoing basis;
and whether the arena-end over-read is a live defect in stock ExecuTorch, which
is a report we owe upstream rather than anything this repo consumes. Given the
annotations, expect positive on the first; the useful output is the
*conditions*, not the yes/no.

A negative from Route A alone is weak evidence: it proves the kernels that model
selected did not over-read on that ISA, not that none will. Record it as such
rather than as a clearance, and record the selected kernel so the claim is
checkable. **A negative does not license skipping the padding** — W7 pads
because XNNPACK's contract says to, and because the uarch table above shows the
dev box is the unrepresentative case, not the representative one.

#### Two harness configurations, one set of machinery

The guard page, the shape rule, and the forced-uarch requirement are identical
in both. Only the setup differs, and **they answer different questions** — (a)
is about us, (b) is about upstream. Build for both from the start; retrofitting
(b) later means re-deriving the allocator plumbing.

| | (a) borrowed input | (b) arena end |
|---|---|---|
| Model | unplanned (`alloc_graph_input=False`), XNNPACK-delegated | stock export defaults, memory-planned, XNNPACK-delegated |
| Buffer under test | our exact-sized host buffer, borrowed via `share_tensor_data` | ExecuTorch's own planned arena |
| Guard placement | caller's buffer abuts `PROT_NONE` page | arena allocator swapped for a guarded one; last planned buffer abuts the page |
| Proves | W7's padding is necessary | **an upstream defect in stock ExecuTorch** |
| Blocked on | nothing | the UNVERIFIED placement question above |

Configuration (b) is reachable because ExecuTorch's memory manager is pluggable:
supply a `HierarchicalAllocator` over `mmap`'d spans instead of `Module`'s
`std::vector<uint8_t>` arena, sized exactly, each followed by a guard page. If
the planner never puts an XNNPACK external input last, (b) reports nothing and
Claim 2 below evaporates — that is a real outcome, not a harness failure, and
should be recorded as one.

#### If it faults: the upstream report

A fault in configuration (b) is reportable against ExecuTorch. Keep the two
claims separate when filing, because they are not equally strong.

**Claim 1 — the borrow path has an undocumented padding requirement.**
Configuration (a). We use public API (`from_blob` → `set_input`, exported
`alloc_graph_input=False`), ExecuTorch takes our exact-sized pointer through
`share_tensor_data` with no alignment or padding check, and `prepare_args` hands
it to XNNPACK unpadded. Real, but upstream has a defensible reply: *the borrow
path implies the caller owns the padding.* Expect a documentation fix. File it,
do not lead with it.

**Claim 2 — stock ExecuTorch over-reads its own arena.** Configuration (b). No
borrowing, stock export defaults, stock `Module`, stock allocator; the user does
nothing unusual. `planned_buffers.emplace_back(size)` sizes the arena exactly and
`XNNExecutor.cpp:97` hands XNNPACK the raw pointer, so an external input placed
last is an out-of-bounds read of ExecuTorch's own heap allocation. **This is the
report** — there is no user-error escape hatch in it.

Anticipated pushback on Claim 2, and the answers:

- *"Your custom allocator faulted; ours doesn't."* True and irrelevant — stock
  `malloc` slack absorbing an OOB read does not make it defined. The guard page
  is an instrument, not the defect.
- *"ASan is clean."* ASan is blinded by construction:
  `XNN_OOB_READS` expands to `__attribute__((no_sanitize("address")))` plus
  `XNN_NO_INLINE_SANITIZER` (`src/xnnpack/common.h:288-321`). Absence of a
  sanitizer report is not evidence here, and that is precisely why this has gone
  unreported.
- *"XNNPACK's problem."* No — XNNPACK documents `XNN_EXTRA_BYTES` and honors it
  in its own harness (`bench/subgraph/benchmark.cc:91` mallocs
  `size + XNN_EXTRA_BYTES` for the identical role). The violation is
  ExecuTorch's, in `backends/xnnpack/runtime/XNNExecutor.cpp`, and that is where
  the fix belongs.

**Precondition: re-verify against ExecuTorch `main` before filing.** Everything
above is checked against `v1.3.1`, which is our pin, not upstream's tip.
Maintainers will ask, and a report against a released tag that is already fixed
on main wastes the exchange. A fix on main is also useful to us — it dates the
pin bump that would remove Claim 2 from this repo's exposure.

#### Deliverables

1. The guard-page harness committed under `native/harness/`, supporting **both**
   configurations above, not wired into any default test target.
2. A run recipe in this brief: the qemu invocation (or the Zen instance type),
   the model and its `N`/`K`, and how to read the result.
3. A dated evidence block appended to §8 recording each run: command, selected
   kernel, outcome. One block per configuration/route actually executed.
4. If (b) faults: a settled answer on the placement question, an
   ExecuTorch-`main` re-verification, and an upstream issue carrying the
   reproducer.

#### Run recipe (implemented 2026-08-05)

Build (QA tree, no JDK; the harness is a plain `et_runtime` consumer):

```bash
cmake --build native/asan --target et_overread_harness
```

Fixtures: `native/spike/clamp5.pte` (Route A), `lin129.pte` (Route B),
`lin129_planned.pte` (config (b)) — exported via
`cd native/spike && PATH=$HOME/workspace/executorch/.venv/bin:$PATH uv run export_w4_models.py`
(flatc must be on `PATH` for delegated exports; see §8). All three verified
through the pinned v1.3.1 runtime: `clamp5`/`lin129` report
`is_memory_planned() == False`, `lin129_planned` `True`, all single
`executorch_call_delegate`.

Run on the repo root as CWD. Each qemu run needs
`ASAN_OPTIONS=detect_leaks=0` (LSan's ptrace scan fatally errors under
qemu-user — verified) plus `handle_segv=0` so the guard-page fault surfaces as
the raw `$? == 139` SIGSEGV instead of ASan's DEADLYSIGNAL handler swallowing
it; and a fresh gdb port per run.

- Route A (f32, forced SSE): `ASAN_OPTIONS=detect_leaks=0:handle_segv=0
  qemu-x86_64 -cpu Nehalem ./native/asan/et_overread_harness borrowed
  native/spike/clamp5.pte` → **expect SIGSEGV (139)**: Nehalem → sse2 →
  `xnn_f32_vclamp_ukernel__sse2_u8`, whose tail load is `xnn_load_tail_f32`
  (`f32-sse2-base.h:195`, `XNN_OOB_READS`, full `_mm_loadu_ps`) reading 12
  bytes past the 20-byte buffer (`N % 4 == 1`). Verified end-to-end 2026-08-05
  in the implementer's smoke: the run faults on the input 0 guard page.
- Route B (plain f32 Linear, AMD Zen): `ASAN_OPTIONS=detect_leaks=0:handle_segv=0
  qemu-x86_64 -cpu EPYC ./native/asan/et_overread_harness borrowed
  native/spike/lin129.pte` → **expect SIGSEGV (139)**. **`-cpu EPYC` (Naples,
  Zen 1), NOT `-cpu EPYC-Rome`** — under EPYC-Rome cpuinfo reports
  `xnn_uarch_zen2` (0x20010A), which misses the gemm-config
  `case xnn_uarch_zen:`/`dhyana:` branch (`gemm-config.c:933-973`) and selects
  the safe default kernels; a negative from EPYC-Rome is void. Under `-cpu
  EPYC` cpuinfo reports `zen` (0x200109) and gdb confirms
  `xnn_f32_gemm_minmax_ukernel_1x16s4__fma3_broadcast` (the annotated
  `XNN_OOB_READS` kernel) executes — K=129 (`K % 4 == 1`) over-reads the
  activation (borrowed input) by 12 bytes.
- Config (b): `ASAN_OPTIONS=detect_leaks=0:handle_segv=0 qemu-x86_64 -cpu EPYC
  ./native/asan/et_overread_harness arena native/spike/lin129_planned.pte` →
  placement line first (input last in its buffer: yes/no), then fault-or-clean.
  If "yes" + SIGSEGV → upstream report track: re-verify against ExecuTorch
  `main` (`~/workspace/executorch`, tag v1.3.1 is not upstream's tip) before
  filing Claim 2.

  **Qualifying a negative — this is one fixture, not a survey.** If the run
  reports "no" or exits clean, the finding is *"`lin129_planned.pte`'s planner
  did not place its XNNPACK external input last in a single 848-byte arena
  buffer."* It is **not** "the planner never does." Record it in that form and
  do not file — but do not write it up as a clearance either; the brief applies
  the same rule to Route A negatives and the reasoning is identical.

  Settling the general question is cheaper by reading than by collecting
  fixtures: memory planning packs by lifetime and graph inputs have the earliest
  start, which predicts they are placed *first* and would make Claim 2
  structurally unreachable. That argument, confirmed against
  `exir/passes/memory_planning_pass.py`, closes the question in a way no number
  of `.pte`s can. Do that before adding a second fixture. If it holds, record it
  as the answer and retire config (b); if it does not, the fixture that breaks
  it is the one worth building.
- Kernel observation per run (the evidence block is unusable without the
  selected-kernel line): gdb under qemu — terminal 1:
  `ASAN_OPTIONS=detect_leaks=0:handle_segv=0 qemu-x86_64 -g 12345 -cpu EPYC
  ./native/asan/et_overread_harness borrowed native/spike/lin129.pte`;
  terminal 2: `gdb -batch -ex 'target remote :12345' -ex 'break
  xnn_f32_gemm_minmax_ukernel_1x16s4__fma3_broadcast' -ex continue -ex 'bt 4'
  ./native/asan/et_overread_harness` (Route A: break
  `xnn_f32_vclamp_ukernel__sse2_u8`). gdb-under-qemu is verified working
  (breakpoints fire, backtraces resolve). On real Zen hardware instead:
  `perf record -F 999 -- ./native/asan/et_overread_harness borrowed
  native/spike/lin129.pte && perf report --stdio | grep ukernel` — the top
  symbol is the selected kernel (ukernel symbols are global in the linked dist
  lib). The user's `perf_users` group wrapper (`/usr/bin/perf`,
  root:perf_users) covers `perf_event_paranoid=4`.
- **Sufficiency arm — run this, it is one command.** Every route above proves
  the padding is *necessary*; none exercises the engine's own path, because
  `et_overread_harness borrowed` deliberately bypasses `EtRuntime::forward` (W7
  staging would pad the slot and absorb the over-read, making every route report
  a meaningless negative). So the same uarch that faults on the raw path should
  be shown **clean** through the real one:

  ```bash
  ASAN_OPTIONS=detect_leaks=0 qemu-x86_64 -cpu EPYC \
    ./native/asan/et_leak_harness native/spike/lin129.pte 1 2
  ```

  Same fixture, same annotated `1x16s4__fma3_broadcast` kernel, but through
  `EtRuntime::forward` and its padded slot. Expect exit 0 and
  `grow=0 staged_input=2 total_input=2`. Verified to run natively during the
  2026-08-05 review; the qemu leg is what remains. Pair it with the Route B run
  in the same sitting — *faults raw / clean staged, on identical hardware and
  fixture* is the whole claim, and either half alone is weak.

  This is still a manual confirmation, not a regression gate: W7's in-repo
  coverage is the alignment and padding-size assertions, the stage-vs-
  pass-through case, the ASan lifetime case, and `grow == 0` (§3/W7). Those run
  on any microarchitecture; this one cannot.

Run evidence goes into §8 using the existing W4 evidence template. A negative
from any route does not license dropping the staging padding (§3/W4).

Not ⚠️-tagged (§7): the ASan rebuild is not part of this item any more — ASan
cannot observe the over-read — and the guard-page harness is a small standalone
binary.

### W5 — Establish the cost ⚠️

Measure, against kernel time:

- the ExecuTorch-side input copy — A/B a MobileNet exported with
  `alloc_graph_input` both ways through the existing `example/src/jmh` harness;
- the heap `byte[]` output copy, including its GC cost, at output sizes well
  past MobileNet's 4 KB.

*Answers:* whether any of this is defensible to users. The IREE spike's own
numbers predict the input answer — copies were ~0.5% of a 61.6 ms MobileNet
kernel there, and our input copy is a single 600 KB memcpy against a comparable
kernel. Expect the input result to be "noise," and treat a surprise as the
finding. The output/GC arm is the one with genuine uncertainty.

**Per standing practice, the user runs benchmarks; this item produces the
harness edit and the run recipe, not the run.** Tagged ⚠️ — see §7; a
large-tensor arm here would reproduce the IREE W2 memory profile exactly.

#### Run recipe (implemented 2026-08-06)

Harness edits live on `feature/w5-w6-direct-outputs`: the `exportMode` A/B in
`MobilenetBenchmark` and the `AddOutputBenchmark` sweep (models from
`tools/scripts/export_w5_add_models.py`, task `:example:exportW5Models`).
Two sessions, because the W6 arm swaps the shim: the heap baseline runs at the
W5-tip commit (pre-W6) against the saved pre-W6 shim `/tmp/et-pre-w6.so` (a file
path, per `LibUtils.loadLibrary`); the W6 comparison runs at the branch tip with
the rebuilt, jar-bundled shim.

Both sessions use the §7 control and a jar built with
`./gradlew :example:jmhJar --no-configuration-cache --rerun-tasks`
(`--no-configuration-cache` is required by the JMH plugin — see
`example/README.md`). All commands below were validated in shape on
2026-08-06 (smoke runs; non-evidence scores in §8/W5).

**S1 — input A/B + output baseline (heap path), at the W5-tip commit:**

```bash
git worktree add /tmp/et-w5-baseline c1ea5dda9c6d9a03a74fbaccd3306a408c5d7c61   # W5 tip
cd /tmp/et-w5-baseline
./gradlew :example:exportModels --no-configuration-cache --rerun-tasks   # uv; flatc fallback if needed
./gradlew :example:exportW5Models --no-configuration-cache --rerun-tasks
./gradlew :example:jmhJar --no-configuration-cache --rerun-tasks

# input A/B: same MobileNet, planned vs unplanned export (the invisible input copy)
# ET_NATIVE (not ET_HYBRID): the fat jar's META-INF/services keeps only the ExecuTorch
# provider (jmhJar duplicatesStrategy=EXCLUDE — see docs/iree-lessons-learned §3), so a
# PyTorch-backed arm cannot load from java -jar. The A/B delta is the export mode alone;
# ET_NATIVE also keeps the fork LibTorch-free.
EXECUTORCH_LIBRARY_PATH=/tmp/et-pre-w6.so systemd-run --user --scope -p MemoryMax=4G taskset -c 0-3 \
  timeout 900 bash -c 'java -jar example/build/libs/example-jmh.jar -f 1 -w 250ms -r 250ms -gc true \
  -jvmArgs "-Xmx1536M -Dexample.models.dir=example/build/models -Dai.djl.pytorch.num_interop_threads=1" \
  -p variant=ET_NATIVE -p exportMode=planned,unplanned MobilenetBenchmark.steadyState'

# output baseline: heap byte[] marshalling + its GC cost, four sizes
EXECUTORCH_LIBRARY_PATH=/tmp/et-pre-w6.so systemd-run --user --scope -p MemoryMax=4G taskset -c 0-3 \
  timeout 900 bash -c 'java -jar example/build/libs/example-jmh.jar -f 1 -w 250ms -r 250ms -gc true -prof gc \
  -jvmArgs "-Xmx1536M -Dexample.models.dir=example/build/models" \
  AddOutputBenchmark.steadyState'
```

`-gc true` + `-Xmx1536M` + `-w/-r 250ms` is the config proven on this host by the
IREE spike (findings §3) — per-iteration GC keeps the W6 arm's Cleaner draining;
`-prof gc` reports the per-op allocation rate and GC time. The export tasks must
run from the repo root; if a delegated export fails with
`FileNotFoundError: 'flatc'`, re-run with
`PATH=$HOME/workspace/executorch/.venv/bin:$PATH` (brief §8 flatc caveat).

**S2 — W6 comparison (direct outputs), at the branch tip after rebuilding the shim:**

```bash
cmake --build native/build -j"$(nproc)"    # native/build is the shipping-shim cache (ET_BUILD_QA/OFF, BENCH/OFF, logging)
cp native/build/libexecutorch_djl.so src/main/resources/native/linux-x86_64/
./gradlew :example:jmhJar --no-configuration-cache --rerun-tasks
# output comparison: SAME AddOutputBenchmark command as S1, no EXECUTORCH_LIBRARY_PATH
# (the jar now bundles the W6 shim)
```

If a cmake reconfigure is needed (stale cache), set `JAVA_HOME` to the Zulu 17
used for the JVM tests first (`CMakeLists.txt` reads `$ENV{JAVA_HOME}` at
configure). The W6 direct arm's numbers go into the same §8/W5 log blocks.

### W6 — Prototype: direct-buffer outputs

Replace copy 3's `NewByteArray` + `ByteBuffer.wrap` with a JNI-allocated block
exposed via `NewDirectByteBuffer`, freed by a `java.lang.ref.Cleaner`. This
stays a copy — `OutputView.data` points into ExecuTorch's arena and is invalid
after the next `forward()`, so the copy is mandatory. Only its *destination*
changes: off-heap instead of on-heap, and direct, which also removes the
model-chaining re-copy at `EtSymbolBlock.java:56`.

**Status: complete (prototype, decision pending on W5 numbers).** Implemented
2026-08-06 on `feature/w5-w6-direct-outputs` (W6-1..W6-4): copy 3's destination
is now a JNI-allocated block exposed via `NewDirectByteBuffer`
(`native/jni/et_output_buffer.h`, `executorch_djl_jni.cpp`), freed by a
`java.lang.ref.Cleaner` registered in `EtOutputBuffers` the moment an output
`EtTensor` is wrapped. The replacement leak signal is the native alive-counter
(`EtNative.aliveOutputBuffers`), asserted to drain by `LeakStressTest`; the
heap-independent marshalling contract is pinned by `EtNativeOomTest` under
`-Xmx128m`. Whether this stays or is reverted is decided on the W5 numbers
(§3/W5, §8/W5).

IREE's W4 prototype ports here nearly verbatim, including its two hard-won
rules: register **only the address primitive** with the Cleaner (capturing the
`ByteBuffer` keeps it strongly reachable and the Cleaner never fires), and make
free idempotent so a mis-registration cannot double-free. Carry its caveat too:
JNI-allocated buffers are **not** counted against `-XX:MaxDirectMemorySize`,
which is precisely the mechanism behind the 20.7 GB OOM-kill recorded in the
IREE brief's §7.

**Required, not optional: W6 silently defeats an existing leak gate.**
`LeakStressTest.inferencePathUnderPressure` detects an output leak *because*
heap `byte[]` and direct buffers count against its `-Xmx256m` /
`-XX:MaxDirectMemorySize=64m` caps. Move outputs to a JNI-allocated block and
they stop counting — the same mechanism as the IREE OOM-kill. The 20,000-predict
loop would then pass regardless of whether outputs leak, and **nothing would
announce the loss of coverage.**

W6 shipped with a replacement signal in the same change: the native
alive-counter `EtNative.aliveOutputBuffers` (IREE's `aliveAlignedBuffers()`
pattern, polled from the test), asserted to drain by `LeakStressTest`'s
deadline-poll helper after each pressure run. Shipping W6 without one would
have been a net reduction in coverage disguised as a performance improvement.

*Answers:* feasibility of the likelier win, and supplies W5's comparison arm.

**Not gated by W4** — outputs are engine-allocated and never handed to XNNPACK
as inputs, so the over-read question does not apply. This is the item most
likely to ship.

### W7 — Grow-only per-slot staging for unplanned inputs

**This is the shipping design for the input path.** It replaces the earlier
"input borrow behind a flag," which mis-stated the problem.

**Provenance note.** The grow-only per-slot staging pattern comes from a
`djl-iree-engine` session that is **not** captured in
`docs/iree-lessons-learned/` — neither the brief nor the findings doc there
mentions it (grep for "grow", "staging buffer", "per-slot"). It is recorded
here as a design this repo adopts on the reasoning below, not as a result
inherited with evidence attached. If the IREE measurement behind it matters
later, it will have to be re-obtained.

**The framing correction.** When `is_memory_planned == false`, we do not have a
choice about borrowing. `share_tensor_data` takes whatever pointer we hand it
and there is no opt-out. The question is never "should we borrow" — it is
**"whose buffer does ExecuTorch borrow: a JVM-owned, unpadded, arbitrarily
aligned one, or an engine-owned one we control?"** So the staging buffer is not
a performance feature. It is the only mechanism that satisfies XNNPACK's
padding contract when the artifact forces a borrow.

**The design.** In `EtRuntime::forward` (`native/core/et_runtime.cpp:70-76`),
for each input whose `is_memory_planned()` is false, memcpy the incoming
borrowed pointer into a per-slot, engine-owned buffer and pass *that* pointer
to `from_blob`. Grow-only, so steady state is zero allocations. Allocate
64-byte aligned and over-allocate by the padding constant, which makes the
over-read land in our own slack on every microarchitecture.

Own the buffers in `RuntimeState`, **not** in Java. Three consequences, all
good:

- The borrowed pointer's lifetime becomes tied to the `Method`'s *by
  construction* — same owner, same destructor. That closes the §4 lifetime
  hazard (`share_tensor_data` leaves a pointer in the `Method` that nothing
  resets) without any ordering argument.
- None of IREE's W4 machinery is needed: no `NewDirectByteBuffer`, no
  `Cleaner`, no address-only capture rule, no `-XX:MaxDirectMemorySize`
  accounting surprise. The buffer is pure native and invisible to Java.
- It is roughly 30 lines at the `from_blob` site, gated on the per-input flag
  from W2. `is_memory_planned()` is per-input, so mixed models fall out
  naturally — stage only the unplanned slots.

**How W7 is actually covered — W4 does not do it.** W4 is a manual,
uarch-dependent run (§3/W4), so nothing in it can guard this code on an ordinary
build. The in-repo coverage W7 must ship with, all of which runs on any
microarchitecture:

| Property | Guard |
|---|---|
| Slot is 64-byte aligned | unit assertion on the allocator, no XNNPACK involved |
| Slot is over-allocated by the padding constant | same |
| Staged only for unplanned inputs, per slot | Catch2 case over `add_unplanned.pte` vs `add.pte` |
| Allocates once, not per call | W8 `staging_grow` exact-count assertion |
| Borrowed pointer outlives a freed Java buffer safely | ASan case: free after `forward()`, run a second `forward()` |

The manual W4 re-run with staging enabled is a confirmation, not a regression
test, and must not be counted as one.

**Be honest that this costs.** Today the unplanned path is genuinely zero-copy
(one copy at `EtNDManager.create`, then ExecuTorch borrows the direct buffer).
Staging makes it two. That is a real regression on that path — safety bought
with a memcpy, not a gain. Do not let "staging buffer" imply speed. What it
buys is proportionate: the over-read, the alignment question, and the lifetime
hazard all close together.

**The separate, genuine perf win is on the *planned* path.**
`EtSymbolBlock.java:56-61` allocates a fresh direct `ByteBuffer` **per forward**
for any non-direct input, and our own outputs are heap `byte[]`
(`executorch_djl_jni.cpp:189-191`), so chaining model A → model B drives B down
that branch every single call. The same grow-only per-slot buffer eliminates
that allocation. Measure it in W5.

So the matrix:

| Model input | Engine behavior |
|---|---|
| `memory_planned=false` | **stage** — padded, aligned, engine-owned (safety, mandatory) |
| `memory_planned=true`, non-direct input | **stage** — kills the per-call `allocateDirect` (perf, optional) |
| `memory_planned=true`, direct input | nothing — ExecuTorch copies into its arena, already correct |

**Stage at `forward()`, not `create()`.** Writing straight into the slot buffer
at create time would keep it at one copy, but per-slot reuse breaks `NDArray`
value semantics the moment a caller creates two inputs for the same slot or
holds one across calls. The copy at forward time is the price of DJL's object
model. Thread safety needs no new constraint: `EtSymbolBlock.forward()` is
already documented as one `Model`/`Predictor` per thread.

**Padding constant.** `XNN_EXTRA_BYTES` is 16 on x86/ARM and 128 on Hexagon,
but it lives in `xnnpack.h`, which is delegate-internal and not on our include
path. Hardcode 128 (the maximum) with a comment citing the source rather than
taking a dependency; the waste is per-slot, not per-call.

**Prerequisite — partially satisfied 2026-08-05 (`7eed3b8`).** The original
statement of this item was "there is no unplanned test fixture, so this path
would ship untested": every `.pte` in the repo was `memory_planned=True`, and
`EtNDManager.create` always returns a direct buffer, so under the matrix above
the entire suite took the pass-through row and never staged.

Now landed: `native/spike/add_unplanned.pte` — the same add model as `add.pte`,
exported with
`ExecutorchBackendConfig(memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False))`
via `native/spike/export_add_unplanned.py` (PEP 723 uv header, pinned
`executorch==1.3.1`). It reports `memory_planned=False` on both inputs and is
consumed by `et_runtime_test.cpp:43` and
`EtMethodMetaTest.readsUnplannedAddModelMetadata`, the latter gated on
`TestSupport.assumeUnplannedModelAvailable()`.

Two gaps remain, and both still land *with* W7:

- **The native leak harness is not pointed at it.** `et_leak_harness.cpp` runs
  `add.pte` only, so W8's exact-count staging assertions have nothing to count.
- **No delegated variant.** `add_unplanned.pte` is plain portable-op add — it
  never reaches XNNPACK, so it exercises `share_tensor_data` but not the
  over-read surface W4 is about. An XNNPACK-partitioned unplanned export is
  confirmed to work on v1.3.1 (remember `flatc` on `PATH`, §8) and is what W4's
  Route A/B models need.

**Both gaps closed 2026-08-05 with W7/W8:** the leak harness now runs
`add_unplanned.pte` with exact-count staging assertions (grow/staged/total,
`build_qa.sh`), and the delegated fixtures `clamp5.pte` / `lin129.pte` /
`lin129_planned.pte` landed via `native/spike/export_w4_models.py`.

*Answers:* whether the unplanned-input case can be made safe without asking
users to change anything about their models. Unlike the old W7, this ships
regardless of the export config — we handle whichever mode the artifact is in.

### W8 — USDT probes and leak-test coverage

Two static probes in `native/core/et_runtime.cpp`, plus the test changes they
make possible.

**Why not on `Method`.** The obvious place to put a slot-size probe is
ExecuTorch's `Method`, and that is the one place it must not go: `Method` is
upstream code shipped prebuilt in the pinned tarball, so instrumenting it means
patching ExecuTorch inside `executorch-runtime-dist` and carrying that patch
across every version bump. The existing `etnp::lstm` USDTs are not a precedent —
that op is first-party dist code. W7's staging lives in our own `RuntimeState`,
so the probes are free and the lifetime question does not arise.

**The slot probably never grows, which changes what to measure.**
`TensorInfo::nbytes()` is available at load for planned *and* unplanned inputs
(the 2026-08-04 probe reported `nbytes=516` on a `memory_planned=False` input).
Static shapes make it exact; dynamic shapes make it an upper bound. Either way
the slots can be sized once in `methodMeta()` and never resized — "grow-only"
degenerates to "allocate once." So:

| Probe | Fires | Job |
|---|---|---|
| `staging_grow(slot, old_bytes, new_bytes)` | **never** — see below | tripwire on the invariants that make it unreachable |
| `staging_input(slot, nbytes, planned, staged)` | per input, per forward | the actual observability |

**Revised 2026-08-05, after implementation.** `staging_grow` ended up stricter
than this table originally described, and the difference matters to anyone
reading the probe's output.

It first shipped firing once per slot per load — routine first-touch, because
slots were sized lazily from the caller's shape rather than at load from
`TensorInfo::nbytes()` as this section specifies. That made a genuine overflow
indistinguishable from normal operation, and `et_leak_harness` asserted the
first-touch count, so it was measuring the wrong steady state. Sizing at load
(commit `934cf38`) fixed that and restored the intended anomaly semantics.

Two later changes then made the probe **unreachable**:

- rejecting any input past its declared bound before the staging copy
  (`0f64c70`) removed the tensor path — `actual` can no longer exceed the bound
  a slot was sized from;
- rejecting methods with non-tensor inputs at load (`94c8174`) removed the
  other — a slot with no declared bound can no longer exist.

So the probe is now a **tripwire, not a detector**: it fires only if slot sizing
and the bound check stop agreeing with each other. `et_leak_harness`'s
assertion is correspondingly `grow == 0` for any number of loads and forwards,
which is strictly stronger than the per-load count it replaced — it catches a
realloc-per-forward slip *and* a regression to sizing from the caller's shape.
The dead branch is deliberate and commented as such in `et_runtime.cpp`; do not
delete it as unreachable code.

`staging_input` is unaffected and remains the per-call observability.

`staging_input` is the complement to W2, not a duplicate: W2 logs the mode once
at load because per-forward logging would be unusable noise, and USDT is exactly
the tool for the per-call view at nop cost when disabled.

**What the probes let the leak tests assert that they cannot today.** The
dominant failure mode of grow-only staging is *reallocate-every-call* — a
`>` vs `>=` slip, byte-count/element-count confusion, or shrink-then-regrow
oscillation on dynamic shapes. That bug frees correctly every time, so **LSan
reports nothing, RSS stays flat, and all three existing gates pass.** Its only
symptom is throughput, which none of them measure. Every current assertion is
aggregate and negative (LSan at exit, or OOM under a cap); none can express
"this allocated once."

- **`native/harness/et_leak_harness.cpp`** — **as built (2026-08-05), the
  assertion is `grow == 0`**, for every fixture and every loads × forwards
  shape. The original plan here (1000 loads × 4 forwards should fire
  `staging_grow` exactly 2000 times, a realloc-per-forward bug gives 8000) was
  written before sizing-at-load; it encoded first-touch allocation as expected
  behavior. The zero-count form supersedes it and is strictly stronger: it fails
  on the realloc-per-forward bug *and* on any regression to sizing from the
  caller's shape, which the 2000-count form would have accepted. The inverted
  variant (**1 load × 10,000 forwards**) is still worth running and is wired in
  `build_qa.sh` — it isolates steady state, where the per-load count could
  otherwise hide a per-forward leak. Equality assertions throughout, not bounds.
  This is the no-JVM binary with a stable build-output path, so it is where the
  probes were developed (§7's attach caveats do not apply).
- **`LeakStressTest.inferencePathUnderPressure`** — already the right shape
  (one model, 20,000 predicts). The unplanned variant landed as a separate test
  (`inferencePathUnderPressureUnplanned`) once the fixture existed. Be clear
  about what it does *not* buy: staging memory is native and counts against
  neither `-Xmx256m` nor `-XX:MaxDirectMemorySize=64m`, and slots are sized once
  at load, so this test cannot observe a staging leak. It guards the JNI/NDArray
  path over a staged model, and the native harness's `grow == 0` is what covers
  staging itself.
- **`LeakStressTest.directBufferLifecycleUnderPressure`** — native-free
  (`NDManager.create` only, no forward). The probes add nothing; recorded so
  nobody instruments it hunting a signal that cannot be there.
- **W6's replacement leak signal** — see W6. Whichever mechanism is chosen,
  this is where it is asserted.

**Bonus: a regression guard on the runtime pin.** `staging_input` asserted
across a QA run fails loudly if a future ExecuTorch changes *when* `set_input`
borrows versus copies. Nothing would notice that today, and the pin is this
repo's supply-chain review gate — so the guard fits the existing posture.

*Answers:* nothing about the design; it makes W6 and W7 testable to a standard
the current suite cannot reach.

**Operational notes.** The shim is `System.load`ed from a content-addressed
cache (`~/.cache/executorch-djl/<sha256>/`), so a `usdt:/path/...` target
changes every build — resolve from `/proc/<pid>/maps` or attach with `-p`. The
library is not mapped until `EtNative` static init, so attaching to the JVM
before first model load sees no probes. Both are avoided entirely by developing
against `et_timing_harness` / `et_leak_harness`, into which the JNIEnv-free core
is also linked. Match the dist's LSTM probes on semaphore usage so one set of
tooling covers both.

### W9 — Shared abstraction assessment

Confirm or overturn IREE's "duplicate, don't extract" verdict now that a second
engine has a concrete aligned-buffer/Cleaner need (W6).

*Answers:* whether to factor out a shared module. The IREE finding was that the
genuine overlap is ~60 lines and the JNI half is per-engine ABI surface;
ExecuTorch's need is narrower still (outputs only, no alignment contract to
honor), which if anything strengthens "duplicate." Expect a short confirmation.
Assessment only; no execution.

---

## 4. Hazards to assess

The IREE brief's three hazards, re-assessed for ExecuTorch, plus two that are
new here.

**Lifetime versus GC — worse than IREE's.** JNI pins the buffer for the duration
of the native call, so mid-call collection is impossible; that much is inherited
and already true of the copy path. But IREE releases its import at the end of
`Invoke`, whereas `share_tensor_data` writes the borrowed pointer into the
`Method`'s own `EValue` and **nothing resets it on return**. The only reset is
`internal::reset_data_ptr`, reached solely from an explicit `FreeCall`
instruction the program may or may not emit (`runtime/executor/method.cpp:1591`).
So a borrowed input pointer plausibly outlives `forward()` inside a live
`Method`. It is not dereferenced until the next execution — but freeing the Java
buffer after `forward()` returns leaves a dangling pointer in the runtime.

**Closed by W7's design, not by an argument.** Owning the staging buffers in
`RuntimeState` ties the borrowed pointer's lifetime to the `Method`'s by
construction: same owner, same destructor, no ordering to reason about. This is
the main reason to put staging in the native core rather than expose engine-
allocated buffers to Java. Still worth an ASan case (free the Java buffer after
`forward()`, then run a second `forward()`) as a regression test for the
property, rather than as the thing that establishes it.

**Completion versus return.** `Module::execute` is synchronous — `set_input`s,
then `method->execute()`, then `get_outputs()` — so call return equals
completion today, as with IREE. The same recorded constraint applies: a borrow
contract built on call return breaks if async delegate execution is ever
introduced.

**Aliasing — bounded by W7, and worth the note anyway.** `et_runtime.cpp:71`
does `from_blob(const_cast<void*>(in.data), ...)`, handing ExecuTorch a mutable
tensor over the caller's memory. IREE bounded this by being inference-only with
no in-place surface; ExecuTorch explicitly supports mutable buffers and stateful
models (KV caches), so "kernels do not write inputs" is a weaker assumption
here. W7 removes the user-visible half of this: a kernel that writes an
unplanned input now scribbles on the engine's staging buffer, not the caller's
`NDArray`. The remaining consequence is that such a write is silently discarded
at the next `forward()` — correct for inference, wrong for a stateful model, and
therefore a thing to revisit if this engine ever grows KV-cache support.

**NEW — no refusal semantics.** IREE's entire safety argument rested on the
allocator refusing bad pointers. ExecuTorch accepts whatever it is given. Every
"bounded either way" conclusion in the IREE findings must be re-derived here
rather than inherited; where the IREE doc says a violation degrades to a copy,
the ExecuTorch equivalent is undefined behavior.

**NEW — the engine's behavior varies by an artifact property nobody inspects.**
The original form of this hazard was "we will have to tell users to re-export
with `alloc_graph_input=False`." W7 removes that ask entirely: the engine
handles whichever mode the artifact is in, and no user changes anything. What
survives is subtler and still real — the engine takes materially different code
paths (stage vs. pass-through) based on a `.pte` property that is invisible in
the filename, the DJL API, and every log line today. A user debugging a
performance difference between two models had no way to see it. W2 was the
mitigation and was not optional. **Mitigated as of 2026-08-05:** `EtModel`
logs `model {} input {} memoryPlanned={}` per input at load, so the variance is
now explicable instead of spooky. What remains owed is the README line telling
users the log exists and what it means (§6).

---

## 5. Dependencies and sequencing

```
W1 (audit, COMPLETE) ──> W2 (observe, COMPLETE) ──> W3 (docs, COMPLETE)
                     │
                     │   [MANUAL, off-CI]
                     │   W4 (over-read confirmation) ..evidence only..┐
                     │      <-- NEXT; qemu or Zen hw                  ¦
                     │                                                v
                     ├─> W7 (staging when unplanned)  [safety; ships regardless]
                     │      + unplanned fixture (partial: non-delegated only)
                     │      + own in-repo guards: alignment/padding units,
                     │        stage-vs-passthrough Catch2, ASan lifetime case
                     │      + W8 staging_grow exact-count  <-- required, not optional
                     ├─> W6 (direct-buffer outputs) ─────┐
                     │      + replacement leak signal    │
                     ├─> W8 (probes) ────────────────────┤
                     └─> W5 (measure) ───────────────────┴──> scope gate ──> W9
```

Dotted line = evidence, not a build dependency. W4 informs W7's padding constant
and never runs in the same environment as W7's tests.

- **W1, W2, and W3 are done** (2026-08-04 / 2026-08-05). W1's answer is what
  makes the rest non-trivial: the copy is real, invisible, and was
  mis-documented. W2 discharges W7's hard dependency — W7 branches per input on
  exactly the flag W2 plumbs through, and that flag now reaches
  `EtMethodMeta.inputMemoryPlanned` with tests on both directions.
- **W4 is the next open item and it is a manual step, not a repo test.** The
  annotated kernels are only selected on hardware we do not test on, so it runs
  under `qemu-x86_64` CPUID masking or on real AMD Zen 1–3. The harness lands in
  `native/harness/`; the *run* is a deliberate act whose output is dated
  evidence in §8 (§3/W4).
- **W4 carries a second, outward-facing track.** Configuration (b) — guarded
  arena, memory-planned model, no borrowing — probes an over-read in *stock*
  ExecuTorch and, if it faults, is an upstream defect report rather than
  anything this repo consumes. It is gated on the UNVERIFIED planner-placement
  question and on re-verifying against ExecuTorch `main`. It blocks nothing
  here: W7 ships either way.
- **W4 gates nothing and tests nothing on an ongoing basis.** It justifies the
  padding constant once. Correcting an earlier draft: the same harness re-run
  with staging on is **not** W7's regression test, because a manual run cannot
  guard code. A negative also does not license dropping the padding (§3/W4).
- **W7 therefore carries its own coverage.** Alignment and padding-size unit
  assertions, a stage-vs-pass-through Catch2 case over `add_unplanned.pte`
  against `add.pte`, the ASan lifetime case, and W8's `staging_grow` exact-count
  assertion. These run on any microarchitecture. Without them W7 is unguarded
  no matter what W4 reported.
- **The delegated fixture is a W4 need, not a W7 need.** `add_unplanned.pte` is
  portable-op add and never reaches XNNPACK, which is fine for W7's staging
  logic and useless for W4's over-read routes; those need an
  XNNPACK-partitioned unplanned export (§3/W7).
- **W7 ships on safety grounds, not on a measurement.** It is the only way to
  satisfy XNNPACK's padding contract for an unplanned input, and we do not get
  to decline the borrow. Its cost (one memcpy on that path) is accepted, not
  justified by W5.
- **W6 is independent** of the entire input question and is where the
  measurable win most likely is. Start it in parallel.
- **W5 depends on W6** for its comparison arm, but the current-path baseline
  can be measured immediately. It also measures W7's *other* effect — removing
  the per-forward `allocateDirect` on the non-direct input path.
- **Scope gate** after W5/W6: this is now a question of how much to ship, not
  whether. W7 is in regardless; W6 and the planned-path staging are the
  discretionary parts, and the measurement decides them.
- **W8 (probes) moved onto the critical path** once W4 stopped being a test.
  W7's realloc-per-call failure mode is invisible to every existing gate, and
  `staging_grow` is now the only thing that catches it — so at minimum that
  probe ships with W7. W6 separately must not ship without *some* replacement
  leak signal. Build the probes against the native harness first, where none of
  the attach caveats apply.
- **Two test artifacts are prerequisites, not follow-ups:** the unplanned
  `.pte` fixture (W7) and W6's replacement leak signal. Both ship in the same
  change as the work they cover, or that work is untested. The first is now
  **half-done** — `add_unplanned.pte` exists and is asserted from both the
  Catch2 and JUnit suites, but the leak harness still runs `add.pte` only and
  there is no delegated variant.
- **W9 is last** and is a confirmation of an existing verdict, not an open
  question. Note W7 weakens it further: staging lives in the native core with
  no JNI or `Cleaner` surface, so the overlap with IREE's aligned-allocator
  work is now nearly nil.

---

## 6. Expected output

- A determination on each of §2's five questions, with pointers. Question 1 is
  already answered above and should be carried forward, not re-derived.
- A hard result on the XNNPACK over-read (W4), stated as positive/negative with
  the negative explicitly qualified as model-specific — recorded as **dated
  evidence from a manual run** (command, selected kernel, outcome), plus the
  committed-but-not-CI-wired harness that produced it. Not a test target.
- A determination on the arena-end question (W4 configuration (b)): either a
  settled "the planner does not place an XNNPACK external input last, so the
  concern is closed," or a fault, a `main` re-verification, and an **upstream
  ExecuTorch issue** carrying the reproducer. This is an output of the work that
  leaves this repository — the only one — and it is not a prerequisite for
  anything here.
- Measured numbers for the invisible input copy and the heap output copy,
  retained regardless of the decision.
- W7 landed: staging for unplanned inputs, carrying its **own** in-repo
  regression tests — alignment and padding-size assertions, a
  stage-vs-pass-through case, the ASan lifetime case, and `staging_grow`'s
  exact-count assertion. Explicitly **not** the W4 harness, which cannot run in
  CI. This is not gated on a measurement.
- A go / go-with-constraints / no-go on the **discretionary** parts — the
  direct-buffer output change (W6) and staging on the planned-path non-direct
  input — decided separately and on the W5 numbers.
- README terms. Note that the earlier expected output here — an export-time
  requirement users must satisfy — is **no longer needed**: W7 handles either
  artifact mode. What the README owes users instead is that the engine's input
  handling depends on how their `.pte` was exported, and how to see which mode
  they are in (W2).
- Leak-test coverage that survives the changes: an unplanned-input fixture, a
  replacement output-leak signal for W6, and an exact-count assertion on
  staging allocations. Stated as a requirement because two of the three
  otherwise represent coverage *lost* rather than gained (W6, W8).
- A confirmation or overturn of IREE's "duplicate, don't extract" (W9), which
  explicitly permits duplication as the outcome.

---

## 7. Execution safety controls

The OOM-kill incident recorded in the IREE brief's §7 happened on **this host**
(Ubuntu 24.04, systemd 255, 31 G RAM) on 2026-08-04: an uncontained JMH fork
grew to 20.7 GB anon-RSS and `systemd-oomd` killed whole units — the Firefox
scope and then the terminal scope, taking the shell with it. The risk here is
inherited rather than reproduced, and this repo's current JMH harness is
MobileNet-only, so its present profile is mild. One item recreates the dangerous
profile:

- **W5**, if it adds a large-tensor arm — that *is* the IREE `CopyCostBenchmark`
  shape, and W6's Cleaner-freed buffers reproduce the exact mechanism, since JNI
  allocations are not counted against `-XX:MaxDirectMemorySize`.

**W4 is no longer in this category.** Its revision dropped the ASan rebuild
(`native/build_qa.sh` at `-j$(nproc)` — concurrent instrumented compilers were
the memory spike), because ASan cannot observe the over-read at all. The
guard-page harness is a small standalone binary. Any *other* work that rebuilds
the ASan tree still belongs under these controls.

Use the control verified on this host:

```bash
systemd-run --user --scope -p MemoryMax=4G taskset -c 0-3 timeout 900 bash <cmd>
```

This confines any kill to a transient scope inside the user manager's delegated
cgroup and — critically — moves the run out of the terminal's own scope, so an
oomd kill cannot take the shell. Always pair with `timeout` (a memory cap does
not stop a hang) and `taskset` (which lowers `nproc`-derived job counts, and is
not a safety mechanism by itself). `ulimit -v` is incompatible with ASan, whose
shadow memory reserves ~16 TB of address space. Full rationale and fallbacks:
`docs/iree-lessons-learned/borrowed-host-buffers-brief.md` §7.

Two project-specific notes:

- **Benchmarks are run by the user, not the agent.** W5 produces the harness
  edit and the recipe; the run is the user's.
- **Container builds leave root-owned outputs.** `bench.sh`, `build_qa.sh`, and
  `build_variants.sh` do not chown their outputs back (only `build.sh` does, via
  its EXIT trap). After any W4/W5 container run:
  `sudo chown -R "$(id -u):$(id -g)" native/bench native/bench-results native/asan`.

---

## 8. Sources and manual-run evidence

### W4 evidence log

W4 is a manual, off-CI run (§3/W4), so its results live here rather than in a
test report. One block per executed route; a block without a
**selected kernel** line is not a usable result.

```
(empty — W4 has not been run)

Template:
  date:            YYYY-MM-DD
  configuration:   (a) borrowed input  |  (b) arena end  |  (c) staged, engine path
  route:           A (forced SSE, qemu) | B (Zen uarch, qemu) | B (native Zen hw)
  environment:     qemu-x86_64 -cpu <model>  |  <cloud instance type>
  model:           <path>, N=<n> / K=<k>, alloc_graph_input=<True|False>
  selected kernel: <what XNNPACK actually chose, however observed>
  uarch detected:  <xnn_uarch_* — required for route B, a qemu negative is void without it>
  arena placement: <(b) only: was an XNNPACK external input last in the arena?>
  outcome:         SIGSEGV on guard page | clean run | harness error
  reading:         positive / negative-and-model-specific / inconclusive
  upstream:        <(b) only: filed / not-filed + why; main re-verified y/n>
  pairs with:      <(c) only: the (a) run it is the clean half of — same uarch,
                    same fixture, or the pair proves nothing>
```

### W5 evidence log

W5's runs are manual, off-CI, user-run sessions (§3/W5). One block per arm: the
input A/B arm, the output heap baseline arm, and the output W6 direct arm. A
block without the `-prof gc` lines on the output arm, or without both A/B arms,
is not a usable result.

```
(empty — W5 has not been run)

Template:
  date:            YYYY-MM-DD
  arm:             input A/B | output heap baseline | output W6 direct
  commit/branch:   <sha> on <branch>  (A/B and output baseline: the W5 tip; output W6: the branch tip)
  models + sizes:  mobilenet_v2.pte vs mobilenet_v2_unplanned.pte
                   | add_4kb (N=1024) .. add_64mb (N=16777216)
  ms/op ± err:     <JMH output, per arm>
  -prof gc:        <allocation rate + GC time/op>  (output arm only)
  CPU/OS/glibc:    <lscpu model name / uname -r / ldd --version | head -1>
  reading:         <what the numbers mean for the W6 decision — see §3/W5>
```

#### Harness validation record (2026-08-06 — smoke runs, NOT evidence)

The recipe's commands were validated in shape before the user-run sessions:
`git worktree add … c1ea5dd` (then removed), both export tasks (all six
artifacts probe-verified: planned/unplanned flags, `(1, N)` f32 outputs),
`jmhJar`, the S2 shim rebuild + stage (`cmake --build native/build -j$(nproc)`
+ copy), and the `EXECUTORCH_LIBRARY_PATH` mechanism (`LibUtils` →
`System.load(override)`; `/tmp/et-pre-w6.so` present, 12,156,592 bytes). Each
arm below executes and prints a score from the fat jar with the recipe's
`-p`/`-jvmArgs` shape, and the logs show the expected artifact load lines
(e.g. `mobilenet_v2 input 0 memoryPlanned=true` vs `mobilenet_v2_unplanned
input 0 memoryPlanned=false`).

The scores are JMH smokes (`-f 1 -wi 0 -i 1 -w 1ms -r 1ms`: no warmup, 1 ms
windows, no `-gc true`, no `-prof gc`) and MUST NOT be cited as measurements —
they only prove the arms run. The full §7 `systemd-run` invocations with
`-gc true`/`-prof gc` and real iteration counts were not run; those are the
S1/S2 user sessions.

| arm | shim | smoke score |
|---|---|---|
| AddOutputBenchmark add_4kb | pre-W6 (heap `byte[]`; jar-bundled copy of the saved shim) | 0.060 ms/op |
| AddOutputBenchmark add_4mb | pre-W6 (heap `byte[]`) | 2.927 ms/op |
| AddOutputBenchmark add_4kb | W6 (direct, rebuilt shim) | 0.064 ms/op |
| MobilenetBenchmark steadyState planned (ET_NATIVE) | pre-W6 (heap `byte[]`) | 22.422 ms/op |
| MobilenetBenchmark steadyState unplanned (ET_NATIVE) | pre-W6 (heap `byte[]`) | 15.134 ms/op |

### Upstream sources

All upstream citations verified 2026-08-04 against `~/workspace/executorch` at
tag `v1.3.1` — the exact runtime version pinned in
`native/cmake/EtRuntimePin.cmake`. The pinned tarball ships headers only, so
upstream `.cpp` behavior is not readable from `native/*/\_deps/`; that checkout
is the authority.

### Reproducing the probes

That checkout's `.venv` has a working `executorch` + `torch` + exir toolchain.
To read a `.pte`'s per-input memory-plan mode — the check behind W1, and the
acceptance test for W2:

```python
from executorch.runtime import Runtime
mm = Runtime.get().load_program(path).load_method("forward").metadata
for i in range(mm.num_inputs()):
    ti = mm.input_tensor_meta(i)
    print(i, ti.dtype(), ti.sizes(), ti.nbytes(), ti.is_memory_planned())
```

Three mechanics that cost time to rediscover:

- **`flatc` must be on `PATH` to export a *delegated* `.pte`.** It ships at
  `.venv/bin/flatc` but is not on `PATH`, and its absence surfaces as a bare
  `FileNotFoundError: 'flatc'` from deep inside `_serialize/_flatbuffer.py`.
  Non-delegated exports (plain `to_executorch()`) do **not** need it, so the
  failure only appears once a partitioner is added — which is exactly when
  building W7's unplanned fixture. Prefix with
  `PATH=<checkout>/.venv/bin:$PATH`.
- **`src/test/resources/lstm/lstm.pte` fails to load in that venv** with
  `error: 0x:14` — the first-party `etnp::lstm` custom op is not registered
  there. Expected, not a signal; use a different fixture.
- The ExecuTorch ScalarType codes that appear in these dumps: `1` = int8,
  `3` = int32, `4` = int64, `6` = **float32**, `7` = float64. `dtype=6` with
  `nbytes == numel * 4` is the confirmation that a quantized model's graph
  input is still f32 (the disproven Route B sub-route, W4).

- `runtime/executor/method.cpp:1143-1255` — `Method::set_input`, the
  memory-planned branch.
- `runtime/executor/method.cpp:1575-1592` — `FreeCall` / `reset_data_ptr`, the
  only borrowed-pointer reset.
- `runtime/core/exec_aten/util/tensor_util_portable.cpp:140-163` —
  `share_tensor_data`, no alignment or padding check.
- `runtime/executor/method_meta.h:63` — `TensorInfo::is_memory_planned()`, public.
- `runtime/core/memory_allocator.h:45` — `kDefaultAlignment = alignof(void*)`.
- `extension/module/module.cpp` — `Module::execute`, synchronous, calls
  `set_input` per input.
- `exir/passes/memory_planning_pass.py:151` — `alloc_graph_input: bool = True`.
- `cmake-out/include/xnnpack.h:24-32` — `XNN_EXTRA_BYTES`, the documented
  over-read.
- `backends/xnnpack/runtime/XNNExecutor.cpp` — `prepare_args`, hands XNNPACK the
  unpadded `mutable_data_ptr()`.

XNNPACK citations are relative to
`backends/xnnpack/third-party/XNNPACK/` in the same checkout:

- `src/xnnpack/common.h:288-321` — `XNN_OOB_READS` and the
  `XNN_DISABLE_{TSAN,MSAN,HWASAN,ASAN}` it expands to; this is why ASan cannot
  observe the over-read.
- `src/xnnpack/simd/f32-sse2-base.h:195` — `xnn_load_tail_f32`, the unmasked
  `_mm_loadu_ps` tail load, annotated `XNN_OOB_READS`.
- `src/xnnpack/simd/f32-avx-base.h:172` — the AVX counterpart,
  `_mm256_maskload_ps`, architecturally safe.
- `bench/subgraph/benchmark.cc:91` — `malloc(size + XNN_EXTRA_BYTES)` for every
  external value; XNNPACK's own harness honoring the contract ExecuTorch does
  not.
- `test/vunary-microkernel-tester.h:225` — unconditional input over-allocation
  by `XNN_EXTRA_BYTES / sizeof(In)`.
- `src/configs/hardware-config.c` — no environment override for ISA selection,
  hence the qemu CPUID-masking approach in W4 Route A.
- `src/configs/gemm-config.c:1444-1471` — the f32 igemm selection chain and the
  `xnn_uarch_zen` / `xnn_uarch_dhyana` case that picks the annotated `s4`
  kernel; the basis for W4 Route B.
- `src/f32-qs8-vcvt/gen/` — f32→qs8 convert, annotated only on sse2/sse41; why
  the quantized sub-route of Route B is safe on AVX and therefore useless as a
  probe.

- `docs/iree-lessons-learned/2026-08-04-borrowed-host-buffers-findings.md` — the
  sibling spike this brief derives from; §4 there holds the JDK direct-buffer
  alignment histogram (`addr % 64 ∈ {0,16,32,48}`, ~40% at 0), which is reusable
  here as an observation but not as a guarantee — the JVM promises 8-byte
  alignment.
