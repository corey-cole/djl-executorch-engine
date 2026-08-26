# Unplanned-Input SIGSEGV Root Cause Investigation

> **Format note:** this is a diagnostic investigation with branching, uncertain outcomes, not
> an implementation plan for a fixed spec. The `superpowers:writing-plans` TDD task template
> (write test → fail → implement → pass → commit) doesn't fit: most experiments here don't
> produce code, and which experiment to run next depends on what the previous one found. This
> document keeps that skill's spirit — concrete steps, exact commands, no placeholders — but is
> structured as a hypothesis/experiment matrix instead.
>
> **The reporter's model is proprietary and cannot leave their environment.** Every experiment
> below falls into one of two kinds, and the distinction matters for what's actually actionable
> without the reporter's continued involvement:
> - **Self-executed** (E1b, E2-E7, E9, E10): run entirely against synthetic fixtures this project
>   builds itself, public tools (`executorch-numpy-runtime`), or local source reading. No reporter
>   dependency once built.
> - **Reporter-executed** (E1a, E8): the reporter runs something in their own environment and
>   reports back the observed result. We package these as exact, self-contained instructions to
>   hand them — never as something run on our own infrastructure against their artifact, since
>   their artifact never reaches our infrastructure at all.

**Goal:** determine why `unplanned` (`alloc_graph_input=False`) inputs SIGSEGV under XNNPACK and
fail with a `RuntimeException` under OpenVINO, for the reporter's model, when `planned` mode works
fine and #67/#68/#70/#72 already closed every failure class this repo's own fixtures could
reproduce.

**Prior work this builds on:**
- `docs/executorch-host-buffer-contract-brief.md` — the original W1-W9 investigation (padding
  sufficiency, the guard-page harness). Its fixtures (`clamp5.pte`, `lin129.pte`) do NOT crash on
  this pin (`1.4.1-3`) or this hardware — verified in this session.
- #67 (padding static_assert), #68 (JNI buffer-capacity guard), #70 (shape-sanity guard), #72
  (rank/shape mismatch verified to fail cleanly, not crash, for static shapes) — all closed, none
  reproduce the reporter's crash.
- #73 (dynamic-shape rank mismatch) — open, unverified, may or may not be related.
- New field evidence (this session): the reporter's own isolation test (raw `EtNative.loadModule`
  + `EtNative.forward` with hand-built, correctly-sized direct `ByteBuffer`s, bypassing all DJL
  marshalling) still fails on both backends. OpenVINO's `unplanned` rejection has precedent already
  documented in this repo: `example/.../Variant.java:44-48` says OpenVINO is "one export, since a
  fully-delegated graph hands its input straight to the delegate" — nobody ever built or tested an
  OpenVINO+unplanned artifact before now.

## Hypotheses

| ID | Hypothesis | Status |
|---|---|---|
| H1 | XNNPACK's real over-read, for the reporter's actual op/kernel, exceeds `kStagingPadding=128` | Open. Verified sufficient for SSE2 `vclamp` and Zen1 GEMM only (#67's guard-page harness). Untested: conv/depthwise, quantized, AVX-512, and anything the reporter's model actually uses. |
| H2 | A dynamic-shaped input reaches the delegate with a wrong-but-in-bound shape | Open — this is exactly #73, unmerged with this investigation unless the reporter's model has dynamic shapes. |
| H3 | A bug in this engine's Java/JNI marshalling (`EtSymbolBlock`, `EtNDManager`, the JNI shim) | **Ruled out.** Reporter's raw `EtNative` test with hand-built correct buffers still fails. |
| H4 | A bug in `native/core/et_runtime.cpp`'s own `forward()`/staging logic, independent of JNI | Open, not yet isolated from H1/H5 — the reporter's test still went through the JNI shim, which itself calls into this same code, so this hasn't been separated from "JNI is fine, native core isn't" vs. "both are fine, something else is wrong." |
| H5 | Something specific to `executorch-runtime-dist`'s `1.4.1-3` build (patches, compiler flags, XNNPACK/OpenVINO vendoring) that a differently-built ExecuTorch wouldn't exhibit | Open, untested. |
| H6 | The `.pte` encodes something the reporter's Python export-time validation didn't actually exercise (i.e. validation ran a different code path than the C++ `Method::execute()`/`share_tensor_data` borrow contract a real consumer uses) | Open — needs to know exactly how the reporter validated in Python. |
| H7 | OpenVINO's C library rejects zero-copy import of a pointer it doesn't recognize/own (registration or provenance, not raw alignment — see below) | Open, OpenVINO-specific, does not by itself explain the XNNPACK SIGSEGV. |
| H8 | The crash isn't in a delegate at all — it's in a non-delegated op that falls back to an ExecuTorch-native kernel, and that kernel has an over-read/UB bug of its own | Open. **Structurally untestable with any current fixture**: every existing `unplanned` fixture (`add_unplanned`, `clamp5`, `lin129`) is a single, fully-delegated `call_delegate` node with zero non-delegated ops (confirmed in the host-buffer-contract-brief) — none of them can exercise a fallback-kernel bug at all. |
| H9 | A **non-float32** `unplanned` input — specifically `int64`, per the reporter's model (25 of 41 inputs) — has a marshalling/staging bug this project has never exercised | Open, and now **structurally confirmed, not just plausible**. `dtypes.pte` is the only int64 fixture in this repo — exported deliberately **without** the XNNPACK partitioner, **planned** not `unplanned`, and never wired into any test. Net: int64 + `unplanned` has zero coverage anywhere in this project. The reporter's actual op list (`aten_cat_default`, `aten_constant_pad_nd_default`, `aten_div_tensor`, `aten_sub_tensor`, `aten_gelu_default`, `aten_unsqueeze_copy_default`, `aten_squeeze_copy_dims`, `aten_embedding_default`) was checked against `~/workspace/executorch/backends/xnnpack/partition/config/generic_node_configs.py` directly: `embedding.default` and `squeeze_copy.dims` have **no partitioner config at all** (never delegated, any dtype); `cat`/`div`/`sub`/`gelu`/`constant_pad_nd`/`unsqueeze_copy` are all restricted to `FP32`/`STATIC_QUANT` precision types (never `int64`). **Every one of the reporter's int64 inputs is therefore structurally guaranteed to reach a non-delegated portable kernel** — this graph cannot be fully XNNPACK-delegated the way every existing fixture in this repo is, regardless of which specific op turns out to matter. Checked the actual portable kernel source for the three int64-relevant, unconditionally-portable ops directly: `kernels/portable/cpu/op_embedding.cpp` (bounds-checks the index against the weight table before use, plain scalar loop, exact-count read — safe by inspection), `op_squeeze_copy.cpp` and `op_unsqueeze_copy.cpp` (both do `memcpy(out, in, in.nbytes())` — exact size, no SIMD, safe by inspection). None of the three show an obvious over-read. **Not yet checked**: the portable *fallback* implementations of `div`/`sub`/`cat`/`constant_pad_nd`/`gelu` (`kernels/portable/cpu/op_div.cpp` etc.) — the ones actually used when these ops operate on `int64` rather than being XNNPACK-delegated — the highest-priority remaining reading target. |

**On H7's mechanism:** a background search of `~/workspace/executorch/backends/openvino/runtime/OpenvinoBackend.cpp`
found no `is_memory_planned()` branch anywhere — planned and unplanned inputs go through the
identical `create_ov_tensor()` → `ov_tensor_create_from_host_ptr()` path. The search's alignment
theory (borrowed pointers less aligned than the planned arena) does **not** fit this engine's
implementation: `staging.h` allocates unplanned buffers 64-byte aligned, *more* aligned than a
typical 16/32-byte planned-arena allocator, and the delegate only ever sees our staged pointer, never
the caller's raw buffer. Whatever OpenVINO's C library actually checks isn't visible from headers
alone — its `ov_tensor.cpp` source isn't vendored locally, only headers and prebuilt `.so`s.

**On H8's mechanism — corrected before it goes further.** The instinct "test the failing model on
Windows, since Windows has a different fallback-op implementation" is right, but not for the reason
it might first appear. `native/CMakeLists.txt:199-203` shows `et_runtime` links **only**
`portable_ops_lib` — never `optimized_ops_lib` or `quantized_ops_lib`, even though the Linux tarball
ships both (`liboptimized_kernels.a`, `liboptimized_ops_lib.a` sit unused in
`_deps/et_runtime-src/lib/`). No comment in the tree explains why; whether that's deliberate or an
unexamined default is itself worth a one-line answer someday, separate from this investigation. The
practical consequence: this engine is *already* on the reference/portable kernel path for
non-delegated ops on every platform, Linux included — so "Linux uses optimized kernels, Windows
falls back to portable" does not differentiate this engine's own behavior between the two OSes the
way it would for, say, `executorch-numpy-runtime` (which does link `optimized`/`quantized` on
Linux per its own platform table).

What Windows *does* still change: the **compiler**. `portable_ops_lib`'s C++ source is the same
across platforms, but MSVC and GCC can compile identical, UB-adjacent source into different runtime
behavior — one crashes, the other doesn't, both are "correct" relative to a spec that permits either.
This project's own UBSan gate (`native/ubsan_gate.sh`, `native-build-job.yml`) runs **Linux only** —
Windows has no equivalent instrumented build, so any UB in a portable kernel, or in this engine's own
code reacting to what a non-delegated op produces, has never been checked there at all. That's a
real, useful asymmetry to test against, just a different one than "Windows lacks optimized kernels."

## Experiment Matrix

Columns show what a **PASS** (no crash/rejection) or **FAIL** (crash/rejection, matching the
report) result does to each hypothesis: **↓** weakens/rules out, **↑** strengthens/rules in, **—**
no information, **±** ambiguous (see notes).

| Experiment | Rebuild? | H1 | H2 | H4 | H5 | H6 | H7 | H8 | H9 |
|---|---|---|---|---|---|---|---|---|---|
| E1a. Ask the reporter for their model's op/shape/dtype characteristics (information only — the artifact itself never leaves their environment) | none | ↑/↓ | ↑/↓ | — | — | ↑/↓ | — | ↑/↓ | ↑/↓ |
| E10. Read the portable-kernel fallback source for `div`/`sub`/`cat`/`constant_pad_nd`/`gelu` | none (read-only) | — | — | — | — | — | — | ↑/↓ | ↑/↓ |
| E1b. Export `conv_unplanned.pte` locally, run through our existing QA harness | container | ↑/↓ | — | — | — | — | — | — | — |
| E2. Run a reproducing artifact through `executorch-numpy-runtime` (XNNPACK) | none (pip) | ↑/↓ | — | ↓* | ±* | ↑/↓ | — | ↑/↓* | ↑/↓* |
| E3. Check whether numpy-runtime pads/stages unplanned inputs itself | none (read source) | context for E2 | — | — | — | — | — | — | — |
| E4. Call `EtRuntime::forward()` from pure C++ (no JNI), same reproducing artifact | container | — | — | ↓ | — | — | — | — | — |
| E5. Diff `executorch-runtime-dist`'s XNNPACK build flags/patches against a stock build | none (read-only) | — | — | — | context | — | — | — | — |
| E6. Export an OpenVINO `unplanned` fixture locally, observe the real failure | container | — | — | — | — | — | ↑/↓ | — | ↑/↓ |
| E7. Build a fixture matching the reporter's real op composition (embedding + squeeze/unsqueeze + sub/div + cat + constant_pad_nd + gelu, mixed int64/float32) | container | — | — | — | — | — | — | ↑/↓ | ↑/↓ |
| E8. Hand the reporter a Windows runbook; they run it in their own environment and report back | none (reporter-executed) | — | — | — | — | — | — | ↑/↓** | ↑/↓** |
| E9. Build a wide-input (~41 scalar), float32-only `unplanned` fixture — isolates *count* from *dtype* | container | — | — | — | — | — | — | — | context |

`*` E2's XNNPACK result is confounded by version: `executorch-numpy-runtime` pins ExecuTorch
**1.3.1** exactly, this engine pins **1.4.1-3**. A FAIL (crashes there too) is clean evidence
against H4/H5 regardless of version, because it proves the failure exists in ExecuTorch's own core
independent of this project's C++/JNI code entirely. A PASS is ambiguous: it could mean H4/H5 (our
code or our build introduces the bug), or it could mean XNNPACK genuinely behaves differently
between 1.3.1 and 1.4.1 with no "bug" on anyone's part (kernel selection changed). Note this
comparison is NOT confounded by microarchitecture the way the original W4 qemu experiments were —
if E2 runs on the same physical machine as the reporter's failure, kernel selection (uarch) is held
constant, which is a real strength over the earlier cross-uarch testing. On H8 specifically: per its
own platform table, `executorch-numpy-runtime` links `optimized`/`quantized` kernels on Linux, unlike
this engine — so if a non-delegated op is involved, E2's Linux run is on a *different* kernel
implementation than this engine ever uses, which weakens what a PASS or FAIL there says about H8 one
way or the other (marked `↑/↓*` rather than a stronger signal).

`**` E8 doesn't isolate the mechanism the way E7 would — a Windows PASS (no crash) is consistent
with H8 (portable-kernel/MSVC-vs-GCC UB) but equally consistent with a delegate-side explanation if
the reporter's model happens to select different XNNPACK kernels on Windows for unrelated reasons
(different available ISA extensions, different `xnn_uarch_*` detection). Treat an E8 result as a
data point that narrows the field, not as a standalone proof — pair it with E7 once a fixture exists
to actually separate "non-delegated op" from "delegate, but Windows picked a different kernel."

## Sequencing

Run in this order — each step either produces the reproducing artifact everything else needs, or is
cheap enough to run regardless. **Step 9 is now the highest-value fixture to build**, ahead of
Step 3's `conv_unplanned.pte` — the op-list evidence structurally confirms the reporter's graph is
partially delegated with int64 flowing entirely through portable kernels, which `conv_unplanned`
(fully delegated, float32-only) cannot exercise at all. Step 3 is still worth keeping as a
dtype-neutral comparison point, just no longer the lead.

### Step 1 — Ask the reporter for their model's characteristics (E1a)

No cost, highest potential value, and the only lever we have on the reporter's *actual* model,
since it cannot leave their environment (no `.pte`, no redacted/synthetic equivalent unless they
build one themselves and confirm it still reproduces before sharing it). Need, as information only:
op types (especially conv/depthwise/quantized, and now confirmed: embedding/index-style ops given
the int64 inputs), the full input list (already partly answered: 41 inputs, 25 `int64` + 16
`float32`, all shape `[1]`), any dynamic shape declarations, and **how** the Python export-time
validation ran — specifically whether it called `ExecutorchProgram.forward()`/an actual C++-backed
`.pte` execution, or only validated the `torch.export`ed graph before lowering (H6). Every answer
here sharpens which of Steps 9/11's synthetic fixtures is worth building, or whether a third design
is needed.

### Step 2 — Read the remaining portable kernel sources (E10)

Zero cost, read-only, no build — and could shortcut every later step if it lands on an obvious bug.
`op_embedding.cpp`, `op_squeeze_copy.cpp`, and `op_unsqueeze_copy.cpp` were already checked and are
safe by inspection (see H9). The remaining candidates are the portable *fallback* implementations
of the FP32-restricted ops — the code paths that only run when `div`/`sub`/`cat`/`constant_pad_nd`/
`gelu` operate on `int64` (or otherwise fail the XNNPACK precision-type check) rather than being
delegated:

- `~/workspace/executorch/kernels/portable/cpu/op_div.cpp`
- `~/workspace/executorch/kernels/portable/cpu/op_sub.cpp`
- `~/workspace/executorch/kernels/portable/cpu/op_cat.cpp`
- `~/workspace/executorch/kernels/portable/cpu/op_constant_pad_nd.cpp`
- `~/workspace/executorch/kernels/portable/cpu/op_gelu.cpp`

Read each the same way the first three were: look for any read/write sized independently of the
tensor's own declared `numel()`/`nbytes()` (a SIMD-width-driven loop bound, an off-by-one, a stride
computed from the wrong tensor), and any place a borrowed/unplanned tensor's raw pointer gets used
without a bounds check the way `op_embedding.cpp`'s `indices_ptr[i] < weight_height` guard has one.
`op_cat.cpp` deserves particular attention — it's the one op here that combines *multiple* input
tensors into one output, a structurally different (and more error-prone) shape than the others'
single-input-to-single-output copies, and the reporter's 41-input model concatenates a lot of them.

### Step 3 — Build `conv_unplanned.pte` and test it locally (E1b)

Only if Step 1 doesn't produce an artifact quickly, or in parallel with waiting on it. Same model
as `native/spike/export_conv.py` (a `Conv2d(3, 8, kernel_size=3, padding=1)` + ReLU over
`[1,3,16,16]`), with the same `alloc_graph_input=False` change `export_add_unplanned.py` made to
`export_add.py`. Write `native/spike/export_conv_unplanned.py`:

```python
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch==2.12.1",
#   "executorch==1.3.1",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Export conv_unplanned.pte -- conv.pte's model, exported with alloc_graph_input=False.

Same model as export_conv.py (Conv2d(3, 8, kernel_size=3, padding=1) + ReLU over [1,3,16,16]),
but with the borrowed-input export config export_add_unplanned.py used. Closes the one op
class the unplanned fixture set has never covered: every existing unplanned fixture
(add_unplanned, clamp5, lin129) is elementwise or a plain GEMM, never a conv, and conv is the
only fixture in this directory proven to grow the XNNPACK workspace arena at all (see
export_conv.py's docstring) -- the op class most likely to select a packing/blocking kernel
with a different real over-read than the SSE2 vclamp / Zen1 GEMM kernels #67's guard-page
harness actually verified.

Run with the checkout venv so flatc is on PATH (delegated exports need it):

    PATH=$HOME/workspace/executorch/.venv/bin:$PATH uv run export_conv_unplanned.py

Writes conv_unplanned.pte into the current working directory.
"""
import torch
from torch.export import export
from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.passes import MemoryPlanningPass
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


class ConvRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv(x))


def main() -> None:
    torch.manual_seed(20260816)  # same seed as export_conv.py -- same weights, comparable output
    model = ConvRelu().eval()
    exported = export(model, (torch.randn(1, 3, 16, 16),))
    lowered = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()]
    ).to_executorch(
        config=ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
        )
    )

    buffer = lowered.buffer
    if b"XnnpackBackend" not in buffer:
        raise SystemExit("conv_unplanned.pte contains no XnnpackBackend delegate")

    with open("conv_unplanned.pte", "wb") as f:
        f.write(buffer)
    print("wrote conv_unplanned.pte")


if __name__ == "__main__":
    main()
```

```bash
cd native/spike
PATH="$HOME/workspace/executorch/.venv/bin:$PATH" uv run export_conv_unplanned.py
```

Run it through the **already-built** `et_leak_harness` in `native/asan` first — no JNI, no JVM,
fastest possible iteration, and it directly reuses the container image already pinned for this repo
rather than needing a fresh toolchain:

```bash
docker run --rm -v "$(pwd)":/workspace -w /workspace "$(cat .engine-build-image)" \
  /bin/bash -c "./native/asan/et_leak_harness native/spike/conv_unplanned.pte 1 2"
```

PASS (`OK: ... grow=0 ...`, exit 0) weakens H1 for conv specifically — not conclusive for other op
classes. FAIL (crash) gives a local, fully-reproducible artifact for every experiment below. Either
way, once the fixture exists, wire it into `native/CMakeLists.txt`'s `et_runtime_test` compile
definitions (`CONV_UNPLANNED_PTE_PATH=...`, same pattern as `LIN129_PTE_PATH` added in #72) so
Step 6 can drive it from a Catch2 `TEST_CASE` without repeating this setup.

### Step 4 — Once a reproducing artifact exists, cross-check via `executorch-numpy-runtime` (E2)

```bash
pip install executorch-numpy-runtime
python3 -c "
import numpy as np, executorch_numpy_runtime as en
prog = en.Runtime.get().load_program('<reproducing.pte>')
method = prog.load_method('forward')
out = method([np.ones(<shape>, np.float32), ...])
print(out)
"
```

Run on the **same machine** used for E1b/E4 to keep microarchitecture constant. Record: does it
crash, raise a clean Python exception, or succeed? Read the result against the confounded-by-version
caveat above before drawing conclusions.

### Step 5 — Read `executorch-numpy-runtime`'s source for its own padding/staging behavior (E3)

Quick, read-only. If E2 succeeds cleanly, check whether that's because it does its own
generous padding (making the comparison less informative for H1) or because it hands XNNPACK the
raw numpy buffer with no padding at all (making a clean E2 PASS much stronger evidence against H1
for this specific artifact, since it would mean an even less defended consumer didn't crash).

### Step 6 — Isolate JNI from native-core (E4)

Only if E2's result doesn't already settle H4 confidently. Add the reproducing fixture to
`et_runtime_test.cpp` (same pattern as #72's `issue71:` tests) and call `EtRuntime::forward()`
directly with a raw `std::vector<float>` — no JNI, no JVM. If this still crashes, H4 (native-core
bug specific to this engine) is essentially ruled out from a second, independent angle beyond the
reporter's own JNI-level isolation test.

### Step 7 — Build an OpenVINO `unplanned` fixture locally (E6)

Independent of the XNNPACK track, closes the OpenVINO side with local evidence instead of inference.
Export a small model (reuse whatever produces `openvino_tiny.pte`, likely under `native/spike/`)
with `alloc_graph_input=False` and the OpenVINO partitioner, then run it through `EtRuntime` with
`OPENVINO_LIB_PATH` set, same as existing OpenVINO tests. Confirm the exact rejection point — does
it fail at `module.load_forward()` (delegate init) or at the first `forward()` (execute)? That
distinguishes an init-time binding problem from a per-call one, which the JNI/Java layer can't see
from the outside (it only sees the resulting `RuntimeException`, and #72's `EtRuntime::forward()`
turns most failures into `std::runtime_error` — worth checking whether the actual OpenVINO error
text survives into that message, or whether it should be added, the same way #70 added
`nameSuffix()` to previously-uninformative messages).

### Step 8 — If H5 is still live after Steps 4-7, diff the build (E5)

Read-only comparison of `~/workspace/executorch-runtime-dist`'s patches/build flags against
whatever `executorch-numpy-runtime`'s own build pipeline uses (check its repo, `docs/`, CI config).
Lowest priority — only worth doing if nothing else has explained a "numpy-runtime doesn't crash"
result.

### Step 9 — Build a fixture matching the reporter's real op composition (E7)

Highest-priority fixture in this plan — not a generic stand-in anymore. Uses the reporter's own
op list, checked one-by-one against the actual partitioner source (see H9 above): `embedding` and
`squeeze_copy.dims` are unconditionally portable; `cat`/`div`/`sub`/`gelu`/`constant_pad_nd`/
`unsqueeze_copy` are FP32-only for delegation. Kept intentionally small (3 int64 + 2 float32
inputs, not the reporter's full 41) — Step 11 isolates *count* separately, so this fixture's job is
purely to exercise every op in the reporter's list, in a dtype context that matches what the real
partitioner would actually decide:

```python
class TabularLike(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeds = torch.nn.ModuleList(
            [torch.nn.Embedding(100, 4) for _ in range(3)]
        )
        self.register_buffer("mean", torch.tensor(0.5))
        self.register_buffer("std", torch.tensor(2.0))
        self.linear = torch.nn.Linear(3 * 4 + 2, 8)

    def forward(self, i0, i1, i2, f0, f1):
        embedded = []
        for idx, emb in zip((i0, i1, i2), self.embeds):
            e = emb(idx)                              # aten_embedding_default -- always portable
            e = e.squeeze(dim=(0,))                    # aten_squeeze_copy_dims -- always portable
            embedded.append(e)

        feats = []
        for x in (f0, f1):
            y = x.unsqueeze(-1)                         # aten_unsqueeze_copy_default -- trailing dim, FP32-eligible
            y = torch.sub(y, self.mean)                 # aten_sub_tensor -- FP32-eligible (tensor, not python scalar,
            y = torch.div(y, self.std)                  #   to trace as .Tensor not .Scalar)
            feats.append(y)                              # aten_div_tensor -- FP32-eligible

        cat = torch.cat(embedded + feats, dim=-1)        # aten_cat_default -- FP32-eligible (2-5 tensors)
        cat = torch.nn.functional.pad(cat, (0, 4))        # aten_constant_pad_nd_default -- FP32-eligible
        return torch.nn.functional.gelu(self.linear(cat))  # aten_gelu_default -- FP32-eligible, post-Linear
```

```python
example_inputs = (
    torch.zeros(1, dtype=torch.int64), torch.zeros(1, dtype=torch.int64), torch.zeros(1, dtype=torch.int64),
    torch.ones(1, dtype=torch.float32), torch.ones(1, dtype=torch.float32),
)
exported = export(TabularLike().eval(), example_inputs)
lowered = to_edge_transform_and_lower(
    exported, partitioner=[XnnpackPartitioner()]
).to_executorch(
    config=ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
    )
)
```

After export, confirm the graph actually is partial before trusting any result from it — check for
**more than one** `executorch_call_delegate` reference, or a mix of delegate and portable-kernel
ops, the same way `export_conv.py` already checks for `XnnpackBackend` presence. A "single
call_delegate" result here (or, symmetrically, zero delegate references — the partitioner declining
to claim anything at all) means the fixture didn't achieve its purpose — reconfirm each op's
precision-type constraint against the actual partitioner source rather than guessing again, since
that's what went wrong. Once confirmed partial, run through `et_leak_harness` same as Step 3.

### Step 10 — Hand the reporter a Windows runbook (E8, reporter-executed)

`winbox` is this project's own test host — it can run *our* synthetic fixtures (Step 9's, if built),
but the reporter's actual `.pte` can never be copied there. The only way to test their real model on
Windows is for the reporter to run it themselves, in their own environment, on a Windows machine,
and report back the observed result — never the artifact itself. Two options to hand them, cheaper
one first, mirroring the isolation test they already ran on Linux:

1. **Repeat their exact JVM-level isolation test, on Windows.** They already built fresh, correctly-
   sized direct `ByteBuffer`s and drove `EtNative.loadModule`/`EtNative.forward` directly on Linux.
   Ask them to repeat that identical test on a Windows machine, using the `windows-x86_64` build of
   this engine (the published artifact, if they consume this project via a dependency, or built
   from source per `docs/building.md`'s Windows section if they build from source already). Report
   back: does it crash, and if so with what symptom (does it match the Linux XNNPACK SIGSEGV, the
   Linux OpenVINO `RuntimeException`, something else, or does it run clean)?
2. **If (1) reproduces and they're willing to dig further: point our native QA harness at their
   local `.pte` themselves, without sharing it.** Hand them build instructions for the
   `-DET_BUILD_QA=ON` tree (`native/build.sh`'s Windows path, no `JAVA_HOME` needed, same as the
   Linux QA tree) and ask them to run `et_leak_harness.exe <their local .pte path> 1 2` locally, on
   their own machine, and report the exit code and any printed output. This isolates JNI/JVM
   entirely from the native core — the same separation Step 6 (E4) gets for our own synthetic
   fixtures — but run entirely inside their environment, so the artifact never has to leave it.

Record whatever they report against both H8's table entries above before drawing a conclusion — a
report of "ran clean on Windows" or "crashed identically on Windows" is exactly as informative here
as if we'd run it ourselves, since the mechanism under test (compiler, kernel selection) lives in
the platform, not in who presses the button.

### Step 11 — Isolate input *count* from input *dtype* (E9)

Only worth doing if Step 9's fixture crashes, to find out which variable actually matters. Export a
model with ~41 `unplanned` scalar inputs, **all float32**, all delegated (e.g. concatenate all 41
into one vector, feed a Linear) — no int64, no non-delegated op. If this crashes too, the count
itself is implicated regardless of dtype, strengthening the original count-based hypothesis this
step exists to test in isolation. If it's clean, that isolates the cause to dtype/op-composition
(H9/H8) rather than sheer input count, since Step 9's fixture and this one differ in exactly that
one respect.

## What this plan deliberately does not do

- Does not assume XNNPACK's over-read is the answer just because it was the first hypothesis
  investigated (#67) — that investigation only proved padding *sufficient for the two kernels it
  tested*, never *sufficient in general*.
- Does not propose a fix before a reproducing artifact exists. Every guard shipped so far (#67,
  #68, #70) was verified against a real, local, reproducing case — this investigation should hold
  the same bar before writing more code.
- Does not treat a `executorch-numpy-runtime` PASS as proof this engine is broken, given the
  version-mismatch confound — flagged explicitly above rather than glossed over.
- Does not carry forward "Windows lacks optimized kernels" as this engine's reason to test there —
  checked against `native/CMakeLists.txt` and found not to apply (this engine never links
  `optimized_ops_lib` on any platform). Windows testing is still in the plan, on the corrected
  rationale (MSVC vs. GCC UB divergence, untested by the Linux-only UBSan gate).
- Does not treat "41 inputs" as self-evidently the cause just because it's an unusually large
  number relative to this repo's fixtures. Checked the vendored ExecuTorch headers for a hardcoded
  small-array input-count limit and found none — inconclusive, not a ruling-out, so Step 11 exists
  to test count in isolation from dtype rather than let the two stay conflated.
