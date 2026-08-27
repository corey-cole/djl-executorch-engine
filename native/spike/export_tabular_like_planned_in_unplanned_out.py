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
"""Export tabular_like_planned_in_unplanned_out.pte -- same fixture as export_tabular_like.py, but with
BOTH alloc_graph_input=False AND alloc_graph_output=False (investigation:
docs/superpowers/plans/2026-08-26-unplanned-sigsegv-root-cause.md).

Critical distinction from every prior fixture in this investigation: alloc_graph_output defaults
to True, and no export script here ever set it explicitly -- meaning every fixture built so far
was actually "unplanned input, PLANNED output", which the reporter confirms WORKS. The reporter's
own failing configuration is "unplanned input, UNPLANNED output" -- this script is the first
fixture in this investigation to actually test it. et_runtime.h's own OutputView doc comment
assumes "Borrowed output: data points into ExecuTorch's host arena" -- an assumption that only
holds when alloc_graph_output=True; this fixture exists to find out what happens when it's False.

3 int64 index inputs (embedding lookups, always portable) + 2 float32 feature inputs (normalized,
then concatenated and padded, all FP32-eligible for XNNPACK delegation). Every op maps to one in
the reporter's list:

  aten_embedding_default        -- always portable (no XNNPACK partitioner config)
  aten_squeeze_copy_dims        -- always portable (no XNNPACK partitioner config)
  aten_unsqueeze_copy_default   -- FP32-eligible, trailing dim only
  aten_sub_tensor                -- FP32-eligible
  aten_div_tensor                -- FP32-eligible
  aten_cat_default               -- FP32-eligible (2-5 tensors)
  aten_constant_pad_nd_default   -- FP32-eligible
  aten_gelu_default              -- FP32-eligible

Run with the checkout venv so flatc is on PATH (delegated exports need it):

    PATH=$HOME/workspace/executorch/.venv/bin:$PATH uv run export_tabular_like_planned_in_unplanned_out.py

Writes tabular_like_planned_in_unplanned_out.pte into the current working directory, and prints the compiled
instruction list (MoveCall/FreeCall counts in particular).
"""
import torch
from torch.export import export
from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.passes import MemoryPlanningPass
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


class TabularLike(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeds = torch.nn.ModuleList(
            [torch.nn.Embedding(100, 4) for _ in range(3)]
        )
        self.register_buffer("mean", torch.tensor(0.5))
        self.register_buffer("std", torch.tensor(2.0))
        self.linear = torch.nn.Linear(3 * 4 + 2 + 4, 8)  # +4 for the constant_pad_nd below

    def forward(self, i0, i1, i2, f0, f1):
        embedded = []
        for idx, emb in zip((i0, i1, i2), self.embeds):
            e = emb(idx)                                  # aten_embedding_default -- [1, 4]
            e = e.unsqueeze(1)                              # aten_unsqueeze_copy_default -- [1, 1, 4]
            e = e.squeeze(dim=(1,))                         # aten_squeeze_copy_dims -- back to [1, 4]
            embedded.append(e)

        feats = []
        for x in (f0, f1):
            y = x.unsqueeze(-1)                            # aten_unsqueeze_copy_default
            y = torch.sub(y, self.mean)                    # aten_sub_tensor
            y = torch.div(y, self.std)                     # aten_div_tensor
            feats.append(y)

        cat = torch.cat(embedded + feats, dim=-1)          # aten_cat_default
        cat = torch.nn.functional.pad(cat, (0, 4))          # aten_constant_pad_nd_default
        return torch.nn.functional.gelu(self.linear(cat))  # aten_gelu_default


def main() -> None:
    torch.manual_seed(20260826)
    model = TabularLike().eval()
    example_inputs = (
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
        torch.ones(1, dtype=torch.float32),
        torch.ones(1, dtype=torch.float32),
    )
    exported = export(model, example_inputs)
    lowered = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()]
    ).to_executorch(
        config=ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(alloc_graph_output=False)
        )
    )

    buffer = lowered.buffer
    n_delegates = buffer.count(b"XnnpackBackend")
    print(f"XnnpackBackend byte-string occurrences in buffer: {n_delegates}")
    if n_delegates == 0:
        raise SystemExit(
            "tabular_like_planned_in_unplanned_out.pte contains no XnnpackBackend delegate -- "
            "the partitioner declined everything, fixture didn't achieve its purpose"
        )

    with open("tabular_like_planned_in_unplanned_out.pte", "wb") as f:
        f.write(buffer)
    print("wrote tabular_like_planned_in_unplanned_out.pte")

    # Dump the compiled instruction list so a MoveCall/FreeCall pair can be confirmed directly,
    # per the investigation plan's H10/H11 convergence finding.
    print("\n--- Instruction list (forward method) ---")
    plan = lowered.executorch_program.execution_plan[0]
    chain = plan.chains[0]
    move_count = 0
    free_count = 0
    delegate_count = 0
    kernel_count = 0
    for i, instr in enumerate(chain.instructions):
        kind = type(instr.instr_args).__name__
        if kind == "MoveCall":
            move_count += 1
        elif kind == "FreeCall":
            free_count += 1
        elif kind == "DelegateCall":
            delegate_count += 1
        elif kind == "KernelCall":
            kernel_count += 1
        print(f"  [{i}] {kind}")
    print(
        f"\nTotals: MoveCall={move_count} FreeCall={free_count} "
        f"DelegateCall={delegate_count} KernelCall={kernel_count}"
    )


if __name__ == "__main__":
    main()
