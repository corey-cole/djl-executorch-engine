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
"""Export the W4 over-read fixtures (clamp5, lin129, lin129_planned).

Same partitioner pattern as export_add_unplanned.py: the whole graph is handed to
XnnpackPartitioner and MemoryPlanningPass decides input memory planning. These three
close the brief's "no delegated variant" gap (§3/W7):

- clamp5.pte        — torch.clamp over 5 floats, alloc_graph_input=False (Route A: N=5, N % 4 == 1).
- lin129.pte        — Linear(129, 64) over 129 floats, alloc_graph_input=False (Route B: K=129, K % 4 == 1).
- lin129_planned.pte — the same Linear with the default (memory-planned) export config (config (b)).

Run with the checkout venv so flatc is on PATH (delegated exports need it):

    PATH=$HOME/workspace/executorch/.venv/bin:$PATH uv run export_w4_models.py

Writes the three .pte files into the current working directory.
"""
import torch
from torch.export import export
from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.passes import MemoryPlanningPass
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


class Clamp(torch.nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=-1.0, max=1.0)


class Linear129(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(129, 64)

    def forward(self, x):
        return self.linear(x)


def main() -> None:
    clamp_model = Clamp().eval()
    clamp_exported = export(clamp_model, (torch.ones(5),))
    clamp_lowered = to_edge_transform_and_lower(
        clamp_exported, partitioner=[XnnpackPartitioner()]
    ).to_executorch(
        config=ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
        )
    )
    with open("clamp5.pte", "wb") as f:
        f.write(clamp_lowered.buffer)
    print("wrote clamp5.pte")

    lin_model = Linear129().eval()
    lin_exported = export(lin_model, (torch.ones(129),))
    lin_lowered = to_edge_transform_and_lower(
        lin_exported, partitioner=[XnnpackPartitioner()]
    )
    lin_unplanned = lin_lowered.to_executorch(
        config=ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
        )
    )
    with open("lin129.pte", "wb") as f:
        f.write(lin_unplanned.buffer)
    print("wrote lin129.pte")

    lin_planned = lin_lowered.to_executorch()  # default config: memory-planned inputs
    with open("lin129_planned.pte", "wb") as f:
        f.write(lin_planned.buffer)
    print("wrote lin129_planned.pte")


if __name__ == "__main__":
    main()
