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
"""Export a trivial (a + b) model to add_unplanned.pte for desktop runtime smoke testing.

Same model as export_add.py, but exported with MemoryPlanningPass(alloc_graph_input=False):
the inputs are borrowed, not copied into ExecuTorch's arena at set_input. This is the
fixture behind the `inputMemoryPlanned == false` path of MethodMeta.

Run with uv so the pinned deps are provisioned automatically:

    uv run export_add_unplanned.py

Writes add_unplanned.pte into the current working directory.
"""
import torch
from torch.export import export
from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.passes import MemoryPlanningPass
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


class Add(torch.nn.Module):
    def forward(self, a, b):
        return a + b


def main() -> None:
    model = Add().eval()
    example_inputs = (torch.ones(1), torch.ones(1))
    exported = export(model, example_inputs)
    lowered = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()]
    ).to_executorch(
        config=ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
        )
    )
    with open("add_unplanned.pte", "wb") as f:
        f.write(lowered.buffer)
    print("wrote add_unplanned.pte")


if __name__ == "__main__":
    main()
