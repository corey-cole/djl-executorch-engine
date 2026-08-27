# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch==2.13.0",
#   "executorch==1.4.1",
#   "openvino==2025.4.1",
#   "nncf==3.1.0",
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
"""Export openvino_planned_in_unplanned_out.pte -- the OpenVINO analog of
native/spike/export_tabular_like_planned_in_unplanned_out.py (investigation:
docs/superpowers/plans/2026-08-26-unplanned-sigsegv-root-cause.md). That fixture isolated
alloc_graph_output=False from alloc_graph_input on XNNPACK and found it crashes the delegate
regardless of input planning. Step 9/E6 of the same investigation tested OpenVINO with
alloc_graph_input=False (planned output) and found it runs clean -- but never tested the output-only
side. This script closes that gap: default (planned) input, alloc_graph_output=False alone, on the
OpenVINO delegate.

Run with the checkout venv so flatc is on PATH (delegated exports need it):

    PATH=$HOME/workspace/executorch/.venv/bin:$PATH uv run export_openvino_planned_in_unplanned_out.py

Writes openvino_planned_in_unplanned_out.pte into the current working directory.
"""
import torch
from torch.export import export
from executorch.backends.openvino.partitioner import OpenvinoPartitioner
from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.passes import MemoryPlanningPass


class TinyLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        return torch.relu(self.linear(x))


def main() -> None:
    torch.manual_seed(20260826)
    model = TinyLinear().eval()
    example_inputs = (torch.ones(1, 4, dtype=torch.float32),)
    exported = export(model, example_inputs)

    compile_spec = [CompileSpec("device", b"CPU")]
    lowered = to_edge_transform_and_lower(
        exported, partitioner=[OpenvinoPartitioner(compile_spec)]
    ).to_executorch(
        config=ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(alloc_graph_output=False)
        )
    )

    buffer = lowered.buffer
    if b"OpenvinoBackend" not in buffer:
        raise SystemExit(
            "openvino_planned_in_unplanned_out.pte contains no OpenvinoBackend delegate"
        )

    with open("openvino_planned_in_unplanned_out.pte", "wb") as f:
        f.write(buffer)
    print("wrote openvino_planned_in_unplanned_out.pte")


if __name__ == "__main__":
    main()
