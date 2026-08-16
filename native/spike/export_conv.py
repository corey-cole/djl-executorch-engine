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
"""Export conv.pte — the fixture that makes XNNPACK allocate a workspace arena.

Delegating and allocating are different properties, and every other delegated fixture in
this directory has the first without the second. An elementwise add and a Linear both hand
their whole graph to XnnpackPartitioner, but a Linear lowers to a direct GEMM over
statically packed weights and an add is a single node with external input and output, so
neither grows the arena in xnn_create_runtime_v4 -- both report a workspace of exactly 0.
A conv does grow it, which is what makes measly::et::xnnpackWorkspaceBytes() assertable
rather than vacuously zero.

Dims mirror the equivalent fixture in the sibling executorch-numpy-runtime, so the two
projects assert the same property against the same shape.

Run with the checkout venv so flatc is on PATH (delegated exports need it):

    PATH=$HOME/workspace/executorch/.venv/bin:$PATH uv run export_conv.py

Writes conv.pte into the current working directory.
"""
import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


class ConvRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)

    def forward(self, x):
        return torch.relu(self.conv(x))


def main() -> None:
    torch.manual_seed(20260816)
    model = ConvRelu().eval()
    exported = export(model, (torch.randn(1, 3, 16, 16),))
    lowered = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()]
    ).to_executorch()

    buffer = lowered.buffer
    # The whole point of this fixture is the delegate. An export that silently fell back to
    # portable kernels would still load and still forward, and would report a workspace of 0
    # -- indistinguishable from the accessor being broken, which is the bug this fixture exists
    # to rule out.
    if b"XnnpackBackend" not in buffer:
        raise SystemExit("conv.pte contains no XnnpackBackend delegate")

    with open("conv.pte", "wb") as f:
        f.write(buffer)
    print("wrote conv.pte")


if __name__ == "__main__":
    main()
