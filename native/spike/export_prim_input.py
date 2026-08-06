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
"""Exports prim_input.pte: a method whose input 1 is a non-tensor (prim double).

The engine builds a tensor for every input, so it cannot drive a method with a prim input --
EtRuntime rejects such a model at load. This fixture is what proves that rejection fires; there is
no supported way to run it.

torch.export keeps `alpha` as a placeholder even though the traced value is specialized into the
multiply, and to_executorch carries it through as a non-tensor input. MethodMeta reports
`Tag: 3 input: 1 is not Tensor`, so input_tensor_meta(1) fails and buildMethodMeta leaves
scalarType = -1 / memoryPlanned = 0 for that slot -- the conflation F3 is about.
"""

import torch
from torch.export import export
from executorch.exir import to_edge


class MulByPrim(torch.nn.Module):
    def forward(self, x, alpha):
        return x * alpha


def main() -> None:
    ep = export(MulByPrim(), (torch.ones(4), 2.0))
    prog = to_edge(ep).to_executorch()
    with open("prim_input.pte", "wb") as f:
        f.write(prog.buffer)
    print("wrote prim_input.pte")


if __name__ == "__main__":
    main()
