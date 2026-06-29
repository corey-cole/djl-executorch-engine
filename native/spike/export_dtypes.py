"""Export a mixed-dtype model (int64 a, float32 b) -> (a+a int64, b+b float32) to dtypes.pte.

Exported WITHOUT the XNNPACK partitioner so the portable kernels handle int64 ops.
"""
# Produced with ExecuTorch v1.3.1 (portable kernels; exported without the XNNPACK partitioner).
import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower


class MixedDtypes(torch.nn.Module):
    def forward(self, a, b):
        return a + a, b + b


def main() -> None:
    model = MixedDtypes().eval()
    example_inputs = (torch.ones(1, dtype=torch.int64), torch.ones(1, dtype=torch.float32))
    exported = export(model, example_inputs)
    lowered = to_edge_transform_and_lower(exported).to_executorch()
    with open("dtypes.pte", "wb") as f:
        f.write(lowered.buffer)
    print("wrote dtypes.pte")


if __name__ == "__main__":
    main()
