"""Export a trivial (a + b) model to add.pte for desktop runtime smoke testing."""
import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
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
    ).to_executorch()
    with open("add.pte", "wb") as f:
        f.write(lowered.buffer)
    print("wrote add.pte")


if __name__ == "__main__":
    main()
