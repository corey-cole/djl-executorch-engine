# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch==2.13.0",
#   "executorch==1.4.1",
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
"""Export the W5 output-size sweep models (add_4kb..add_64mb).

Same Add() class and partitioner pattern as native/spike/export_add.py, at four
(1, N) f32 output sizes spanning MobileNet's 4 KB logits to a 64 MB tensor — the
output/GC arm of W5 (the heap `byte[]` output copy's cost at scale):

- add_4kb.pte    N=1024     (4 KB f32 output)
- add_256kb.pte  N=65536    (256 KB)
- add_4mb.pte    N=1048576  (4 MB)
- add_64mb.pte   N=16777216 (64 MB)

Default memory planning throughout; the W5 output arm measures the marshalling
cost (copy 3), not the input path. A size that fails to export is dropped and
the sweep continues — the evidence log records the gap.

Run with the checkout venv so flatc is on PATH (delegated exports need it):

    PATH=$HOME/workspace/executorch/.venv/bin:$PATH uv run export_w5_add_models.py

Writes the four .pte files into the current working directory.
"""
import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


class Add(torch.nn.Module):
    def forward(self, a, b):
        return a + b


# model name -> N; output is a (1, N) f32 tensor, 4 * N bytes.
SIZES = {
    "add_4kb": 1024,
    "add_256kb": 65536,
    "add_4mb": 1048576,
    "add_64mb": 16777216,
}


def export_one(name: str, n: int) -> None:
    model = Add().eval()
    example_inputs = (torch.ones(1, n), torch.ones(1, n))
    exported = export(model, example_inputs)
    lowered = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()]
    ).to_executorch()
    with open(f"{name}.pte", "wb") as f:
        f.write(lowered.buffer)
    print(f"wrote {name}.pte")


def main() -> None:
    for name, n in SIZES.items():
        try:
            export_one(name, n)
        except Exception as e:  # drop one size, keep the rest — W5 must not block on it
            print(f"SKIPPED {name} (N={n}): {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
