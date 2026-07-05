# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch==2.12.1",
#   "torchvision==0.27.1",
#   "executorch==1.3.1",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchvision = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Export MobileNetV2 to both ExecuTorch (.pte) and TorchScript (.pt) from the SAME weights.

Run with uv so the pinned deps are provisioned automatically:

    uv run tools/scripts/export_mobilenet.py

Writes into the current working directory:
  - mobilenet_v2.pte  (torch.export -> XNNPACK to_edge_transform_and_lower -> to_executorch)
  - mobilenet_v2.pt   (torch.jit.trace -> torch.jit.save)  [.pt: DJL PyTorch resolves by model name]
  - versions.json     ({torch, torchvision, executorch} for reproducibility)

The .pte uses the general single-tensor path, so NO model_spec.json is emitted.

Note on pins: executorch==1.3.1 requires torch>=2.12.0a0 (see its PyPI metadata), which is
newer than the torch/torchvision pair one might guess from a torchvision release alone.
torch==2.12.1 / torchvision==0.27.1 is the pair `uv` resolves for executorch==1.3.1. The
`[tool.uv]` index override pulls torch/torchvision from the CPU-only wheel index
(download.pytorch.org/whl/cpu) so this script doesn't drag in multi-GB CUDA dependencies -
executorch itself still comes from the default PyPI index.
"""
import json
from importlib.metadata import PackageNotFoundError, version

import torch
import torchvision
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


def _v(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "unknown"


def main() -> None:
    weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
    model = torchvision.models.mobilenet_v2(weights=weights).eval()
    example = (torch.randn(1, 3, 224, 224),)

    # ExecuTorch .pte, XNNPACK-lowered.
    lowered = to_edge_transform_and_lower(
        export(model, example),
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()
    with open("mobilenet_v2.pte", "wb") as f:
        f.write(lowered.buffer)

    # TorchScript .pt from the SAME weights.
    traced = torch.jit.trace(model, example)
    torch.jit.save(traced, "mobilenet_v2.pt")

    with open("versions.json", "w") as f:
        json.dump(
            {
                "torch": _v("torch"),
                "torchvision": _v("torchvision"),
                "executorch": _v("executorch"),
            },
            f,
            indent=2,
        )

    print("wrote mobilenet_v2.pte, mobilenet_v2.pt, versions.json")


if __name__ == "__main__":
    main()
