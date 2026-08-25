# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch==2.13.0",
#   "torchvision==0.28.0",
#   "executorch==1.4.1",
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
  - mobilenet_v2.pte  (torch.export -> XNNPACK to_edge_transform_and_lower -> to_executorch,
                       default memory planning: inputs are memory-planned)
  - mobilenet_v2_unplanned.pte  (same weights, alloc_graph_input=False: inputs are borrowed)
  - mobilenet_v2.pt   (torch.jit.trace -> torch.jit.save)  [.pt: DJL PyTorch resolves by model name]
  - versions.json     ({torch, torchvision, executorch} for reproducibility)
  - mobilenet_v2.etrecord  (only with --etrecord: the ExecuTorch Inspector needs it to attribute
                            runtime events to graph ops)

The .pte uses the general single-tensor path, so NO model_spec.json is emitted.

Note on pins: executorch==1.4.1 requires torch>=2.13.0a0 (see its PyPI metadata) and
torchvision==0.28.0 requires exactly torch==2.13.0, which fixes the trio. The
`[tool.uv]` index override pulls torch/torchvision from the CPU-only wheel index
(download.pytorch.org/whl/cpu) so this script doesn't drag in multi-GB CUDA dependencies -
executorch itself still comes from the default PyPI index.
"""
import argparse
import json
from importlib.metadata import PackageNotFoundError, version

import torch
import torchvision
from torch.export import export
from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.passes import MemoryPlanningPass
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


def _v(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--etrecord",
        action="store_true",
        help="also emit mobilenet_v2.etrecord, which the ExecuTorch Inspector needs to attribute "
        "runtime events to graph ops. Off by default: an ETRecord embeds the program buffer and "
        "the graph modules, and the common case for this script is producing a demo model.",
    )
    args = parser.parse_args()
    weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
    model = torchvision.models.mobilenet_v2(weights=weights).eval()
    example = (torch.randn(1, 3, 224, 224),)

    # ExecuTorch .pte, XNNPACK-lowered (default export config: memory-planned inputs).
    lowered = to_edge_transform_and_lower(
        export(model, example),
        partitioner=[XnnpackPartitioner()],
    )
    program = lowered.to_executorch()
    with open("mobilenet_v2.pte", "wb") as f:
        f.write(program.buffer)
    if args.etrecord:
        # Imported here, not at module top: the default path (./gradlew :example:exportModels never
        # passes --etrecord) must not depend on executorch.devtools importing. A broken devtools in
        # the pinned env would otherwise take down the default export too.
        from executorch.devtools import generate_etrecord
        generate_etrecord("mobilenet_v2.etrecord", lowered, program)

    # Same weights, alloc_graph_input=False: ExecuTorch borrows the input pointer
    # (share_tensor_data) instead of memcpy'ing into the arena — the W5 input A/B arm.
    # Pattern: native/spike/export_w4_models.py calls to_executorch() twice on one lowered.
    unplanned = lowered.to_executorch(
        config=ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
        )
    )
    with open("mobilenet_v2_unplanned.pte", "wb") as f:
        f.write(unplanned.buffer)

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

    etrecord = ", mobilenet_v2.etrecord" if args.etrecord else ""
    print(f"wrote mobilenet_v2.pte, mobilenet_v2_unplanned.pte, mobilenet_v2.pt, versions.json{etrecord}")


if __name__ == "__main__":
    main()
