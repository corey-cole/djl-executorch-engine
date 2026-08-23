# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch==2.13.0",
#   "torchvision==0.28.0",
#   "executorch==1.4.1",
#   "openvino==2025.4.1",
#   "nncf==3.1.0",
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
"""Export MobileNetV2 to ExecuTorch (.pte) lowered through the OpenVINO delegate.

Run with uv so the pinned deps are provisioned automatically:

    uv run tools/scripts/export_mobilenet_openvino.py

Writes into the current working directory:
  - mobilenet_v2_openvino.pte  (torch.export -> OpenvinoPartitioner(device=CPU) -> to_executorch)
  - versions_openvino.json     ({torch, torchvision, executorch, openvino} for reproducibility)

Separate from export_mobilenet.py rather than a fourth output in it. The OpenVINO AOT path needs
two extra wheels that together dominate the environment, for a delegate that runs on linux-x86_64
only -- folding them in would make the default `:example:exportModels` heavier for everyone who
never touches OpenVINO.

nncf is a dependency of the *import*, not of anything this script does: importing
executorch.backends.openvino.partitioner executes the package __init__, which imports
OpenVINOQuantizer, which imports nncf at module scope. Nothing here quantizes.

The openvino wheel is likewise an AOT requirement, not just a runtime one -- the partitioner
imports openvino.frontend.pytorch at module scope. It is unrelated to the OpenVINO *runtime* the
engine ships as its own opt-in jar; this script only produces a .pte.
"""
import json
from importlib.metadata import PackageNotFoundError, version

import torch
import torchvision
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.backend.backend_details import CompileSpec
from executorch.backends.openvino.partitioner import OpenvinoPartitioner


def _v(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "unknown"


def _name(target) -> str:
    return getattr(target, "__name__", str(target))


def main() -> None:
    # Same seed as the other fixture exports: the example must be reproducible run to run.
    # MobileNetV2 loads pretrained weights, so this only fixes the tracing input.
    torch.manual_seed(20260816)

    weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
    model = torchvision.models.mobilenet_v2(weights=weights).eval()
    example = (torch.randn(1, 3, 224, 224),)

    # "device" CompileSpec picks the OpenVINO device; CPU is the only plugin the engine ships.
    partitioner = OpenvinoPartitioner([CompileSpec("device", b"CPU")])
    lowered = to_edge_transform_and_lower(export(model, example), partitioner=[partitioner])
    buffer = lowered.to_executorch().buffer

    # Two assertions, because a partitioner that declines nodes fails SILENTLY -- it still emits a
    # valid .pte, one that benchmarks portable CPU while claiming to benchmark OpenVINO.
    if b"OpenvinoBackend" not in buffer:
        raise SystemExit("export_mobilenet_openvino: .pte contains no OpenvinoBackend delegate")

    # Stronger than the substring check: assert the WHOLE graph went to the delegate. A partial
    # lowering leaves the undelegated ops as extra call_function nodes beside the delegate call.
    nodes = [n for n in lowered.exported_program().graph_module.graph.nodes if n.op == "call_function"]
    delegated = [n for n in nodes if _name(n.target) == "executorch_call_delegate"]
    residual = [n for n in nodes if _name(n.target) not in ("executorch_call_delegate", "getitem")]
    if len(delegated) != 1 or residual:
        raise SystemExit(
            "export_mobilenet_openvino: MobileNetV2 did not fully lower to OpenVINO "
            f"({len(delegated)} delegate calls, {len(residual)} undelegated ops: "
            f"{[str(n.target) for n in residual[:10]]}). Benchmarking this would compare "
            "OpenVINO against a mix of OpenVINO and portable CPU."
        )

    with open("mobilenet_v2_openvino.pte", "wb") as f:
        f.write(buffer)

    with open("versions_openvino.json", "w") as f:
        json.dump(
            {
                "torch": _v("torch"),
                "torchvision": _v("torchvision"),
                "executorch": _v("executorch"),
                "openvino": _v("openvino"),
            },
            f,
            indent=2,
        )

    print(f"wrote mobilenet_v2_openvino.pte ({len(buffer)} bytes), versions_openvino.json")


if __name__ == "__main__":
    main()
