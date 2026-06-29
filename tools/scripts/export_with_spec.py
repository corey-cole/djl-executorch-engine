"""Generate the `priced` named-parameter fixture: priced.pte + model_spec.json.

This is the fixture generator for one specific model (Priced below). To make a fixture for a
different named-parameter model, copy this script and change the model class + example_inputs;
the metadata extraction (names from forward()'s signature, dtype/shape from example_inputs) is reusable.

Produced with ExecuTorch v1.3.1 (portable kernels; exported without the XNNPACK partitioner).
Run from the output directory: it writes priced.pte + model_spec.json into the cwd.

Note: parameter names/order come from the Python forward() signature, which matches the export
graph's input order for simple models (no *args/**kwargs/elided inputs).
"""
import inspect
import json
from importlib.metadata import PackageNotFoundError, version
import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower


class Priced(torch.nn.Module):
    def forward(self, price, qty):
        return price * qty.to(torch.float32)


def main() -> None:
    model = Priced().eval()
    example_inputs = (torch.ones(1, dtype=torch.float32), torch.ones(1, dtype=torch.int64))

    names = list(inspect.signature(model.forward).parameters)  # ['price', 'qty']
    inputs_meta = [
        {
            "name": name,
            "position": pos,
            "dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(t.shape),
        }
        for pos, (name, t) in enumerate(zip(names, example_inputs))
    ]

    lowered = to_edge_transform_and_lower(export(model, example_inputs)).to_executorch()
    with open("priced.pte", "wb") as f:
        f.write(lowered.buffer)
    try:
        et_version = version("executorch")
    except PackageNotFoundError:
        et_version = "unknown"
    with open("model_spec.json", "w") as f:
        json.dump(
            {"runtime": "executorch", "executorch_version": et_version, "inputs": inputs_meta},
            f,
            indent=2,
        )
    print("wrote priced.pte and model_spec.json")


if __name__ == "__main__":
    main()
