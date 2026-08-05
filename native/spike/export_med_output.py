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
"""Export a model with a large (512 MiB) f32 output to med_output.pte.

The output (2**27 f32 values) sits far above any constrained test heap but below the 2 GiB JNI
jsize limit, so the OOM test exercises the output-marshalling allocation path without tripping
the issue-10 size guard. Exported WITHOUT the XNNPACK partitioner (portable kernels, like
export_dtypes.py); the expand op is implemented by the portable expand_copy_out kernel.

Run with uv so the pinned deps are provisioned automatically:

    uv run export_med_output.py

Writes med_output.pte into the current working directory.
"""
import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower


class BigOutput(torch.nn.Module):
    def forward(self, x):
        return x.expand((2 ** 27,))


def main() -> None:
    model = BigOutput().eval()
    example_inputs = (torch.ones(1),)
    exported = export(model, example_inputs)
    lowered = to_edge_transform_and_lower(exported).to_executorch()
    with open("med_output.pte", "wb") as f:
        f.write(lowered.buffer)
    print("wrote med_output.pte")


if __name__ == "__main__":
    main()
