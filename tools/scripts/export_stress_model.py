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
"""Export the threading/workspace stress model: stress_mlp.pte + stress_golden.json.

Both files are written in ONE run, on purpose. The goldens are digests of this exact .pte's
output, so a regenerated model with stale goldens is a silent wrong-answer bug. Writing them
together is the mechanism that prevents drift -- never regenerate one without the other.

Shape of the model (see the design doc, section 3):

  branch(x, layers):
      sel = clamp(int64(x[0,0] * N_BUCKETS), 0, N_BUCKETS-1)   # data-dependent
      g   = index_select(table, 0, sel)                        # portable kernel -> SERIAL arm
      h   = x + g
      for lin in layers: h = relu(lin(h))                      # XNNPACK -> PARALLEL arm
  forward(x1, x2) = branch(x1, A) + branch(x2, B)

ExecuTorch has NO inter-op parallelism -- the graph is walked as a single instruction stream, so
the two branches run one after the other. The serial/parallel mix comes from KERNEL choice, not
graph topology: index_select is not lowered by the XNNPACK partitioner and runs single-threaded,
while Linear is lowered and runs on the shared pthreadpool holding an XNNPACK workspace. The
workspace mutex is only observable in the Linear stack.

BATCH is load-bearing. At batch 1 the Linear stack is a GEMV -- memory-bandwidth-bound, so DRAM
becomes the bottleneck and masks the workspace lock entirely. At batch 32 it is a real GEMM,
compute-bound, with ~1 MB of weights that stay resident in L2.

TUNED CONSTANTS (measured, not estimated -- see the design doc, section 3.4). On the measurement
host (11th Gen Intel Core i7-1185G7 @ 3.00 GHz, 4P/8T) the authoritative native figure is
ET_INTRAOP_THREADS=1 ./native/bench/et_scaling_harness ... 1 2000 200:

  DEPTH=4: 284 us/forward at one intra-op thread -- just under the 300-500 us target band.
  DEPTH=5: 354 us/forward (350.9/354.4/357.6) -- in band. Cost scales linearly with DEPTH; the
           ~1.25 MB of weights per branch stay L2-resident, so DEPTH is the cheapest knob.

Run from the repo root:

    PATH=$HOME/workspace/executorch/.venv/bin:$PATH \
        uv run tools/scripts/export_stress_model.py

Writes both files into src/test/resources/models/stress/.
"""
import json
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# --- tuned constants; see the design doc section 3.4 -------------------------------------------
SEED = 20260808
BATCH = 32
HIDDEN = 256
DEPTH = 5
N_BUCKETS = 64
RAMP = 1e-5          # input ramp step; travels to Java via stress_golden.json (config.ramp) — do not hardcode it on the Java side
SAMPLE_COUNT = 16    # strided output samples recorded per case
# -----------------------------------------------------------------------------------------------

OUT_DIR = Path("src/test/resources/models/stress")

# (name, v1, v2). v is in [0, 1); bucket = int(v * N_BUCKETS). Chosen to sit on the first bucket,
# the last bucket, and either side of two interior boundaries, for each input independently -- so
# an off-by-one or mis-marshalled bucket index is caught rather than averaged away.
CASES = [
    ("b0_b0",       0.0000,  0.0000),
    ("b0_blast",    0.0000,  0.9999),
    ("blast_b0",    0.9999,  0.0000),
    ("blast_blast", 0.9999,  0.9999),
    ("b15_lo",      15.999 / N_BUCKETS, 0.5),
    ("b16_hi",      16.001 / N_BUCKETS, 0.5),
    ("b31_lo",      0.5, 31.999 / N_BUCKETS),
    ("b32_hi",      0.5, 32.001 / N_BUCKETS),
]


class StressNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Parameter(
            torch.randn(N_BUCKETS, HIDDEN) * 0.05, requires_grad=False
        )
        self.a = torch.nn.ModuleList(
            [torch.nn.Linear(HIDDEN, HIDDEN) for _ in range(DEPTH)]
        )
        self.b = torch.nn.ModuleList(
            [torch.nn.Linear(HIDDEN, HIDDEN) for _ in range(DEPTH)]
        )

    def branch(self, x, layers):
        sel = (x[0, 0] * N_BUCKETS).to(torch.int64).clamp(0, N_BUCKETS - 1)
        g = torch.index_select(self.table, 0, sel.reshape(1))  # (1, HIDDEN), broadcasts over BATCH
        h = x + g
        for lin in layers:
            h = torch.relu(lin(h))
        return h

    def forward(self, x1, x2):
        return self.branch(x1, self.a) + self.branch(x2, self.b)


def build_input(v: float) -> torch.Tensor:
    """Deterministic input tensor from a single scalar.

    Reproduced bit-for-bit in Java by StressTranslator.buildInput. Every arithmetic step is done
    in float32 in BOTH implementations, in the same order -- `(float) i * RAMP + v` -- so the two
    agree exactly. Doing the ramp in float64 and narrowing at the end would differ by an ulp and
    silently break the bitwise self-reference check.

    Element [0, 0] is exactly v (index 0 contributes a zero ramp term), which is what steers the
    bucket.
    """
    idx = torch.arange(BATCH * HIDDEN, dtype=torch.float32)
    flat = idx * torch.tensor(RAMP, dtype=torch.float32) + torch.tensor(v, dtype=torch.float32)
    return flat.reshape(BATCH, HIDDEN)


def main() -> None:
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = StressNet().eval()
    example_inputs = (build_input(0.5), build_input(0.5))

    lowered = to_edge_transform_and_lower(
        export(model, example_inputs), partitioner=[XnnpackPartitioner()]
    ).to_executorch()
    pte_path = OUT_DIR / "stress_mlp.pte"
    pte_path.write_bytes(lowered.buffer)
    print(f"wrote {pte_path} ({pte_path.stat().st_size} bytes)")

    # Goldens come from running the exported .pte through the ExecuTorch runtime, NOT from torch
    # eager: eager uses different kernels than XNNPACK, so eager goldens would either fail
    # spuriously or force a tolerance so loose it proves nothing.
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    method = program.load_method("forward")

    stride = (BATCH * HIDDEN) // SAMPLE_COUNT
    cases = []
    for name, v1, v2 in CASES:
        out = method.execute([build_input(v1), build_input(v2)])[0]
        flat = out.reshape(-1)
        cases.append(
            {
                "name": name,
                "v1": v1,
                "v2": v2,
                "sum": float(flat.to(torch.float64).sum()),
                "absSum": float(flat.to(torch.float64).abs().sum()),
                "maxAbs": float(flat.abs().max()),
                "samples": [float(flat[i * stride]) for i in range(SAMPLE_COUNT)],
            }
        )

    # Coarse cost signal only. The Python runtime is a different build with different overheads --
    # the authoritative one-intra-op-thread figure for tuning against the 300-500us target comes
    # from  ET_INTRAOP_THREADS=1 ./native/bench/et_scaling_harness <pte> 1 2000 200
    # (per_thread_mean_ms at threads=1 IS the per-forward cost at one intra-op thread;
    # et_timing_harness does NOT read ET_INTRAOP_THREADS and measures at the default pool).
    warm = [build_input(0.5), build_input(0.5)]
    for _ in range(50):
        method.execute(warm)
    t0 = time.perf_counter()
    reps = 300
    for _ in range(reps):
        method.execute(warm)
    us = (time.perf_counter() - t0) / reps * 1e6

    try:
        et_version = version("executorch")
    except PackageNotFoundError:
        et_version = "unknown"

    golden = {
        "executorchVersion": et_version,
        "seed": SEED,
        "config": {
            "batch": BATCH,
            "hidden": HIDDEN,
            "depth": DEPTH,
            "nBuckets": N_BUCKETS,
            "ramp": RAMP,
        },
        "sampleStride": stride,
        "measuredUsPerForward": round(us, 1),
        "cases": cases,
    }
    golden_path = OUT_DIR / "stress_golden.json"
    golden_path.write_text(json.dumps(golden, indent=2) + "\n")
    print(f"wrote {golden_path}")
    print(f"measured {us:.1f} us/forward (python runtime, default intra-op threads)")
    print("target is 300-500 us at ONE intra-op thread; retune BATCH/HIDDEN/DEPTH if far off")


if __name__ == "__main__":
    main()
