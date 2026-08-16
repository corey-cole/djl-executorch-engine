# Documentation

Current reference material lives here. Point-in-time records live in
[research/](research), which keeps the reasoning behind past decisions without presenting it as
current guidance.

## Current

| Document | What it covers |
|---|---|
| [building.md](building.md) | Building the native shim and running the test suites from source |
| [native-architecture.md](native-architecture.md) | The C++ layer: core/JNI split, ownership, staging, process-global state |
| [benchmarking.md](benchmarking.md) | The timing harness and how to read its output |
| [ci-native-build.md](ci-native-build.md) | How the native build matrix works in CI |
| [executorch-build-notes.md](executorch-build-notes.md) | Building an ExecuTorch runtime from source (escape hatch) |
| [executorch-host-buffer-contract-brief.md](executorch-host-buffer-contract-brief.md) | The host-buffer contract: when ExecuTorch copies an input and when it borrows |
| [openvino-version-bump.md](openvino-version-bump.md) | Checklist for bumping the vendored OpenVINO runtime |

Design specs and implementation plans are under [superpowers/](superpowers).

## Research and historical records

[research/](research) holds completed work orders, superseded working notes, and open research.
Each file carries a header saying what it is and when it was written. Nothing there is current
guidance; read it for reasoning, not instructions.
