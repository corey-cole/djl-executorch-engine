# ExecuTorch Engine for DJL

As of the most recent version (0.36.0), DJL only supports the TorchScript export API.  This API has been deprecated for several point release versions
of PyTorch.  One of the export APIs present in PyTorch is the [ExecuTorch](https://executorch.ai/) backend.  This is a lightweight integration layer
that was built to be cross-language and is intended for edge model deployment.

The goal of this project is to provide an ExecuTorch engine for DJL such that PyTorch based models can make use of the newer export APIs.
As a separate engine, it also allows for slow migration from TorchScript/PyTorch to this new backend.

## Building and testing

> **Status:** desktop **linux-x86_64** only, and a work in progress. These steps build what exists
> today — the ExecuTorch runtime, our JNI shim, and the JVM + native test suites.

### Prerequisites

- **Docker** — the native library is built inside a `manylinux_2_28` container. ExecuTorch 1.3
  pins `torch==2.12.0`, whose wheel needs **glibc ≥ 2.28**, so the artifact's floor is glibc 2.28
  (covers RHEL/Rocky 8+, Ubuntu 20.04+, Debian 11+).
- No ExecuTorch checkout needed — CMake downloads the pinned runtime. Network access is required for
  the tarball fetch (and Catch2 in QA).
- **JDK 17** on the host for Gradle. (The native build fetches its own JDK *headers* — you do not
  need a JDK inside the container.)

### 1. Build the native library

The engine loads a native `libexecutorch_djl.so` that is **built from source, not committed**. The
ExecuTorch **runtime** it links against is **not built here** — CMake downloads a hash-pinned,
attested tarball published by [`executorch-runtime-dist`](https://github.com/measly-java-learning/executorch-runtime-dist)
(`native/cmake/EtRuntimePin.cmake`). Build the shim with the container wrapper:

```bash
./native/local_build_wrapper.sh
```

The wrapper launches a `manylinux_2_28` container and runs the build **inside it**, so the staged
`.so` keeps its **glibc-2.28 floor** (RHEL8+). Inside the container, CMake `FetchContent`s the pinned
`logging` runtime, compiles the shim, and stages it into `src/main/resources/native/linux-x86_64/`.
It is fast — there is no ExecuTorch build.

**Local fast path (do NOT ship):** to iterate quickly you can run `./native/build.sh` directly on the
host (no Docker). The resulting `.so` links against a host-glibc runtime and **breaks the 2.28 floor**
— fine for your own `./gradlew test`, never for a release.

**Escape hatch / custom runtime:** set `ET_INSTALL=/path/to/et-install` to link an existing runtime
tree (e.g. one you built from source per `docs/executorch-build-notes.md`); CMake skips the download.

**Verifying runtime provenance (optional, local):** CI verifies every pinned tarball with a build
attestation. To check by hand:
```bash
gh attestation verify <downloaded-tarball> --repo measly-java-learning/executorch-runtime-dist
```

### 2. Run the tests

The JVM integration tests load the native `.so`, so **build it first (step 1)**. Then:

```bash
./gradlew test        # unit + native integration tests
./gradlew leakTest    # JVM-side memory-leak stress test
```

### 3. Native QA (optional)

AddressSanitizer/LeakSanitizer Catch2 units + the leak harness. The runtime is fetched by CMake
(or set `ET_INSTALL` for the escape hatch); run in the same `manylinux_2_28` container:

```bash
./native/build_qa.sh
```

### Container file ownership (known gap)

The container builds run as **root**, so anything written into the bind-mounted repo ends up
root-owned on the host. `native/build.sh` mitigates this for **its own** outputs — when the wrapper
passes `HOST_UID`/`HOST_GID`, an `EXIT` trap `chown`s them back to the invoking user
(`native/build` and the staged `src/main/resources/native/linux-*`).

The sibling scripts do **not** yet do this, so they leave root-owned directories behind:

- `native/bench.sh` → `native/bench/`
- `native/build_variants.sh` → `native/bench-results/` (and drives `bench.sh` → `native/bench/`)
- `native/build_qa.sh` → `native/asan/`

Until these grow the same trap, fix ownership by hand after running them, e.g.:

```bash
sudo chown -R "$(id -u):$(id -g)" native/bench native/bench-results native/asan
```