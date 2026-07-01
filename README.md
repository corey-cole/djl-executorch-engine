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
- **An ExecuTorch v1.3.x checkout**, default `$HOME/workspace/executorch` (override with `ET_ROOT`).
- **JDK 17** on the host for Gradle. (The native build fetches its own JDK *headers* — you do not
  need a JDK inside the container.)
- **Network access** — the container downloads the pinned torch wheel and (for QA) fetches Catch2.

### 1. Build the native library

The engine loads a native `libexecutorch_djl.so` that is **built from source, not committed to the
repo**. Build it with the container wrapper (it downloads the JDK RPM and runs `native/build.sh`
inside `manylinux_2_28`):

```bash
./native/local_build_wrapper.sh
```

This builds the ExecuTorch runtime (installed to `et-install/`), then our JNI shim, and stages
`libexecutorch_djl.so` into `src/main/resources/native/linux-x86_64/`. The first run is slow
(multi-minute ExecuTorch build). To rebuild only the shim afterward and reuse the runtime:

```bash
SKIP_ET_BUILD=1 ./native/local_build_wrapper.sh
```

### 2. Run the tests

The JVM integration tests load the native `.so`, so **build it first (step 1)** — native tests
require a build. Then:

```bash
./gradlew test        # unit + native integration tests
./gradlew leakTest    # JVM-side memory-leak stress test (constrained heap/direct memory)
```

### 3. Native QA (optional)

AddressSanitizer/LeakSanitizer Catch2 units + the leak harness, built against the runtime from
step 1:

```bash
./native/build_qa.sh
```