# CI Native Build — ExecuTorch Desktop Binaries

> **Status:** design note (2026-06-28). Captures the CI/packaging implications discovered while
> executing the [native-build feasibility plan](superpowers/plans/2026-06-28-executorch-desktop-linux-native-build.md).
> Informs a future "Task 3" (production packaging) and the eventual engine release pipeline.
> Companion to the top-level [`djl-executorch-engine-design.md`](../djl-executorch-engine-design.md).

## Problem

The DJL ExecuTorch engine will bundle a native shared library (`libexecutorch_djl.so` and
its per-platform siblings) so users get inference without a local C++ toolchain. Producing that
binary required **non-standard build options**, so CI cannot simply pull a prebuilt runtime — it
must reproduce the build. This note records what that means.

## What we actually ship (measured)

The spike `.so` (linux-x86_64, built against ExecuTorch v1.3.1) is **self-contained**: the
ExecuTorch runtime, XNNPACK, and op kernels are statically linked in. Its only dynamic
dependencies are the universal C/C++ runtime libraries.

```
$ ldd libexecutorch_djl.so
    libstdc++.so.6, libm.so.6, libgcc_s.so.1, libc.so.6, ld-linux-x86-64.so.2
```

Implication: **distribution is one self-contained `.so` per platform**, dropped into the JAR
under `native/<platform>/`. No transitive native dependency chain to manage (unlike libtorch).

## The central constraint: the build environment sets the portability floor

Because the runtime + XNNPACK are statically baked in, the **glibc/libstdc++ versions of the
build host become the *minimum* the user's machine must have.** Measured on the spike artifact,
which was built on Ubuntu 24.04 (glibc 2.39):

```
$ objdump -T libexecutorch_djl.so | grep -oE 'GLIBC_[0-9.]+' | sort -V | tail -1
GLIBC_2.38
```

A floor of **glibc 2.38** excludes most deployment targets: RHEL/Rocky 8 (2.28), Ubuntu
20.04/22.04 (2.31/2.35), Debian 11 (2.31), Amazon Linux 2 (2.26). **A binary built on a dev box
is not distributable.**

**Fix (industry standard):** build inside an **old-glibc container** so the floor is low:
- `manylinux_2_28` → glibc 2.28 (covers RHEL 8+, modern Ubuntu/Debian). Recommended default.
- `manylinux2014` → glibc 2.17 (maximum reach; matches what PyTorch/ONNX Runtime ship).

This applies to **both** the ExecuTorch runtime build **and** the shim build, since they are
statically combined into one object. macOS and Windows have analogous floors
(`MACOSX_DEPLOYMENT_TARGET`; the MSVC runtime version) that must be pinned the same way.

## Build container

**Baseline requirement:** a manylinux build environment (glibc floor as above). The canonical
choice is the **PyPA** base images — `quay.io/pypa/manylinux2014_x86_64` (glibc 2.17) and the
`_aarch64` variant — but note these ship **no JDK**, so we'd `yum install java-devel` to get the
`jni.h` headers our shim needs.

**Convenient prebuilt alternative:** the **`rayproject`** manylinux2014 image (Ray builds its
production wheels on it). Compared to stock PyPA it adds two things we want for free:
- a **bundled JDK**, which provides `jni.h` directly (slots into our headers-only `JAVA_HOME`
  CMake path), and
- **amd64 + arm64** coverage from one image lineage, knocking out two matrix rows and helping the
  aarch64 question.

It's a convenience, not a requirement — stock PyPA manylinux + `java-devel` is equivalent.

**Headers-only nuance (why the image's JDK version is irrelevant):** our shim includes
`jni.h`/`jni_md.h` and links **nothing** from the JDK — the JVM supplies those symbols at the
user's runtime. So the build image's JDK affects the build only, never what users run; any
JDK 8+ provides stable-enough headers.

**Both builds run in this container.** The ExecuTorch runtime build and the shim build are
statically combined, so the chosen image must satisfy the runtime build's needs too — not just
the shim's.

**Verify before adopting any image** (these double as in-container smoke checks):
1. **C++17 toolchain active.** manylinux2014's *system* gcc is 4.8; C++17 comes from a devtoolset
   (gcc 9/10) that must be enabled (`scl enable` / `/opt/rh/...`). Confirm `g++ --version` ≥ 7 once
   it's on PATH (ExecuTorch is C++17).
2. **Build deps present/installable:** `cmake`, `ninja`, `git`, and a Python 3.10–3.13 for
   `install_executorch.sh` (manylinux images carry several under `/opt/python`).
3. **JDK headers locatable on both arches:** `find "$JAVA_HOME/include" -name jni.h` and
   `include/linux/jni_md.h`; set `JAVA_HOME` accordingly.
4. **Prove the floor empirically** (not from the image's claims) — after building, run the
   portability gate below and assert `<= GLIBC_2.17` (or the floor we commit to).

`rayproject/manylinux2014:260624.8167734-jdk-x86_64` contains:
* gcc (GCC) 10.2.1 20210130 (Red Hat 10.2.1-11)
* g++ (GCC) 10.2.1 20210130 (Red Hat 10.2.1-11)
* Python 3.10.19 (as `python3`)
* cmake version 4.2.1
* openjdk version "1.8.0_412"
* git version 2.52.0

`ninja-build` is not present, so will have to work out whether it's easier to add ninja or use the `PyPA` image

`quay.io/pypa/manylinux2014_x86_64:latest` contains:
* gcc (GCC) 10.2.1 20210130 (Red Hat 10.2.1-11)
* g++ (GCC) 10.2.1 20210130 (Red Hat 10.2.1-11)
* Python 3.10.20 (main, Jun 19 2026, 23:23:54) [GCC 10.2.1 20210130 (Red Hat 10.2.1-11)] on linux (available via `python3.10`)
* Python 3.11.15 (available via `python3.11`)
* Python 3.12.13 (available via `python3.12`)
* cmake version 4.3.4
* git version 2.54.0

Neither `ninja-build` nor `JDK` are present.  Base OS for the PyPA image is CentOS 7, may have slight inconvenience finding packages for those.
(`ninja` is pip installable, so probably not a huge concern)
Amazon provides a stable Java 8 build via the Corretto project: https://corretto.aws/downloads/latest/amazon-corretto-8-x64-linux-jdk.rpm
(This is amd64, but there are also aarch64, Windows, and macOS binaries)

Note that local builds used GCC 13.x, and GCC 10.2.1, while supporting C++17 may have some quirks.



## Which "non-standard options" actually burden CI

| Category | Examples | CI burden |
|---|---|---|
| **Consumption-side** (live in *our* CMake) | `tokenizers_DIR` + prefix path; bare target names (no `executorch::`); rely on targets' self-`--whole-archive`; JNI headers-only via `JAVA_HOME`; config dir is `lib/cmake/ExecuTorch` | **None** — CI just runs our `CMakeLists.txt` |
| **Runtime build-side** (must be reproduced) | `-DCMAKE_POSITION_INDEPENDENT_CODE=ON`; `EXECUTORCH_BUILD_EXTENSION_{MODULE,TENSOR,DATA_LOADER}=ON`; `EXECUTORCH_BUILD_XNNPACK=ON`; `cmake --install` | **High** — forces building ExecuTorch from source (below) |

The CMake quirks we fought through during the spike are essentially free in CI. The cost is
entirely in **having to build the ExecuTorch runtime ourselves.**

## Why CI must build ExecuTorch from source

No consumable prebuilt desktop runtime exists:
- The Maven `org.pytorch:executorch-android` AAR is **Android-only** (and hard-gated; see the
  feasibility plan).
- The `executorch` **pip wheels are the AOT/Python side**, not the C++ static libs +
  `executorch-config.cmake` we link against — and not built with our PIC flag.

So CI must clone ExecuTorch + submodules at a pinned tag and compile it. XNNPACK is the slow
part (multi-minute, cold).

## Recommended pipeline: two stages, decoupled by cost of change

**Stage A — runtime (slow, changes rarely).**
In a manylinux container, per platform:
1. Clone ExecuTorch at a **pinned commit** (not just a tag) with submodules **locked to exact
   SHAs** for reproducibility.
2. Configure with our flags (PIC + extensions + XNNPACK), build, `cmake --install`.
3. Publish the install tree as a cache/artifact **keyed on `(ET_VERSION, flags-hash, platform,
   toolchain)`**. Heavy `ccache`.

   Re-runs only when the version, flags, or toolchain change.

**Stage B — shim + package (fast, per-commit).**
1. Restore the cached runtime install tree.
2. Build `libexecutorch_djl.so` from `native/jni/` against it (our `native/CMakeLists.txt`).
3. Run the JVM smoke test (`add.pte` → `RESULT=5.0`).
4. Stage the `.so` into `src/main/resources/native/<platform>/` for JAR bundling.

**Platform matrix** multiplies Stage A across `linux-x86_64`, `linux-aarch64`, `osx-x86_64`,
`osx-aarch64`, `win-x86_64` — each on a matching runner with its own runtime build.

**Distribution shape:** mirror DJL's own native layout — a separate `...-native-<platform>`
artifact per platform — so the heavy native build is fully decoupled from the pure-Java engine
build and release.

## CI guardrails to add

- **Portability gate.** Fail the build if the glibc floor regresses above the target:
  ```bash
  max=$(objdump -T libexecutorch_djl.so | grep -oE 'GLIBC_[0-9.]+' | sort -V | tail -1)
  # assert: $max <= GLIBC_2.28   (or whichever floor we commit to)
  ```
  Run the equivalent check for `GLIBCXX_`/`CXXABI_`.
- **Self-containment check.** Assert `ldd` shows no `libexecutorch*`/`libxnnpack*` deps — i.e.
  the static linking didn't silently regress to dynamic.
- **License bundling.** XNNPACK, cpuinfo, pthreadpool, eigen_blas (and tokenizers, if linked)
  are now *inside* the `.so`. Their (BSD-family) licenses and notices must ship in the JAR.

## Open questions for the packaging task

- **Build the runtime in CI every release, or pre-build per `ET_VERSION` and store the install
  tree as a release asset / internal artifact?** The latter turns Stage A into a manual/scheduled
  job and makes per-commit CI cheap. Likely preferable given the build cost.
- **Which glibc floor do we commit to?** `manylinux_2_28` (2.28) is a good balance; `manylinux2014`
  (2.17) maximizes reach at some toolchain-age cost.
- **aarch64 strategy** — native ARM runners vs. cross-compile vs. emulation (slow).
- **Reproducibility** — pin ExecuTorch submodule SHAs (the spike used `--depth 1
  --shallow-submodules`, which is *not* reproducible and must be replaced for CI).
