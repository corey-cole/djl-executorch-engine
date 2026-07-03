# ExecuTorch Runtime — Engine-Side Build Notes

> **The runnable recipe now lives in `measly-java-learning/executorch-runtime-dist` (Repo A).**
> This file preserves the *why* behind the ExecuTorch runtime build that the engine used to do
> in-tree (`native/build.sh` Stage A, removed 2026-07-03). Recreating a build script from these
> notes is cheap; the reasoning is the part worth keeping.

## Why a separate, pinned runtime
The ExecuTorch runtime build is heavy and, outside the variant matrix, rarely changes. Repo A builds
it once per ET version, attests it, and publishes relocatable `et-install` tarballs. The engine
downloads a hash-pinned tarball (`native/cmake/EtRuntimePin.cmake`) and links its JNI shim against it.

## glibc 2.28 floor (load-bearing)
ExecuTorch 1.3.x pins `torch==2.12.0+cpu`, whose wheel requires **glibc ≥ 2.28**. That sets the
artifact floor at 2.28 (RHEL/Rocky 8+, Ubuntu 20.04+, Debian 11+). We have real users on RHEL8, so
the floor is non-negotiable: the shipped shim is compiled inside `manylinux_2_28` so it links the
runtime at the 2.28 floor. A host-native shim build (newer glibc) runs locally but must not ship.

## AVX512-VNNI toolchain check (why it's gone from the engine)
XNNPACK compiles its x86 VNNI microkernels with `-mavx512vnni` and dispatches at runtime via cpuinfo;
the build host only needs a toolchain that can *encode* VNNI (gcc ≥ 8, binutils ≥ 2.30), not a CPU
that runs it. That check mattered while the engine compiled XNNPACK (Stage A). It no longer does —
the shim only *links* the prebuilt `xnnpack_backend` from the tarball — so the check lives in Repo A now.

## JDK headers only (never link libjvm)
The shim is a JNI library loaded *by* the JVM; it needs only `jni.h` + `jni_md.h`, never `libjvm`/`libjawt`.
The manylinux image lacks `cpio`, so we extract the Corretto RPM via `rpm2archive` → `.tgz` → `tar`
(not `rpm2cpio | cpio`) and derive `JAVA_HOME` from the extracted `jni.h`.

## flatcc / install-destination gotcha
Repo A carries a workaround for an ExecuTorch install-destination bug (`pytorch/executorch#20709`);
irrelevant to tarball consumption. When ET merges the fix and we bump to an ET tag that includes it,
the patch becomes a no-op.

## Building a custom runtime (from source)
Not automated in the engine. Run Repo A's `build-runtime.sh --variant <v> --prefix <dir> --et-src
<et-checkout>` inside `manylinux_2_28`, then build the engine with `ET_INSTALL=<dir>` (the escape
hatch — CMake skips the download and links your tree). See Repo A's README for the full recipe.
