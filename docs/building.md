# Building and testing

> **Status:** desktop **linux-x86_64**, **linux-aarch64** and **windows-x86_64** are supported.
> These steps build what exists today — the ExecuTorch runtime, our JNI shim, and the JVM + native
> test suites. The Docker prerequisite below applies to the Linux build only; Windows builds with
> MSVC 2022 and no container.

## 1. Prerequisites

- **Docker** — the Linux native library is built inside the pinned shared engine-build image, a
  `manylinux_2_28` derivative (see [The glibc floor](#2-the-glibc-floor-and-why-the-container-is-not-optional)
  below). Docker is needed for a **pull** of that image's pinned digest, not to build it — the
  first build pays the pull.
- **JDK 17** on the host for Gradle. (The native build fetches its own JDK *headers* for JNI — you
  do not need a JDK inside the container.)
- No ExecuTorch checkout is needed — CMake `FetchContent`s the pinned runtime tarball. Network
  access is required for that fetch (and for Catch2 in the native QA build).

## 2. The glibc floor, and why the container is not optional

ExecuTorch 1.3 pins `torch==2.12.0`, whose wheel needs **glibc ≥ 2.28**. So the shipped `.so` must
be built inside a `manylinux_2_28` container to keep that floor (covers RHEL/Rocky 8+,
Ubuntu 20.04+, Debian 11+). Building on the host instead produces a `.so` linked against host glibc
that **breaks the floor** — fine for local `./gradlew test`, never for a release.

## 3. Building the native shim

The engine loads a native `libexecutorch_djl.so` that is **built from source, not committed**. The
ExecuTorch **runtime** it links against is **not built here** — CMake downloads a hash-pinned,
attested tarball published by
[`executorch-runtime-dist`](https://github.com/measly-java-learning/executorch-runtime-dist)
(pinned in `native/cmake/EtRuntimePin.cmake`). Build the shim with the container wrapper:

```bash
./native/local_build_wrapper.sh
```

The wrapper runs the **pinned shared engine-build image** — a `manylinux_2_28` derivative whose
digest lives in `.engine-build-image` (see [Bumping the toolchain image](#bumping-the-toolchain-image)
below) — and runs the build **inside it**, so the staged `.so` keeps its **glibc-2.28 floor**
(RHEL8+). The image is pulled by digest, never built locally, and the first run pays the pull.
Inside the container, CMake `FetchContent`s the pinned `logging` runtime, compiles the shim, and
stages it into `src/main/resources/native/linux-x86_64/`. It is fast — there is no ExecuTorch
build.

**Host builds now work but break the floor.** `native/build.sh` is what `local_build_wrapper.sh`
invokes *inside* the container, but it no longer requires one — it **asserts** its toolchain
(JDK headers via `JAVA_HOME`, `ninja` on PATH) instead of installing it. A Linux host that already
has ninja, cmake, a C++17 compiler, and JDK headers can therefore run `native/build.sh` directly
and get a shim. What such a build costs: it links **host glibc** and **breaks the 2.28 floor**, so
it is fine for local `./gradlew test` and never for a release. `local_build_wrapper.sh` remains the
blessed path because it holds the floor. (On Windows, `build.sh` *is* run directly on the host —
see [Windows](#4-windows) below — because there is no equivalent container image for that
platform.)

**Escape hatch / custom runtime:** set `ET_INSTALL=/path/to/et-install` to link an existing runtime
tree (e.g. one you built from source per `docs/executorch-build-notes.md`); CMake then skips the
download.

When the pinned runtime provides the first-party custom ops (the `logging` linux-x86_64 tarball
ships an `etnp::lstm` op), the shim auto-detects the tarball's `ETNPExtras.cmake` and
whole-archives the op in. The Windows tarball has no such extras, so the op is simply absent there.

### Bumping the toolchain image

The engine-build image is pinned by digest in `.engine-build-image` (exactly one line). CI's
`native-build-job.yml` and `local_build_wrapper.sh` both read that file, so a bump is a one-line
change both pick up together. Digests are per-run, not per-commit — take the new digest from the
`Publish Engine Images` run you intend to consume, not from a commit in this repo. The image is
published by `measly-java-learning/base-docker-images`; see its
`docs/consuming-engine-build.md` for what the image guarantees and for `gh attestation verify`.

## 4. Windows

There is no container on Windows (the manylinux image only bakes the glibc floor for Linux), so the
shim is built directly on the host by the same `native/build.sh` — it detects Git-Bash
(`uname -s` = `MINGW*`/`MSYS*`) and takes the Windows path. Requirements, all generic (no
assumptions about VS edition or a specific machine):

- **Visual Studio 2022 with the C++ toolchain** (any edition — Community/Professional/Enterprise).
  CI discovers it edition-agnostically via `vswhere -latest -products *` and activates it with
  `Launch-VsDevShell.ps1 -Arch amd64`. `build.sh` does **not** activate VS itself — the caller must
  already have the MSVC dev shell active (it just asserts `cl` and `ninja` are on PATH).
- **Ninja** and **CMake** on PATH (both ship with the VS C++ workload).
- **Git-Bash** to run `build.sh` (invoke it by explicit path so PATH order can't pick WSL's
  `bash.exe`; use a non-login shell so the profile doesn't reset the VS env).
- **A JDK for headers only** — set `JAVA_HOME` to any JDK; the build compiles against
  `include/win32/jni_md.h` and never links `libjvm`. CI binds JDK 8 deliberately (oldest supported
  `jni.h` = widest runtime compatibility), but any JDK's headers work.

Key ABI constraint: the build passes `-DCMAKE_BUILD_TYPE=Release` on Windows because MSVC encodes
the CRT flavour into every object and refuses to mix them. The pinned runtime tarball is built
Release (`/MD`), so a non-Release shim fails to link with `LNK2038` `RuntimeLibrary`/
`_ITERATOR_DEBUG_LEVEL` mismatches. GCC/ELF has no such ABI tag, so the Linux leg leaves the build
type unset. MSVC does **not** reliably diagnose a CRT mismatch (no `LNK2038`, not even an
`LNK4098`), so `native/tests/check_windows_crt.sh` is the real gate; it runs over both the shim tree
and the QA tree. Output is `executorch_djl.dll` (no `lib` prefix), staged into
`src/main/resources/native/windows-x86_64/`.

## 5. Running the tests

**The JVM integration tests load the native library, so the native shim must be built and staged
first** (see [Building the native shim](#3-building-the-native-shim) above). Then:

```bash
./gradlew test        # unit + native integration tests
./gradlew leakTest    # JVM-side memory-leak stress test
./gradlew build        # full build incl. jacoco coverage report
```

## 6. Native QA and benchmarking (optional)

`native/build_qa.sh` (AddressSanitizer/LeakSanitizer Catch2 units + leak harness), `native/bench.sh`
(Release timing harness), and `native/build_variants.sh` (times all three runtime variants) each
fetch the runtime via CMake (or set `ET_INSTALL` for the escape hatch). Run them in the **same
pinned engine-build image** as the shim build so the toolchain matches — pass the script to the
wrapper:

```bash
./native/local_build_wrapper.sh native/build_qa.sh
./native/local_build_wrapper.sh native/bench.sh
ITERS=2000 ./native/local_build_wrapper.sh native/build_variants.sh
```

Running them directly on the host works but is unsupported: the runtime toolchain won't match, and
a `native/bench`/`native/asan` tree left over from a container run has a different source root —
the scripts wipe their own tree to avoid that collision, but the host toolchain mismatch remains.
The wrapper is the blessed path.

## 7. Container file ownership (known gap)

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

## 8. Verifying runtime provenance (optional, local)

CI verifies every pinned tarball with a build attestation. To check by hand:

```bash
gh attestation verify <downloaded-tarball> --repo measly-java-learning/executorch-runtime-dist
```

## Editor setup (clangd)

`.clangd` points at `native/build-clangd`, a compile database that no build script touches.
Generate or refresh it with:

```bash
./native/gen_clangd_db.sh
```

It runs two CMake configures and merges them, because `native/CMakeLists.txt` builds the JNI
shim only when `ET_BUILD_QA` and `ET_BUILD_BENCH` are both off — so no single configure covers
both `jni/` and `test/`. Configure only; nothing is compiled.

Re-run it after bumping `native/cmake/EtRuntimePin.cmake` or changing compile flags. The
database is refreshed only by that script, so a stale one keeps resolving against the previous
runtime's headers, silently and with no warning.
