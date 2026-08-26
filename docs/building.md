# Building and testing

> **Status:** desktop **linux-x86_64**, **linux-aarch64** and **windows-x86_64** are supported.
> These steps build what exists today — the ExecuTorch runtime, our JNI shim, and the JVM + native
> test suites. The Docker prerequisite below applies to the Linux build only; Windows builds with
> MSVC 2022 and no container.

## 1. Prerequisites

- **Docker** — the Linux native library is built inside the pinned shared engine-build image, a
  `manylinux_2_28` derivative (see [The glibc floor](#2-the-glibc-floor-and-why-the-container-is-required-for-a-release)
  below). Docker is needed for a **pull** of that image's pinned digest, not to build it — the
  first build pays the pull.
- **JDK 17** on the host for Gradle. (The native build uses the JDK *headers* from JAVA_HOME for JNI —
  the shared image bakes JAVA_HOME; a host build points build.sh at any JDK.)
- No ExecuTorch checkout is needed — CMake `FetchContent`s the pinned runtime tarball. Network
  access is required for that fetch (and for Catch2 in the native QA build).

## 2. The glibc floor, and why the container is required for a release

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
Inside the container, CMake `FetchContent`s the pinned runtime for this platform's variant
(`devtools` on all three shipped platforms; see `native/variant_select.sh`), compiles the shim, and
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

When the pinned runtime provides the first-party custom ops (both Linux tarballs ship an
`etnp::lstm` op), the shim auto-detects the tarball's `ETNPExtras.cmake` and whole-archives the op
in. The Windows tarball has no such extras, so the op is simply absent there.

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

Every root-project `Test` task runs with `-Xcheck:jni`, the JVM's JNI-contract checker. The flag is attached to
the test-task umbrella in `build.gradle.kts` (`tasks.withType<Test>().configureEach`) rather than to
`tasks.test`, because `tasks.test` excludes eight tags including `oom`; `oomTest`, which drives the
allocation-failure paths this checker exists to police, is among the excluded. A JNI contract
violation surfaces as a `WARNING in native method:` line or a VM abort rather than a test failure.
`JniCheckFlagTest` and `JniCheckFlagTaggedTest` prove the flag is attached — the tagged subclass
carries the assertion into the eight tag-filtered tasks — so deleting either silently removes the
proof.

## 6. Native QA and benchmarking (optional)

`native/build_qa.sh` (Catch2 units + leak harness under ASan **and** UBSan), `native/bench.sh`
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

`build_qa.sh` builds the QA tree under **both** ASan and UBSan, so the Catch2 units and the leak
harness run under UndefinedBehaviorSanitizer too. UBSan is a gate, not a log: UB **aborts** the run
rather than printing, so a QA failure may be a `runtime error:` line rather than a failed
assertion — treat either as a finding. The check set lives in the `ET_UBSAN_CHECKS` CMake cache
variable (`undefined,float-cast-overflow,float-divide-by-zero`, minus `vptr`) and can be narrowed
for a one-off run. GCC has no ignorelist, so the way to exempt a function is
`__attribute__((no_sanitize("undefined")))` (or a per-TU compile-option override).
`implicit-signed-integer-truncation` is clang-only and therefore uncovered by this gate.

The same gate runs in CI on both Linux arches — `native-build-job.yml` calls `build_qa.sh` in its
`linux-x86_64` and `linux-aarch64` rows — and UBSan adds compile and run time to both.

### The JNI shim UBSan gate (JVM-driven)

`build_qa.sh` never touches the JNI shim: `native/CMakeLists.txt` skips `jni/` under
`ET_BUILD_QA`, and the QA scripts are JVM-free. The **only** configuration that instruments
`jni/executorch_djl_jni.cpp` is `native/ubsan_gate.sh`, which runs the JVM suite against an
UBSan-instrumented shim.

It runs in **two phases** because no single environment has both the toolchain and a JDK Gradle
can use: the pinned image ships Corretto 8 (the oldest supported `jni.h`), which cannot start
Gradle 9.6.1's JDK 17 toolchain. `ET_UBSAN_MODE` selects the phase — `build` | `test` | `all`,
default `auto` (build-only inside the pinned image, both phases outside it). The local invocation
is the two commands the script prints:

```bash
./native/local_build_wrapper.sh native/ubsan_gate.sh   # build phase (in-container)
ET_UBSAN_MODE=test ./native/ubsan_gate.sh              # JVM phase (host, JDK 17)
```

A UB hit presents as a **JVM hard crash** mid-test, not a Java exception or assertion failure: the
`runtime error:` line and its stack trace appear **above** the JVM's own crash dump — that is the
gate working, not a flake. The instrumented library is never staged into
`src/main/resources/native/`: it is reached through `EXECUTORCH_LIBRARY_PATH`, which `LibUtils`
honours ahead of the classpath copy and which `build.gradle.kts` declares as a `Test` task input,
so the ordinary tree is untouched and no rebuild is needed afterwards. The link uses
`-static-libubsan` so the UBSan runtime travels inside the `.so` and a stock JVM can `dlopen` it;
the script asserts the result has no dynamic `libubsan` dependency.

CI runs the gate on **`linux-x86_64` only** — `native-build-job.yml` builds the instrumented shim
in that matrix row, and `native-build.yml`'s `ubsan-jvm-gate` job downloads the
`executorch-ubsan-linux-x86_64` artifact and runs the JVM phase. The aarch64 row is deliberately
not gated (a second native build plus a `--rerun-tasks` JVM suite is real CI time, and it would
double for no new defect class).

## 7. Verifying runtime provenance (optional, local)

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
