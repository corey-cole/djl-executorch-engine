# Engine Runtime Consumption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the DJL ExecuTorch engine **download** a prebuilt, hash-pinned, attested `et-install` tarball from `measly-java-learning/executorch-runtime-dist` (Repo A) instead of building the ExecuTorch runtime from source.

**Architecture:** `native/CMakeLists.txt` gains a two-branch runtime resolution — an `ET_INSTALL` escape hatch, else `FetchContent` the pinned tarball selected by `ET_RUNTIME_VARIANT` (default `logging`). All consumers (`build.sh`, `bench.sh`, `build_qa.sh`, `build_variants.sh`) shed their from-source machinery and route through that seam. CI verifies provenance with `gh attestation verify`; the ET build recipe survives as a distilled doc.

**Tech Stack:** CMake ≥ 3.24 (`FetchContent`, `URL_HASH SHA256`), Bash, GitHub Actions, `manylinux_2_28` container, C++20 shim.

## Global Constraints

- **glibc-2.28 floor is sacred** — RHEL8 users depend on it. The shipped shim MUST be compiled in the `manylinux_2_28` container. A host-native shim build is a documented "local test only, do not ship" shortcut.
- **Never commit native binaries** (`*.so`/`*.dylib`/`*.dll`) — supply-chain: a committed binary makes this repo a distribution vector. Only source + hash-pins are committed.
- **Container is the blessed default** for producing a shippable `.so`; it is now *fast* (no runtime build), not optional.
- **Pinned runtime:** ET version `1.3.1`, Repo A release tag `v1.3.1-2`. The committed `EtRuntimePin.cmake` is the verbatim release artifact — do not hand-edit hashes.
- **Variants:** `bare` (LOGGING=OFF), `logging` (LOGGING=ON, **ship default**), `devtools` (LOGGING=OFF + DEVTOOLS=ON + EVENT_TRACER=ON). Platform token: `linux-x86_64` only.
- **Supply-chain:** `URL_HASH SHA256` integrity always (local + CI); `gh attestation verify` provenance in CI YAML; local `gh` verification is **docs only** — no engine-owned verify script.
- **Two resolution paths only** — the engine never invokes Repo A's `build-runtime.sh`. From-source is a documented procedure.
- **Testing reality:** these are shell/CMake/YAML/docs changes. Unit tests are **host-fast** and use diagnostic early-exit + grep (the repo's established idiom). The real end-to-end (container fetch → `find_package` → shim link → JVM tests) runs only in-container/CI; each task names that manual verification separately — it is NOT a host-fast unit gate.
- **Commits:** frequent, one per task. End every commit message with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `native/cmake/EtRuntimePin.cmake` | Verbatim pin: URLs + SHA256s + versions for all 3 variants | Create |
| `native/CMakeLists.txt` | Two-branch runtime resolution + `ET_PRINT_RESOLUTION` diagnostic | Modify |
| `native/tests/cmake_resolution.sh` | Host-fast test of the resolution branches | Create |
| `native/build.sh` | Shim-only build (Stage A removed); forwards `ET_RUNTIME_VARIANT` | Modify |
| `native/local_build_wrapper.sh` | Container wrapper (ET mount removed) | Modify |
| `native/tests/build_config.sh` | Host-fast test of `build.sh`'s config seam | Create |
| `native/tests/build_variants_flags.sh` | Tested the removed `ET_VARIANT` map | Delete |
| `native/bench.sh` | Route through resolution seam (variant/escape-hatch) | Modify |
| `native/build_qa.sh` | Route through resolution seam; own its `libasan` install | Modify |
| `native/build_variants.sh` | Loop variants via download (no from-source) | Modify |
| `native/tests/build_variants_matrix.sh` | Host-fast stub-driven matrix test (reshaped) | Modify |
| `.github/workflows/native-build.yml` | Drop ET checkout; add attestation step | Modify |
| `native/tests/ci_workflow.sh` | Host-fast assertions on the workflow | Create |
| `docs/executorch-build-notes.md` | Distilled ET build reasoning (recipe now in Repo A) | Create |
| `README.md` | Rewrite build/QA sections for the fetch model | Modify |
| `docs/benchmarking.md` | Variant section reflects download model | Modify |
| `.gitignore` | Drop obsolete `et-*` trees; ignore QA/bench outputs | Modify |

---

### Task 1: Runtime pin + CMake resolution

**Files:**
- Create: `native/cmake/EtRuntimePin.cmake`
- Modify: `native/CMakeLists.txt:10-16` (the `ET_INSTALL` default + `find_package` block)
- Test: `native/tests/cmake_resolution.sh`

**Interfaces:**
- Produces: cache var `ET_RUNTIME_VARIANT` (default `"logging"`); cache var `ET_INSTALL` (default `""`, escape hatch); option `ET_PRINT_RESOLUTION` (OFF); a `message(STATUS "ET_RESOLUTION resolution=… variant=… stem=… url=… et_install=…")` line printed under `-DET_PRINT_RESOLUTION=ON`, then `return()`.
- Consumes (later tasks): `build.sh`/`bench.sh`/`build_qa.sh` pass `-DET_RUNTIME_VARIANT=<v>` and optionally `-DET_INSTALL=<path>`.

- [ ] **Step 1: Create the pin file (verbatim Repo A `v1.3.1-2` artifact)**

`native/cmake/EtRuntimePin.cmake`:
```cmake
# Generated by executorch-runtime-dist release v1.3.1-2. Do not edit by hand.
# All three variant rows are committed on purpose: `logging` is the SHIPPED/default runtime;
# `bare` and `devtools` exist ONLY for native/build_variants.sh benchmarking. Shipping only uses logging.
# Bump procedure: replace this whole file with the EtRuntimePin.cmake asset from the next
# `v<etver>-<pkgrev>` Repo A release. The SHA256 change is the supply-chain review gate.
set(ET_RUNTIME_VERSION "1.3.1-2")
set(ET_RUNTIME_ET_VERSION "1.3.1")

set(ET_RUNTIME_URL_bare_linux-x86_64
  "https://github.com/measly-java-learning/executorch-runtime-dist/releases/download/v1.3.1-2/executorch-runtime-1.3.1-bare-linux-x86_64.tar.gz")
set(ET_RUNTIME_SHA256_bare_linux-x86_64 "7d583d7f4cb3ba0d400cc36800f0b88fc79aeb4e48866f1f7fe7dbaf946675b3")

set(ET_RUNTIME_URL_logging_linux-x86_64
  "https://github.com/measly-java-learning/executorch-runtime-dist/releases/download/v1.3.1-2/executorch-runtime-1.3.1-logging-linux-x86_64.tar.gz")
set(ET_RUNTIME_SHA256_logging_linux-x86_64 "79456966eafc280506eed60eb9327c8dfbf48fcc9e5bed06a20bc45b9061e57a")

set(ET_RUNTIME_URL_devtools_linux-x86_64
  "https://github.com/measly-java-learning/executorch-runtime-dist/releases/download/v1.3.1-2/executorch-runtime-1.3.1-devtools-linux-x86_64.tar.gz")
set(ET_RUNTIME_SHA256_devtools_linux-x86_64 "cda457775de0e22b32330f4e13f4ec642577f8a5067d8c57fbdac7885a80d832")
```

- [ ] **Step 2: Write the failing test**

`native/tests/cmake_resolution.sh`:
```bash
#!/usr/bin/env bash
# Host-fast test of native/CMakeLists.txt runtime resolution via the ET_PRINT_RESOLUTION early-exit.
# No network / no find_package: the diagnostic returns BEFORE FetchContent_MakeAvailable.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

probe() {  # extra -D args; echoes the ET_RESOLUTION status line
  local b err; b="$(mktemp -d)"; err="${b}/err"
  cmake -S native -B "${b}" -DET_PRINT_RESOLUTION=ON "$@" >/dev/null 2>"${err}" \
    || { cat "${err}"; rm -rf "${b}"; fail "cmake configure failed ($*)"; }
  grep -h 'ET_RESOLUTION' "${err}" || { rm -rf "${b}"; fail "no ET_RESOLUTION line ($*)"; }
  rm -rf "${b}"
}

out="$(probe)"                                            # default => fetch logging
grep -q 'resolution=fetch'                                     <<<"${out}" || fail "default not fetch"
grep -q 'variant=logging'                                     <<<"${out}" || fail "default variant not logging"
grep -q 'stem=executorch-runtime-1.3.1-logging-linux-x86_64'  <<<"${out}" || fail "default stem wrong"
grep -q 'executorch-runtime-1.3.1-logging-linux-x86_64.tar.gz' <<<"${out}" || fail "default url wrong"

out="$(probe -DET_RUNTIME_VARIANT=bare)"
grep -q 'stem=executorch-runtime-1.3.1-bare-linux-x86_64'      <<<"${out}" || fail "bare stem wrong"
grep -q 'executorch-runtime-1.3.1-bare-linux-x86_64.tar.gz'    <<<"${out}" || fail "bare url wrong"

out="$(probe -DET_RUNTIME_VARIANT=devtools)"
grep -q 'stem=executorch-runtime-1.3.1-devtools-linux-x86_64'  <<<"${out}" || fail "devtools stem wrong"

out="$(probe -DET_INSTALL=/tmp/my-et)"                    # escape hatch
grep -q 'resolution=escape-hatch'                             <<<"${out}" || fail "escape hatch not detected"
grep -q 'et_install=/tmp/my-et'                              <<<"${out}" || fail "escape hatch path wrong"

echo "PASS: cmake resolution"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `chmod +x native/tests/cmake_resolution.sh && native/tests/cmake_resolution.sh`
Expected: FAIL — `native/CMakeLists.txt` has no `ET_PRINT_RESOLUTION` path yet (configure errors on the old hardcoded `find_package`, or prints no `ET_RESOLUTION` line).

- [ ] **Step 4: Implement the resolution block**

In `native/CMakeLists.txt`, replace lines 10-16 (the `# TODO: ET_INSTALL …` comment through the `find_package(executorch …)` line) with:
```cmake
# --- Runtime resolution: escape hatch (ET_INSTALL set) OR FetchContent the pinned tarball. ---
# The ExecuTorch runtime is NOT built here; it is downloaded (hash-pinned + attested by Repo A).
# See docs/executorch-build-notes.md. ET_PRINT_RESOLUTION is a host-test seam: it prints the
# resolved config and returns before any download or find_package.
option(ET_PRINT_RESOLUTION "Print resolved runtime config and exit (host-test seam)" OFF)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EtRuntimePin.cmake)
set(ET_RUNTIME_VARIANT "logging" CACHE STRING "Runtime variant: logging (ship) | bare | devtools (bench)")
set(ET_INSTALL "" CACHE PATH "Explicit ExecuTorch install prefix (escape hatch); empty => fetch the pinned tarball")
set(_ET_PLATFORM "linux-x86_64")

if(ET_INSTALL)
  set(_ET_RESOLUTION "escape-hatch")
  set(_ET_STEM "")
  set(_ET_RESOLVED_URL "")
else()
  set(_ET_RESOLUTION "fetch")
  set(_ET_STEM "executorch-runtime-${ET_RUNTIME_ET_VERSION}-${ET_RUNTIME_VARIANT}-${_ET_PLATFORM}")
  set(_ET_RESOLVED_URL "${ET_RUNTIME_URL_${ET_RUNTIME_VARIANT}_${_ET_PLATFORM}}")
  set(_ET_RESOLVED_SHA "${ET_RUNTIME_SHA256_${ET_RUNTIME_VARIANT}_${_ET_PLATFORM}}")
  if(NOT _ET_RESOLVED_URL)
    message(FATAL_ERROR "No pin row for variant='${ET_RUNTIME_VARIANT}' platform='${_ET_PLATFORM}' in EtRuntimePin.cmake")
  endif()
endif()

if(ET_PRINT_RESOLUTION)
  message(STATUS "ET_RESOLUTION resolution=${_ET_RESOLUTION} variant=${ET_RUNTIME_VARIANT} stem=${_ET_STEM} url=${_ET_RESOLVED_URL} et_install=${ET_INSTALL}")
  return()
endif()

if(_ET_RESOLUTION STREQUAL "fetch")
  include(FetchContent)
  FetchContent_Declare(et_runtime URL "${_ET_RESOLVED_URL}" URL_HASH "SHA256=${_ET_RESOLVED_SHA}")
  FetchContent_MakeAvailable(et_runtime)
  # The tarball has a single top-level dir (the stem) under the extracted source dir.
  set(ET_INSTALL "${et_runtime_SOURCE_DIR}/${_ET_STEM}")
endif()

list(PREPEND CMAKE_PREFIX_PATH "${ET_INSTALL}")
# The tarball ships lib/cmake/{ExecuTorch,tokenizers,absl,re2,pcre2}; prepending CMAKE_PREFIX_PATH
# lets find_package resolve them all. tokenizers_DIR kept for parity with the prior build.
set(tokenizers_DIR "${ET_INSTALL}/lib/cmake/tokenizers" CACHE PATH "" FORCE)
find_package(executorch CONFIG REQUIRED PATHS "${ET_INSTALL}/lib/cmake/ExecuTorch")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `native/tests/cmake_resolution.sh`
Expected: `PASS: cmake resolution`

- [ ] **Step 6: (Manual/CI, not a unit gate) End-to-end fetch smoke**

In a network-capable env: `cmake -S native -B /tmp/etfetch` (no flags) should download the logging tarball, pass `URL_HASH`, and reach `find_package(executorch)` without error. Record the result in the task report; do not add it to the host-fast suite.

- [ ] **Step 7: Commit**
```bash
git add native/cmake/EtRuntimePin.cmake native/CMakeLists.txt native/tests/cmake_resolution.sh
git commit -m "feat(native): CMake resolves ET runtime via FetchContent pin + escape hatch"
```

---

### Task 2: `build.sh` sheds Stage A; wrapper drops the ET mount

**Files:**
- Modify: `native/build.sh` (remove the `ET_VARIANT` map, Stage A, VNNI check, `libasan` install; keep JDK headers, shim build, `STAGE_SO`, chown trap)
- Modify: `native/local_build_wrapper.sh` (remove `ET_ROOT` mount + `SKIP_ET_BUILD`)
- Create: `native/tests/build_config.sh`
- Delete: `native/tests/build_variants_flags.sh`

**Interfaces:**
- Consumes: Task 1's `-DET_RUNTIME_VARIANT` / `-DET_INSTALL`.
- Produces: `PRINT_BUILD_CONFIG=1` early-exit printing `ET_RUNTIME_VARIANT=<v> STAGE_SO=<n> NATIVE_BUILD_DIR=<d> ET_INSTALL=<path-or-empty>`; env seams `ET_RUNTIME_VARIANT` (default `logging`), `STAGE_SO` (default `1`), `NATIVE_BUILD_DIR` (default `native/build`), optional `ET_INSTALL` passthrough.

- [ ] **Step 1: Write the failing test**

`native/tests/build_config.sh`:
```bash
#!/usr/bin/env bash
# Host-fast test of build.sh's config seam via PRINT_BUILD_CONFIG, plus a guard that Stage-A is gone.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

out="$(PRINT_BUILD_CONFIG=1 bash native/build.sh)"
grep -q 'ET_RUNTIME_VARIANT=logging' <<<"${out}" || fail "default variant not logging"
grep -q 'STAGE_SO=1'                  <<<"${out}" || fail "default STAGE_SO not 1"
grep -q 'NATIVE_BUILD_DIR=native/build\b' <<<"${out}" || fail "default NATIVE_BUILD_DIR changed"

out="$(ET_RUNTIME_VARIANT=bare STAGE_SO=0 NATIVE_BUILD_DIR=/tmp/nb ET_INSTALL=/tmp/xi \
       PRINT_BUILD_CONFIG=1 bash native/build.sh)"
grep -q 'ET_RUNTIME_VARIANT=bare'  <<<"${out}" || fail "variant override ignored"
grep -q 'STAGE_SO=0'               <<<"${out}" || fail "STAGE_SO override ignored"
grep -q 'NATIVE_BUILD_DIR=/tmp/nb\b' <<<"${out}" || fail "NATIVE_BUILD_DIR override ignored"
grep -q 'ET_INSTALL=/tmp/xi\b'     <<<"${out}" || fail "ET_INSTALL passthrough missing"

# Stage-A must be fully gone from build.sh.
grep -qE 'SKIP_ET_BUILD|EXECUTORCH_ENABLE_LOGGING|torch==2\.12|avx512vnni' native/build.sh \
  && fail "Stage-A remnants still present in build.sh"

echo "PASS: build.sh config"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `chmod +x native/tests/build_config.sh && native/tests/build_config.sh`
Expected: FAIL — `build.sh` has no `PRINT_BUILD_CONFIG` path and still contains Stage-A remnants.

- [ ] **Step 3: Rewrite `native/build.sh`**

Replace the entire file with:
```bash
#!/bin/bash
set -ex # Fail on error, print commands to log

# Container bind-mount outputs are root-owned on the host; chown them back on exit (any status).
cleanup() {
  rc=$?
  if [ -n "${HOST_UID:-}" ]; then
    chown -R "${HOST_UID}:${HOST_GID}" "${NATIVE_BUILD_DIR}" src/main/resources/native/linux* 2>/dev/null || true
  fi
  exit "$rc"
}
trap cleanup EXIT

# --- Shim build config. The ExecuTorch runtime is NOT built here anymore: native/CMakeLists.txt
#     resolves it (FetchContent the pinned tarball, or -DET_INSTALL escape hatch). The runtime
#     recipe now lives in measly-java-learning/executorch-runtime-dist; see
#     docs/executorch-build-notes.md for the engine-side reasoning. ---
ET_RUNTIME_VARIANT="${ET_RUNTIME_VARIANT:-logging}"
STAGE_SO="${STAGE_SO:-1}"
NATIVE_BUILD_DIR="${NATIVE_BUILD_DIR:-native/build}"

# Fast diagnostic: print resolved shim-build config and exit before any heavy setup.
if [ -n "${PRINT_BUILD_CONFIG:-}" ]; then
  echo "ET_RUNTIME_VARIANT=${ET_RUNTIME_VARIANT} STAGE_SO=${STAGE_SO} NATIVE_BUILD_DIR=${NATIVE_BUILD_DIR} ET_INSTALL=${ET_INSTALL:-}"
  exit 0
fi

# This script expects:
# 1. To be running inside quay.io/pypa/manylinux_2_28_x86_64 (glibc-2.28 floor for the shipped .so)
# 2. The Corretto RPM downloaded to /workspace
# The runtime tarball is fetched by CMake during the shim configure (also inside the container,
# so the fetched runtime is linked on glibc 2.28).

echo "--- Extracting Corretto JDK headers (headers-only; we never link libjvm) ---"
JDK_EXTRACT=/opt/corretto
mkdir -p "${JDK_EXTRACT}"
cp /workspace/amazon-corretto-linux-jdk.rpm /tmp/corretto.rpm
rpm2archive /tmp/corretto.rpm            # -> /tmp/corretto.rpm.tgz (no cpio in this image)
tar -C "${JDK_EXTRACT}" -xzf /tmp/corretto.rpm.tgz
JNI_H="$(find "${JDK_EXTRACT}" -path '*/include/jni.h' | head -1)"
export JAVA_HOME="${JNI_H%/include/jni.h}"
test -f "${JAVA_HOME}/include/linux/jni_md.h" \
  || { echo "JDK headers not found under JAVA_HOME=${JAVA_HOME}"; exit 1; }
echo "JAVA_HOME=${JAVA_HOME}"

echo "--- Setting up Ninja (the shim configures with -G Ninja) ---"
export PATH="/opt/python/cp312-cp312/bin:${PATH}"
pip install ninja

echo "--- Toolchain Versions ---"
gcc --version; g++ --version; cmake --version; ninja --version

# In GitHub Actions, publish JAVA_HOME for any downstream shim-building step. ET_INSTALL is no
# longer exported — the runtime is resolved inside cmake now (FetchContent), per configure.
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "JAVA_HOME=${JAVA_HOME}" >> "${GITHUB_ENV}"
fi

JOBS="${JOBS:-$(nproc)}"
cd /workspace
# native/build is disposable; its cached absolute paths won't match a fresh container, so wipe it.
rm -rf "${NATIVE_BUILD_DIR}"

# Forward ET_INSTALL as an escape hatch only if the caller set it; otherwise CMake FetchContents
# the pinned ${ET_RUNTIME_VARIANT} tarball.
ET_INSTALL_ARG=()
[ -n "${ET_INSTALL:-}" ] && ET_INSTALL_ARG=(-DET_INSTALL="${ET_INSTALL}")
cmake -B "${NATIVE_BUILD_DIR}" -S native -G Ninja \
  -DET_RUNTIME_VARIANT="${ET_RUNTIME_VARIANT}" "${ET_INSTALL_ARG[@]}"
cmake --build "${NATIVE_BUILD_DIR}" -j"${JOBS}"

if [ "${STAGE_SO}" = "1" ]; then
  OUT="src/main/resources/native/linux-x86_64"
  mkdir -p "${OUT}"
  cp "${NATIVE_BUILD_DIR}/libexecutorch_djl.so" "${OUT}/"
  echo "Artifact: ${OUT}/libexecutorch_djl.so"
  ls -lh "${OUT}/libexecutorch_djl.so"
else
  echo "STAGE_SO=0: built shim but not staging into resources"
  ls -lh "${NATIVE_BUILD_DIR}/libexecutorch_djl.so"
fi
```

- [ ] **Step 4: Rewrite `native/local_build_wrapper.sh`**

Replace the file with (Corretto download kept; ET checkout + mount + `SKIP_ET_BUILD` removed):
```bash
#!/bin/bash
set -ex # Fail on error, print commands to log

# Arranges the environment for build.sh when run locally — the tasks the GHA workflow does in CI.
# The ExecuTorch runtime is downloaded by CMake (FetchContent) during the build; there is NO ET
# checkout to mount anymore. This wrapper is the BLESSED default: it builds the shim inside
# manylinux_2_28 so the staged .so keeps its glibc-2.28 floor (RHEL8). For a quick local shim that
# you will NOT ship, you can run native/build.sh directly on the host (breaks the floor).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "${REPO_ROOT}/amazon-corretto-linux-jdk.rpm" ]; then
  echo "Downloading Amazon Corretto JDK RPM to ${REPO_ROOT}/amazon-corretto-linux-jdk.rpm"
  curl -L -o "${REPO_ROOT}/amazon-corretto-linux-jdk.rpm" \
    https://corretto.aws/downloads/latest/amazon-corretto-8-x64-linux-jdk.rpm
fi

# Must use manylinux_2_28 (glibc >= 2.28) so the shim links the fetched runtime at the 2.28 floor.
# Override the runtime variant with ET_RUNTIME_VARIANT (default logging).
docker run --rm \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    -e ET_RUNTIME_VARIANT="${ET_RUNTIME_VARIANT:-logging}" \
    -v "${REPO_ROOT}":/workspace \
    -w /workspace \
    quay.io/pypa/manylinux_2_28_x86_64:latest \
    /bin/bash /workspace/native/build.sh
```

- [ ] **Step 5: Delete the obsolete flag-map test**

Run: `git rm native/tests/build_variants_flags.sh`
(It tested the removed `ET_VARIANT` map; `build_config.sh` + `cmake_resolution.sh` replace its coverage.)

- [ ] **Step 6: Run test to verify it passes**

Run: `native/tests/build_config.sh`
Expected: `PASS: build.sh config`

- [ ] **Step 7: Commit**
```bash
git add native/build.sh native/local_build_wrapper.sh native/tests/build_config.sh
git rm native/tests/build_variants_flags.sh
git commit -m "refactor(native): build.sh builds only the shim; wrapper drops the ET mount"
```

---

### Task 3: `bench.sh`, `build_qa.sh`, and `build_variants.sh` route through the seam

**Files:**
- Modify: `native/bench.sh` (variant/escape-hatch instead of required `ET_INSTALL`)
- Modify: `native/build_qa.sh` (same; own its `libasan` install)
- Modify: `native/build_variants.sh` (loop variants via download; no from-source)
- Modify: `native/tests/build_variants_matrix.sh` (stub `bench.sh`, reshaped assertions)

**Interfaces:**
- Consumes: Task 1 resolution (`-DET_RUNTIME_VARIANT`, `-DET_INSTALL`); `bench.sh` prints the existing `et_timing:` line (`… warm_mean_ms=<f> …`).
- Produces: `build_variants.sh` results file with 3 `et_timing:` lines + a table (`variant load_ms cold_ms warm_mean_ms`) and deltas-vs-bare; override seams `BENCH_SH`/`WORKSPACE`/`ITERS`/`WARMUP`.

- [ ] **Step 1: Write the failing test**

Replace `native/tests/build_variants_matrix.sh` with:
```bash
#!/usr/bin/env bash
# Host-fast test: build_variants.sh loops the 3 variants through a STUBBED bench.sh (keyed on
# ET_RUNTIME_VARIANT) and produces the table + deltas. No real build/download.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

STUB_DIR="$(mktemp -d)"
BENCH_STUB="${STUB_DIR}/bench.sh"
cat > "${BENCH_STUB}" <<'EOF'
#!/usr/bin/env bash
case "${ET_RUNTIME_VARIANT}" in
  bare)     echo "et_timing: model=add.pte iters=10 warmup=1 load_ms=5.0 cold_ms=2.0 warm_min_ms=1.0 warm_mean_ms=1.100 warm_max_ms=1.2 sink=0";;
  logging)  echo "et_timing: model=add.pte iters=10 warmup=1 load_ms=5.5 cold_ms=2.1 warm_min_ms=1.1 warm_mean_ms=1.250 warm_max_ms=1.3 sink=0";;
  devtools) echo "et_timing: model=add.pte iters=10 warmup=1 load_ms=6.0 cold_ms=2.2 warm_min_ms=1.2 warm_mean_ms=1.400 warm_max_ms=1.5 sink=0";;
  *) echo "bad variant" >&2; exit 1;;
esac
EOF
chmod +x "${BENCH_STUB}"

out="$(BENCH_SH="${BENCH_STUB}" WORKSPACE="${STUB_DIR}" bash native/build_variants.sh)"

grep -qE '^bare\s' <<<"${out}"     || fail "no bare row"
grep -qE '^logging\s' <<<"${out}"  || fail "no logging row"
grep -qE '^devtools\s' <<<"${out}" || fail "no devtools row"
grep -q '1.250' <<<"${out}"        || fail "logging warm_mean missing from table"
# deltas vs bare: logging - bare warm_mean = +0.150; devtools - bare = +0.300
grep -q '+0.150' <<<"${out}"       || fail "logging delta wrong"
grep -q '+0.300' <<<"${out}"       || fail "devtools delta wrong"

RESULTS="$(ls -t native/bench-results/variants-*.txt | head -1)"
[ "$(grep -c '^et_timing:' "${RESULTS}")" -eq 3 ] || fail "results file should hold 3 et_timing lines"

echo "PASS: build_variants matrix"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `native/tests/build_variants_matrix.sh`
Expected: FAIL — current `build_variants.sh` calls `build.sh` with the removed `ET_VARIANT`/`STAGE_SO`/`SKIP_ET_BUILD` machinery and expects a shim-size column.

- [ ] **Step 3: Rewrite `native/build_variants.sh`**

```bash
#!/usr/bin/env bash
# Benchmark all three ExecuTorch runtime variants (bare/logging/devtools) and print a comparison
# table with warm-mean deltas vs. bare. Each variant's prebuilt tarball is DOWNLOADED by CMake
# (FetchContent) inside bench.sh — no from-source build. Run INSIDE manylinux_2_28 (matches the
# shipped toolchain). See docs/benchmarking.md.
#
# Env: ITERS/WARMUP forwarded to bench.sh. BENCH_SH/WORKSPACE are override seams (tests inject a
# stub bench.sh; WORKSPACE is unused by the real path but kept so the stub test can sandbox).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BENCH_SH="${BENCH_SH:-native/bench.sh}"
VARIANTS=(bare logging devtools)

RESULTS_DIR="native/bench-results"
mkdir -p "${RESULTS_DIR}"
RESULTS="${RESULTS_DIR}/variants-$(date -u +%Y%m%dT%H%M%SZ).txt"

declare -A LOAD COLD WARM
for v in "${VARIANTS[@]}"; do
  echo "=== variant ${v} ==="
  line="$(ET_RUNTIME_VARIANT="${v}" ITERS="${ITERS:-1000}" WARMUP="${WARMUP:-100}" \
          bash "${BENCH_SH}" | grep '^et_timing:')"
  echo "${line}" >> "${RESULTS}"
  LOAD[$v]="$(sed -n 's/.*[[:space:]]load_ms=\([0-9.]*\).*/\1/p' <<<"${line}")"
  COLD[$v]="$(sed -n 's/.*[[:space:]]cold_ms=\([0-9.]*\).*/\1/p' <<<"${line}")"
  WARM[$v]="$(sed -n 's/.*[[:space:]]warm_mean_ms=\([0-9.]*\).*/\1/p' <<<"${line}")"
done

table() {
  printf '%-10s %10s %10s %14s\n' variant load_ms cold_ms warm_mean_ms
  for v in "${VARIANTS[@]}"; do
    printf '%-10s %10s %10s %14s\n' "$v" "${LOAD[$v]}" "${COLD[$v]}" "${WARM[$v]}"
  done
  echo
  echo "deltas vs bare (warm_mean isolates the runtime cost of each build flag):"
  awk -v lw="${WARM[logging]}" -v bw="${WARM[bare]}" -v dw="${WARM[devtools]}" 'BEGIN{
    printf "  logging - bare : warm_mean %+.3f ms\n", lw-bw
    printf "  devtools - bare: warm_mean %+.3f ms\n", dw-bw
  }'
}

table | tee -a "${RESULTS}"
echo "Wrote ${RESULTS}"
```

- [ ] **Step 4: Rewrite `native/bench.sh` to route through the seam**

Replace the body from the `ET_INSTALL="${ET_INSTALL:-…}"` line through the `cmake -B native/bench …` configure with:
```bash
ET_RUNTIME_VARIANT="${ET_RUNTIME_VARIANT:-logging}"
ITERS="${ITERS:-1000}"
WARMUP="${WARMUP:-100}"

# Runtime comes from CMake resolution: default fetches the pinned ${ET_RUNTIME_VARIANT} tarball;
# set ET_INSTALL to point at an existing install (escape hatch). No precondition to check here.
ET_ARGS=(-DET_RUNTIME_VARIANT="${ET_RUNTIME_VARIANT}")
[ -n "${ET_INSTALL:-}" ] && ET_ARGS+=(-DET_INSTALL="${ET_INSTALL}")

# Release, no sanitizer, own build tree (distinct from native/asan QA and native/build shim).
cmake -B native/bench -S native -G "Unix Makefiles" "${ET_ARGS[@]}" \
  -DET_BUILD_BENCH=ON -DCMAKE_BUILD_TYPE=Release
cmake --build native/bench --target et_timing_harness
```
Keep the `#!`, header comment (update the "build the runtime first" line to "runtime is fetched by CMake"), `set -euo pipefail`, `REPO_ROOT`, and the final `et_timing_harness` run line unchanged.

- [ ] **Step 5: Rewrite `native/build_qa.sh` to route through the seam + own libasan**

Replace the body from the `ET_INSTALL="${ET_INSTALL:-…}"` line through the `cmake -B native/asan …` configure with:
```bash
ITERS="${ITERS:-1000}"

# QA is the only ASan consumer; install the toolset's ASan runtime here (moved out of build.sh).
if command -v dnf >/dev/null 2>&1; then
  TOOLSET_VER="$(gcc -dumpversion | cut -d. -f1)"
  dnf install -y -q "gcc-toolset-${TOOLSET_VER}-libasan-devel" || true
fi

# Runtime comes from CMake resolution: default fetches the pinned logging tarball; set ET_INSTALL
# to point at an existing install (escape hatch). No precondition to check here.
ET_ARGS=(-DET_RUNTIME_VARIANT="${ET_RUNTIME_VARIANT:-logging}")
[ -n "${ET_INSTALL:-}" ] && ET_ARGS+=(-DET_INSTALL="${ET_INSTALL}")

cmake -B native/asan -S native -G "Unix Makefiles" "${ET_ARGS[@]}" -DET_BUILD_QA=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"
```
Keep the `#!`, header comment (update the ET_INSTALL prerequisite note to "runtime fetched by CMake, or set ET_INSTALL"), `set -euo pipefail`, `REPO_ROOT`, the `cmake --build native/asan …` line, and both harness run lines unchanged.

- [ ] **Step 6: Run test to verify it passes**

Run: `native/tests/build_variants_matrix.sh`
Expected: `PASS: build_variants matrix`

- [ ] **Step 7: (Manual/CI, not a unit gate) Real variant bench**

In-container: `bash native/build_variants.sh` downloads all three tarballs and prints real timings. Record in the task report.

- [ ] **Step 8: Commit**
```bash
git add native/bench.sh native/build_qa.sh native/build_variants.sh native/tests/build_variants_matrix.sh
git commit -m "refactor(native): bench/QA/variants consume the runtime resolution seam"
```

---

### Task 4: CI workflow — drop ET checkout, add provenance verification

**Files:**
- Modify: `.github/workflows/native-build.yml`
- Create: `native/tests/ci_workflow.sh`

**Interfaces:**
- Consumes: `EtRuntimePin.cmake` (Task 1) for the logging URL; `native/build.sh` (Task 2).
- Produces: a workflow with no `pytorch/executorch` checkout and a `gh attestation verify` gate before the container build.

- [ ] **Step 1: Write the failing test**

`native/tests/ci_workflow.sh`:
```bash
#!/usr/bin/env bash
# Host-fast assertions on the native-build workflow (no runner needed).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }
WF=".github/workflows/native-build.yml"

grep -q 'pytorch/executorch' "${WF}" && fail "ExecuTorch checkout must be removed"
grep -q 'gh attestation verify' "${WF}" || fail "attestation verify step missing"
grep -q 'measly-java-learning/executorch-runtime-dist' "${WF}" || fail "attestation repo missing"
grep -q 'native/build.sh' "${WF}" || fail "shim build step missing"
# Python is optional; if present, assert the file is valid YAML.
if command -v python3 >/dev/null 2>&1 && python3 -c 'import yaml' 2>/dev/null; then
  python3 -c "import yaml,sys; yaml.safe_load(open('${WF}')); print('yaml ok')" || fail "workflow is not valid YAML"
fi
echo "PASS: ci workflow"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `chmod +x native/tests/ci_workflow.sh && native/tests/ci_workflow.sh`
Expected: FAIL — the workflow still checks out `pytorch/executorch` and has no attestation step.

- [ ] **Step 3: Edit `.github/workflows/native-build.yml`**

Delete the entire `- name: Checkout ExecuTorch` step (the `uses: actions/checkout@v7` block with `repository: pytorch/executorch`). Insert, immediately **before** the `- name: Build in manylinux_2_28 container` step:
```yaml
      # Provenance gate: prove the pinned runtime tarball was built by Repo A's CI before we link
      # against it. URL_HASH in native/CMakeLists.txt covers integrity; this covers provenance.
      - name: Verify ExecuTorch runtime provenance
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          url="$(grep -oE 'https://[^"]*logging-linux-x86_64\.tar\.gz' native/cmake/EtRuntimePin.cmake | head -1)"
          test -n "${url}" || { echo "no logging URL in EtRuntimePin.cmake"; exit 1; }
          curl -fL -o runtime.tgz "${url}"
          gh attestation verify runtime.tgz --repo measly-java-learning/executorch-runtime-dist
          rm -f runtime.tgz
```
Leave the `Build in manylinux_2_28 container` step, the Corretto download, the shim upload, and the `build-java-package` job unchanged. (The container build's `cmake` configure fetches the runtime itself; this step only pre-verifies provenance.)

- [ ] **Step 4: Run test to verify it passes**

Run: `native/tests/ci_workflow.sh`
Expected: `PASS: ci workflow`

- [ ] **Step 5: Commit**
```bash
git add .github/workflows/native-build.yml native/tests/ci_workflow.sh
git commit -m "ci(native): drop ExecuTorch checkout, verify runtime provenance attestation"
```

---

### Task 5: Documentation + gitignore housekeeping

**Files:**
- Create: `docs/executorch-build-notes.md`
- Modify: `README.md` (build/QA sections)
- Modify: `docs/benchmarking.md` (variant note)
- Modify: `.gitignore`
- Test: `native/tests/docs_present.sh`

**Interfaces:**
- Consumes: all prior tasks (documents the new flow).

- [ ] **Step 1: Write the failing test**

`native/tests/docs_present.sh`:
```bash
#!/usr/bin/env bash
# Host-fast presence checks for the documentation deliverables.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

test -f docs/executorch-build-notes.md || fail "build-notes doc missing"
grep -q 'executorch-runtime-dist' docs/executorch-build-notes.md || fail "build-notes must point at Repo A"
grep -qi 'glibc' docs/executorch-build-notes.md || fail "build-notes must cover the glibc floor"
grep -qi 'vnni' docs/executorch-build-notes.md || fail "build-notes must cover the VNNI check rationale"

grep -q 'gh attestation verify' README.md || fail "README must document local attestation verify"
grep -qi 'FetchContent\|downloads\|prebuilt' README.md || fail "README must describe the fetch model"

grep -q 'et-install/' .gitignore && fail "obsolete et-install/ still ignored (no longer produced)"
echo "PASS: docs present"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `chmod +x native/tests/docs_present.sh && native/tests/docs_present.sh`
Expected: FAIL — the doc doesn't exist and README/`.gitignore` are unchanged.

- [ ] **Step 3: Create `docs/executorch-build-notes.md`**

```markdown
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
```

- [ ] **Step 4: Rewrite the `README.md` build sections**

Replace the block from `### 1. Build the native library` through the end of `### 3. Native QA (optional)` (the fenced `./native/build_qa.sh` block) with:
````markdown
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
````
Also update the existing "Prerequisites" bullet that says "An ExecuTorch v1.3.x checkout … (override with `ET_ROOT`)" — replace it with: "No ExecuTorch checkout needed — CMake downloads the pinned runtime. Network access is required for the tarball fetch (and Catch2 in QA)."

- [ ] **Step 5: Update `docs/benchmarking.md` variant note**

Find the section describing the bare/logging/devtools variant builds and append this note (adjust the anchor sentence to match the actual prose):
```markdown
> **Note (2026-07-03):** `native/build_variants.sh` no longer builds the three ExecuTorch variants
> from source — it downloads the prebuilt `bare`/`logging`/`devtools` tarballs from Repo A (selected
> via `ET_RUNTIME_VARIANT`) and times each. Same table, faster, no torch/ET build.
```

- [ ] **Step 6: Update `.gitignore`**

Remove the two obsolete lines `et-cmake-out/` and `et-install/` (no longer produced by the default build). Add, under a `# Native QA / bench outputs (container-produced)` comment:
```
native/bench/
native/bench-results/
native/asan/
native/qa_noasan/
```
(The `FetchContent` runtime cache lives under `native/build/_deps`, already covered by `**/build/`.)

- [ ] **Step 7: Run test to verify it passes**

Run: `native/tests/docs_present.sh`
Expected: `PASS: docs present`

- [ ] **Step 8: Commit**
```bash
git add docs/executorch-build-notes.md README.md docs/benchmarking.md .gitignore native/tests/docs_present.sh
git commit -m "docs(native): document the fetch model; archive ET build reasoning"
```

---

## Self-Review

**Spec coverage:**
- Component A (CMake resolution) → Task 1. ✅ (incl. `tokenizers_DIR` resolved to "keep, repointed".)
- Component B (build.sh drop Stage A) → Task 2. ✅ (VNNI + libasan relocated.)
- Component C (wrapper drop ET mount) → Task 2. ✅
- Component D (build_variants download loop) → Task 3. ✅ (+ bench.sh/build_qa.sh seam, which the spec implied via "consumers of ET_INSTALL".)
- Component E (verification: URL_HASH always, attestation in CI YAML, local docs) → Task 1 (`URL_HASH`) + Task 4 (CI) + Task 5 (README `gh` command). ✅
- Component F (build-notes doc) → Task 5. ✅
- Component G (gitignore, README, benchmarking.md, pin-bump) → Task 5 (+ pin-bump documented in the pin file header, Task 1). ✅
- Contract feedback (⚠ C8 docs-only; C2 tokenizers) → resolved in-plan (C2: tarball ships tokenizers; C8: from-source is docs in build-notes). No engine code. ✅

**Placeholder scan:** No TBD/TODO/"handle edge cases". All code blocks are complete; the two "Manual/CI" steps are explicitly labeled non-unit-gates, not deferred work.

**Type/seam consistency:** `ET_RUNTIME_VARIANT` (default `logging`), `ET_INSTALL` (escape hatch), `ET_PRINT_RESOLUTION`/`PRINT_BUILD_CONFIG` diagnostics, and the `et_timing: … warm_mean_ms=` parse are used identically across Tasks 1–5. `bench.sh` prints the same line `build_variants.sh` and its stub parse. Pin variable names (`ET_RUNTIME_URL_<variant>_linux-x86_64`) match the committed pin and the CMake nested-deref.

**Out-of-scope confirmed untouched:** JNI shim / `et_runtime` core / QA harness sources; multi-platform; `publish.yml` (still a Java-only publish — the shim-artifact question there predates this work and is not part of the fetch reshape).
