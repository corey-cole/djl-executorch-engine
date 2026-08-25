#!/usr/bin/env bash
# Build + run the native QA targets (Catch2 units + ASan/LSan leak harness) against the resolved
# ExecuTorch runtime (default fetches the pinned tarball for the platform-keyed variant — devtools
# on linux-x86_64, logging elsewhere — or set ET_INSTALL). Not part of the shipping build — the QA
# targets are gated behind -DET_BUILD_QA=ON and built with AddressSanitizer/LeakSanitizer, so they
# are a distinct build tree (native/asan) from the Release .so (native/build via native/build.sh).
#
# Prerequisites: a C++ compiler with -fsanitize=address, cmake + make, and network access (Catch2
# and the ExecuTorch runtime tarball are fetched at configure time).
#
# CI env vars: ET_RUNTIME_VARIANT (default: platform-keyed — devtools on linux-x86_64),
# ET_INSTALL (escape hatch). The QA targets are JVM-free and the shared native/CMakeLists.txt skips
# the JNI shim under -DET_BUILD_QA=ON, so NO JAVA_HOME/JDK is needed. In GitHub Actions, run this
# in the shared engine-build image as native/build.sh (matching gcc-toolset toolchain).
set -euo pipefail

# shellcheck source=native/container_env.sh
. "${BASH_SOURCE[0]%/*}/container_env.sh"

# Run from the repo root regardless of where this script is invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ITERS="${ITERS:-1000}"

# Host fork; see native/build.sh. Windows QA is Catch2-only: MSVC has ASan but no LeakSanitizer, so
# et_leak_harness is structurally Linux-only (design §6). The Catch2 suite is also Windows' XNNPACK
# gate, standing in for the nm post-link guard that cannot run under MSVC (design §3.4).
case "$(uname -s)" in
  MINGW*|MSYS*) ET_HOST_OS=windows ;;
  *)            ET_HOST_OS=linux ;;
esac

# Platform identity, mirroring build.sh's OUT_PLATFORM. The QA tree must exercise the runtime that
# ships, so the variant is resolved from the same platform list build.sh uses (build.sh is not
# sourceable here: set -ex, immediate PRINT_* exits). The default list is linux-x86_64, so Windows
# and aarch64 QA fall back to logging -- unchanged behaviour there.
if [ "${ET_HOST_OS}" = "windows" ]; then
  OUT_PLATFORM="windows-x86_64"
else
  case "$(uname -m)" in
    aarch64|arm64) OUT_PLATFORM="linux-aarch64" ;;
    *)             OUT_PLATFORM="linux-x86_64"  ;;
  esac
fi

ET_DEVTOOLS_SUPPORTED_PLATFORMS="${ET_DEVTOOLS_SUPPORTED_PLATFORMS-linux-x86_64}"
if [ -z "${ET_RUNTIME_VARIANT:-}" ]; then
  case " ${ET_DEVTOOLS_SUPPORTED_PLATFORMS} " in
    *" ${OUT_PLATFORM} "*) ET_RUNTIME_VARIANT=devtools ;;
    *)                      ET_RUNTIME_VARIANT=logging ;;
  esac
fi

ET_ARGS=(-DET_RUNTIME_VARIANT="${ET_RUNTIME_VARIANT}")
[ -n "${ET_INSTALL:-}" ] && ET_ARGS+=(-DET_INSTALL="${ET_INSTALL}")

if [ "${ET_HOST_OS}" = "windows" ]; then
  command -v cl >/dev/null 2>&1 || { echo "cl.exe not on PATH: activate the VS dev shell first"; exit 1; }
  JOBS="${JOBS:-${NUMBER_OF_PROCESSORS:-4}}"
  bash native/clean_stale_tree.sh native/asan native
  # No sanitizers: MSVC has no LeakSanitizer, and the leak harness is not built here at all.
  # RelWithDebInfo, NOT Debug: Debug would compile against the Debug CRT while the pinned runtime is
  # Release, and MSVC refuses to mix them — the same LNK2038/LNK1319 wall the shim build hits (see
  # build.sh). RelWithDebInfo gives us the matching release CRT while keeping symbols, so a Catch2
  # failure is still debuggable. The static-vs-dynamic CRT choice is separate and is made by
  # CMAKE_MSVC_RUNTIME_LIBRARY in native/CMakeLists.txt, which also propagates into the FetchContent'd
  # Catch2 build — a /MD Catch2 inside a /MT test exe links silently and corrupts at runtime.
  cmake -B native/asan -S native -G Ninja "${ET_ARGS[@]}" -DET_BUILD_QA=ON \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
  cmake --build native/asan --target et_runtime_test -j"${JOBS}"

  # Catch2 comes in via FetchContent, so it is the one target whose CRT we do not set directly — the
  # global CMAKE_MSVC_RUNTIME_LIBRARY has to propagate into a subproject to reach it. A /MD Catch2
  # inside this /MT test exe links with no LNK2038 and not even an LNK4098, then corrupts the heap at
  # runtime. Assert it here, before running the suite, so the failure names its own cause instead of
  # surfacing as an inexplicable Catch2 crash. No DLL argument: this tree builds a test exe, not a DLL.
  echo "--- CRT check: QA tree must be uniformly static (/MT) ---"
  bash native/tests/check_windows_crt.sh native/asan

  echo "--- Catch2 unit suite (no sanitizers; MSVC has no LSan) ---"
  # --order decl: same rationale as the Linux run below -- Catch2 v3 randomizes by default and the
  # intra-op pool tests must precede ANY EtRuntime construction (issue #26).
  ./native/asan/et_runtime_test.exe --order decl
  echo "--- Leak harness SKIPPED: no LeakSanitizer under MSVC (Linux-only coverage) ---"
else
  # <sys/sdt.h> ships in systemtap-sdt-devel and native/core/et_probes.h requires it. Assert it so a
  # missing header fails here by name rather than as a fatal error part-way through the compile.
  test -e /usr/include/sys/sdt.h || {
    echo "missing /usr/include/sys/sdt.h (systemtap-sdt-devel): native/core/et_probes.h needs it;"
    echo "install it, or run QA via ./native/local_build_wrapper.sh native/build_qa.sh"; exit 1; }

  JOBS="${JOBS:-$(nproc)}"
  # Drop the tree only if it was configured for a different source root (container vs host); a same-root
  # re-run keeps it so the cached Catch2 build (native/asan/_deps) is reused, not rebuilt. CLEAN=1 forces.
  bash native/clean_stale_tree.sh native/asan native
  et_chown_outputs_on_exit native/asan
  export UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1
  cmake -B native/asan -S native -G "Unix Makefiles" "${ET_ARGS[@]}" -DET_BUILD_QA=ON \
    -DET_UBSAN=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"
  cmake --build native/asan --target et_runtime_test et_leak_harness et_stress_harness -j"${JOBS}"

  echo "--- Catch2 unit suite ---"
  # --order decl: Catch2 v3 randomizes test order by default, but the intra-op tests (registered
  # first in et_runtime_test.cpp) MUST run before any EtRuntime construction -- the reset guard
  # (issue #26) refuses a reset once a runtime has been constructed, so a randomized order would
  # flake the pool tests.
  ./native/asan/et_runtime_test --order decl

  echo "--- ASan/LSan leak harness (${ITERS} iterations) ---"
  ./native/asan/et_leak_harness native/spike/add.pte "${ITERS}"                # planned: grow==0, guards the runtime pin
  ./native/asan/et_leak_harness native/spike/add_unplanned.pte "${ITERS}" 4    # staged 8*ITERS times, grow == 0 (slots sized at load)
  ./native/asan/et_leak_harness native/spike/add_unplanned.pte 1 10000         # inverted: isolates steady state; grow == 0

  # Opt-in: saturates every core for its duration, which is not something to do on a free CI runner.
  # ET_STRESS_SECONDS tunes the per-arm duration (default 20).
  if [ "${ET_STRESS:-0}" = "1" ]; then
    STRESS_PTE="src/test/resources/models/stress/stress_mlp.pte"
    STRESS_SECS="${ET_STRESS_SECONDS:-20}"
    if [ ! -f "${STRESS_PTE}" ]; then
      echo "--- Stress harness SKIPPED: ${STRESS_PTE} not found"
      echo "    (build it via tools/scripts/export_stress_model.py)"
    else
      echo "--- ASan/LSan stress harness: 8 threads, global sharing, ${STRESS_SECS}s ---"
      ET_SHARING_MODE=2 ./native/asan/et_stress_harness "${STRESS_PTE}" 8 "${STRESS_SECS}"
      echo "--- ASan/LSan stress harness: 8 threads, sharing disabled, ${STRESS_SECS}s ---"
      ET_SHARING_MODE=0 ./native/asan/et_stress_harness "${STRESS_PTE}" 8 "${STRESS_SECS}"
    fi
  fi
fi
