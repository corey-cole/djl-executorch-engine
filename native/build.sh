#!/bin/bash
set -ex # Fail on error, print commands to log

# Host fork. Under Git-Bash on Windows `uname -s` is MINGW64_NT-* or MSYS_NT-*. The caller must have
# already activated the MSVC dev shell (see .github/workflows/native-build-job.yml); this script does
# not activate VS itself. Everything Linux-only below (Corretto RPM, chown, dnf, nproc) is skipped.
case "$(uname -s)" in
  MINGW*|MSYS*) ET_HOST_OS=windows ;;
  *)            ET_HOST_OS=linux ;;
esac

# Container bind-mount outputs are root-owned on the host; chown them back on exit (any status).
cleanup() {
  rc=$?
  if [ -n "${HOST_UID:-}" ]; then
    chown -R "${HOST_UID}:${HOST_GID}" "${NATIVE_BUILD_DIR}" src/main/resources/native/linux* 2>/dev/null || true
  fi
  exit "$rc"
}
[ "${ET_HOST_OS}" = "linux" ] && trap cleanup EXIT

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

if [ "${ET_HOST_OS}" = "windows" ]; then
  echo "--- Using the runner's JDK headers (headers-only; we never link libjvm) ---"
  test -n "${JAVA_HOME:-}" || { echo "JAVA_HOME must be set on Windows (see setup-java)"; exit 1; }
  # Git-Bash gives JAVA_HOME as a Windows path; cmake accepts it, but the test below needs a POSIX path.
  JAVA_HOME="$(cygpath -u "${JAVA_HOME}" 2>/dev/null || echo "${JAVA_HOME}")"
  export JAVA_HOME
  test -f "${JAVA_HOME}/include/win32/jni_md.h" \
    || { echo "JDK headers not found under JAVA_HOME=${JAVA_HOME} (expected include/win32/jni_md.h)"; exit 1; }
  echo "JAVA_HOME=${JAVA_HOME}"
else
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
fi

if [ "${ET_HOST_OS}" = "windows" ]; then
  echo "--- Toolchain Versions (MSVC dev shell must already be activated by the caller) ---"
  command -v cl >/dev/null 2>&1 || { echo "cl.exe not on PATH: activate the VS dev shell first"; exit 1; }
  command -v ninja >/dev/null 2>&1 || { echo "ninja not on PATH: activate the VS dev shell first"; exit 1; }
  cl 2>&1 | head -1; cmake --version; ninja --version
else
  echo "--- Setting up Ninja (the shim configures with -G Ninja) ---"
  export PATH="/opt/python/cp312-cp312/bin:${PATH}"
  pip install ninja
  echo "--- Toolchain Versions ---"
  gcc --version; g++ --version; cmake --version; ninja --version
fi

# In GitHub Actions, publish JAVA_HOME for any downstream shim-building step. ET_INSTALL is no
# longer exported — the runtime is resolved inside cmake now (FetchContent), per configure.
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "JAVA_HOME=${JAVA_HOME}" >> "${GITHUB_ENV}"
fi

if [ "${ET_HOST_OS}" = "windows" ]; then
  JOBS="${JOBS:-${NUMBER_OF_PROCESSORS:-4}}"
else
  JOBS="${JOBS:-$(nproc)}"
  cd /workspace
fi
# native/build is disposable; its cached absolute paths won't match a fresh container, so wipe it.
rm -rf "${NATIVE_BUILD_DIR}"

# Forward ET_INSTALL as an escape hatch only if the caller set it; otherwise CMake FetchContents
# the pinned ${ET_RUNTIME_VARIANT} tarball.
ET_INSTALL_ARG=()
[ -n "${ET_INSTALL:-}" ] && ET_INSTALL_ARG=(-DET_INSTALL="${ET_INSTALL}")
# MSVC encodes the CRT flavour into every object and the linker refuses to mix them. The pinned runtime
# tarball is built Release (/MD — see its BUILDINFO cmake_flags), so the shim must be Release too or the
# link dies with LNK2038 'RuntimeLibrary' / '_ITERATOR_DEBUG_LEVEL' mismatches. GCC/ELF has no such ABI
# tag, so the Linux leg stays as-is (unset) and its artifact is unchanged.
BUILD_TYPE_ARG=()
[ "${ET_HOST_OS}" = "windows" ] && BUILD_TYPE_ARG=(-DCMAKE_BUILD_TYPE=Release)
cmake -B "${NATIVE_BUILD_DIR}" -S native -G Ninja \
  -DET_RUNTIME_VARIANT="${ET_RUNTIME_VARIANT}" "${BUILD_TYPE_ARG[@]}" "${ET_INSTALL_ARG[@]}"
cmake --build "${NATIVE_BUILD_DIR}" -j"${JOBS}"

if [ "${ET_HOST_OS}" = "windows" ]; then
  OUT_PLATFORM="windows-x86_64"; OUT_LIB="executorch_djl.dll"
else
  OUT_PLATFORM="linux-x86_64";   OUT_LIB="libexecutorch_djl.so"
fi

if [ "${STAGE_SO}" = "1" ]; then
  OUT="src/main/resources/native/${OUT_PLATFORM}"
  mkdir -p "${OUT}"
  cp "${NATIVE_BUILD_DIR}/${OUT_LIB}" "${OUT}/"
  echo "Artifact: ${OUT}/${OUT_LIB}"
  ls -lh "${OUT}/${OUT_LIB}"
else
  echo "STAGE_SO=0: built shim but not staging into resources"
  ls -lh "${NATIVE_BUILD_DIR}/${OUT_LIB}"
fi
