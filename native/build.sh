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
