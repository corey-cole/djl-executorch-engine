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
