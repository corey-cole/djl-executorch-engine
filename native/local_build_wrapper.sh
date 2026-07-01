#!/bin/bash
set -ex # Fail on error, print commands to log

#
# This script arranges the environment for `build.sh` when run locally.
# It's performing the tasks that the GitHub Action workflow manages when running in CI.
#
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Default is specific to my local directory structure
# This assumes that ExecuTorch v1.3.x has already been cloned locally
ET_ROOT="${ET_ROOT:-$HOME/workspace/executorch}"

# Download the Corretto RPM from Amazon
# Extras are notes for eventual cross-platform CI
# Linux arm64: https://corretto.aws/downloads/latest/amazon-corretto-8-aarch64-linux-jdk.rpm
# Windows: https://corretto.aws/downloads/latest/amazon-corretto-8-x64-windows-jdk.zip
# MacOS x64: https://corretto.aws/downloads/latest/amazon-corretto-8-x64-macos-jdk.tar.gz
# MacOS arm64: https://corretto.aws/downloads/latest/amazon-corretto-8-aarch64-macos-jdk.tar.gz
if [ ! -f ${REPO_ROOT}/amazon-corretto-linux-jdk.rpm ]; then
  echo "Downloading Amazon Corretto JDK RPM to ${REPO_ROOT}/amazon-corretto-linux-jdk.rpm"
  curl -L -o ${REPO_ROOT}/amazon-corretto-linux-jdk.rpm https://corretto.aws/downloads/latest/amazon-corretto-8-x64-linux-jdk.rpm
fi

# quay.io/pypa/manylinux2014_aarch64 is the arm64 version of this
# Can move to quay.io/pypa/manylinux_2_28_x86_64 / quay.io/pypa/manylinux_2_28_aarch64 when we want to bump glibc to 2.28

# Specific instructions for ExecuTorch 1.3, covers major operating systems
# https://docs.pytorch.org/executorch/1.3/using-executorch-building-from-source.html
# Note that there's a `profiling` target for ET, check into what that entails later

# Must use manylinux_2_28 as ExecuTorch 1.3 requires glibc >= 2.28 via PyTorch 2.12.0+cpu
# Pass SKIP_ET_BUILD=1 to reuse a previously-built ExecuTorch runtime (in ${REPO_ROOT}/et-install)
# and only rebuild the shim: `SKIP_ET_BUILD=1 ./native/local_build_wrapper.sh`
docker run --rm \
    -e SKIP_ET_BUILD="${SKIP_ET_BUILD:-0}" \
    -v ${REPO_ROOT}:/workspace \
    -v ${ET_ROOT}:/workspace/executorch \
    -w /workspace \
    quay.io/pypa/manylinux_2_28_x86_64:latest \
    /bin/bash /workspace/native/build.sh
