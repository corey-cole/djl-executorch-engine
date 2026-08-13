#!/bin/bash
set -ex # Fail on error, print commands to log

# Runs a native/ script inside the manylinux_2_28 toolchain container — the environment the GHA
# workflow uses in CI. The ExecuTorch runtime is downloaded by CMake (FetchContent) during the run;
# there is NO ET checkout to mount. The JDK headers come from the image itself (JAVA_HOME), not a
# downloaded RPM, and the image is a manifest list covering amd64 and arm64, so this wrapper works
# unmodified on an aarch64 workstation. This is the BLESSED way to run the native scripts: the
# toolchain matches, and a shim built here keeps its glibc-2.28 floor (RHEL8). Running these
# scripts directly on the host works but breaks the floor (build.sh) or collides on a container-made
# cache (bench/qa wipe theirs).
#
# Usage: ./native/local_build_wrapper.sh [script]   (default: native/build.sh)
#   ./native/local_build_wrapper.sh native/bench.sh
#   ITERS=2000 ./native/local_build_wrapper.sh native/build_qa.sh
#   ./native/local_build_wrapper.sh native/build_variants.sh
# Note: only build.sh chowns its outputs back to you; bench/qa/variants leave root-owned dirs
# (see the "Container file ownership" note in README.md).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Optional first arg: the native/ script to run in the container (default the shim build).
TARGET_SCRIPT="${1:-native/build.sh}"

# The pinned shared toolchain image, digest not tag. It is a manifest list covering amd64 and
# arm64, so Docker resolves the architecture -- there is no Dockerfile to pick and no --platform
# to pass. The digest lives in one file so a bump is a one-line change that CI and this wrapper
# pick up together; a second copy here would drift, and the failure mode is CI green while you
# build against a different toolchain. Published by measly-java-learning/base-docker-images; see
# its docs/consuming-engine-build.md for what the image guarantees.
ET_BUILD_IMAGE="${ET_BUILD_IMAGE:-$(cat "${REPO_ROOT}/.engine-build-image")}"
test -n "${ET_BUILD_IMAGE}" || { echo "empty .engine-build-image" >&2; exit 1; }

# Override the runtime variant with ET_RUNTIME_VARIANT (default logging). ITERS/WARMUP forward to
# the bench/QA scripts when set (harmless for build.sh, which ignores them).
docker run --rm \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    -e ET_RUNTIME_VARIANT="${ET_RUNTIME_VARIANT:-logging}" \
    -e ITERS \
    -e WARMUP \
    -e MODEL \
    -e THREADS \
    -e MODES \
    -e BUILD_ONLY \
    -e REPS \
    -e INTRAOP \
    -e ET_STRESS \
    -e ET_STRESS_SECONDS \
    -v "${REPO_ROOT}":/workspace \
    -w /workspace \
    "${ET_BUILD_IMAGE}" \
    /bin/bash "/workspace/${TARGET_SCRIPT}"
