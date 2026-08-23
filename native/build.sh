#!/bin/bash
set -ex # Fail on error, print commands to log

# shellcheck source=native/container_env.sh
. "${BASH_SOURCE[0]%/*}/container_env.sh"

# Host fork. Under Git-Bash on Windows `uname -s` is MINGW64_NT-* or MSYS_NT-*. The caller must have
# already activated the MSVC dev shell (see .github/workflows/native-build-job.yml); this script does
# not activate VS itself. Everything Linux-only below (Corretto RPM, chown, dnf, nproc) is skipped.
case "$(uname -s)" in
  MINGW*|MSYS*) ET_HOST_OS=windows ;;
  *)            ET_HOST_OS=linux ;;
esac
# Platform identity of the artifact this run produces. Derived from the host alone, so it is safe to
# compute here -- PRINT_OPENVINO_RESOLUTION reads it without building anything.
if [ "${ET_HOST_OS}" = "windows" ]; then
  OUT_PLATFORM="windows-x86_64"; OUT_LIB="executorch_djl.dll"
else
  case "$(uname -m)" in
    aarch64|arm64) OUT_PLATFORM="linux-aarch64" ;;
    *)             OUT_PLATFORM="linux-x86_64"  ;;
  esac
  OUT_LIB="libexecutorch_djl.so"
fi

PIN="native/cmake/EtRuntimePin.cmake"

# The engine's OpenVINO support set: what the JAVA layer can load, which is not the same question as
# what the pin publishes. The producer ships a windows-x86_64 bundle that OpenVinoRuntime cannot
# extract -- its library list, ABI-suffix naming and OPENVINO_LIB_PATH handling are all .so-shaped --
# so staging it would put ~21 MB in a jar nothing can use. Adding a platform here is the LAST step of
# supporting it, never the first.
ET_OPENVINO_SUPPORTED_PLATFORMS="${ET_OPENVINO_SUPPORTED_PLATFORMS:-linux-x86_64}"

# Sets OV_DECISION (stage | unsupported | unpublished) and, when staging, OV_URL / OV_SHA / OV_VER.
#
# Keyed on the PLATFORM identity, not the pin row -- the opposite of the ExecuTorch tarball, which is
# row-keyed because the CRT flavour is part of its identity. An OpenVINO bundle is per platform: the
# wheel's DLLs are /MD however a consumer links, so one bundle serves both Windows rows and the pin
# expresses that with an ALIAS row whose value is a CMake variable reference. Shell cannot
# dereference that, so a row-keyed grep would return the literal text rather than a URL.
#
# An absent row is legitimate (linux-aarch64 has no bundle upstream), so this mirrors the pin's
# et_runtime_openvino_url(): empty, not an error. The deprecated singular
# ET_RUNTIME_OPENVINO_{PLATFORM,URL,SHA256} vars are deliberately not read.
et_openvino_resolve() {
  local platform="$1"
  OV_URL=""; OV_SHA=""; OV_VER=""

  case " ${ET_OPENVINO_SUPPORTED_PLATFORMS} " in
    *" ${platform} "*) ;;
    *) OV_DECISION="unsupported"; return 0 ;;
  esac

  OV_URL="$(grep -oPz "set\(ET_RUNTIME_OPENVINO_URL_${platform}\s+\"\K[^\"]+" "${PIN}" | tr -d '\0' || true)"
  OV_SHA="$(grep -oPz "set\(ET_RUNTIME_OPENVINO_SHA256_${platform}\s+\"\K[^\"]+" "${PIN}" | tr -d '\0' || true)"
  OV_VER="$(grep -oP 'set\(ET_RUNTIME_OPENVINO_VERSION "\K[^"]+' "${PIN}" || true)"

  if [ -z "${OV_URL}" ] || [ -z "${OV_SHA}" ]; then
    OV_DECISION="unpublished"
    return 0
  fi

  # A non-literal value means the lookup reached an alias row. Fail loudly rather than handing curl
  # an unexpanded cmake reference, which would 404 with a message naming none of this.
  case "${OV_URL}" in
    https://*) ;;
    *) echo "OpenVINO pin row for ${platform} is not a literal URL: ${OV_URL}"; exit 1 ;;
  esac

  OV_DECISION="stage"
}

# Fast diagnostic: print the staging decision for a platform and exit, mirroring PRINT_BUILD_CONFIG.
# ET_STAGE_PLATFORM overrides the host's identity so native/tests/openvino_pin_selector.sh can assert
# a foreign platform's decision -- the pin is a file, so no foreign hardware is involved.
if [ -n "${PRINT_OPENVINO_RESOLUTION:-}" ]; then
  ov_platform="${ET_STAGE_PLATFORM:-${OUT_PLATFORM}}"
  et_openvino_resolve "${ov_platform}"
  echo "OV_RESOLUTION platform=${ov_platform} decision=${OV_DECISION} version=${OV_VER} url=${OV_URL}"
  exit 0
fi

# --- Shim build config. The ExecuTorch runtime is NOT built here anymore: native/CMakeLists.txt
#     resolves it (FetchContent the pinned tarball, or -DET_INSTALL escape hatch). The runtime
#     recipe now lives in measly-java-learning/executorch-runtime-dist; see
#     docs/executorch-build-notes.md for the engine-side reasoning. ---
ET_RUNTIME_VARIANT="${ET_RUNTIME_VARIANT:-logging}"
STAGE_SO="${STAGE_SO:-1}"
NATIVE_BUILD_DIR="${NATIVE_BUILD_DIR:-native/build}"

[ "${ET_HOST_OS}" = "linux" ] && \
  et_chown_outputs_on_exit "${NATIVE_BUILD_DIR}" 'src/main/resources/native/linux*'

# Fast diagnostic: print resolved shim-build config and exit before any heavy setup.
if [ -n "${PRINT_BUILD_CONFIG:-}" ]; then
  echo "ET_RUNTIME_VARIANT=${ET_RUNTIME_VARIANT} STAGE_SO=${STAGE_SO} NATIVE_BUILD_DIR=${NATIVE_BUILD_DIR} ET_INSTALL=${ET_INSTALL:-}"
  exit 0
fi

# This script expects a toolchain — supplied by the pinned engine-build image via
# ./native/local_build_wrapper.sh, or by the host (JAVA_HOME, ninja, gcc/g++, cmake). It asserts
# requirements; it never installs them. Building outside the image does not hold the glibc-2.28
# floor (a warning is printed; see the toolchain block below).
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
  # Headers only; we never link libjvm. JAVA_HOME comes from the pinned image or from the host --
  # either way it is supplied, never installed here.
  echo "--- JDK headers ---"
  test -n "${JAVA_HOME:-}" \
    || { echo "JAVA_HOME is unset: point it at any JDK, or build via ./native/local_build_wrapper.sh"; exit 1; }
  test -f "${JAVA_HOME}/include/linux/jni_md.h" \
    || { echo "no JDK headers under JAVA_HOME=${JAVA_HOME} (want include/linux/jni_md.h); or build via ./native/local_build_wrapper.sh"; exit 1; }
  echo "JAVA_HOME=${JAVA_HOME}"
fi

if [ "${ET_HOST_OS}" = "windows" ]; then
  echo "--- Toolchain Versions (MSVC dev shell must already be activated by the caller) ---"
  command -v cl >/dev/null 2>&1 || { echo "cl.exe not on PATH: activate the VS dev shell first"; exit 1; }
  command -v ninja >/dev/null 2>&1 || { echo "ninja not on PATH: activate the VS dev shell first"; exit 1; }
  cl 2>&1 | head -1; cmake --version; ninja --version
else
  echo "--- Toolchain (asserted, never installed) ---"
  command -v ninja >/dev/null 2>&1 \
    || { echo "ninja not on PATH: install it, or build via ./native/local_build_wrapper.sh"; exit 1; }
  # Building outside the image is supported but does NOT hold the glibc-2.28 floor -- the artifact
  # links host glibc. Fine for local ./gradlew test, never for a release (see CLAUDE.md).
  if [ "${MEASLY_DJL_PINNED_IMAGE:-}" != "1" ]; then
    echo "WARNING: not the pinned engine-build image -- this .so links host glibc and breaks the" >&2
    echo "         2.28 floor. Local testing only; release builds go through local_build_wrapper.sh." >&2
  fi
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
# tarball is built Release with the STATIC CRT (/MT — see its BUILDINFO cmake_flags:
# CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded), so the shim must be Release too or the link dies with
# LNK2038 '_ITERATOR_DEBUG_LEVEL' mismatches. Release is about the debug/release CRT split only; the
# static-vs-dynamic choice is made by CMAKE_MSVC_RUNTIME_LIBRARY in native/CMakeLists.txt, not here.
# GCC/ELF has no such ABI tag, so the Linux leg stays as-is (unset) and its artifact is unchanged.
BUILD_TYPE_ARG=()
[ "${ET_HOST_OS}" = "windows" ] && BUILD_TYPE_ARG=(-DCMAKE_BUILD_TYPE=Release)
cmake -B "${NATIVE_BUILD_DIR}" -S native -G Ninja \
  -DET_RUNTIME_VARIANT="${ET_RUNTIME_VARIANT}" "${BUILD_TYPE_ARG[@]}" "${ET_INSTALL_ARG[@]}"
cmake --build "${NATIVE_BUILD_DIR}" -j"${JOBS}"

if [ "${STAGE_SO}" = "1" ]; then
  OUT="src/main/resources/native/${OUT_PLATFORM}"
  mkdir -p "${OUT}"
  cp "${NATIVE_BUILD_DIR}/${OUT_LIB}" "${OUT}/"
  echo "Artifact: ${OUT}/${OUT_LIB}"
  ls -lh "${OUT}/${OUT_LIB}"

  # Third-party notices from the resolved runtime tree: escape-hatch (ET_INSTALL set) or the
  # FetchContent extraction under the build dir. Required — never ship a binary without them.
  ET_RUNTIME_ROOT="${ET_INSTALL:-${NATIVE_BUILD_DIR}/_deps/et_runtime-src}"
  test -f "${ET_RUNTIME_ROOT}/LICENSE" && test -d "${ET_RUNTIME_ROOT}/THIRD-PARTY-NOTICES" \
    || { echo "runtime notices missing under ${ET_RUNTIME_ROOT} (LICENSE + THIRD-PARTY-NOTICES/)"; exit 1; }
  LIC_OUT="${OUT}/licenses"
  rm -rf "${LIC_OUT}"
  mkdir -p "${LIC_OUT}"
  cp "${ET_RUNTIME_ROOT}/LICENSE" "${LIC_OUT}/"
  cp -r "${ET_RUNTIME_ROOT}/THIRD-PARTY-NOTICES" "${LIC_OUT}/"
  echo "Notices: ${LIC_OUT} ($(find "${LIC_OUT}" -type f | wc -l) files)"

  # --- OpenVINO runtime bundle (optional, published as a separate opt-in jar) ---
  # Fetched here rather than by CMake because nothing links against it: the delegate dlopens the C
  # API at runtime. The decision is made by et_openvino_resolve above; this only acts on it.
  et_openvino_resolve "${OUT_PLATFORM}"
  case "${OV_DECISION}" in
  stage)
    OV_OUT="${OUT}/openvino"
    TARBALL="native/build/openvino-runtime.tar.gz"

    curl -fsSL -o "${TARBALL}" "${OV_URL}"
    echo "${OV_SHA}  ${TARBALL}" | sha256sum -c - \
      || { echo "OpenVINO bundle SHA256 mismatch -- refusing to stage"; exit 1; }

    rm -rf "${OV_OUT}"
    mkdir -p "${OV_OUT}"
    # --strip-components=1 drops the single top-level dir, keeping lib/, licenses/ and BUILDINFO.
    tar xzf "${TARBALL}" --strip-components=1 -C "${OV_OUT}"
    # The symlink is deliberately not shipped: jars do not preserve symlinks, and it is unnecessary --
    # OPENVINO_LIB_PATH names the versioned file directly and $ORIGIN resolves the rest. Verified
    # against this exact bundle.
    rm -f "${OV_OUT}/lib/libopenvino_c.so"

    {
      echo "openvino_version=${OV_VER}"
      echo "tarball_sha256=${OV_SHA}"
      echo "tarball_url=${OV_URL}"
    } > "${OV_OUT}/MANIFEST"

    echo "OpenVINO bundle staged: ${OV_OUT} ($(du -sh "${OV_OUT}" | cut -f1))"
    ;;
  unsupported)
    echo "OpenVINO bundle: ${OUT_PLATFORM} is not in the engine's supported set (${ET_OPENVINO_SUPPORTED_PLATFORMS}); skipping"
    ;;
  unpublished)
    echo "OpenVINO bundle: the pin publishes no bundle for ${OUT_PLATFORM}; skipping"
    ;;
  esac
else
  echo "STAGE_SO=0: built shim but not staging into resources"
  ls -lh "${NATIVE_BUILD_DIR}/${OUT_LIB}"
fi
