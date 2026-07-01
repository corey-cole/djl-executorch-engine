#!/bin/bash
set -ex # Fail on error, print commands to log

# This script expects:
# 1. To be running inside quay.io/pypa/manylinux_2_28_x86_64
# 2. The Corretto RPM downloaded to /workspace
# 3. ExecuTorch checked out to /workspace/executorch

echo "--- Extracting Corretto JDK headers (headers-only; we never link libjvm) ---"
# manylinux_2_28 is AlmaLinux 8, where 'yum' IS 'dnf'. Installing a local RPM still hits the
# network to resolve deps and hangs on a slow/blocked mirror. Our shim needs only jni.h, so we
# extract the RPM directly (no network, no dnf). This minimal image has 'rpm' (=> rpm2archive) and
# tar, but NOT cpio, so we go RPM -> .tgz -> tar rather than rpm2cpio | cpio.
JDK_EXTRACT=/opt/corretto
mkdir -p "${JDK_EXTRACT}"
# rpm2archive writes "<input>.tgz" next to its input; stage in /tmp so we don't litter /workspace.
cp /workspace/amazon-corretto-linux-jdk.rpm /tmp/corretto.rpm
rpm2archive /tmp/corretto.rpm            # -> /tmp/corretto.rpm.tgz
tar -C "${JDK_EXTRACT}" -xzf /tmp/corretto.rpm.tgz
# Derive JAVA_HOME from the extracted jni.h so we don't hardcode the corretto version dir.
JNI_H="$(find "${JDK_EXTRACT}" -path '*/include/jni.h' | head -1)"
export JAVA_HOME="${JNI_H%/include/jni.h}"
test -f "${JAVA_HOME}/include/linux/jni_md.h" \
  || { echo "JDK headers not found under JAVA_HOME=${JAVA_HOME}"; exit 1; }
echo "JAVA_HOME=${JAVA_HOME}"

echo "--- Setting up Python and Ninja ---"
export PATH="/opt/python/cp312-cp312/bin:${PATH}"
pip install ninja   # needed by BOTH the ET runtime build (Stage A) and the shim build (-G Ninja)


echo "--- Toolchain Versions ---"
gcc --version
g++ --version
cmake --version
ninja --version

echo "--- Asserting AVX512-VNNI toolchain support ---"
# XNNPACK builds its x86 VNNI microkernels as C-with-intrinsics compiled with -mavx512vnni,
# then dispatches at runtime via cpuinfo. We only need the toolchain to *encode* VNNI here; the
# build host CPU need not support it. Fail early if gcc/binutils is too old to encode it.
echo '#include <immintrin.h>
__m512i f(__m512i a, __m512i b, __m512i c) { return _mm512_dpbusd_epi32(a, b, c); }' \
  | gcc -x c -c -mavx512vnni - -o /dev/null \
  || { echo "ERROR: toolchain cannot encode AVX512-VNNI (need gcc>=8, binutils>=2.30)"; exit 1; }

ET_BUILD="/workspace/et-cmake-out"
ET_INSTALL="/workspace/et-install"

# In GitHub Actions, publish the values a later step (e.g. native/build_qa.sh) needs, since our
# `export`s here don't survive into a separate step. ET_INSTALL is required by build_qa.sh; JAVA_HOME
# is exported for any downstream step that builds the JNI shim.
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "ET_INSTALL=${ET_INSTALL}" >> "${GITHUB_ENV}"
  echo "JAVA_HOME=${JAVA_HOME}" >> "${GITHUB_ENV}"
fi

# --- Stage A: build the ExecuTorch runtime (the slow part). Set SKIP_ET_BUILD=1 to reuse an
#     existing ${ET_INSTALL} from a prior run and jump straight to the (fast) shim build below.
#     The install tree persists on the host because ${ET_INSTALL} is under the /workspace mount. ---
if [ "${SKIP_ET_BUILD:-0}" != "1" ]; then
  echo "--- Installing ExecuTorch build dependencies (torch etc.) ---"
  pip install -U pip setuptools wheel pyyaml
  pip install torch==2.12.0+cpu --index-url https://download.pytorch.org/whl/cpu

  echo "--- Configuring CMake ---"
  # TODO: We'll need to manage backends based on input once we go multiplatform
  # TODO: There's a separate profiling preset that we might conditionally enable depending on how benchmarking works out
  cmake -B ${ET_BUILD} -S /workspace/executorch -G Ninja --preset linux \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${ET_INSTALL} \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON

  echo "--- Building ExecuTorch ---"
  cmake --build ${ET_BUILD} -j"$(nproc)"

  echo "--- Installing Artifacts ---"
  cmake --install ${ET_BUILD} --prefix ${ET_INSTALL}  # REQUIRED: emits executorch-config.cmake for find_package
  echo "ExecuTorch runtime installed to ${ET_INSTALL}"
else
  echo "--- SKIP_ET_BUILD=1: reusing existing ExecuTorch runtime at ${ET_INSTALL} ---"
fi


JOBS="${JOBS:-$(nproc)}"

test -f "${ET_INSTALL}/lib/cmake/ExecuTorch/executorch-config.cmake" \
  || { echo "ET_INSTALL=${ET_INSTALL} has no executorch-config.cmake; build the runtime first (see Prerequisite)"; exit 1; }

cd /workspace
# native/build is a disposable (gitignored) tree that a host-side build_desktop.sh may also use;
# its cached absolute paths (/home/corey/...) won't match the container's (/workspace/...), so
# wipe it for a clean configure. et-cmake-out doesn't need this — it's only ever built in-container.
rm -rf native/build
cmake -B native/build -S native -G Ninja -DET_INSTALL="${ET_INSTALL}"
cmake --build native/build -j"${JOBS}"

OUT="src/main/resources/native/linux-x86_64"
mkdir -p "${OUT}"
cp native/build/libexecutorch_djl.so "${OUT}/"
echo "Artifact: ${OUT}/libexecutorch_djl.so"
ls -lh "${OUT}/libexecutorch_djl.so"
