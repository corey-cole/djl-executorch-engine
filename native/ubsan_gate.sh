#!/usr/bin/env bash
# UBSan gate for the JNI shim, driven by the JVM suite.
#
# This is the ONLY configuration in which native/jni/executorch_djl_jni.cpp is instrumented:
# native/build_qa.sh covers et_runtime, the Catch2 suite and the harnesses, but
# native/CMakeLists.txt skips the shim under ET_BUILD_QA so QA stays JVM-free.
#
# NOTE: a UB hit here presents as a JVM HARD CRASH mid-test, not a Java exception or an assertion
# failure. That is the gate working. Look for the "runtime error:" line and its stack trace above
# the JVM's own crash output.
#
# TWO PHASES, because they need different environments. The pinned image has the right toolchain and
# the wrong JDK (Corretto 8, for the oldest supported jni.h); Gradle 9.6.1 with this project's JDK 17
# toolchain cannot run there. So: build in the container, test on the host. ET_UBSAN_MODE selects a
# phase and defaults to `auto` -- build-only inside the image, both phases outside it.
#
# The instrumented .so is NEVER staged into src/main/resources: it is reached through
# EXECUTORCH_LIBRARY_PATH, which LibUtils honours ahead of the classpath copy and which
# build.gradle.kts already declares as a Test task input. The plain tree is left untouched.
#
# Linux only: MSVC has no UndefinedBehaviorSanitizer.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BUILD_DIR="${BUILD_DIR:-native/ubsan}"
JOBS="${JOBS:-$(nproc)}"

# The build phase runs as root under native/local_build_wrapper.sh, so without this the tree comes
# back root-owned and the next run's `rm -rf` dies with a bare "Permission denied". No-op on a host,
# where HOST_UID is unset.
# shellcheck source=native/container_env.sh
. "${REPO_ROOT}/native/container_env.sh"
et_chown_outputs_on_exit "${BUILD_DIR}"

# Resolve the shipped runtime variant through the shared rule (build.sh and build_qa.sh source the
# same file). This gate must instrument the configuration that actually ships: without it the CMake
# cache default (logging) would compile the etDump() body out -- ET_HAVE_DEVTOOLS undefined -- and
# the get_etdump_data()/free() path this gate exists to cover would never run under UBSan.
. "${REPO_ROOT}/native/variant_select.sh"

MODE="${ET_UBSAN_MODE:-auto}"
if [ "${MODE}" = "auto" ]; then
  if [ -n "${MEASLY_DJL_PINNED_IMAGE:-}" ]; then MODE=build; else MODE=all; fi
fi
case "${MODE}" in
  build|test|all) ;;
  *) echo "ET_UBSAN_MODE must be build, test, all or auto (got '${MODE}')" >&2; exit 1 ;;
esac

# tasks.test excludes eight tags; oomTest and leakTest are where the marshalling loop meets
# allocation failure and memory pressure, which is the exposure this gate is for.
TEST_TASKS="${TEST_TASKS:-test leakTest oomTest}"

# --no-daemon is not a preference. A pre-existing daemon lives in whatever cgroup it was first
# started in, so ./gradlew would hand the work -- including every forked test JVM -- to a process
# outside any resource scope wrapping this script. oomTest exhausts a heap on purpose.
GRADLE_FLAGS="${GRADLE_FLAGS:---no-daemon}"

# -fno-sanitize-recover (native/CMakeLists.txt) makes UBSan abort; these make the abort legible.
export UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1

if [ "${MODE}" = "build" ] || [ "${MODE}" = "all" ]; then
  echo "--- Building the UBSan-instrumented shim (variant ${ET_RUNTIME_VARIANT}) ---"
  rm -rf "${BUILD_DIR}"
  # No ET_BUILD_QA: that is what makes CMakeLists build the shim rather than skip it. JAVA_HOME is
  # needed here for jni.h only -- we never link libjvm. ET_RUNTIME_VARIANT is resolved above
  # through variant_select.sh so the gate instruments the configuration that ships (devtools on
  # linux-x86_64), not the CMake cache default.
  cmake -S native -B "${BUILD_DIR}" -G "Unix Makefiles" \
    -DET_RUNTIME_VARIANT="${ET_RUNTIME_VARIANT}" \
    -DET_UBSAN=ON -DCMAKE_BUILD_TYPE=Debug
  cmake --build "${BUILD_DIR}" --target executorch_djl -j"${JOBS}"

  # A dynamic libubsan dependency means -static-libubsan did not apply and System.load would fail.
  # Assert here so the failure names its own cause: the build may happen in a container and the load
  # on a host hours later, in a different job.
  if ! _ldd_out="$(ldd "${BUILD_DIR}/libexecutorch_djl.so" 2>&1)"; then
    echo "FAIL: ldd cannot read ${BUILD_DIR}/libexecutorch_djl.so;" >&2
    echo "      the build did not produce the instrumented shim" >&2
    exit 1
  fi
  if printf '%s\n' "${_ldd_out}" | grep -qi ubsan; then
    echo "FAIL: ${BUILD_DIR}/libexecutorch_djl.so has a dynamic libubsan dependency;" >&2
    echo "      -static-libubsan did not apply (native/CMakeLists.txt)" >&2
    exit 1
  fi
  echo "--- UBSan runtime is statically linked ---"
fi

if [ "${MODE}" = "build" ]; then
  echo "--- UBSan shim built at ${BUILD_DIR}/libexecutorch_djl.so; JVM phase skipped ---"
  echo "--- Run the JVM phase where a JDK 17 lives: ET_UBSAN_MODE=test ./native/ubsan_gate.sh ---"
  exit 0
fi

# Refuse the JVM phase rather than letting Gradle fail obscurely.
if [ -n "${MEASLY_DJL_PINNED_IMAGE:-}" ]; then
  echo "REFUSING the JVM phase inside the pinned image: JAVA_HOME is Corretto 8, and Gradle 9.6.1" >&2
  echo "with a JDK 17 toolchain cannot run there. Build here, test on the host:" >&2
  echo "  ./native/local_build_wrapper.sh native/ubsan_gate.sh   # build phase, in-container" >&2
  echo "  ET_UBSAN_MODE=test ./native/ubsan_gate.sh              # JVM phase, on the host" >&2
  exit 1
fi

# Refuse the JVM phase with a legible message rather than letting Gradle fail obscurely. The
# probe avoids a java|head pipeline (head exits after line 1; java can then SIGPIPE, and
# pipefail would abort the assignment silently).
_java_bin="${JAVA_HOME:-/usr}/bin/java"
if [ ! -x "${_java_bin}" ]; then
  echo "no java at ${_java_bin}; the JVM phase needs JDK 17+. Set JAVA_HOME and rerun." >&2
  exit 1
fi
_java_version="$("${_java_bin}" -version 2>&1 || true)"
_java_major="$(sed -nE -e 's/.*"1\.([0-9]+).*/\1/p' -e 's/.*"([1-9][0-9]*).*/\1/p' <<<"${_java_version}")"
case "${_java_major}" in
  ''|*[!0-9]*)
    echo "could not parse the java version (\"${_java_version}\"); Gradle 9.6.1 and this project need JDK 17+." >&2
    exit 1
    ;;
esac
if [ "${_java_major}" -lt 17 ]; then
  echo "JAVA_HOME points at Java ${_java_major}; Gradle 9.6.1 and this project need 17+." >&2
  exit 1
fi

if [ ! -f "${BUILD_DIR}/libexecutorch_djl.so" ]; then
  echo "no instrumented shim at ${BUILD_DIR}/libexecutorch_djl.so -- run the build phase first" >&2
  exit 1
fi

echo "--- JVM suite against the instrumented shim (${TEST_TASKS}) ---"
# --rerun-tasks: a cached UP-TO-DATE would report a pass for a run that never loaded this library.
EXECUTORCH_LIBRARY_PATH="${REPO_ROOT}/${BUILD_DIR}/libexecutorch_djl.so" \
  ./gradlew ${GRADLE_FLAGS} ${TEST_TASKS} --rerun-tasks

echo "--- UBSan gate PASS ---"
