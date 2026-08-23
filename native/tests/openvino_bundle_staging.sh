#!/usr/bin/env bash
# The staged bundle must be usable as-is: one flat directory, every library present, no symlink
# required, and a MANIFEST whose version agrees with the pin. A bundle that is merely "downloaded"
# but split across directories or missing a library fails at model load with an import error that
# names none of these causes.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

# Platform-parameterized: CI runs this once per row that stages a bundle. Default keeps every
# existing caller working unchanged.
PLATFORM="${1:-linux-x86_64}"
DIR="build/native-staging/${PLATFORM}/openvino"
# Absent is normally fine -- most builds never stage a bundle, and skipping keeps this runnable
# everywhere. On the release path it is the opposite: a publish whose bundle silently failed to
# arrive is exactly the failure this guards, so OPENVINO_BUNDLE_REQUIRED=1 turns absence into a
# failure. publish.yml sets it.
if [ ! -d "${DIR}" ]; then
  if [ "${OPENVINO_BUNDLE_REQUIRED:-0}" = "1" ]; then
    fail "no bundle at ${DIR}, but OPENVINO_BUNDLE_REQUIRED=1"
  fi
  echo "SKIP: bundle not staged"
  exit 0
fi

# Presence first: everything below reads these, and a bundle that arrived truncated should say
# which file is missing rather than fail inside a grep.
[ -f "${DIR}/BUILDINFO" ] || fail "missing BUILDINFO"
[ -f "${DIR}/MANIFEST" ]  || fail "missing MANIFEST"
[ -d "${DIR}/licenses" ]  || fail "missing licenses/"
[ -d "${DIR}/lib" ]       || fail "missing lib/"

# The expected member SET is the part worth reviewing on a version bump -- an OpenVINO release can
# add or drop a transitive dependency, and a missing one fails at model load with an error naming
# none of this. Kept independent of MANIFEST on purpose: a manifest generated from a truncated
# bundle would describe that truncation accurately. See docs/openvino-version-bump.md.
case "${PLATFORM}" in
  linux-x86_64)
    # The ABI suffix is DERIVED from BUILDINFO, never hardcoded: it tracks the OpenVINO version
    # (2025.4.1 -> 2541), so a literal would be a thing to edit on every bump.
    abi="$(grep -oP '^ov_abi=\K.*' "${DIR}/BUILDINFO")"
    [ -n "${abi}" ] || fail "BUILDINFO carries no ov_abi"
    expected="libopenvino_c.so.${abi} libopenvino.so.${abi} libopenvino_intel_cpu_plugin.so"
    expected="${expected} libopenvino_ir_frontend.so.${abi} libtbb.so.12 libtbbbind_2_5.so.3"
    expected="${expected} libhwloc.so.15"
    ;;
  windows-x86_64)
    # No ov_abi key at all here, and its ABSENCE is asserted rather than tolerated: the DLLs are
    # unversioned, so a bundle that grew one would mean the upstream layout changed under us.
    # `grep && fail` is safe under set -e -- a failing non-final member of an AND-list does not
    # exit, which is the same idiom docs_present.sh uses for its policy bans.
    grep -q '^ov_abi=' "${DIR}/BUILDINFO" && fail "windows BUILDINFO must carry no ov_abi"
    # Six, not seven: hwloc is folded into tbbbind_2_5.dll on Windows.
    expected="openvino_c.dll openvino.dll openvino_intel_cpu_plugin.dll openvino_ir_frontend.dll"
    expected="${expected} tbb12.dll tbbbind_2_5.dll"
    ;;
  *) fail "no expected library set for platform '${PLATFORM}'" ;;
esac

for f in ${expected}; do
  [ -f "${DIR}/lib/${f}" ] || fail "missing library: ${f} (see docs/openvino-version-bump.md)"
done

# Nothing may be shipped that no one enumerated: an unlisted library means the bundle grew and the
# expectations above have not caught up.
want="$(printf '%s\n' ${expected} | wc -l)"
count="$(ls -1 "${DIR}/lib" | wc -l)"
[ "${count}" -eq "${want}" ] \
  || fail "expected ${want} libraries, found ${count} -- the bundle changed; see docs/openvino-version-bump.md"

# Flat, not nested: RPATH=$ORIGIN is what resolves the graph.
find "${DIR}/lib" -mindepth 1 -type d | grep -q . && fail "lib/ must be flat, found a subdirectory"

pin_ver="$(grep -oP 'set\(ET_RUNTIME_OPENVINO_VERSION "\K[^"]+' native/cmake/EtRuntimePin.cmake)"
man_ver="$(grep -oP '^openvino_version=\K.*' "${DIR}/MANIFEST")"
[ "${pin_ver}" = "${man_ver}" ] || fail "MANIFEST openvino_version=${man_ver} != pin ${pin_ver}"

# The bundle declares its own contents so the Java extractor copies names rather than reconstructing
# them -- which is what lets one code path serve an ABI-versioned Linux bundle and an unversioned
# Windows one. Asserted against the actual lib/ directory, not just for presence: a MANIFEST that
# disagrees with the tree would send the extractor after a file that is not there, failing at
# dlopen rather than here.
man_libs="$(grep -oP '^libs=\K.*' "${DIR}/MANIFEST" || true)"
[ -n "${man_libs}" ] || fail "MANIFEST carries no libs"

actual_libs="$(ls -1 "${DIR}/lib" | LC_ALL=C sort | tr '\n' ' ')"
[ "${man_libs} " = "${actual_libs}" ] \
  || fail "MANIFEST libs disagree with lib/: manifest='${man_libs}' actual='${actual_libs%% }'"

man_clib="$(grep -oP '^c_library=\K.*' "${DIR}/MANIFEST" || true)"
[ -n "${man_clib}" ] || fail "MANIFEST carries no c_library"
[ -f "${DIR}/lib/${man_clib}" ] || fail "c_library names a file that is not in lib/: ${man_clib}"
# It must be the C API library specifically -- pointing OPENVINO_LIB_PATH at any other library in
# the bundle loads something that resolves no ov_* symbols.
case "${man_clib}" in
  libopenvino_c.so.*|openvino_c.dll) ;;
  *) fail "c_library is not the OpenVINO C API library: ${man_clib}" ;;
esac

echo "PASS: openvino bundle staging"
