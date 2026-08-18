#!/usr/bin/env bash
# The staged bundle must be usable as-is: one flat directory, every library present, no symlink
# required, and a MANIFEST whose version agrees with the pin. A bundle that is merely "downloaded"
# but split across directories or missing a library fails at model load with an import error that
# names none of these causes.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

DIR="build/native-staging/linux-x86_64/openvino"
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

# The ABI suffix is DERIVED from BUILDINFO, never hardcoded: it tracks the OpenVINO version
# (2025.4.1 -> 2541), so a hardcoded literal would make this test a thing to edit on every bump,
# and a stale one would fail with "missing library" rather than "you bumped OpenVINO".
abi="$(grep -oP '^ov_abi=\K.*' "${DIR}/BUILDINFO")"
[ -n "${abi}" ] || fail "BUILDINFO carries no ov_abi"

# The SET of libraries is the part worth reviewing on a version bump -- an OpenVINO release can add
# or drop a transitive dependency, and a missing one fails at model load with an error naming none
# of this. Keep in sync with OpenVinoRuntime.LIBS; docs/openvino-version-bump.md is the checklist.
for f in "libopenvino_c.so.${abi}" "libopenvino.so.${abi}" libopenvino_intel_cpu_plugin.so \
         "libopenvino_ir_frontend.so.${abi}" libtbb.so.12 libtbbbind_2_5.so.3 libhwloc.so.15; do
  [ -f "${DIR}/lib/${f}" ] || fail "missing library: ${f} (see docs/openvino-version-bump.md)"
done

# Nothing may be shipped that no one enumerated: an unlisted library means the bundle grew and
# OpenVinoRuntime.LIBS will not extract it, which fails at dlopen rather than here.
count="$(find "${DIR}/lib" -maxdepth 1 -type f | wc -l)"
[ "${count}" -eq 7 ] \
  || fail "expected 7 libraries, found ${count} -- the bundle changed; see docs/openvino-version-bump.md"

# Flat, not nested: RPATH=$ORIGIN is what resolves the graph.
find "${DIR}/lib" -mindepth 1 -type d | grep -q . && fail "lib/ must be flat, found a subdirectory"

pin_ver="$(grep -oP 'set\(ET_RUNTIME_OPENVINO_VERSION "\K[^"]+' native/cmake/EtRuntimePin.cmake)"
man_ver="$(grep -oP '^openvino_version=\K.*' "${DIR}/MANIFEST")"
[ "${pin_ver}" = "${man_ver}" ] || fail "MANIFEST openvino_version=${man_ver} != pin ${pin_ver}"

echo "PASS: openvino bundle staging"
