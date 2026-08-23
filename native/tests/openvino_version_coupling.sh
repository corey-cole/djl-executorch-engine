#!/usr/bin/env bash
# The OpenVINO version lives in three places. They must agree, or a rebuild can vendor a runtime
# that cannot import the fixture's precompiled blob -- which surfaces at model load rather than at
# build time. OpenVINO versions independently of ExecuTorch, so an OV re-roll can invalidate the
# committed fixture with no ET bump; this is what makes that a build failure.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

pin="$(grep -oP 'set\(ET_RUNTIME_OPENVINO_VERSION "\K[^"]+' native/cmake/EtRuntimePin.cmake)"
[ -n "${pin}" ] || fail "no ET_RUNTIME_OPENVINO_VERSION in the pin"

fixture="$(grep -oP '^openvino_version=\K.*' src/test/resources/models/openvino/MANIFEST)"
[ "${pin}" = "${fixture}" ] \
  || fail "fixture MANIFEST openvino_version=${fixture} != pin ${pin} (see docs/openvino-version-bump.md)"

staged="build/native-staging/${1:-linux-x86_64}/openvino/MANIFEST"
if [ -f "${staged}" ]; then
  bundle="$(grep -oP '^openvino_version=\K.*' "${staged}")"
  [ "${pin}" = "${bundle}" ] || fail "bundle MANIFEST openvino_version=${bundle} != pin ${pin}"
fi

echo "PASS: openvino version coupling (${pin})"
