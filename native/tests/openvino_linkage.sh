#!/usr/bin/env bash
# The OpenVINO delegate must be linked into the shim wherever the runtime tarball provides it, and
# must be ABSENT wherever it does not -- both are correctness, not just presence. Asserted against
# the built artifact rather than the CMake source, so a link line that silently GC's the
# registration fails here too.
set -eu
# NOTE: deliberately NO pipefail. `grep -q` exits at the first match and closes the pipe, so nm
# (which writes ~10k lines here) dies of SIGPIPE; under pipefail that turns a MATCH into a failing
# pipeline and the PASS case would report FAIL. The `|| fail` below already handles the real
# negative, so pipefail adds nothing.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

SO="src/main/resources/native/linux-x86_64/libexecutorch_djl.so"
[ -f "${SO}" ] || { echo "SKIP: no staged linux-x86_64 shim"; exit 0; }

RUNTIME_LIB="native/build/_deps/et_runtime-src/lib/libopenvino_backend.a"
if [ -f "${RUNTIME_LIB}" ]; then
  nm -C --defined-only "${SO}" 2>/dev/null | grep -q 'OpenvinoBackend' \
    || fail "runtime ships libopenvino_backend.a but the shim carries no OpenvinoBackend symbols"
  echo "PASS: OpenVINO delegate linked"
else
  echo "PASS: runtime ships no OpenVINO delegate on this platform; nothing to link"
fi
