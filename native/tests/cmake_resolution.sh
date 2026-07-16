#!/usr/bin/env bash
# Host-fast test of native/CMakeLists.txt runtime resolution via the ET_PRINT_RESOLUTION early-exit.
# No network / no find_package: the diagnostic returns BEFORE FetchContent_MakeAvailable.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

probe() {  # extra -D args; echoes the ET_RESOLUTION status line
  local b err; b="$(mktemp -d)"; err="${b}/err"
  cmake -S native -B "${b}" -DET_PRINT_RESOLUTION=ON "$@" >"${err}" 2>&1 \
    || { cat "${err}"; rm -rf "${b}"; fail "cmake configure failed ($*)"; }
  grep -h 'ET_RESOLUTION' "${err}" || { rm -rf "${b}"; fail "no ET_RESOLUTION line ($*)"; }
  rm -rf "${b}"
}

out="$(probe)"                                            # default => fetch logging
grep -q 'resolution=fetch'                                     <<<"${out}" || fail "default not fetch"
grep -q 'variant=logging'                                     <<<"${out}" || fail "default variant not logging"
grep -q 'stem=executorch-runtime-1.3.1-logging-linux-x86_64'  <<<"${out}" || fail "default stem wrong"
grep -q 'executorch-runtime-1.3.1-logging-linux-x86_64.tar.gz' <<<"${out}" || fail "default url wrong"
# The pin must point at the v1.3.1-6 release: the ET version (1.3.1) is unchanged from v1.3.1-2, so
# the tarball stem alone cannot distinguish the two. Assert the release-tag path segment instead.
grep -q '/download/v1.3.1-6/'                                 <<<"${out}" || fail "pin is not at release v1.3.1-6"

out="$(probe -DET_RUNTIME_VARIANT=bare)"
grep -q 'stem=executorch-runtime-1.3.1-bare-linux-x86_64'      <<<"${out}" || fail "bare stem wrong"
grep -q 'executorch-runtime-1.3.1-bare-linux-x86_64.tar.gz'    <<<"${out}" || fail "bare url wrong"

out="$(probe -DET_RUNTIME_VARIANT=devtools)"
grep -q 'stem=executorch-runtime-1.3.1-devtools-linux-x86_64'  <<<"${out}" || fail "devtools stem wrong"

out="$(probe -DET_INSTALL=/tmp/my-et)"                    # escape hatch
grep -q 'resolution=escape-hatch'                             <<<"${out}" || fail "escape hatch not detected"
grep -q 'et_install=/tmp/my-et'                              <<<"${out}" || fail "escape hatch path wrong"

echo "PASS: cmake resolution"
