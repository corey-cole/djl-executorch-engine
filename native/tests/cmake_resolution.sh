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

probe_fails() {  # extra -D args; asserts configure FAILS and echoes the captured error text
  local b err; b="$(mktemp -d)"; err="${b}/err"
  if cmake -S native -B "${b}" -DET_PRINT_RESOLUTION=ON "$@" >"${err}" 2>&1; then
    cat "${err}"; rm -rf "${b}"; fail "cmake configure unexpectedly succeeded ($*)"
  fi
  cat "${err}"; rm -rf "${b}"
}

out="$(probe)"                                            # default => fetch logging
grep -q 'resolution=fetch'                                     <<<"${out}" || fail "default not fetch"
grep -q 'variant=logging'                                     <<<"${out}" || fail "default variant not logging"
grep -q 'stem=executorch-runtime-1.3.1-logging-linux-x86_64'  <<<"${out}" || fail "default stem wrong"
grep -q 'executorch-runtime-1.3.1-logging-linux-x86_64.tar.gz' <<<"${out}" || fail "default url wrong"
# The pin must point at the v1.3.1-8 release: the ET version (1.3.1) is unchanged across pkgrevs, so
# the tarball stem alone cannot distinguish them. Assert the release-tag path segment instead.
grep -q '/download/v1.3.1-8/'                                 <<<"${out}" || fail "pin is not at release v1.3.1-8"

out="$(probe -DET_RUNTIME_VARIANT=bare)"
grep -q 'stem=executorch-runtime-1.3.1-bare-linux-x86_64'      <<<"${out}" || fail "bare stem wrong"
grep -q 'executorch-runtime-1.3.1-bare-linux-x86_64.tar.gz'    <<<"${out}" || fail "bare url wrong"

out="$(probe -DET_RUNTIME_VARIANT=devtools)"
grep -q 'stem=executorch-runtime-1.3.1-devtools-linux-x86_64'  <<<"${out}" || fail "devtools stem wrong"

# Windows resolution, asserted from a Linux host: ET_PLATFORM is a cache var, so the ET_PRINT_RESOLUTION
# seam can resolve a foreign platform's pin row without that platform being present.
#
# Windows resolves the -static (/MT) row by default: the shipped DLL must not need a VC++ redist.
# ET_PLATFORM is the platform IDENTITY and stays 'windows-x86_64'; ET_RUNTIME_ROW is the pin key.
out="$(probe -DET_PLATFORM=windows-x86_64)"
grep -q 'platform=windows-x86_64'                                       <<<"${out}" || fail "windows platform not echoed"
grep -q 'row=windows-x86_64-static'                                     <<<"${out}" || fail "windows must default to the -static row"
grep -q 'stem=executorch-runtime-1.3.1-logging-windows-x86_64-static'   <<<"${out}" || fail "windows stem wrong"
grep -q 'executorch-runtime-1.3.1-logging-windows-x86_64-static.tar.gz' <<<"${out}" || fail "windows url wrong"

# The /MD row must remain selectable, not merely present in the pin file. Without this the row would be
# a hardcode with extra steps, and a typo'd row name would be indistinguishable from a deleted one.
out="$(probe -DET_PLATFORM=windows-x86_64 -DET_RUNTIME_ROW=windows-x86_64)"
grep -q 'row=windows-x86_64 '                                    <<<"${out}" || fail "dynamic row not selectable"
grep -q 'executorch-runtime-1.3.1-logging-windows-x86_64.tar.gz' <<<"${out}" || fail "dynamic row url wrong"

# Linux identity and row coincide; assert the split did not introduce a Linux-side divergence.
out="$(probe)"
grep -q 'row=linux-x86_64' <<<"${out}" || fail "linux row must equal linux platform"

# The default on this (Linux) host must be unaffected by ET_PLATFORM existing.
out="$(probe)"
grep -q 'platform=linux-x86_64'                                  <<<"${out}" || fail "default platform not linux-x86_64"

# windows-x86_64 has no bare/devtools pin row upstream. Fail with the real reason, not "no pin row".
out="$(probe_fails -DET_PLATFORM=windows-x86_64 -DET_RUNTIME_VARIANT=bare)"
grep -q "ships the 'logging' variant only" <<<"${out}" || fail "windows+bare should fail with a named error"

# The escape hatch bypasses the pin entirely, so the variant constraint must NOT apply there.
out="$(probe -DET_PLATFORM=windows-x86_64 -DET_RUNTIME_VARIANT=bare -DET_INSTALL=/tmp/my-et)"
grep -q 'resolution=escape-hatch' <<<"${out}" || fail "escape hatch must bypass the windows variant guard"

out="$(probe -DET_INSTALL=/tmp/my-et)"                    # escape hatch
grep -q 'resolution=escape-hatch'                             <<<"${out}" || fail "escape hatch not detected"
grep -q 'et_install=/tmp/my-et'                              <<<"${out}" || fail "escape hatch path wrong"

echo "PASS: cmake resolution"
