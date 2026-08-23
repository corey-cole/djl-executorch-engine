#!/usr/bin/env bash
# The Windows arm of the OpenVINO precision probe must load its library with BOTH alternate-search
# flags. This is a source-level policy ban, not a style check: the failing shapes it forbids are
# measured, and one of them fails in a way that looks like success.
#
#   LoadLibrary{A,W}(abs)                     cannot resolve the bundle's own siblings -- Windows
#                                             has no $ORIGIN -- but SUCCEEDS whenever some other
#                                             OpenVINO sits on PATH or in System32, so a green run
#                                             on a contaminated machine proves nothing.
#   LoadLibraryExW(.., DLL_LOAD_DIR) alone    drops System32 with the wheel's CRT and fails.
#
# The behavioural gate is OpenVinoColdProbeTest, which only runs where a bundle is staged. This runs
# everywhere, including the Linux rows, so the reversion is caught by the cheapest job rather than
# only by the Windows leg.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

SRC="native/core/et_runtime.cpp"
[ -f "${SRC}" ] || fail "missing ${SRC}"

# Code only: the comment above ovLoadLibrary names the banned forms deliberately, to say what they
# do and why they are banned. Banning the words rather than the calls would forbid documenting them.
CODE="$(sed 's://.*::' "${SRC}")"

# `grep && fail` is safe under set -e: a failing non-final member of an AND-list does not exit.
printf '%s' "${CODE}" | grep -qE '\bLoadLibrary[AW]?\(' \
  && fail "${SRC} uses a plain LoadLibrary; it cannot resolve the bundle graph (see the comment on ovLoadLibrary)"

printf '%s' "${CODE}" | grep -q 'LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR' \
  || fail "${SRC} must search the DLL's own directory: Windows has no \$ORIGIN"
printf '%s' "${CODE}" | grep -q 'LOAD_LIBRARY_SEARCH_DEFAULT_DIRS' \
  || fail "${SRC} must keep the default dirs: the alternate search order drops System32, where the OpenVINO wheel's CRT lives"

# The A-suffixed entry points convert through the ANSI codepage, and the bundle path runs through
# %LOCALAPPDATA% -- i.e. the Windows profile name, which need not be representable there.
printf '%s' "${CODE}" | grep -qE '\bLoadLibraryExA\(|\bGetModuleHandleA\(' \
  && fail "${SRC} uses an ANSI loader entry point; the bundle path carries the Windows profile name"

echo "PASS: openvino windows loader flags"
