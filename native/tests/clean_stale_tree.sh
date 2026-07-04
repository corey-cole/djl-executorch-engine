#!/usr/bin/env bash
# Host-fast test for native/clean_stale_tree.sh: wipe only on source-root mismatch (or CLEAN=1),
# preserve a same-root tree so its _deps cache survives.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

tmp="$(mktemp -d)"
bd="${tmp}/build"; srcdir="${tmp}/src"
mkdir -p "${bd}" "${srcdir}"
srcabs="$(cd "${srcdir}" && pwd)"

# 1. Matching source root => tree preserved (cache reuse).
printf 'CMAKE_HOME_DIRECTORY:INTERNAL=%s\n' "${srcabs}" > "${bd}/CMakeCache.txt"
touch "${bd}/keep"
bash native/clean_stale_tree.sh "${bd}" "${srcdir}"
[ -f "${bd}/keep" ] || fail "same-root tree was wiped (cache would be lost)"

# 2. Mismatched source root => wiped.
printf 'CMAKE_HOME_DIRECTORY:INTERNAL=/workspace/native\n' > "${bd}/CMakeCache.txt"
touch "${bd}/keep"
bash native/clean_stale_tree.sh "${bd}" "${srcdir}"
[ -f "${bd}/keep" ] && fail "cross-root stale tree was NOT wiped"

# 3. CLEAN=1 => wiped regardless of matching root.
mkdir -p "${bd}"
printf 'CMAKE_HOME_DIRECTORY:INTERNAL=%s\n' "${srcabs}" > "${bd}/CMakeCache.txt"
touch "${bd}/keep"
CLEAN=1 bash native/clean_stale_tree.sh "${bd}" "${srcdir}"
[ -f "${bd}/keep" ] && fail "CLEAN=1 did not wipe"

# 4. No cache (fresh tree) => left alone.
mkdir -p "${bd}"
touch "${bd}/keep"
bash native/clean_stale_tree.sh "${bd}" "${srcdir}"
[ -f "${bd}/keep" ] || fail "fresh tree with no cache should be left alone"

rm -rf "${tmp}"
echo "PASS: clean_stale_tree"
