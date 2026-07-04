#!/usr/bin/env bash
# Wipe a CMake build tree ONLY if its cached source root differs from the requested one. CMake cannot
# reconfigure across a changed absolute source path — the exact failure when a tree configured in the
# container (/workspace/...) is reused on the host (or vice versa). Same-root re-runs are left intact
# so the tree's _deps caches (Catch2's compiled libs, the FetchContent'd runtime tarball) are reused
# instead of rebuilt/re-downloaded. Set CLEAN=1 to force a wipe regardless.
#
# Usage: clean_stale_tree.sh <build_dir> <source_dir>
set -euo pipefail

build="$1"
src="$(cd "$2" && pwd)"                 # absolute, matching what CMake stores in the cache
cache="${build}/CMakeCache.txt"

if [ "${CLEAN:-0}" = "1" ]; then
  rm -rf "${build}"
  exit 0
fi

# CMAKE_HOME_DIRECTORY is the source root CMake baked into the cache. If it doesn't match ours, the
# tree is unusable here and CMake would abort — wipe it. No cache => fresh tree => nothing to do.
if [ -f "${cache}" ] && ! grep -qxF "CMAKE_HOME_DIRECTORY:INTERNAL=${src}" "${cache}"; then
  echo "clean_stale_tree: ${build} was configured for a different source root; wiping"
  rm -rf "${build}"
fi
