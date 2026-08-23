#!/usr/bin/env bash
# The staged notices tree is a distribution obligation, not a build artifact. build.sh already hard-
# fails when LICENSE or THIRD-PARTY-NOTICES/ is missing entirely; this is the floor for the case that
# fails silently instead -- a tree that arrives present but hollow, or one that shrinks on a bump.
#
# Deliberately NOT a filename list. Notice filenames are path-derived upstream and shift with the
# vendoring path, so a filename check would fail for the wrong reason on the next bump. The one
# content check is Eigen: whole-archiving optimized_native_cpu_ops_lib pulls MPL-2.0 code into the
# shipped .so through optimized_kernels -> cpublas -> eigen_blas, which makes that notice an
# obligation on OUR distribution rather than only on the runtime tarball.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

DIR="${NOTICES_DIR:-src/main/resources/native/linux-x86_64/licenses}"
# Absent is fine on a host that has not staged a build. The release path is the opposite: publish.yml
# must never ship a jar whose notices silently failed to arrive, so it sets NOTICES_REQUIRED=1 and
# points NOTICES_DIR at the staged artifact tree (build/native-staging/linux-x86_64/licenses). The
# default keeps local runs and native-build-job.yml (which stages into the source tree) unchanged.
if [ ! -d "${DIR}" ]; then
  if [ "${NOTICES_REQUIRED:-0}" = "1" ]; then
    fail "no notices at ${DIR}, but NOTICES_REQUIRED=1"
  fi
  echo "SKIP: notices not staged"
  exit 0
fi

[ -s "${DIR}/LICENSE" ]             || fail "LICENSE missing or empty"
[ -d "${DIR}/THIRD-PARTY-NOTICES" ] || fail "THIRD-PARTY-NOTICES/ missing"

count="$(find "${DIR}/THIRD-PARTY-NOTICES" -type f | wc -l)"
# A floor, not the exact count: the set legitimately grows when the runtime vendors a new dependency.
# A tree that drops below this has lost notices rather than gained them.
[ "${count}" -ge 30 ] || fail "only ${count} third-party notices; the tree has shrunk"

empty="$(find "${DIR}" -type f -empty | head -5)"
[ -z "${empty}" ] || fail "empty notice files: ${empty}"

grep -ril eigen "${DIR}" >/dev/null \
  || fail "no Eigen notice: the shim whole-archives optimized_native_cpu_ops_lib, which links MPL-2.0 eigen_blas"

echo "PASS: notices staged (${count} third-party notices)"
