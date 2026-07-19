#!/usr/bin/env bash
# Host-fast assertions on the native-build workflow (no runner needed).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }
# The caller workflow. The build steps themselves live in the reusable job below — asserting them
# against this file is what made this suite red on main.
WF=".github/workflows/native-build.yml"
# The reusable workflow that actually builds the shim (both platforms).
WFJOB=".github/workflows/native-build-job.yml"

grep -q 'pytorch/executorch' "${WFJOB}" && fail "ExecuTorch checkout must be removed"
grep -q 'gh attestation verify' "${WFJOB}" || fail "attestation verify step missing"
grep -q 'measly-java-learning/executorch-runtime-dist' "${WFJOB}" || fail "attestation repo missing"
grep -q 'native/build.sh' "${WFJOB}" || fail "shim build step missing"
grep -q 'native-build-job.yml' "${WF}" || fail "caller must delegate to the reusable job"

# Windows shim job: a sibling job in the reusable workflow (not a separate file), so publish.yml's
# single `uses:` + executorch-libs-* pattern keeps working unchanged.
grep -q 'build-executorch-shim-windows' "${WFJOB}" || fail "windows shim job missing"
grep -q 'runs-on: windows-2022'         "${WFJOB}" || fail "windows job must run on windows-2022"
grep -q 'vswhere'                       "${WFJOB}" || fail "windows job must discover VS via vswhere"
grep -q 'products \*'                   "${WFJOB}" || fail "vswhere must be edition-agnostic (-products *)"
grep -q 'executorch-libs-windows-x86_64' "${WFJOB}" || fail "windows artifact name missing"
# Scope the QA assertion to the windows job block (it is the last job in the file, so from its header to
# EOF). A bare `grep build_qa.sh "${WFJOB}"` is vacuous — the linux job runs build_qa.sh too, so it would
# stay green even if the windows QA step were deleted.
awk '/^  build-executorch-shim-windows:/{f=1} f' "${WFJOB}" | grep -q 'build_qa.sh' \
  || fail "windows QA step missing (build_qa.sh not invoked in the windows job)"
# The aarch64 rows exist in the pin but are out of scope: the matrix entry must stay commented out.
grep -qE '^\s*- platform: linux-aarch64' "${WFJOB}" && fail "linux-aarch64 is out of scope for this PR"

# The Windows provenance gate must attest the row the build actually links (-static, /MT). The /MD row
# is still in the pin file, so a pattern without the suffix keeps matching and keeps PASSING while
# verifying the wrong tarball — a silent supply-chain hole, which is why this is asserted here.
# Scoped to the windows job block: the linux job has its own attestation step, so an unscoped grep
# would stay green even if the windows one regressed.
awk '/^  build-executorch-shim-windows:/{f=1} f' "${WFJOB}" \
  | grep -q 'logging-windows-x86_64-static' \
  || fail "windows provenance gate must attest the -static tarball"

# Python is optional; if present, assert the file is valid YAML.
if command -v python3 >/dev/null 2>&1 && python3 -c 'import yaml' 2>/dev/null; then
  python3 -c "import yaml,sys; yaml.safe_load(open('${WF}')); yaml.safe_load(open('${WFJOB}')); print('yaml ok')" \
    || fail "workflow is not valid YAML"
fi
echo "PASS: ci workflow"
