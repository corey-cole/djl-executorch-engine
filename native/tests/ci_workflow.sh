#!/usr/bin/env bash
# Host-fast assertions on the native-build workflow (no runner needed).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }
WF=".github/workflows/native-build.yml"

grep -q 'pytorch/executorch' "${WF}" && fail "ExecuTorch checkout must be removed"
grep -q 'gh attestation verify' "${WF}" || fail "attestation verify step missing"
grep -q 'measly-java-learning/executorch-runtime-dist' "${WF}" || fail "attestation repo missing"
grep -q 'native/build.sh' "${WF}" || fail "shim build step missing"
# Python is optional; if present, assert the file is valid YAML.
if command -v python3 >/dev/null 2>&1 && python3 -c 'import yaml' 2>/dev/null; then
  python3 -c "import yaml,sys; yaml.safe_load(open('${WF}')); print('yaml ok')" || fail "workflow is not valid YAML"
fi
echo "PASS: ci workflow"
