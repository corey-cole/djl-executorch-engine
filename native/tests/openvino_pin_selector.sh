#!/usr/bin/env bash
# The OpenVINO staging decision is TWO independent questions: "does this ENGINE support a bundle on
# this platform" (a statement about what the Java layer can load) and "does the PIN publish one for
# it". Conflating them is what let the deprecated singular ET_RUNTIME_OPENVINO_PLATFORM silently
# ignore every non-Linux bundle. Driven through build.sh's PRINT_OPENVINO_RESOLUTION seam, so this
# asserts foreign platforms with no foreign hardware and no build.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

decide() {  # <platform> [supported-set]; echoes the OV_RESOLUTION line
  local platform="$1"
  if [ "$#" -ge 2 ]; then
    PRINT_OPENVINO_RESOLUTION=1 ET_STAGE_PLATFORM="${platform}" ET_OPENVINO_SUPPORTED_PLATFORMS="$2" \
      bash native/build.sh 2>/dev/null | grep -m1 'OV_RESOLUTION' || fail "no OV_RESOLUTION for ${platform}"
  else
    PRINT_OPENVINO_RESOLUTION=1 ET_STAGE_PLATFORM="${platform}" \
      bash native/build.sh 2>/dev/null | grep -m1 'OV_RESOLUTION' || fail "no OV_RESOLUTION for ${platform}"
  fi
}

# The shipped default: linux-x86_64 is the one platform whose bundle OpenVinoRuntime can extract.
out="$(decide linux-x86_64)"
grep -q 'decision=stage'   <<<"${out}" || fail "linux-x86_64 must stage: ${out}"
grep -q 'url=https://'     <<<"${out}" || fail "linux-x86_64 url must be literal: ${out}"

# The version must come from the pin, not a literal here -- OpenVINO versions independently of
# ExecuTorch, so a hardcode would make this a file to edit on an unrelated bump.
pin_ver="$(grep -oP 'set\(ET_RUNTIME_OPENVINO_VERSION "\K[^"]+' native/cmake/EtRuntimePin.cmake)"
grep -q "version=${pin_ver}" <<<"${out}" || fail "version not taken from the pin: ${out}"

# Windows is supported: the runtime tarballs ship openvino_backend.lib, the pin publishes a
# win_amd64 bundle, and OpenVinoRuntime extracts whatever the bundle's MANIFEST declares.
out="$(decide windows-x86_64)"
grep -q 'decision=stage' <<<"${out}" || fail "windows-x86_64 must stage: ${out}"
# The literal-URL guard. The pin expresses "both Windows CRT rows share one bundle" as an ALIAS row
# whose VALUE is a CMake variable reference. A row-keyed lookup would hand curl the unexpanded text
# ${ET_RUNTIME_OPENVINO_URL_windows-x86_64}; keying on the platform never reads the alias.
grep -q 'url=https://' <<<"${out}" || fail "windows url must be literal, not a cmake reference: ${out}"

# The publication question must be answerable independently too: linux-aarch64 has no bundle
# upstream and never will, so a supported-but-unpublished platform must skip rather than fail.
out="$(decide linux-aarch64 'linux-x86_64 linux-aarch64')"
grep -q 'decision=unpublished' <<<"${out}" || fail "linux-aarch64 has no bundle row: ${out}"

# And with the shipped set it is simply unsupported -- support is checked first, so the answer is
# about our decision rather than about upstream's publication schedule.
out="$(decide linux-aarch64)"
grep -q 'decision=unsupported' <<<"${out}" || fail "linux-aarch64 must be unsupported by default: ${out}"

echo "PASS: openvino pin selector"
