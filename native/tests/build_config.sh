#!/usr/bin/env bash
# Host-fast test of build.sh's config seam via PRINT_BUILD_CONFIG, plus a guard that Stage-A is gone.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

out="$(PRINT_BUILD_CONFIG=1 bash native/build.sh)"
grep -q 'ET_RUNTIME_VARIANT=logging' <<<"${out}" || fail "default variant not logging"
grep -q 'STAGE_SO=1'                  <<<"${out}" || fail "default STAGE_SO not 1"
grep -q 'NATIVE_BUILD_DIR=native/build\b' <<<"${out}" || fail "default NATIVE_BUILD_DIR changed"

out="$(ET_RUNTIME_VARIANT=bare STAGE_SO=0 NATIVE_BUILD_DIR=/tmp/nb ET_INSTALL=/tmp/xi \
       PRINT_BUILD_CONFIG=1 bash native/build.sh)"
grep -q 'ET_RUNTIME_VARIANT=bare'  <<<"${out}" || fail "variant override ignored"
grep -q 'STAGE_SO=0'               <<<"${out}" || fail "STAGE_SO override ignored"
grep -q 'NATIVE_BUILD_DIR=/tmp/nb\b' <<<"${out}" || fail "NATIVE_BUILD_DIR override ignored"
grep -q 'ET_INSTALL=/tmp/xi\b'     <<<"${out}" || fail "ET_INSTALL passthrough missing"

# Stage-A must be fully gone from build.sh.
grep -qE 'SKIP_ET_BUILD|EXECUTORCH_ENABLE_LOGGING|torch==2\.12|avx512vnni' native/build.sh \
  && fail "Stage-A remnants still present in build.sh"

# Requirement assertions must fail BY NAME, not as a confusing failure ten steps later. These drive
# the JAVA_HOME and ninja assertions; they are behavioural (run the script, read what it says) and
# host-independent, since each one removes a requirement rather than depending on what this host has.
rc=0
out="$(JAVA_HOME=/nonexistent bash native/build.sh 2>&1)" || rc=$?
test "${rc}" -ne 0 || fail "build.sh must fail when the JDK headers are absent"
grep -q 'jni_md.h\|JAVA_HOME' <<<"${out}" || fail "JDK failure must name what is missing"
grep -q 'local_build_wrapper.sh' <<<"${out}" || fail "JDK failure must point at the wrapper"

# Ninja must still be absent AFTER the JDK assertion passes, so supply a real JDK root derived
# host-independently from `java`; if the host has no java, the ninja path cannot be exercised
# (build.sh asserts JDK first) and the subtest is skipped.
JDK_ROOT="$(readlink -f "$(command -v java 2>/dev/null)" 2>/dev/null)" \
  && JDK_ROOT="$(dirname "$(dirname "${JDK_ROOT}")")"
if [ -n "${JDK_ROOT:-}" ] && [ -f "${JDK_ROOT}/include/linux/jni_md.h" ]; then
  rc=0
  out="$(JAVA_HOME="${JDK_ROOT}" PATH=/nonexistent-bin /bin/bash native/build.sh 2>&1)" || rc=$?
  test "${rc}" -ne 0 || fail "build.sh must fail when ninja is absent"
  grep -q 'ninja not on PATH' <<<"${out}" || fail "toolchain failure must name ninja"
fi

echo "PASS: build.sh config"
