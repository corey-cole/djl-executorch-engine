#!/usr/bin/env bash
# Behavioural tests for native/variant_select.sh, the single definition of "which runtime variant
# does this platform ship". Exercises every platform identity from ONE host: the selector takes
# ET_PLATFORM_IDENTITY as an input, so a foreign platform's rule is testable without that hardware
# (the same trick native/tests/cmake_resolution.sh uses for the Windows pin rows).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

fail() { echo "FAIL: $1"; exit 1; }

# Resolves in a subshell so no case leaks state into the next. Echoes the resolved variant.
resolve() {
  ( set -e
    unset ET_RUNTIME_VARIANT
    export ET_PLATFORM_IDENTITY="$1"
    shift
    # Remaining args are VAR=VALUE overrides for this case only.
    for kv in "$@"; do export "${kv?}"; done
    # shellcheck source=native/variant_select.sh
    . native/variant_select.sh
    echo "${ET_RUNTIME_VARIANT}" )
}

# Every platform this project ships is provisioned for profiling.
for p in linux-x86_64 linux-aarch64 windows-x86_64; do
  got="$(resolve "${p}")"
  [ "${got}" = devtools ] || fail "${p} must ship devtools, got '${got}'"
done

# A platform absent from the list falls back to logging. Uses a name that is not any host's
# identity, so the assertion does not depend on where this runs.
got="$(resolve linux-x86_64 "ET_DEVTOOLS_SUPPORTED_PLATFORMS=some-other-platform")"
[ "${got}" = logging ] || fail "a platform off the list must fall back to logging, got '${got}'"

# Emptying the list is how the last platform leaves; it must not re-add the default. This is the
# single-dash ${VAR-default} behaviour, which an edit to ${VAR:-default} would silently break.
got="$(resolve linux-x86_64 "ET_DEVTOOLS_SUPPORTED_PLATFORMS=")"
[ "${got}" = logging ] || fail "an empty list must mean no platforms, got '${got}'"

# An explicit variant beats the list, so benchmarking (bare) and the negative QA arm (logging)
# stay reachable on a provisioned platform.
for v in bare logging; do
  got="$( ( export ET_RUNTIME_VARIANT="${v}" ET_PLATFORM_IDENTITY=linux-x86_64
            . native/variant_select.sh; echo "${ET_RUNTIME_VARIANT}" ) )"
  [ "${got}" = "${v}" ] || fail "explicit ET_RUNTIME_VARIANT=${v} must win, got '${got}'"
done

# The identity is derived from the host when the caller supplies none -- the path build_qa.sh and
# ubsan_gate.sh take. Assert it resolves to SOMETHING known rather than to an empty string.
got="$( ( unset ET_RUNTIME_VARIANT ET_PLATFORM_IDENTITY
          . native/variant_select.sh; echo "${ET_PLATFORM_IDENTITY}" ) )"
case "${got}" in
  linux-x86_64|linux-aarch64|windows-x86_64) ;;
  *) fail "host-derived identity must be a known platform, got '${got}'" ;;
esac

echo "PASS: variant selection"
