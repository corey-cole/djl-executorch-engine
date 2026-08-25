#!/usr/bin/env bash
# Shared resolution of the shipped ExecuTorch runtime variant, used by the three scripts that
# build against the runtime: native/build.sh, native/build_qa.sh and native/ubsan_gate.sh.
# Sourced, never executed -- one definition of the rule, because nothing binds the copies:
# native/tests/build_config.sh only exercises build.sh, so duplicated blocks can drift and the
# symptom is QA (or the UBSan gate) silently exercising a variant that is not the one shipping.
#
# On return, ET_RUNTIME_VARIANT is set. An explicit value exported by the caller always wins, so
# benchmarking (bare) and the negative QA arm (logging) are reachable on a devtools platform.
# Otherwise the default is keyed on the PLATFORM IDENTITY:
#
#   ET_PLATFORM_IDENTITY (required input): the platform this run produces for. Callers with a
#   richer identity set it first -- build.sh assigns OUT_PLATFORM, its artifact platform identity
#   (which also covers the Windows Git-Bash fork). When it is unset, the identity is derived from
#   the host via uname, the same derivation build_qa.sh and ubsan_gate.sh used to carry copies of.
#
#   ET_DEVTOOLS_SUPPORTED_PLATFORMS (input, default "linux-x86_64"): the platforms whose shipped
#   artifact links a devtools runtime, enabling the per-model profiling option. An engine-side
#   decision, NOT a mirror of the pin: the pin publishes devtools for every platform as of
#   1.4.1-3, and a platform joins this list once a test proves profiling works there. The cost of
#   carrying it is +138 KB of .so and steady-state latency bounded under 0.35%
#   (see docs/profiling.md).
#
#   The default is ${ET_DEVTOOLS_SUPPORTED_PLATFORMS-linux-x86_64} with a SINGLE dash,
#   deliberately NOT ${...:-...}: an explicitly EMPTY list must mean "no platforms" -- that is how
#   a platform leaves the list -- not "re-add the default". native/tests/build_config.sh pins the
#   distinction (ET_DEVTOOLS_SUPPORTED_PLATFORMS= must resolve to logging), so an edit to `:-`
#   breaks that test silently.

if [ -z "${ET_PLATFORM_IDENTITY:-}" ]; then
  case "$(uname -s)" in
    MINGW*|MSYS*) ET_PLATFORM_IDENTITY="windows-x86_64" ;;
    *)
      case "$(uname -m)" in
        aarch64|arm64) ET_PLATFORM_IDENTITY="linux-aarch64" ;;
        *)             ET_PLATFORM_IDENTITY="linux-x86_64"  ;;
      esac
      ;;
  esac
fi

ET_DEVTOOLS_SUPPORTED_PLATFORMS="${ET_DEVTOOLS_SUPPORTED_PLATFORMS-linux-x86_64}"
if [ -z "${ET_RUNTIME_VARIANT:-}" ]; then
  case " ${ET_DEVTOOLS_SUPPORTED_PLATFORMS} " in
    *" ${ET_PLATFORM_IDENTITY} "*) ET_RUNTIME_VARIANT=devtools ;;
    *)                             ET_RUNTIME_VARIANT=logging ;;
  esac
fi
