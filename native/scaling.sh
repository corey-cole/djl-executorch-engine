#!/usr/bin/env bash
# Build + run the thread-scaling harness: does concurrent forward() on independent EtRuntime
# instances actually scale, or does it serialize on the XNNPACK shared-workspace mutex?
#
# ExecuTorch 1.3.1 defaults EXECUTORCH_XNNPACK_SHARED_WORKSPACE=ON, and our runtime tarball is
# built `--preset linux` (BUILDINFO cmake_flags) which does not override it -- so the shipped
# runtime uses WorkspaceSharingMode::Global, whose XNNWorkspace::acquire() takes a mutex on every
# delegate call. This sweep is what turns that reading of the source into a number.
#
# Sweeps thread count x workspace sharing mode. The mode is a RUNTIME backend option in 1.3.1, so
# every arm comes from the same binary and the same runtime tarball -- no rebuild, no pin bump.
#
#   default  - leave the runtime alone (Global). This is what ships today.
#   permodel - one workspace per program; lock kept but uncontended across instances.
#   disabled - one workspace per delegate instance, locking off. The ceiling with no lock.
#
# Read the result as a shape, not a single number: if `default` throughput is ~flat from 1 to N
# threads while `disabled` rises with cores, the lock is the ceiling. peak_rss_kb on each line is
# the memory that `disabled` spends to buy it.
#
# Env: MODEL (default the mobilenet demo artifact, falling back to add.pte), THREADS (default
# "1 2 4 8" capped at nproc), MODES (default "default permodel disabled"), ITERS (default 200),
# WARMUP (default 20), ET_RUNTIME_VARIANT (default logging), ET_INSTALL (escape hatch).
#
# Run in the SAME manylinux_2_28 container as native/build.sh so the toolchain matches the runtime
# (./native/local_build_wrapper.sh native/scaling.sh). No JDK needed: ET_BUILD_BENCH skips the shim.
# Container caveat: like bench.sh, this leaves native/bench/ root-owned. Fix with
#   sudo chown -R "$(id -u):$(id -g)" native/bench
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ET_RUNTIME_VARIANT="${ET_RUNTIME_VARIANT:-logging}"
ITERS="${ITERS:-200}"
WARMUP="${WARMUP:-20}"
MODES="${MODES:-default permodel disabled}"
REPS="${REPS:-3}"
# INTRAOP=1 is the arm that actually isolates the workspace lock: XNNPACK runs on ExecuTorch's
# shared pthreadpool, sized to the core count, so at the default size ONE forward already saturates
# the machine and app-thread scaling is flat no matter what any mutex does. Leave unset to measure
# the shipped configuration; set to 1 to measure the lock.
[ -n "${INTRAOP:-}" ] && export ET_INTRAOP_THREADS="${INTRAOP}"

# Prefer a real XNNPACK-delegated conv model: add.pte is small enough that per-call overhead, not
# the delegate, dominates -- which would overstate the lock's share of the time.
if [ -z "${MODEL:-}" ]; then
  if [ -f example/build/models/mobilenet_v2.pte ]; then
    MODEL=example/build/models/mobilenet_v2.pte
  else
    MODEL=native/spike/add.pte
    echo "WARNING: mobilenet_v2.pte not found (./gradlew :example:exportModels); falling back to" >&2
    echo "         ${MODEL}, which is too small to represent a realistic delegate workload." >&2
  fi
fi

NPROC="$(nproc)"
if [ -z "${THREADS:-}" ]; then
  THREADS=""
  for t in 1 2 4 8; do
    [ "$t" -le "$NPROC" ] && THREADS="${THREADS} ${t}"
  done
  [ -z "${THREADS}" ] && THREADS="1"
fi

ET_ARGS=(-DET_RUNTIME_VARIANT="${ET_RUNTIME_VARIANT}")
[ -n "${ET_INSTALL:-}" ] && ET_ARGS+=(-DET_INSTALL="${ET_INSTALL}")

# Shares the native/bench tree with bench.sh: same Release/ET_BUILD_BENCH configure, so the
# FetchContent'd runtime tarball is reused rather than downloaded twice.
bash native/clean_stale_tree.sh native/bench native
cmake -B native/bench -S native -G "Unix Makefiles" "${ET_ARGS[@]}" \
  -DET_BUILD_BENCH=ON -DCMAKE_BUILD_TYPE=Release
cmake --build native/bench --target et_scaling_harness

# BUILD_ONLY=1 stops here: a compile/link check that does not burn a measurement run. The harness
# cannot be built on the host (the runtime's exported config hardcodes the container's
# /usr/lib64/libm.so -- et_timing_harness has the same constraint), so this is the cheap way to
# verify a source change from outside a measurement session.
if [ -n "${BUILD_ONLY:-}" ]; then
  echo "BUILD_ONLY set: built native/bench/et_scaling_harness, skipping the sweep."
  exit 0
fi

echo "--- thread scaling: model=${MODEL} iters=${ITERS} warmup=${WARMUP} nproc=${NPROC}"
echo "    intraop=${INTRAOP:-default} reps=${REPS} modes='${MODES}' threads='${THREADS}' ---"

# Loop order is reps -> threads -> modes, NOT modes -> threads. Running each mode's whole thread
# sweep as one block puts minutes of sustained 8-core load between the first arm and the last, and
# the resulting thermal/turbo drift lands entirely on whichever mode ran last -- a first pass this
# way made `disabled` look 2.6x slower than `default` at ONE thread, where the two are provably
# identical. Interleaving spreads drift evenly across arms; the reps expose whatever is left.
# Compare cells across reps before believing any gap between modes.
for rep in $(seq 1 "${REPS}"); do
  for t in ${THREADS}; do
    for mode in ${MODES}; do
      case "${mode}" in
        default)  unset ET_SHARING_MODE ;;
        disabled) export ET_SHARING_MODE=0 ;;
        permodel) export ET_SHARING_MODE=1 ;;
        global)   export ET_SHARING_MODE=2 ;;
        *) echo "unknown mode '${mode}' (default|permodel|disabled|global)" >&2; exit 2 ;;
      esac
      printf 'rep=%d ' "${rep}"
      ./native/bench/et_scaling_harness "${MODEL}" "${t}" "${ITERS}" "${WARMUP}"
    done
  done
done
