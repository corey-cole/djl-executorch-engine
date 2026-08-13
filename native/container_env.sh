#!/usr/bin/env bash
# Ownership handling for scripts run under native/local_build_wrapper.sh.
#
# The wrapper bind-mounts the repo into a container that runs as root, so anything a build or QA run
# creates comes back root-owned on the host, where the next run's `rm -rf` then fails with a bare
# "Permission denied". The wrapper passes HOST_UID/HOST_GID so we can hand the outputs back on exit.
#
# Sourced, never executed. Keeping it in one place is the point: two copies of this trap will drift.

# Paths registered by et_chown_outputs_on_exit, chowned by et_chown_cleanup.
ET_CHOWN_PATHS=()

et_chown_cleanup() {
  rc=$?
  if [ -n "${HOST_UID:-}" ] && [ "${#ET_CHOWN_PATHS[@]}" -gt 0 ]; then
    # Deliberately unquoted: entries may be globs (src/main/resources/native/linux*) that must expand
    # HERE, at exit, rather than at registration time when the directories may not exist yet.
    # `|| true` so a chown failure never masks the script's real exit status.
    chown -R "${HOST_UID}:${HOST_GID}" ${ET_CHOWN_PATHS[@]} 2>/dev/null || true
  fi
  exit "$rc"
}

# Usage: et_chown_outputs_on_exit native/build 'src/main/resources/native/linux*'
# Quote glob arguments at the call site so they survive to exit-time expansion.
et_chown_outputs_on_exit() {
  ET_CHOWN_PATHS=("$@")
  trap et_chown_cleanup EXIT
}
