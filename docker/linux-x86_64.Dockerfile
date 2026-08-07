# Build container for the `linux-x86_64` platform. The filename encodes the platform token so the
# image tag, its Dockerfile, and the artifact platform stay one name; an aarch64 platform would add
# a sibling docker/linux-aarch64.Dockerfile (manylinux_2_28_aarch64 base) with no change to how
# tag/Dockerfile are resolved. Same layout as measly-java-learning/iree-runtime-dist's docker/.
#
# Thin specialization of the manylinux_2_28 build container: the base image is already the thing
# that bakes our glibc-2.28 floor (see CLAUDE.md), and this only adds packages the shim and its QA
# need but the base does not ship.
#
# Pinned to a dated tag, not `latest`, and `latest` is exactly what bit us. The base image was
# rebuilt on 2026-08-05T21:10Z (digest sha256:e0b40ace…) and that rebuild dropped
# systemtap-sdt-devel, so every container build of native/core/et_probes.h started failing with
#     native/core/et_probes.h:5:10: fatal error: sys/sdt.h: No such file or directory
# on a tree whose last green run, hours earlier the same day, had compiled the identical header.
# A floating base leaves no way to tell "the image changed" from "our code changed"; that question
# cost real time here.
#
# Note this pin alone would NOT have prevented it: the 2026.06.04-1 base does not ship
# <sys/sdt.h> either. The base's package set is simply not something to depend on. Hence both
# halves -- pin the base so it stops moving, AND install what we need explicitly below.
FROM quay.io/pypa/manylinux_2_28_x86_64:2026.06.04-1

# Pinned to exact NEVRAs rather than bare package names, so a repo update to a newer default module
# stream cannot silently change what this image bakes in. If AlmaLinux/EPEL retire these specific
# builds, this RUN starts failing loudly (dnf cannot resolve the NEVRA) instead of drifting --
# re-resolve with `dnf list --showduplicates <pkg>` and update the pins.
#
# systemtap-sdt-devel  provides <sys/sdt.h>, required by native/core/et_probes.h for the W8 USDT
#                      tracepoints. Note the in-process probe dispatch that the Catch2 suite and
#                      et_leak_harness assert on does NOT need this header -- only the external
#                      bpftrace/perf tracepoints do -- so a missing header is a silently
#                      observability-only loss at runtime but a hard compile failure at build time.
#
# gcc-toolset-14-libasan-devel  the ASan runtime for native/build_qa.sh. Held to 14.2.1-11.el8_10 to
#                      match the base image's compiler exactly (gcc 14.2.1-11, at
#                      /opt/rh/gcc-toolset-14/root/usr/bin/gcc); a libasan from a different toolset
#                      revision than the gcc that emitted the instrumentation is the classic source
#                      of confusing ASan link errors. build_qa.sh still derives the toolset number
#                      from `gcc -dumpversion` and dnf-installs as a fallback, which is now a fast
#                      no-op here and keeps host runs working.
RUN dnf install -y \
      systemtap-sdt-devel-4.9-3.el8 \
      gcc-toolset-14-libasan-devel-14.2.1-11.el8_10 \
    && dnf clean all \
    && rm -rf /var/cache/dnf

# Fail at image-build time, not three steps into a shim build, if a pin ever stops delivering what
# it is here for.
RUN test -e /usr/include/sys/sdt.h \
    || { echo "systemtap-sdt-devel installed but /usr/include/sys/sdt.h is missing"; exit 1; }
