# Build container for the `linux-aarch64` platform. Sibling of linux-x86_64.Dockerfile; the
# filename encodes the platform token so the image tag, its Dockerfile, and the artifact platform
# stay one name. Same layout as measly-java-learning/iree-runtime-dist's docker/.
#
# This image is what the live `linux-aarch64` matrix row in native-build-job.yml builds inside: it
# runs on the ubuntu-24.04-arm runner, so the shim and its QA are compiled and sanitizer-tested on
# native aarch64 hardware.
#
# Pinned to a dated tag, not `latest` -- see linux-x86_64.Dockerfile for why that pin exists.
FROM quay.io/pypa/manylinux_2_28_aarch64:2026.06.04-1

# NEVRAs verified 2026-08-06 by running this exact base tag on a native aarch64 host (not qemu):
# gcc is 14.2.1-11, and both packages resolve to the same versions as the x86_64 image. That
# agreement is a convenience, not a rule -- re-resolve per arch with
# `dnf list --showduplicates <pkg>` on the next base bump rather than copying the x86_64 values.
#
# systemtap-sdt-devel  provides <sys/sdt.h> for native/core/et_probes.h (W8 USDT tracepoints).
# gcc-toolset-14-libasan-devel  ASan runtime for native/build_qa.sh, held to the base image's own
#                      compiler revision so libasan and the gcc that emitted the instrumentation
#                      always match.
RUN dnf install -y \
      systemtap-sdt-devel-4.9-3.el8 \
      gcc-toolset-14-libasan-devel-14.2.1-11.el8_10 \
    && dnf clean all \
    && rm -rf /var/cache/dnf

# Fail at image-build time, not three steps into a shim build, if a pin ever stops delivering what
# it is here for.
RUN test -e /usr/include/sys/sdt.h \
    || { echo "systemtap-sdt-devel installed but /usr/include/sys/sdt.h is missing"; exit 1; }
