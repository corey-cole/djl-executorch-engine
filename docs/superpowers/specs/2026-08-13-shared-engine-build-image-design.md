# Migrating onto the shared `engine-build` image

**Status:** approved design, not yet implemented
**Contract:** `measly-java-learning/base-docker-images` → `docs/consuming-engine-build.md`

## 1. What changes and why

This repo builds its own Linux toolchain container from `docker/`, on every CI run, backed by a
GitHub Actions layer cache that a second workflow (`warm-build-image.yml`) exists solely to warm.
That machinery is replaced by a `docker run` against a digest-pinned image published by
`measly-java-learning/base-docker-images`.

The pin:

```
ghcr.io/measly-java-learning/engine-build@sha256:725884538caa4f7f8444847e34b3928bb90089da95d5b77ce560aa2e624f905b
```

Three things make this a net simplification rather than a lateral move:

1. **The GHA layer cache disappears entirely.** With no cache there is no cache scope, so the
   cross-arch scope collision fixed in #38/#39 cannot recur — and a whole workflow
   (`warm-build-image.yml`) whose only job was feeding that cache is deleted.
2. **The image is a manifest list** (`linux/amd64` + `linux/arm64`, verified against the digest
   above with `docker manifest inspect`, anonymously). One reference serves both matrix rows and
   both developer architectures. The per-row `image:` variable and the wrapper's hardcoded
   `linux-x86_64.Dockerfile` both go away, and `local_build_wrapper.sh` starts working unmodified
   on an aarch64 workstation.
3. **The package is public**, so the cross-org pull from `corey-cole/djl-executorch-engine` needs
   no login step, no token, and no cross-org secret.

The glibc-2.28 floor is unaffected: the shared image is itself a manylinux_2_28 derivative, which
is the property the floor depends on (CLAUDE.md, "glibc floor").

## 2. Verified premise correction

The migration brief assumed `native/build.sh` and `native/build_qa.sh` "already read
`MEASLY_DJL_*` from the environment", needing verification rather than edits. **They do not.** The
string `MEASLY_DJL` does not appear anywhere in this repository. What the scripts do today, all of
which the image supersedes:

| Location | Today | Under the image |
| --- | --- | --- |
| `build.sh:73-74` | `export PATH=/opt/python/cp312-cp312/bin:…`; `pip install ninja` | ninja is at `/usr/local/bin/ninja`, version `MEASLY_DJL_NINJA_VERSION` |
| `build.sh:53-63` | `cp /workspace/amazon-corretto-linux-jdk.rpm`; `rpm2archive`; untar into `/opt/corretto` | `JAVA_HOME` is set and its headers are present; **`/opt/corretto` is already populated by the image**, so this is a live collision, not mere redundancy |
| `build_qa.sh:64-68` | `dnf install -y gcc-toolset-N-libasan-devel \|\| true` | baked at `MEASLY_DJL_TOOLSET_NEVRA`, assertable with `rpm -q` |
| `docker/*.Dockerfile` | asserts `/usr/include/sys/sdt.h` at image-build time | `systemtap-sdt-devel-4.9-3.el8` is baked; the assertion must move into a script or it is lost |

So the scripts are edited, not merely verified.

## 3. No non-image Linux fallback

`build.sh`'s Linux branch has never been runnable outside the container. It assumes `rpm2archive`
(absent on Debian/Ubuntu), `dnf`, `/opt/python/cp312-cp312` on PATH, and write access to
`/opt/corretto`. On an Ubuntu workstation or a stock GitHub runner it does not degrade — it dies,
historically with `mkdir /opt/corretto: Permission denied`. Both this project's development
workstation and its CI runners are Ubuntu, so the "keep the install paths as a fallback" option
preserves nothing that has ever worked.

Therefore: inside the image, **assert**; on Linux outside it, **fail immediately** with a message
naming `native/local_build_wrapper.sh`. Deleting the install paths removes code that only ever ran
where the packages were already present. The Windows branch never had pip/dnf/RPM logic and is
untouched throughout.

`MEASLY_DJL_PINNED_IMAGE=1` is the branch signal, per the contract: inside the image a missing tool
is a broken image and must fail loudly rather than be installed at run time.

## 4. Design

### 4.1 `.engine-build-image` — single source of truth

A new repo-root file holding exactly one line: the digest reference, no comments. Both consumers
read it with a plain `cat` plus a non-empty check — the rationale comment lives at each point of
use, not in the data file:

```bash
IMAGE="$(cat "${REPO_ROOT}/.engine-build-image")"
test -n "${IMAGE}" || { echo "empty .engine-build-image" >&2; exit 1; }
```

- `native/local_build_wrapper.sh` — as the default for `ET_BUILD_IMAGE`, so an explicit override
  still wins.
- `.github/workflows/native-build-job.yml` — one step after checkout publishing `ET_BUILD_IMAGE`
  to `$GITHUB_ENV` for the `docker run` steps, with the same non-empty check.

The alternative — a workflow `env:` var plus a wrapper default — writes the digest twice and lets
local builds silently disagree with CI, which is the drift the pin exists to prevent.

### 4.2 `.github/workflows/native-build-job.yml`

Deleted: the `image:` and `corretto-jdk-url:` matrix keys (the matrix collapses to `platform` +
`runner`), the "Download Corretto JDK 8 RPM" step, the `docker/setup-buildx-action` step, the
`docker/build-push-action` step, and its `cache-from: type=gha` line.

Added: a "Resolve the pinned build image" step reading `.engine-build-image` into `$GITHUB_ENV`.

Changed: both `docker run` steps reference `$ET_BUILD_IMAGE`; the pull is implicit.

Unchanged: the runtime-provenance `gh attestation verify` gate, the artifact upload, and the entire
`build-executorch-shim-windows` job.

### 4.3 Deletions

- `docker/linux-x86_64.Dockerfile`, `docker/linux-aarch64.Dockerfile`, and the `docker/` directory.
- `.github/workflows/warm-build-image.yml` in its entirety.

### 4.4 `native/local_build_wrapper.sh`

Deleted: the `curl` that downloads and caches `amazon-corretto-linux-jdk.rpm`, the `docker build`
block, and `SKIP_IMAGE_BUILD` (meaningless with no build). The `docker run` invocation, its `-e`
forwarding list, and the `/workspace` mount are unchanged. Comments are rewritten: the image is
pulled, not built, and is arch-agnostic.

### 4.5 `native/build.sh`, Linux branch

- **Guard**, placed *after* the `PRINT_BUILD_CONFIG` early-exit so that diagnostic keeps working on
  any host: on Linux, `MEASLY_DJL_PINNED_IMAGE != 1` exits with a message naming
  `native/local_build_wrapper.sh` and `docs/building.md`.
- **JDK**: assert `JAVA_HOME` is set and `$JAVA_HOME/include/linux/jni_md.h` exists. The
  `rpm2archive` extraction block is deleted.
- **Ninja**: the `pip install ninja` and `cp312` PATH export are deleted. Assert `ninja` resolves
  and that `ninja --version` **reports** `MEASLY_DJL_NINJA_VERSION` — the contract warns this
  string (`1.13.0.git.kitware.jobserver-pipe-1`) differs from the pip metadata version
  (`1.13.0`), so the comparison is against reported output only.

Unchanged: `cd /workspace`, `JOBS`, the `HOST_UID` chown trap, the `GITHUB_ENV` `JAVA_HOME`
publication, CMake configure/build, staging, and licence copying.

### 4.6 `native/build_qa.sh`, Linux branch

The `dnf install … || true` block is replaced by two assertions, both reading values from the
environment rather than hardcoding them (the contract's troubleshooting section calls out
hardcoding as the cause of a script/image disagreement):

- `rpm -q "gcc-toolset-${MEASLY_DJL_TOOLSET_VER}-libasan-devel-${MEASLY_DJL_TOOLSET_NEVRA}"`
- `test -e /usr/include/sys/sdt.h` — the assertion the deleted Dockerfile carried. It must land
  somewhere or a future pin bump that drops `systemtap-sdt-devel` reappears as
  `native/core/et_probes.h:5:10: fatal error: sys/sdt.h: No such file or directory`, the exact
  failure the Dockerfile comment documents.

The same non-image guard as `build.sh` applies, inside the Linux branch only. The Windows branch is
untouched.

### 4.7 `native/tests/ci_workflow.sh`

This suite is load-bearing, not incidental cleanup: lines 43-45 assert the aarch64 matrix row uses
`amazon-corretto-8-aarch64-linux-jdk.rpm`. That assertion becomes false under this migration and the
suite goes red. It is replaced with assertions on the invariants that matter afterwards:

- `.engine-build-image` exists and contains an `@sha256:` digest.
- The workflow references `$ET_BUILD_IMAGE` and `ghcr.io/measly-java-learning/engine-build`.
- **No** `build-push-action`, `setup-buildx-action`, or `type=gha` appears anywhere in it.
- The `linux-aarch64` row still exists and still runs on `ubuntu-24.04-arm`.

The third bullet is what keeps #38's bug class retired by test rather than by memory.

### 4.8 Documentation

- `CLAUDE.md` — the "Build & test" section and the container-only note: the image is pulled, and
  the pin lives in `.engine-build-image`.
- `docs/building.md` — the Docker prerequisite is a pull rather than a build; the wrapper
  description; a pin-bump procedure pointing at the contract doc.
- `README.md` (~line 250) — the manylinux sentence gains the shared-image reference.
- `docs/ci-native-build.md` is point-in-time research whose manylinux content remains accurate;
  untouched.
- `.gitignore` — drop `amazon-corretto-linux-jdk.rpm` if present, since nothing produces it now.

## 5. Verification

1. `native/tests/ci_workflow.sh` and `native/tests/docs_present.sh` — fast, host-only.
2. `./native/local_build_wrapper.sh` — pulls the pinned image and builds the shim.
3. `./native/local_build_wrapper.sh native/build_qa.sh` — Catch2 suite plus the ASan/LSan leak
   harness inside the image.
4. `./gradlew test` — JVM integration tests against the freshly staged `.so`.

Steps 2-4 prove the migration on amd64 only. The arm64 half of the manifest list is provable only
on the `ubuntu-24.04-arm` runner in CI, or on the aarch64 host on the LAN; that is stated as a
known gap rather than papered over.

## 6. Out of scope

- Bumping the ExecuTorch runtime pin (`native/cmake/EtRuntimePin.cmake`) — unrelated supply chain.
- The Windows job and the Windows branches of both build scripts.
- `gh attestation verify` on the image digest. The contract documents the command; adding it as a
  CI gate is a separate decision from this migration, and the runtime-tarball attestation gate
  already in the workflow is untouched either way.
