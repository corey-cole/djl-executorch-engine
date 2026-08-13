# Shared `engine-build` Image Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace this repo's per-run Linux toolchain container build with a `docker run` against a digest-pinned, multi-arch image published by `measly-java-learning/base-docker-images`.

**Architecture:** The digest lives in one file, `.engine-build-image`, read by both `native/local_build_wrapper.sh` and `.github/workflows/native-build-job.yml`. The image is a manifest list, so one reference serves amd64 and arm64 and no `--platform` is passed. `native/build.sh` and `native/build_qa.sh` stop installing ninja/JDK/libasan and instead assert the image supplies them, branching on the image's own `MEASLY_DJL_PINNED_IMAGE=1` signal. `docker/` and `warm-build-image.yml` are deleted outright.

**Tech Stack:** GitHub Actions, Docker, Bash, CMake/Ninja, manylinux_2_28.

**Spec:** `docs/superpowers/specs/2026-08-13-shared-engine-build-image-design.md`

## Global Constraints

- **The pin, verbatim** (one line, no trailing comment):
  `ghcr.io/measly-java-learning/engine-build@sha256:725884538caa4f7f8444847e34b3928bb90089da95d5b77ce560aa2e624f905b`
- **Digest, never a tag.** `:main` moves on every publish; the `sha-<short>-amd64` / `-arm64` tags are manifest-list children and must not be referenced.
- **No login, no buildx, no image build.** The package is public and pulls anonymously. Any `docker/setup-buildx-action`, `docker/build-push-action`, `cache-from: type=gha`, or `cache-to: type=gha` reintroduced into this repo is a regression of #38.
- **Image-supplied environment**, to be read and never hardcoded: `MEASLY_DJL_PINNED_IMAGE=1`, `MEASLY_DJL_TOOLSET_VER=14`, `MEASLY_DJL_TOOLSET_NEVRA=14.2.1-11.el8_10`, `MEASLY_DJL_NINJA_VERSION=1.13.0.git.kitware.jobserver-pipe-1`, `JAVA_HOME=/opt/corretto-jdk`.
- **`MEASLY_DJL_NINJA_VERSION` is the string `ninja --version` reports**, not the pip metadata version (`1.13.0`). Compare against reported output only.
- **The Windows job and both scripts' Windows branches are untouched** by every task in this plan.
- **The `/workspace` mount path stays.** `native/build.sh` hardcodes `cd /workspace`; the contract doc's `/src` example is arbitrary and must not be copied.
- **Bare pin file.** `.engine-build-image` holds exactly one line and is read with `cat` plus a non-empty check — no comment-stripping pipeline.

## Intermediate CI state — read before pushing

Tasks 1-4 cannot leave the tree green in CI individually: after Task 2 the build scripts require the new image, but the workflow does not use it until Task 4. **Do not push the branch until Task 5 is committed.** Task 6's local run against the real image is the gate that matters; CI's arm64 leg is the only thing that must wait for a push.

---

### Task 1: Pin file and local wrapper

**Files:**
- Create: `.engine-build-image`
- Modify: `native/local_build_wrapper.sh:21-37`

**Interfaces:**
- Produces: `.engine-build-image` (one-line digest reference), consumed by Task 4's workflow step. `ET_BUILD_IMAGE` remains the wrapper's override env var.

**No test of its own.** Asserting the wrapper's text with greps would pin its exact wording and break on the next edit while proving nothing. The wrapper is verified where it actually matters — Task 6 runs it against the real image — and the one durable invariant about the pin file (digest, not tag) is asserted once in Task 4.

- [ ] **Step 1: Create the pin file**

`.engine-build-image`, exactly one line and a trailing newline:

```
ghcr.io/measly-java-learning/engine-build@sha256:725884538caa4f7f8444847e34b3928bb90089da95d5b77ce560aa2e624f905b
```

- [ ] **Step 2: Rewrite the wrapper's preamble**

In `native/local_build_wrapper.sh`, delete the Corretto `curl` block (lines 21-25) and the image-build block (lines 27-37) and put this in their place:

```bash
# The pinned shared toolchain image, digest not tag. It is a manifest list covering amd64 and
# arm64, so Docker resolves the architecture -- there is no Dockerfile to pick and no --platform
# to pass. The digest lives in one file so a bump is a one-line change that CI and this wrapper
# pick up together; a second copy here would drift, and the failure mode is CI green while you
# build against a different toolchain. Published by measly-java-learning/base-docker-images; see
# its docs/consuming-engine-build.md for what the image guarantees.
ET_BUILD_IMAGE="${ET_BUILD_IMAGE:-$(cat "${REPO_ROOT}/.engine-build-image")}"
test -n "${ET_BUILD_IMAGE}" || { echo "empty .engine-build-image" >&2; exit 1; }
```

Also update the header comment: the JDK headers now come from the image (`JAVA_HOME`), not a downloaded RPM, and the wrapper works unmodified on an aarch64 workstation.

`.gitignore` needs no edit — line 39 is a generic `*.rpm`, not a named Corretto entry. Delete any stale `amazon-corretto-linux-jdk.rpm` left in your working tree from an earlier build; nothing produces it now.

- [ ] **Step 3: Sanity-check the wrapper's own resolution**

Run: `ET_BUILD_IMAGE=echo-only bash -n native/local_build_wrapper.sh && head -c 200 .engine-build-image`
Expected: `bash -n` reports no syntax error, and the pin file prints the digest on one line.

- [ ] **Step 4: Commit**

```bash
git add .engine-build-image native/local_build_wrapper.sh
git commit -m "build: pin the shared engine-build image and stop building one locally"
```

---

### Task 2: `native/build.sh` asserts its requirements instead of installing them

**Files:**
- Modify: `native/build.sh:52-64` (JDK), `native/build.sh:71-77` (ninja)
- Test: `native/tests/build_config.sh`

**Interfaces:**
- Consumes: `JAVA_HOME` (required on any Linux host); `MEASLY_DJL_PINNED_IMAGE` and `MEASLY_DJL_NINJA_VERSION` (used only to switch on the in-image version-equality check).
- Produces: a missing requirement exits non-zero with a message naming the requirement and pointing at `local_build_wrapper.sh`. A host build still succeeds when the tools are present, with a glibc-floor warning. `PRINT_BUILD_CONFIG=1` behaviour is unchanged.

- [ ] **Step 1: Write the failing test**

Append to `native/tests/build_config.sh`, before the final `echo "PASS: ..."`:

```bash
# Requirement assertions must fail BY NAME, not as a confusing failure ten steps later. These drive
# the JAVA_HOME and ninja assertions; they are behavioural (run the script, read what it says) and
# host-independent, since each one removes a requirement rather than depending on what this host has.
rc=0
out="$(JAVA_HOME=/nonexistent bash native/build.sh 2>&1)" || rc=$?
test "${rc}" -ne 0 || fail "build.sh must fail when the JDK headers are absent"
grep -q 'jni_md.h\|JAVA_HOME' <<<"${out}" || fail "JDK failure must name what is missing"
grep -q 'local_build_wrapper.sh' <<<"${out}" || fail "JDK failure must point at the wrapper"

rc=0
out="$(PATH=/nonexistent-bin bash native/build.sh 2>&1)" || rc=$?
test "${rc}" -ne 0 || fail "build.sh must fail when ninja is absent"
grep -q 'ninja' <<<"${out}" || fail "toolchain failure must name ninja"
```

Behavioural, not shape. Do **not** add greps for the removed `pip install ninja` / `rpm2archive` lines.

- [ ] **Step 2: Run test to verify it fails**

Run: `bash native/tests/build_config.sh`
Expected: FAIL — today the script gets as far as the Corretto RPM step and dies on `cp: cannot stat '/workspace/amazon-corretto-linux-jdk.rpm'` or a `/opt/corretto` permission error, so the message assertions fail. That failure mode *is* what this task replaces.

- [ ] **Step 3: Replace the JDK block**

There is **no permission guard** — a host with the tools may build (spec §3). Replace the `else` branch of the JDK `if` (lines 52-64, the `JDK_EXTRACT`/`rpm2archive`/`find` block) with:

```bash
else
  # Headers only; we never link libjvm. JAVA_HOME comes from the pinned image or from the host --
  # either way it is supplied, never installed here.
  echo "--- JDK headers ---"
  test -n "${JAVA_HOME:-}" \
    || { echo "JAVA_HOME is unset: point it at any JDK, or build via ./native/local_build_wrapper.sh"; exit 1; }
  test -f "${JAVA_HOME}/include/linux/jni_md.h" \
    || { echo "no JDK headers under JAVA_HOME=${JAVA_HOME} (want include/linux/jni_md.h); or build via ./native/local_build_wrapper.sh"; exit 1; }
  echo "JAVA_HOME=${JAVA_HOME}"
fi
```

Update the "This script expects:" comment above it (lines 36-41): it no longer expects a Corretto RPM in `/workspace`. It expects a toolchain — supplied by the pinned image via `local_build_wrapper.sh`, or by the host.

- [ ] **Step 4: Replace the ninja block**

Replace the `else` branch of the toolchain `if` (lines 71-77) with:

```bash
else
  echo "--- Toolchain (asserted, never installed) ---"
  command -v ninja >/dev/null 2>&1 \
    || { echo "ninja not on PATH: install it, or build via ./native/local_build_wrapper.sh"; exit 1; }
  # In the pinned image a version mismatch means a broken image, so equality is checked there and
  # only there; on a host any ninja will do. Compare what ninja REPORTS, not pip metadata: the
  # Kitware jobserver-pipe wheel is `1.13.0` to pip but prints `1.13.0.git.kitware.jobserver-pipe-1`.
  if [ "${MEASLY_DJL_PINNED_IMAGE:-}" = "1" ] && [ "$(ninja --version)" != "${MEASLY_DJL_NINJA_VERSION}" ]; then
    echo "ninja reports $(ninja --version), image declares ${MEASLY_DJL_NINJA_VERSION}"; exit 1
  fi
  # Building outside the image is supported but does NOT hold the glibc-2.28 floor -- the artifact
  # links host glibc. Fine for local ./gradlew test, never for a release (see CLAUDE.md).
  if [ "${MEASLY_DJL_PINNED_IMAGE:-}" != "1" ]; then
    echo "WARNING: not the pinned engine-build image -- this .so links host glibc and breaks the" >&2
    echo "         2.28 floor. Local testing only; release builds go through local_build_wrapper.sh." >&2
  fi
  gcc --version; g++ --version; cmake --version; ninja --version
fi
```

- [ ] **Step 5: Run test to verify it passes**

Run: `bash native/tests/build_config.sh`
Expected: `PASS: build.sh config`

- [ ] **Step 6: Commit**

```bash
git add native/build.sh native/tests/build_config.sh
git commit -m "build: assert the toolchain instead of installing it"
```

---

### Task 3: `native/build_qa.sh` asserts its QA requirements

**Files:**
- Modify: `native/build_qa.sh:63-68`
- Test: `native/tests/ci_workflow.sh`

**Interfaces:**
- Consumes: `MEASLY_DJL_PINNED_IMAGE`, `MEASLY_DJL_TOOLSET_VER`, `MEASLY_DJL_TOOLSET_NEVRA`.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the failing test**

Append to `native/tests/ci_workflow.sh`, after the Task 1 block:

```bash
# build_qa.sh is the only thing asserting <sys/sdt.h>. Drop that assertion and a toolchain without
# systemtap-sdt-devel surfaces as `fatal error: sys/sdt.h: No such file` inside et_probes.h instead.
grep -q '/usr/include/sys/sdt.h' native/build_qa.sh \
  || fail "build_qa.sh must assert <sys/sdt.h> is present"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bash native/tests/ci_workflow.sh`
Expected: `FAIL: build_qa.sh must assert <sys/sdt.h> is present`

- [ ] **Step 3: Replace the dnf block**

In `native/build_qa.sh`, replace the `if command -v dnf …` block (lines 63-68) with the following. Note there is **no permission guard**: a host with a working `-fsanitize=address` toolchain and `<sys/sdt.h>` can run QA, and if its libasan is missing or mismatched the ASan link fails and says so — remediating it here was the actual mistake.

```bash
  # QA requirements are asserted, never installed. <sys/sdt.h> ships in systemtap-sdt-devel and is
  # required by native/core/et_probes.h for its USDT tracepoints; assert it here so a missing header
  # fails by name instead of as a fatal error mid-compile.
  test -e /usr/include/sys/sdt.h || {
    echo "missing /usr/include/sys/sdt.h (systemtap-sdt-devel): native/core/et_probes.h needs it;"
    echo "install it, or run QA via ./native/local_build_wrapper.sh native/build_qa.sh"; exit 1; }

  # Pinned image only. A libasan from a different toolset revision than the gcc that emitted the
  # instrumentation produces confusing ASan link errors, so the image holds both to one revision and
  # this asserts it held. Read both values from the environment: a hardcoded NEVRA is how a script
  # and its image come to disagree. Skipped on a host, where rpm need not exist and a different ASan
  # runtime is legitimate -- there the -fsanitize=address link is the check.
  if [ "${MEASLY_DJL_PINNED_IMAGE:-}" = "1" ]; then
    ASAN_PKG="gcc-toolset-${MEASLY_DJL_TOOLSET_VER}-libasan-devel-${MEASLY_DJL_TOOLSET_NEVRA}"
    rpm -q "${ASAN_PKG}" >/dev/null \
      || { echo "libasan NEVRA not installed as pinned: ${ASAN_PKG}"; exit 1; }
  fi
```

Also update the script's header comment (lines 11-14): CI runs this in the shared engine-build image, not "the SAME manylinux_2_28 container".

- [ ] **Step 4: Run test to verify it passes**

Run: `bash native/tests/ci_workflow.sh`
Expected: `PASS: ci workflow`

- [ ] **Step 5: Commit**

```bash
git add native/build_qa.sh native/tests/ci_workflow.sh
git commit -m "build: assert the pinned image's ASan runtime and sdt.h in QA"
```

---

### Task 4: Workflow migration and deletions

**Files:**
- Modify: `.github/workflows/native-build-job.yml:12-84`
- Delete: `.github/workflows/warm-build-image.yml`, `docker/linux-x86_64.Dockerfile`, `docker/linux-aarch64.Dockerfile`
- Test: `native/tests/ci_workflow.sh:38-45`

**Interfaces:**
- Consumes: `.engine-build-image` from Task 1.
- Produces: `ET_BUILD_IMAGE` in `$GITHUB_ENV` for the two `docker run` steps.

- [ ] **Step 1: Write the failing test**

In `native/tests/ci_workflow.sh`, **replace** the aarch64 Corretto assertion (lines 43-45, the `awk '/platform: linux-aarch64/{f=1} f' … amazon-corretto-8-aarch64-linux-jdk.rpm` block) with:

```bash
# The image is a manifest list, so the aarch64 row needs no image of its own and no arch-specific
# JDK -- it needs only an arm runner. Its identity is the runner, asserted just above.

# The image is pulled, never built here. Layer caching stays banned: its scopes collide across the
# arch rows (#38).
grep -q 'build-push-action'   "${WFJOB}" && fail "workflow must not build an image"
grep -q 'setup-buildx-action' "${WFJOB}" && fail "workflow must not set up buildx"
grep -q 'type=gha'            "${WFJOB}" && fail "workflow must not use the GHA layer cache"
grep -q 'corretto'            "${WFJOB}" && fail "workflow must not download a JDK (image supplies JAVA_HOME)"
grep -q 'ET_BUILD_IMAGE'      "${WFJOB}" || fail "workflow must run against the pinned image"
test -f .github/workflows/warm-build-image.yml && fail "warm-build-image.yml must be deleted"
test -d docker && fail "docker/ must be deleted"

# Digest, not tag: `:main` moves on every publish, which is the exact failure the pin exists to
# prevent -- a toolchain rebuilding underneath a green tree.
grep -q 'engine-build@sha256:' .engine-build-image || fail ".engine-build-image must pin a digest, not a tag"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bash native/tests/ci_workflow.sh`
Expected: `FAIL: workflow must not build an image`

- [ ] **Step 3: Rewrite the Linux job**

In `.github/workflows/native-build-job.yml`, the matrix (lines 13-23) becomes:

```yaml
    strategy:
      matrix:
        include:
          - platform: linux-x86_64
            runner: ubuntu-latest
          - platform: linux-aarch64
            runner: ubuntu-24.04-arm
```

Delete the "Download Corretto JDK 8 RPM" step (lines 32-34) — the image supplies `JAVA_HOME`.

Replace the buildx + build-push-action steps (lines 48-66) with:

```yaml
      # The shared toolchain image, published by measly-java-learning/base-docker-images. It is a
      # public manifest list covering amd64 and arm64, so both matrix rows use one reference and the
      # pull needs no credentials. The digest is read from .engine-build-image rather than written
      # here, so a bump is one line that this workflow and native/local_build_wrapper.sh both pick up.
      - name: Resolve the pinned build image
        run: |
          image="$(cat .engine-build-image)"
          test -n "${image}" || { echo "empty .engine-build-image" >&2; exit 1; }
          echo "ET_BUILD_IMAGE=${image}" >> "$GITHUB_ENV"
          echo "Build image: ${image}"
```

Both `docker run` steps (lines 70-84) use `${{ env.ET_BUILD_IMAGE }}` in place of `${{ matrix.image }}`; the mount, `-w /workspace`, and the script paths are unchanged. Update the stale comment at lines 68-69 to say the wrapper runs the same image.

- [ ] **Step 4: Delete the retired machinery**

```bash
git rm .github/workflows/warm-build-image.yml
git rm -r docker
```

- [ ] **Step 5: Run test to verify it passes**

Run: `bash native/tests/ci_workflow.sh`
Expected: `PASS: ci workflow` (the embedded `yaml.safe_load` check also proves the edited YAML still parses)

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/native-build-job.yml native/tests/ci_workflow.sh
git commit -m "ci: run the shared engine-build image; delete docker/ and the warm job"
```

---

### Task 5: Documentation

**Files:**
- Modify: `CLAUDE.md` ("Build & test" / native shim section), `docs/building.md:5-46,102`, `README.md:250`
- Test: `native/tests/docs_present.sh`

**Interfaces:**
- Consumes: nothing. Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the failing test**

Append to `native/tests/docs_present.sh`, before the final `echo`:

```bash
# Current guidance must not send a contributor to a Dockerfile this repo does not contain. Scoped to
# the three current-guidance docs: docs/superpowers/ and docs/research/ are point-in-time records and
# are expected to name things that are gone.
grep -q 'docker/linux-.*\.Dockerfile' docs/building.md README.md CLAUDE.md \
  && fail "current docs reference a per-platform Dockerfile that does not exist"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bash native/tests/docs_present.sh`
Expected: `FAIL: docs still reference the deleted per-platform Dockerfiles` (CLAUDE.md and `docs/building.md` both name them today)

- [ ] **Step 3: Update `docs/building.md`**

- Prerequisites (lines 5-10): Docker is still required, but for a **pull**, not a build. First build pays a pull of the pinned digest.
- The wrapper description (lines 36-46): it runs the shared image. **This is a behaviour change to document, not just a mechanism swap** — `build.sh` no longer installs anything, so a host build is now possible where the host already has ninja, cmake, a C++17 compiler and JDK headers. Say plainly what such a build costs: it links host glibc and breaks the 2.28 floor, so it is for local testing only and never for a release. The wrapper remains the blessed path.
- Add a short "Bumping the toolchain image" section: the digest lives in `.engine-build-image`; a bump is a one-line change; digests are per-run, not per-commit, so take the digest from the `Publish Engine Images` run you intend to consume; point at `base-docker-images/docs/consuming-engine-build.md` for what the image guarantees and for `gh attestation verify`.
- Line 102 (native QA section): same image, same wrapper.

- [ ] **Step 4: Update `CLAUDE.md` and `README.md`**

- `CLAUDE.md`, "Native shim (do this first)": `local_build_wrapper.sh` runs the pinned shared image rather than building one. **The "`build.sh` is container-only, there is no host fast path" claim must be rewritten, not merely reworded** — it was true because the script extracted an RPM and shelled out to `dnf`, and after Task 2 neither exists. The accurate replacement: the wrapper is the blessed path because it holds the glibc floor; `build.sh` on a suitably equipped host now works but yields a floor-breaking artifact.
- `README.md:250`: the `manylinux_2_28` sentence gains the shared-image reference — the floor is unchanged because the shared image is itself a manylinux_2_28 derivative.

- [ ] **Step 5: Run tests to verify they pass**

Run: `bash native/tests/docs_present.sh && bash native/tests/ci_workflow.sh && bash native/tests/build_config.sh`
Expected: three `PASS:` lines

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md docs/building.md README.md native/tests/docs_present.sh
git commit -m "docs: describe the shared engine-build image and where its pin lives"
```

---

### Task 6: End-to-end verification against the real image

**Files:** none expected; commit any fix the run forces.

**Interfaces:** consumes everything above.

- [ ] **Step 1: Confirm the digest resolves anonymously**

Run: `docker manifest inspect "$(cat .engine-build-image)"`
Expected: an OCI image index listing `linux/amd64` and `linux/arm64`, with no credential prompt.

- [ ] **Step 2: Build the shim inside the pinned image**

Run: `./native/local_build_wrapper.sh`
Expected: a multi-GB pull on first run, then the toolchain banner showing gcc 14.2.1 and ninja `1.13.0.git.kitware.jobserver-pipe-1`, and finally `Artifact: src/main/resources/native/linux-x86_64/libexecutorch_djl.so` plus the `Notices:` line. No `pip install`, no `rpm2archive`, no `dnf` anywhere in the log.

- [ ] **Step 3: Run native QA inside the pinned image**

Run: `./native/local_build_wrapper.sh native/build_qa.sh`
Expected: Catch2 suite passes, then three `et_leak_harness` runs pass under ASan/LSan. Then fix ownership, which this script does not do for itself:

```bash
sudo chown -R "$(id -u):$(id -g)" native/asan
```

- [ ] **Step 4: Run the JVM tests against the freshly staged `.so`**

Run: `./gradlew test`
Expected: BUILD SUCCESSFUL.

- [ ] **Step 5: Confirm the tree is clean and push**

Run: `git status --short`
Expected: no unexpected modifications (the staged `.so` under `src/main/resources/native/` is gitignored).

Then push the branch and open the PR. **CI's `linux-aarch64` row is the only proof of the arm64 half of the manifest list** — nothing local covers it, so watch that row specifically.

---

## Known gap

Local verification proves amd64 only. arm64 is proven by the `ubuntu-24.04-arm` CI row, or by running `./native/local_build_wrapper.sh` on the aarch64 host on the LAN — which the wrapper now supports unmodified, since it no longer picks a per-arch Dockerfile.
