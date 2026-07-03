# Engine Runtime Consumption — Design

> **What this is:** the design for reshaping the DJL ExecuTorch engine (Repo B) to **consume**
> the attested, hash-pinned `et-install` tarballs published by `measly-java-learning/executorch-runtime-dist`
> (Repo A), instead of building the ExecuTorch runtime from source in its own CI.
>
> **Input:** `docs/handover-to-engine.md` (the frozen Contract C1–C9 + the one C8 delta). This
> design is the engine-side response to that hand-off.

---

## Goal

Stop building the ExecuTorch runtime inside the engine. The engine's native build should **download
a prebuilt, hash-pinned, attested `et-install` tarball** and link the JNI shim against it. Building the
runtime from source becomes a *documented procedure* (run Repo A's `build-runtime.sh`, point the engine
at the result), not engine code.

## Architecture (the seven decisions)

1. **CMake owns runtime resolution** — `native/CMakeLists.txt` resolves the runtime declaratively via
   `FetchContent`, treading CMake's paved path rather than a hand-rolled shell downloader.
2. **Two resolution paths only** — `ET_INSTALL` escape hatch, or `FetchContent` the pinned tarball. No
   automated from-source path in the engine (Repo A owns that recipe; escape hatch consumes its output).
3. **Container stays the blessed default** — the shipped shim must keep its **glibc-2.28 floor** (RHEL8
   users). Dropping Stage A makes the container *fast*, not optional. A no-Docker native shim build is a
   documented "local test only, breaks the floor, don't ship it" shortcut.
4. **Pin file committed verbatim, variant via cache var** — `native/cmake/EtRuntimePin.cmake` carries all
   three variant rows; `ET_RUNTIME_VARIANT` (default `logging`) selects one; hand-bumped per Repo A release.
5. **Benchmarking reshaped to downloads** — `build_variants.sh` loops over the three variants via
   `-DET_RUNTIME_VARIANT`, `FetchContent`-ing each prebuilt tarball; all from-source machinery is deleted.
6. **ET build knowledge distilled to a doc** — `docs/executorch-build-notes.md` preserves the *reasoning*;
   the runnable recipe now lives in Repo A.
7. **Two-tier supply-chain verification** — `URL_HASH SHA256` always (integrity, local + CI); `gh
   attestation verify` provenance in CI's GHA YAML; a documented local `gh` command as the opt-in floor.

---

## Component changes

### A. `native/CMakeLists.txt` — resolution logic (the core change)

Replace the hardcoded `ET_INSTALL` default (`native/CMakeLists.txt:11`) + single `find_package` with a
two-branch resolution:

```cmake
include(cmake/EtRuntimePin.cmake)     # ET_RUNTIME_ET_VERSION, ET_RUNTIME_URL_*, ET_RUNTIME_SHA256_*
set(ET_RUNTIME_VARIANT "logging" CACHE STRING "Runtime variant: logging (ship) | bare | devtools (bench)")

if(ET_INSTALL)
  # Escape hatch: caller supplied an install tree (e.g. a from-source build). Use as-is.
else()
  include(FetchContent)
  set(_plat "linux-x86_64")
  FetchContent_Declare(et_runtime
    URL      "${ET_RUNTIME_URL_${ET_RUNTIME_VARIANT}_${_plat}}"
    URL_HASH "SHA256=${ET_RUNTIME_SHA256_${ET_RUNTIME_VARIANT}_${_plat}}")
  FetchContent_MakeAvailable(et_runtime)
  # FetchContent strips the tarball's single top-level dir on extraction, so SOURCE_DIR IS the install root.
  set(ET_INSTALL "${et_runtime_SOURCE_DIR}")
endif()

list(PREPEND CMAKE_PREFIX_PATH "${ET_INSTALL}")
find_package(executorch CONFIG REQUIRED PATHS "${ET_INSTALL}/lib/cmake/ExecuTorch")
```

Everything past `find_package` (the `et_runtime` static lib, the shim, QA/bench targets) is unchanged.

**Integration flag to verify during implementation:** the current `tokenizers_DIR` hint
(`native/CMakeLists.txt:14`) points into the install tree. Contract C2 says the tarball contains only
`lib/` + `include/` + `LICENSE` + `THIRD-PARTY-NOTICES/` + `BUILDINFO`. Confirm whether the tarball ships
`lib/cmake/tokenizers`:
- If yes → keep the line, repointed at the resolved `ET_INSTALL`.
- If no → drop the line (`find_package(executorch)` does not require it). Raise a **C2 clarification** to
  Repo A's Impact Map if tokenizers is expected but absent.

### B. `native/build.sh` — drop Stage A

**Delete:** the ExecuTorch runtime build (torch wheel install, ET cmake configure/build/install), the
`ET_VARIANT` flag map, `ET_BUILD`, `SKIP_ET_BUILD`, the `executorch-config.cmake` precondition check on a
prebuilt `ET_INSTALL`.

**Keep:** JDK-header extraction (Corretto RPM → jni.h), the AVX512-VNNI toolchain assertion, the shim
`cmake` configure + build, `STAGE_SO` staging into `src/main/resources/native/linux-x86_64/`, the
`HOST_UID`/`HOST_GID` chown-on-EXIT trap, the `GITHUB_ENV` export of `JAVA_HOME`.

The tarball download now happens **inside the shim's `cmake configure`** (`FetchContent` in component A),
which still runs in-container → the fetched runtime is linked on glibc 2.28 → the staged shim keeps its
floor. `build.sh` forwards `-DET_RUNTIME_VARIANT` when the env var is set. `NATIVE_BUILD_DIR`,
`PRINT_ET_FLAGS`-equivalent diagnostics, and path overrides are revisited (much of the flag surface
disappears with Stage A).

### C. `native/local_build_wrapper.sh` — drop the ET mount

Remove `-v ${ET_ROOT}:/workspace/executorch` and the "ExecuTorch checked out" prerequisite — nothing
builds ET anymore. Keep the Corretto RPM download and the `manylinux_2_28` `docker run`. The wrapper stays
the **blessed default** (guarantees a shippable, 2.28-floor `.so`), now fast because Stage A is gone.

Document the **native fast path** separately: run `native/build.sh` directly on the host (no Docker) for a
quick local `.so` to run `./gradlew test` — explicitly flagged as **floor-breaking, not shippable**.

### D. `native/build_variants.sh` — reshape to downloads

Loop over `bare`/`logging`/`devtools`, driving `-DET_RUNTIME_VARIANT=<v>` so `FetchContent` pulls each
prebuilt tarball; build the timing harness (`ET_BUILD_BENCH`) against each; run `bench.sh`; keep the
comparison table + deltas-vs-bare. **Delete** the from-source machinery: the `ET_VARIANT` flag map, torch
wheel reuse, `SKIP_ET_BUILD`, `STAGE_SO`, per-variant `et-install-*`/`et-cmake-out-*` build trees.
`bench.sh`, `ET_BUILD_BENCH`, and the timing harness are unchanged. Ongoing value: cheap re-benchmark on
each ET version bump.

### E. Supply-chain verification

- **Integrity (always):** `URL_HASH SHA256` in `FetchContent` (component A) — the download must match the
  SHA committed in `EtRuntimePin.cmake`. Runs everywhere, no tooling required. A pin change is a visible,
  reviewable commit.
- **Provenance (CI):** a `gh attestation verify <tarball> --repo measly-java-learning/executorch-runtime-dist`
  step **in the engine's GHA workflow YAML**, gating the shim build. This is where a malicious-PR-swapped
  binary is caught — the threat the hash alone can't stop.
- **Provenance (local floor):** **documentation only.** README / build-notes carry the copy-pasteable
  `gh attestation verify` command for opt-in local checking. No engine-owned verify script.

### F. `docs/executorch-build-notes.md` — distilled knowledge (new)

Preserve the engine-side *reasoning* being removed with Stage A, opening with: "the runnable recipe now
lives in `measly-java-learning/executorch-runtime-dist`; this file preserves the engine-side *why*."
Content: the glibc-2.28 floor rationale (torch 2.12 wheel), the AVX512-VNNI assertion as a **build-host**
encode check (not a runtime requirement), the JDK-headers-only extraction trick, the flatcc/install-
destination gotcha (`pytorch/executorch#20709`), the torch-2.12 pin. Curated prose, not a runnable
duplicate recipe (recreating shell from this doc is cheap).

### G. Housekeeping

- **`.gitignore`:** `et-install*/` and `et-cmake-out*/` are no longer produced by the default build — drop
  or leave harmless. Add the `FetchContent` cache/build dir under `native/build`.
- **README §1–§3 rewrite:** default build = fast fetch + shim compile; container-default rationale (glibc
  floor / RHEL8); the native fast-path caveat; the CI provenance step + the local `gh` command; the
  pin-bump procedure.
- **Pin-bump procedure (documented):** on a new Repo A `v<etver>-<pkgrev>`, replace
  `native/cmake/EtRuntimePin.cmake` with the release's published copy and update any fallback-tag reference.
  One reviewable commit; the SHA256 change is the supply-chain gate.

---

## Contract feedback to Repo A (Impact Map)

- **⚠ C8 resolves to "no engine code."** The hand-off flagged C8 (from-source `--et-src`) as the one engine
  touch-point needing work. Decision Q2 (two paths only) means the engine **never invokes**
  `build-runtime.sh` — the from-source story is a documented procedure in `executorch-build-notes.md`. Record
  in Repo A's Impact Map that C8's engine impact is docs-only, not orchestration code.
- **Possible C2 clarification:** whether the tarball ships `lib/cmake/tokenizers` (see component A). Only
  raised if the engine turns out to need it.

## Out of scope

- Automated from-source builds in the engine (Repo A owns the recipe; escape hatch consumes its output).
- Multi-platform (`linux-x86_64` only; the `<platform>` token in the pin scales later).
- Moving the build-notes doc into Repo A (may happen eventually; stays in the engine for now).
- Any change to the JNI shim, `et_runtime` core, or the QA/leak harness beyond the runtime-resolution seam.
