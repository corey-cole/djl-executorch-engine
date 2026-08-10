# ExecuTorch Runtime Distribution (`executorch-runtime-dist`) — Design & Hand-off

> **What this is:** a cold-start hand-off spec for a **new agent** implementing
> `measly-java-learning/executorch-runtime-dist` (referred to here as **Repo A**). It assumes no
> prior context. Three delineated parts:
> 1. **Repo A implementation spec** — what to build.
> 2. **The Contract (C1–C9)** — the interface the engine (**Repo B**) depends on; treat as frozen.
> 3. **The Repo B Impact Map** — which engine touch-point each contract point drives, with a
>    Contract Deltas log for anything you're *forced* to change.
>
> **Receiving agent:** plan and implement Part 1 yourself (run your own writing-plans pass). This
> document is your input, not your plan. Treat Part 2 as fixed; if implementation forces a change,
> follow the drift rule in Part 3 — that is the *only* thing that flows back to Repo B.

---

## Background (why Repo A exists)

An out-of-tree DJL engine (Repo B, in a personal namespace — **out of scope to move**) loads a
custom JNI shim (`libexecutorch_djl.so`) that statically links an **ExecuTorch runtime**. Producing
that runtime is a heavy, ~10-minute build: it runs in a `manylinux_2_28` container, downloads
`torch==2.12.0` (whose wheel sets a **glibc ≥ 2.28** floor), and compiles ExecuTorch from source.
It changes rarely — only on an ET version bump or a build-flag change.

Repo A extracts that build so it runs **once per ET version**, publishing the resulting `et-install`
tree as versioned, hash-pinned, attested GitHub Release tarballs. Repo B then **downloads** the
runtime instead of building it, cutting the ET compile out of the engine's CI entirely.

**Supply-chain posture (non-negotiable):** Repo B deliberately never commits native binaries. Repo A
must therefore be the **sole, auditable builder** — every artifact SHA-256-pinned *and* accompanied
by a GitHub build-provenance attestation, so "my CI built this from this commit" is *verifiable*, not
promised.

---

# Part 1 — Repo A implementation spec

## Scope

**In:** build the ET runtime from source in `manylinux_2_28`, for three variants
(`bare`/`logging`/`devtools`) on one platform (`linux-x86_64`); package each as a **relocatable**
`et-install` tarball + `.sha256` + `BUILDINFO`; on a manual version tag, run a CI matrix that builds,
attests, and publishes a GitHub Release and emits a ready-to-paste `EtRuntimePin.cmake` snippet.

**Out:** the engine consumption side (Repo B); model-export tooling; multi-platform builds (design
the *naming* to scale, but only `linux-x86_64` is built now).

## The build recipe (migrated verbatim from the engine's Stage A)

Repo A owns the canonical recipe. It is exactly what the engine's `build.sh` Stage A does today, plus
the `ET_VARIANT` flag map. Reproduce faithfully:

**Environment:** `quay.io/pypa/manylinux_2_28_x86_64` (AlmaLinux 8; gcc-toolset-14). Use cp312:
`export PATH=/opt/python/cp312-cp312/bin:$PATH`.

**ExecuTorch source:** clone `https://github.com/pytorch/executorch` at the tag matching the release's
ET version (e.g. `v1.3.1`) **with submodules** (`git clone --recursive`, or
`git submodule update --init --recursive`) — the ET build links in-tree third-party (flatcc, XNNPACK,
etc.). Do **not** rely on `install_requirements.sh`; the working recipe installs deps directly (below).

**Python deps (in the container):**
```bash
pip install ninja
pip install -U pip setuptools wheel pyyaml
pip install torch==2.12.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

**Variant flag map** (`ET_VARIANT` → cmake flags):

| variant | flags |
|---|---|
| `bare`     | `-DEXECUTORCH_ENABLE_LOGGING=OFF` |
| `logging`  | `-DEXECUTORCH_ENABLE_LOGGING=ON` |
| `devtools` | `-DEXECUTORCH_ENABLE_LOGGING=OFF -DEXECUTORCH_BUILD_DEVTOOLS=ON -DEXECUTORCH_ENABLE_EVENT_TRACER=ON` |

**Configure / build / install:**
```bash
cmake -B "${ET_BUILD}" -S "${ET_SRC}" -G Ninja --preset linux \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  "${ET_VARIANT_FLAGS[@]}" \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON
cmake --build "${ET_BUILD}" -j"$(nproc)"
cmake --install "${ET_BUILD}" --prefix "${PREFIX}"   # emits lib/cmake/ExecuTorch/executorch-config.cmake
```

> **devtools flag risk (known unknown):** the `devtools` combo (`EXECUTORCH_BUILD_DEVTOOLS=ON` +
> `EXECUTORCH_ENABLE_EVENT_TRACER=ON`) has **not** been exercised against a real ET 1.3.x configure.
> It likely pulls additional required targets (`etdump`, `flatccrt`). Expect the first `devtools`
> build to fail; resolve the flag set against ET 1.3.x's devtools docs and pin it in the recipe's
> `devtools` arm. This is anticipated, not a regression.

### Single recipe entrypoint (used by BOTH CI and Repo B)

Factor the recipe into **`build-runtime.sh`** so CI and Repo B's from-source fallback call the *same*
code (this is what guarantees a local from-source runtime matches the released one):

```
./build-runtime.sh --variant <bare|logging|devtools> --prefix <install-dir> [--et-tag <tag>]
```

- Produces a **relocatable** `et-install` at `<install-dir>` (see next section).
- Runs inside `manylinux_2_28`. Non-zero exit on any failure.
- `--et-tag` defaults to the ET version the repo is currently targeting.

This entrypoint's name, flags, and prefix semantics are **contract point C8** — Repo B invokes it.

## Relocatability — the FIRST milestone, gates everything

A tarball is worthless if `find_package(executorch)` breaks when extracted to a path other than the
build machine's `CMAKE_INSTALL_PREFIX`. CMake package configs sometimes bake **absolute** paths into
`lib/cmake/ExecuTorch/*.cmake` (`IMPORTED_LOCATION`, `INTERFACE_INCLUDE_DIRECTORIES`).

**Step 1 — measure.** After a `logging` install, grep the generated configs for the absolute prefix:
```bash
grep -rn "${PREFIX}" "${PREFIX}/lib/cmake/"    # any hit = non-relocatable
```
Relocatable configs derive paths from `${CMAKE_CURRENT_LIST_DIR}` / `${PACKAGE_PREFIX_DIR}` and show
no absolute build-prefix hits.

**Step 2 — mitigate if needed.** If absolute paths are present, post-process the generated `*.cmake`
in the staging prefix to recompute paths relative to `${CMAKE_CURRENT_LIST_DIR}` before tarring. (Do
not attempt to make the consumer replicate the build prefix — that's the fragile anti-pattern.)

**Step 3 — acceptance gate (required test).** Extract a built tarball to a **different** directory and
prove consumption:
```bash
tar -C /tmp/relotest -xzf executorch-runtime-1.3.1-logging-linux-x86_64.tar.gz
# tiny consumer: find_package(executorch CONFIG REQUIRED PATHS /tmp/relotest/<stem>/lib/cmake/ExecuTorch)
#                + link one target that pulls an ET lib; configure+build must succeed
```
This extract-elsewhere-and-`find_package` test is the go/no-go for the whole approach. Build it early.

## Artifacts

### Tarball layout (C2)
Each tarball contains a **single top-level directory** named after the asset stem:
```
executorch-runtime-<etver>-<variant>-<platform>/
  lib/            # incl. cmake/ExecuTorch/executorch-config.cmake (relocatable)
  include/
  BUILDINFO
```
So the consumer's `ET_INSTALL` is `<extracted>/executorch-runtime-<etver>-<variant>-<platform>`, and
`executorch-config.cmake` sits at `<ET_INSTALL>/lib/cmake/ExecuTorch/`.

### `BUILDINFO` (C5)
Plain `key=value` lines (human/greppable; not machine-consumed by Repo B):
```
et_version=1.3.1
et_commit=<full sha of the ET checkout>
torch_version=2.12.0+cpu
variant=logging
platform=linux-x86_64
cmake_flags=-DEXECUTORCH_ENABLE_LOGGING=ON ...   # the full effective flag set
toolchain=manylinux_2_28 gcc-toolset-14
build_utc=2026-07-02T00:00:00Z
package_tag=v1.3.1-1
```

### Checksums + attestation (C7)
- `<asset>.sha256` in `sha256sum` format (hash + filename), one per tarball.
- Each tarball gets a GitHub build-provenance attestation via `actions/attest-build-provenance`.
- Verify: `gh attestation verify <tarball> --repo measly-java-learning/executorch-runtime-dist`.

### Pin snippet (C6)
CI's final job emits an `EtRuntimePin.cmake` (a **release asset** and a job-summary block) that Repo B
pastes in wholesale. Flat `set()` variables keyed by `<variant>_<platform>`:
```cmake
# Generated by executorch-runtime-dist release v1.3.1-1. Do not edit by hand.
set(ET_RUNTIME_VERSION "1.3.1-1")
set(ET_RUNTIME_ET_VERSION "1.3.1")

set(ET_RUNTIME_URL_logging_linux-x86_64
  "https://github.com/measly-java-learning/executorch-runtime-dist/releases/download/v1.3.1-1/executorch-runtime-1.3.1-logging-linux-x86_64.tar.gz")
set(ET_RUNTIME_SHA256_logging_linux-x86_64 "<sha256>")
# ...bare and devtools rows, same shape...
```

## CI / release pipeline

- **Trigger:** push of tag `v<etver>-<pkgrev>` (e.g. `v1.3.1-1`) — **manual**, you decide when to
  support a new ET. (C9)
- **Build matrix:** `variant ∈ {bare, logging, devtools}` × `platform ∈ {linux-x86_64}`, each job in
  the `manylinux_2_28` container calling `build-runtime.sh --variant <v> --prefix <stage>`.
- **Per job:** build → assemble the C2 tarball → `sha256sum` → `attest-build-provenance`.
- **Aggregate job:** create/update the GitHub Release for the tag; upload all tarballs + `.sha256`;
  generate and attach the `EtRuntimePin.cmake` snippet (C6) covering every `(variant, platform)`.
- Permissions: the attestation action needs `id-token: write` and `attestations: write`.

## Repo A file structure

- `build-runtime.sh` — the recipe entrypoint (C8); the migrated Stage A + variant map.
- `scripts/` — package a prefix into the C2 tarball; emit `BUILDINFO`; generate `EtRuntimePin.cmake`.
- `.github/workflows/release.yml` — the matrix build + attest + publish + pin emission.
- `test/relocatability.sh` — the extract-elsewhere + `find_package` gate.
- `README.md` — what the repo is, how to cut a release, how to verify an artifact.

## Testing / acceptance

1. **Relocatability gate** (above) — extract-elsewhere + `find_package` + link, on `logging`.
2. **`build-runtime.sh` local dry run** for one variant → a usable prefix.
3. **CI dry run on a pre-release tag** → all six assets (3 tarballs + 3 sha256) + attestations, and a
   valid `EtRuntimePin.cmake`; `gh attestation verify` passes on each tarball.

---

# Part 2 — The Contract (C1–C9)

> This block is the interface Repo B depends on. **Frozen.** If implementation forces a change,
> apply the drift rule in Part 3. Written to be liftable into a standalone `et-runtime-contract.md`
> if it ever needs to graduate.

- **C1 — Asset names:** `executorch-runtime-<etver>-<variant>-<platform>.tar.gz`, plus a sibling
  `<asset>.sha256`.
- **C2 — Tarball layout:** one top-level dir `executorch-runtime-<etver>-<variant>-<platform>/`
  containing `lib/` (with `cmake/ExecuTorch/executorch-config.cmake`), `include/`, and `BUILDINFO`.
  The tree is **relocatable** (no absolute build-prefix paths in any `*.cmake`).
- **C3 — Variants:** `bare` (LOGGING=OFF), `logging` (LOGGING=ON), `devtools` (LOGGING=OFF +
  DEVTOOLS=ON + EVENT_TRACER=ON). **`logging` is the engine's ship default.**
- **C4 — Platforms:** `linux-x86_64` (glibc ≥ 2.28 floor). The `<platform>` token scales to future
  targets.
- **C5 — `BUILDINFO`:** `key=value` lines with keys `et_version, et_commit, torch_version, variant,
  platform, cmake_flags, toolchain, build_utc, package_tag`.
- **C6 — `EtRuntimePin.cmake`:** flat cmake `set()`s: `ET_RUNTIME_VERSION`, `ET_RUNTIME_ET_VERSION`,
  and per row `ET_RUNTIME_URL_<variant>_<platform>` + `ET_RUNTIME_SHA256_<variant>_<platform>`.
- **C7 — Integrity/provenance:** `<asset>.sha256` (sha256sum format); GitHub build-provenance
  attestation per tarball; `gh attestation verify <tarball> --repo measly-java-learning/executorch-runtime-dist`.
- **C8 — From-source entrypoint:** `build-runtime.sh --variant <v> --prefix <dir> [--et-tag <tag>]`
  → relocatable `et-install` at `<dir>`; runs in `manylinux_2_28`; non-zero on failure. Repo B clones
  Repo A at tag `v<etver>-<pkgrev>` (C9) to invoke it.
- **C9 — Release tag:** `v<etver>-<pkgrev>` (e.g. `v1.3.1-1`); `pkgrev` increments for re-rolls of the
  same ET version.

---

# Part 3 — Repo B Impact Map

> **Drift rule (for the Repo A agent):** the Contract is frozen. If you are *forced* to change a
> `Cn`, do two things: (1) append a **Contract Delta** entry below; (2) mark that row's Impact
> column with `⚠`. The set of `⚠` rows is the complete, targeted punch-list for Repo B — nothing
> else in Repo B can be affected by your work.

### Contract Deltas (append-only; empty at hand-off)

_None yet._

### Map: Contract point → Repo B touch-point

| C# | Repo B artifact it drives | Status |
|----|---------------------------|--------|
| C1 | `EtRuntimePin.cmake` URL/filename construction; `native/build_variants.sh` download filenames | ok |
| C2 | `native/CMakeLists.txt`: `ET_INSTALL` derived as `<fetched>/executorch-runtime-<etver>-<variant>-<platform>`; `find_package` PATHS | ok |
| C3 | `EtRuntimePin.cmake` rows; `build_variants.sh` variant list; `docs/benchmarking.md` variant meaning | ok |
| C4 | `EtRuntimePin.cmake` platform key; engine platform detection (currently `linux-x86_64` only) | ok |
| C5 | Informational only — no functional consumer (engine may log it). Low. | ok |
| C6 | `native/CMakeLists.txt`: `include(EtRuntimePin.cmake)` + read `ET_RUNTIME_*` vars + `FetchContent(URL, URL_HASH SHA256=…)`. **Primary consumer.** | ok |
| C7 | `FetchContent` `URL_HASH` uses the SHA; optional engine-CI step running `gh attestation verify` before use | ok |
| C8 | `native/build.sh` from-source fallback: clone Repo A at the pinned tag, invoke `build-runtime.sh --variant logging --prefix <tmp>` | ok |
| C9 | Engine from-source fallback clone tag; `EtRuntimePin.cmake` version comment | ok |

### Repo B work these imply (for the later Repo B spec — NOT this agent's job)

The engine's ET-runtime resolution becomes 3-way in `native/CMakeLists.txt` / `build.sh`:
1. `ET_INSTALL` provided → use it (unchanged escape hatch).
2. build-from-source toggle → clone Repo A @ pinned tag, run `build-runtime.sh` (C8), point `ET_INSTALL` at it.
3. default → `FetchContent` the pinned `logging`/`linux-x86_64` tarball with `URL_HASH` (C1/C2/C6).
Plus: `build.sh` **drops Stage A** (recipe now lives in Repo A); `build_variants.sh` **downloads** the
three variant tarballs instead of building them.

---

## Hand-off checklist for the receiving agent

- [ ] Read Part 1; run your own writing-plans pass to produce Repo A's implementation plan.
- [ ] Prove the **relocatability gate** before anything else — it can invalidate C2.
- [ ] Keep Part 2 frozen; log any forced change as a Contract Delta + `⚠` the mapped row.
- [ ] On completion, the `⚠` rows (if any) + the Deltas log are what returns to the engine repo.
