# ExecuTorch Runtime — Engine Consumption Handover (Repo B)

> **What this is:** a cold-start hand-off for the agent working in the **DJL ExecuTorch engine**
> (Repo B — the out-of-tree engine that builds `libexecutorch_djl.so`). `executorch-runtime-dist`
> (**Repo A**) now builds, attests, and **publishes** the ExecuTorch runtime. The engine should
> **download** it instead of building it from source in its own CI.
>
> This document is your input: the **frozen contract** (C1–C9), the **one contract change** made
> during Repo A's implementation, your **punch-list**, and **concrete consumption recipes**. Assume no
> prior context. Producer repo: `measly-java-learning/executorch-runtime-dist`.

---

## 0. TL;DR — what changes for the engine

1. Repo A publishes **hash-pinned, attested** `et-install` tarballs per ET version (3 variants ×
   `linux-x86_64`) on a GitHub Release. The engine consumes those.
2. The engine's ET-runtime resolution becomes **3-way** (see §4): explicit `ET_INSTALL` escape hatch →
   build-from-source via Repo A → **default: `FetchContent` the pinned tarball**.
3. **`native/build.sh` drops Stage A** — the ET build recipe now lives in Repo A. `native/build_variants.sh`
   **downloads** the three variant tarballs instead of building them.
4. **One contract change** from the original plan (§3): Repo A's `build-runtime.sh` **no longer clones
   ExecuTorch** — the caller provides the source via `--et-src`. This only affects the engine's
   *from-source fallback*, which must now check out ET itself.

---

## 1. What's published (concrete)

- **Producer:** `measly-java-learning/executorch-runtime-dist`
- **Release tag:** `v1.3.1-2` (format `v<etver>-<pkgrev>`; `pkgrev` bumps on a re-roll of the same ET version)
- **Per release, for `variant ∈ {bare, logging, devtools}` × `platform = linux-x86_64`:**
  - `executorch-runtime-1.3.1-<variant>-linux-x86_64.tar.gz`
  - sibling `…​.tar.gz.sha256` (sha256sum format)
  - one GitHub **build-provenance attestation** per tarball
  - `EtRuntimePin.cmake` — a paste-in snippet carrying the URLs + SHA256s (see §5a)
- **Ship default variant:** `logging`.
- **Asset URL shape:**
  `https://github.com/measly-java-learning/executorch-runtime-dist/releases/download/v1.3.1-2/executorch-runtime-1.3.1-<variant>-linux-x86_64.tar.gz`

> Do **not** hardcode SHA256s in the engine — paste the release's `EtRuntimePin.cmake`, which is the
> single source of truth for URLs + hashes for that release.

---

## 2. The Contract (C1–C9) — frozen

- **C1 — Asset names:** `executorch-runtime-<etver>-<variant>-<platform>.tar.gz`, plus a sibling `<asset>.sha256`.
- **C2 — Tarball layout:** one top-level dir `executorch-runtime-<etver>-<variant>-<platform>/` containing
  **exactly** `lib/` (incl. `cmake/ExecuTorch/executorch-config.cmake`), `include/`, `LICENSE`,
  `THIRD-PARTY-NOTICES/`, `BUILDINFO`. The tree is **relocatable** (no absolute build-prefix in any
  `*.cmake`) and `lib/` binaries are **position-independent** (proven by linking into a `.so`).
- **C3 — Variants:** `bare` (LOGGING=OFF), `logging` (LOGGING=ON — **ship default**), `devtools`
  (LOGGING=OFF + DEVTOOLS=ON + EVENT_TRACER=ON).
- **C4 — Platform:** `linux-x86_64` (glibc ≥ 2.28 floor). The `<platform>` token scales to future targets.
- **C5 — `BUILDINFO`:** `key=value` lines — `et_version, et_commit, torch_version, variant, platform,
  cmake_flags, toolchain, build_utc, package_tag`. Informational (engine may log it); no functional consumer.
- **C6 — `EtRuntimePin.cmake`:** flat cmake `set()`s — `ET_RUNTIME_VERSION`, `ET_RUNTIME_ET_VERSION`, and per
  row `ET_RUNTIME_URL_<variant>_<platform>` + `ET_RUNTIME_SHA256_<variant>_<platform>`. **Primary consumer.**
- **C7 — Integrity/provenance:** `<asset>.sha256` (sha256sum format); one build-provenance attestation per
  tarball; verify via `gh attestation verify <tarball> --repo measly-java-learning/executorch-runtime-dist`.
- **C8 — From-source entrypoint:** `build-runtime.sh --variant <v> --prefix <dir> --et-src <et-checkout>
  [--et-tag <label>] [--build-dir <dir>]` → relocatable `et-install` at `<dir>`; non-zero on failure.
  **Two caller-owned boundaries:** (1) must run **inside `manylinux_2_28`** (container), and (2) the caller
  **provides the ExecuTorch checkout** (with submodules) via `--et-src` — **the recipe does not clone.**
  `--et-tag` is a version label (default `v1.3.1`); `et_commit` is read from the checkout. `--build-dir`
  defaults to `<dirname of --prefix>/et-build-<variant>` (persisted/inspectable). Env `SKIP_ET_BUILD=1`
  reuses an existing `--prefix` install (guarded by `executorch-config.cmake`), mirroring the engine's own
  Stage A flag. The engine clones Repo A at tag `v<etver>-<pkgrev>` (C9) to invoke it.
- **C9 — Release tag:** `v<etver>-<pkgrev>` (e.g. `v1.3.1-2`); `pkgrev` increments for re-rolls of the same ET version.

---

## 3. Contract Delta since the original hand-off (the ONE thing that changed)

- **C8 reshaped — caller-owned source (`--et-src`, no clone) + build reuse.** The original C8 had
  `build-runtime.sh` clone ExecuTorch itself. That was inconsistent with how it actually lands: in CI the
  source arrives via `actions/checkout`, and the engine's fallback likewise checks out ET. So `build-runtime.sh`
  now **requires `--et-src <checkout>`** (with submodules) and never clones — symmetric with the already
  caller-owned container boundary. `--et-tag` became a label only. `SKIP_ET_BUILD=1` reuses an existing
  `--prefix` install (keyed to the install prefix, no marker written — **C2 unaffected**).
  **Engine impact (⚠ C8 only):** the from-source fallback must `actions/checkout` (or `git clone --recursive`)
  `pytorch/executorch` at the pinned ET tag and pass `--et-src`, instead of relying on the recipe to fetch it.

Everything else in C1–C9 is unchanged from the original hand-off.

---

## 4. Your punch-list (Contract point → engine touch-point)

| C# | Engine artifact it drives | Status |
|----|---------------------------|--------|
| C1 | `EtRuntimePin.cmake` URL/filename construction; `native/build_variants.sh` download filenames | ok |
| C2 | `native/CMakeLists.txt`: `ET_INSTALL = <fetched>/executorch-runtime-<etver>-<variant>-<platform>`; `find_package` PATHS. `LICENSE`/`THIRD-PARTY-NOTICES/` are informational. | ok |
| C3 | `EtRuntimePin.cmake` rows; `build_variants.sh` variant list; `docs/benchmarking.md` variant meaning | ok |
| C4 | `EtRuntimePin.cmake` platform key; engine platform detection (`linux-x86_64` only for now) | ok |
| C5 | Informational only (engine may log it) | ok |
| C6 | `native/CMakeLists.txt`: `include(EtRuntimePin.cmake)` + read `ET_RUNTIME_*` + `FetchContent(URL, URL_HASH SHA256=…)`. **Primary consumer.** | ok |
| C7 | `FetchContent` `URL_HASH` uses the SHA; optional engine-CI `gh attestation verify` before use | ok |
| C8 | `native/build.sh` from-source fallback: clone Repo A @ pinned tag, enter `manylinux_2_28`, **check out `pytorch/executorch` with submodules**, then `build-runtime.sh --variant logging --prefix <tmp> --et-src <et-checkout>` | **⚠ source now caller-provided via `--et-src`; recipe no longer clones — the fallback must check out ET itself** |
| C9 | Engine from-source fallback clone tag; `EtRuntimePin.cmake` version comment | ok |

**The only ⚠ is C8.** Every other row is unchanged behavior.

### Implied engine work
The ET-runtime resolution in `native/CMakeLists.txt` / `native/build.sh` becomes **3-way**:
1. **`ET_INSTALL` provided** → use it as-is (unchanged escape hatch).
2. **build-from-source toggle** → clone Repo A @ pinned tag, check out ET, run `build-runtime.sh` **inside
   `manylinux_2_28`** with `--et-src` (C8), point `ET_INSTALL` at the produced prefix.
3. **default** → `FetchContent` the pinned `logging` / `linux-x86_64` tarball with `URL_HASH` (C1/C2/C6).

Plus: `native/build.sh` **drops Stage A** (the ET build recipe now lives in Repo A); `native/build_variants.sh`
**downloads** the three variant tarballs instead of building them.

---

## 5. Consumption recipes (concrete)

### 5a. Default — `FetchContent` the pinned tarball (C6, primary path)
Paste the release's `EtRuntimePin.cmake`, then in `native/CMakeLists.txt`:
```cmake
include(EtRuntimePin.cmake)   # sets ET_RUNTIME_URL_* / ET_RUNTIME_SHA256_* / ET_RUNTIME_VERSION
include(FetchContent)
FetchContent_Declare(et_runtime
  URL       "${ET_RUNTIME_URL_logging_linux-x86_64}"
  URL_HASH  "SHA256=${ET_RUNTIME_SHA256_logging_linux-x86_64}")
FetchContent_MakeAvailable(et_runtime)
set(ET_INSTALL "${et_runtime_SOURCE_DIR}/executorch-runtime-${ET_RUNTIME_ET_VERSION}-logging-linux-x86_64")
find_package(executorch CONFIG REQUIRED PATHS "${ET_INSTALL}/lib/cmake/ExecuTorch")
# link the JNI shim (a SHARED lib) against the `executorch` target
# ⚠ but the XNNPACK backend and the kernel/ops archives register via STATIC INITIALIZERS and
#    MUST be whole-archive'd at this final .so link, or they GC away — see §6 "Whole-archive".
```

### 5b. From-source fallback (C8) — **the ⚠ change is here**
Must run **inside `manylinux_2_28`**; the engine (caller) checks out ET and passes `--et-src`:
```bash
git clone --branch v1.3.1-2 https://github.com/measly-java-learning/executorch-runtime-dist runtime-dist
git clone --recursive --branch v1.3.1 https://github.com/pytorch/executorch et-src   # caller-owned source
export PATH=/opt/python/cp312-cp312/bin:$PATH
./runtime-dist/build-runtime.sh --variant logging --prefix "$PWD/et-install" --et-src "$PWD/et-src"
# ET_INSTALL="$PWD/et-install"; set SKIP_ET_BUILD=1 on a re-run to reuse an existing install
```

### 5c. Escape hatch
`ET_INSTALL` explicitly provided → use as-is; skip both fetch and build.

### 5d. Verify an artifact (C7)
```bash
sha256sum -c executorch-runtime-1.3.1-logging-linux-x86_64.tar.gz.sha256
gh attestation verify executorch-runtime-1.3.1-logging-linux-x86_64.tar.gz \
  --repo measly-java-learning/executorch-runtime-dist
```

---

## 6. Notes / gotchas

- **Tarball is exactly the C2 members** — no `bin/`, no `share/`, no dotfiles. The `find_package` target is
  `executorch`; consume it into a **SHARED** library (the whole point of the PIC guarantee).
- **⚠ Whole-archive the backend + kernel/ops archives at the JNI `.so` link (engine-side responsibility).**
  The tarball ships **relocatable static archives** — `--whole-archive`/`-force_load` is a property of the
  *final* link that produces `libexecutorch_djl.so`, so it cannot be baked into a `.a` and Repo A cannot do
  it for you. `libxnnpack_backend.a`, `libportable_ops_lib.a`, `liboptimized_native_cpu_ops_lib.a`,
  `libquantized_ops_lib.a` (and friends) are **pure static-initializer registration TUs**
  (`RegisterCodegenUnboxedKernelsEverything.cpp`, XNNPACK backend registration). Nothing references their
  symbols, so a normal archive link — especially with `--gc-sections` — drops them and you get
  **backend-not-found / unregistered-operator** errors at *load/execute* time even though everything
  compiled and linked cleanly. The shipped upstream `executorch-config.cmake` exposes these as plain target
  names in `EXECUTORCH_LIBRARIES` with **no** `--whole-archive` wrapping, so linking `${EXECUTORCH_LIBRARIES}`
  — or just the `executorch` target — does **not** protect you. Wrap them explicitly at the JNI link:
  ```cmake
  target_link_libraries(executorch_djl PRIVATE
    executorch
    -Wl,--whole-archive
      xnnpack_backend portable_ops_lib optimized_native_cpu_ops_lib quantized_ops_lib
    -Wl,--no-whole-archive)
  # Apple: -Wl,-force_load,<path/to/libxnnpack_backend.a> per archive instead of --whole-archive.
  ```
  Wrap only the registration archives you actually need for the shipped variant (at minimum
  `xnnpack_backend` + the portable/optimized ops lib); over-wrapping just inflates the `.so`. This is the
  same fix already applied on the JNI path — it re-emerges here because a setuptools/CMake extension links
  differently. Repo A's PIC gate links only the `executorch` target and never loads a model, so it will
  **not** catch a whole-archive regression for you.
- **glibc ≥ 2.28 floor** — comes from the `torch==2.12.0+cpu` wheel used to build. Consumers must be on
  glibc ≥ 2.28 (RHEL8 / AlmaLinux 8 / Ubuntu 20.04+).
- **BUILDINFO `cmake_flags`** records the exact effective build flags (including `-DCMAKE_BUILD_TYPE=Release`)
  from a single source shared with the build — useful for auditing what produced the artifact.
- **Upstream ET bug:** Repo A carries a workaround patch for an ExecuTorch install-destination bug
  (`pytorch/executorch#20709`) — the from-source path (§5b) applies it automatically; irrelevant to
  tarball consumption. When ET merges the fix and the engine bumps to an ET tag that includes it, the patch
  becomes a no-op.
- **New ET version / re-roll:** Repo A pushes a new `v<etver>-<pkgrev>` tag; the engine updates its pasted
  `EtRuntimePin.cmake` and the fallback clone tag accordingly.
