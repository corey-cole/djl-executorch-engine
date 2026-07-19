# Windows static CRT (`/MT`) adoption — design

**Date:** 2026-07-18
**Source:** `docs/handover-windows-static-cxx17.md` (work order from `measly-java-learning/executorch-runtime-dist`)
**Scope:** Stream A (adopt the `windows-x86_64-static` runtime row) with Stream B (C++17 propagation)
folded in as a verification-and-documentation item.

---

## 1. Goal

Ship a Windows JNI DLL that folds the C runtime in statically, so end users need **no VC++
redistributable**. Windows consumers of this engine are developers on potentially locked-down
workstations who may not be able to run an installer.

The producer repo now publishes two Windows rows. The engine must select the `/MT` one:

| Pin row | CRT | Intended consumer |
|---|---|---|
| `logging_windows-x86_64` | `/MD` dynamic | CPython extensions (must match CPython's CRT) |
| `logging_windows-x86_64-static` | `/MT` static | **the JNI DLL — this engine** |

Non-goals: no change to any Linux behaviour; no change to the Java-side platform token; no upstream
issue filed against `pytorch/executorch` (that is the producer repo's open item).

---

## 2. Evidence gathered before design

These were measured, not assumed, by downloading both `v1.3.1-8` Windows tarballs and inspecting
them from Linux.

| Check | Result |
|---|---|
| File trees, dynamic vs static | **Identical** — same 20 `.lib` files, same `lib/cmake/ExecuTorch` (no `tokenizers`/`absl`/`re2`/`pcre2`, no `ETNPExtras`, on either row) |
| CRT directives (`strings` over the `.lib`s, grepping `DEFAULTLIB`) | dynamic: 1288× `MSVCRT`, static: 1288× `LIBCMT`; zero crossover in either direction |
| static `BUILDINFO` | `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded`, `CMAKE_BUILD_TYPE=Release`, `EXECUTORCH_BUILD_XNNPACK=ON`, `toolchain=msvc-2022`, `torch_version=2.12.0+cpu` |
| static tarball SHA256 | matches the `EtRuntimePin.cmake` asset from `v1.3.1-8` |

Two consequences that shape the design:

1. **Identical target surface** means the pin-row swap is a pure string change. No `find_package`,
   `ETNPExtras`, or link-line restructuring is implied. The existing unconditional
   `set(tokenizers_DIR ...)` at `native/CMakeLists.txt:66` remains harmless on Windows, exactly as
   it is today.
2. **The A4 CRT scan does not require a Windows runner for the runtime side.** MSVC records
   `/DEFAULTLIB` directives as plain strings inside the archive, so `strings` reproduces what
   `dumpbin -directives` reports. The *shim's* own `.lib` is produced on Windows and still needs
   `dumpbin` there.

`v1.3.1-8` is the current latest release; there is no public `-7`.

---

## 3. Phasing

Sequenced so that each phase is independently reviewable and the CMake design is settled only after
the unknowns are measured.

### Phase 1 — Pin bump to `v1.3.1-8` (standalone commit)

Replace `native/cmake/EtRuntimePin.cmake` wholesale with the release asset, then re-apply the
existing comment header (the file is generated; this is the documented bump procedure). The header
gains a line explaining the two Windows rows and which one the engine resolves.

**This is not a Windows-only diff.** `v1.3.1-8` rebuilt every artifact — all six Linux hashes
changed as well as the Windows one. The bump therefore re-triggers the full supply-chain review gate
and requires a Linux rebuild plus `./gradlew test` before it lands, independent of any `/MT` work.

The `/MD` row stays in the file and stays reachable by explicit override. It is not dead weight: it
is what `native/tests/cmake_resolution.sh` uses to prove row selection is a real choice rather than a
hardcode.

### Phase 2 — Winbox spike (throwaway, nothing committed)

Requires **zero source edits**. `ET_PLATFORM` and `ET_RUNTIME_VARIANT` are already `CACHE` variables
and `CMAKE_MSVC_RUNTIME_LIBRARY` is a standard CMake cache variable, so the whole spike is
command-line overrides on top of Phase 1:

```
-DET_PLATFORM=windows-x86_64-static -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
```

It must answer exactly four questions:

- **S1 — does the QA path link?** `native/build_qa.sh` builds Catch2 from source via `FetchContent`.
  This is the highest-risk item in the change, because per the work order's A3 a CRT mismatch links
  **silently** — measured, with no `LNK2005` and not even an `LNK4098`.
- **S2 — is the redistributable actually gone?** `dumpbin -dependents executorch_djl.dll` must show
  no `VCRUNTIME140.dll` and no `MSVCP140.dll`.
- **S3 — does it still work?** Catch2 suite green (which is also what proves XNNPACK registered on
  Windows, since the ELF `nm` guard cannot run there), and `./gradlew test` green.
- **S4 — Stream B / B5.** `cmake --build <dir> -- -v 2>&1 | grep -o '/std:c++[0-9a-z]*' | sort -u`
  must show `/std:c++20` reaching the translation units that include ExecuTorch headers.

If S1 fails, Phase 3's CRT mechanism is revisited before anything lands.

### Phase 3 — Land the change

Five edits, all located.

#### 3.1 `native/CMakeLists.txt` — split the double-duty variable

`ET_PLATFORM` currently serves two roles at once: the pin-row lookup key, and the identity of "the
Windows platform" used in comparisons. The `-static` suffix separates those roles, so the variable
splits:

- `ET_PLATFORM` — **identity**, stays `windows-x86_64` / `linux-x86_64`.
- `ET_RUNTIME_ROW` — **pin lookup key and stem**, defaults to `windows-x86_64-static` on `WIN32`
  and `linux-x86_64` otherwise. Both remain `CACHE` variables so the `ET_PRINT_RESOLUTION` host-test
  seam can force either.

Rationale over simply changing `ET_PLATFORM`'s value: the variant guard at line 37
(`ET_PLATFORM STREQUAL "windows-x86_64"`) would silently stop matching, converting its actionable
"windows ships logging only" error back into the generic "No pin row" message the guard was written
to prevent. Splitting keeps every existing comparison correct by construction.

Only the lookup and stem move to the new variable:

```cmake
set(_ET_STEM "executorch-runtime-${ET_RUNTIME_ET_VERSION}-${ET_RUNTIME_VARIANT}-${ET_RUNTIME_ROW}")
set(_ET_RESOLVED_URL "${ET_RUNTIME_URL_${ET_RUNTIME_VARIANT}_${ET_RUNTIME_ROW}}")
set(_ET_RESOLVED_SHA "${ET_RUNTIME_SHA256_${ET_RUNTIME_VARIANT}_${ET_RUNTIME_ROW}}")
```

`_ET_STEM` feeds only the `ET_PRINT_RESOLUTION` diagnostic — `FetchContent` strips the tarball's
top-level directory, so nothing on the real build path depends on it. That bounds the blast radius of
getting the stem wrong to a test assertion.

#### 3.2 CRT selection — global cache variable, not per-target

```cmake
if(WIN32)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded" CACHE STRING "Static CRT: ship without a VC++ redist")
endif()
```

Set before the first `FetchContent_MakeAvailable`, i.e. near the top of the file.

**This deliberately diverges from the work order's A2**, which prescribes per-target
`set_property(TARGET <t> PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")`. Per-target is insufficient
here: Catch2 enters the build through `FetchContent_MakeAvailable`, so its targets are not declared
by this project and cannot have a property set on them. The global variable is the initializer for
every target's `MSVC_RUNTIME_LIBRARY` property, including targets created inside added
subdirectories, which is the only mechanism that covers the FetchContent case.

A2's underlying requirement — *every* object linked into the DLL must be `/MT`, because a `/MD`
static library inside a `/MT` DLL reintroduces exactly the problem being removed — is fully honoured;
only the mechanism differs. `CMP0091` is already `NEW`, since `cmake_minimum_required(VERSION 3.24)`
exceeds the 3.15 threshold.

#### 3.3 `native/build.sh` — correct a comment that becomes false

The Windows leg passes `-DCMAKE_BUILD_TYPE=Release` and justifies it in a comment stating the pinned
runtime is built `/MD`. After this change that is wrong for the row the engine resolves. `Release` is
still **required** — it is what matches `_ITERATOR_DEBUG_LEVEL` and the release CRT flavour, and
omitting it still produces `LNK2038` — so only the explanation changes, not the flag.

#### 3.4 `.github/workflows/native-build-job.yml` — provenance gate targets the wrong tarball

Line 99 extracts the URL to attest with:

```bash
grep -oE 'https://[^"]*logging-windows-x86_64\.tar\.gz' native/cmake/EtRuntimePin.cmake | head -1
```

The `/MD` row remains in the pin file, so this keeps matching and keeps passing — while attesting a
tarball the build no longer links. This is a silent correctness failure in a supply-chain gate, not a
cosmetic one. The pattern must select the `-static` asset.

#### 3.5 `native/tests/cmake_resolution.sh` — extend row-selection coverage

Update the Windows stem/URL assertions to `windows-x86_64-static`, and add a case that forces the
dynamic row explicitly and asserts it still resolves. That second case is what makes the row a
selectable parameter rather than a hardcode with extra steps, and it runs on Linux with no Windows
hardware.

---

## 4. What does **not** change

- **`build.gradle.kts:56` and `LibUtils.java:60`.** `windows-x86_64` there is the Java resource path
  (`/native/windows-x86_64/`) and the per-platform jar classifier — a different namespace from the
  runtime pin row, and one that is part of the published artifact's coordinates. It stays.
- **`native/build.sh` staging path**, for the same reason: it writes to
  `src/main/resources/native/windows-x86_64/`.
- **`native/tests/ci_workflow.sh`**, which asserts the CI artifact name `executorch-libs-windows-x86_64`.
- **Every Linux path**, including the `nm`-based `assert_xnnpack_registered.cmake` guard, which is
  already `if(NOT WIN32)`-gated.

---

## 5. Verification

### 5.1 Replacing A5

The work order's A5 asks for a load test on a Windows image that has never had a VC++
redistributable or Visual Studio installed. **No such machine is available**, and a dev box proves
nothing because it already has the runtime. Two static gates substitute:

- **CRT directive scan** over the shim's own `.lib`, adapted from the producer repo's
  `scripts/check-windows-crt.sh`. Expect `LIBCMT`/`LIBCPMT`.
- **Import-table assertion**: `dumpbin -dependents` on the built DLL shows no `VCRUNTIME140.dll` and
  no `MSVCP140.dll`.

Both rules the producer script paid for in debugging time carry over verbatim:

- **Dash-form flags** (`-nologo -directives`, never `/nologo`). Under MSYS/Git-Bash a leading `/` is
  path-converted into something like `C:\Program Files\Git\nologo` and the tool fails.
- **Assert on presence of the expected marker, never absence of the wrong one.** An absence check
  reports PASS when the tool failed to run at all — which is precisely how an earlier version of the
  producer's script passed on 18 libraries while `dumpbin` was erroring on every one of them.

Accepted gap, recorded explicitly: the "no redist required" claim ships on import-table evidence and
has not been confirmed by execution on a clean image. Tracked as a follow-up, not a blocker.

### 5.2 Full gate list

- Linux: `./native/local_build_wrapper.sh` then `./gradlew test` — green, artifact unchanged in kind.
- `native/tests/cmake_resolution.sh` — both Windows rows resolve, Linux rows unaffected.
- Windows CI: shim build, `build_qa.sh` Catch2 suite (which also stands in for the XNNPACK
  registration guard), CRT scan, import-table check.
- `./gradlew test` on Windows — `System.load` succeeds and inference runs.

---

## 6. Stream B — resolved, no code change

**Where the C++17 requirement comes from on Windows:** `native/CMakeLists.txt:3-4` sets
`CMAKE_CXX_STANDARD 20` together with `CMAKE_CXX_STANDARD_REQUIRED ON`, at the top of the project's
single `CMakeLists.txt`, before any target is declared.

That source is trustworthy on all four axes the work order asks about:

- **Explicit and first-party** — not implicit, not supplied by a dependency, so it cannot vanish
  under an unrelated bump (B1).
- **`REQUIRED ON`** — CMake cannot silently decay to an older standard (B2).
- **Scope** — the setting precedes every `add_library`/`add_executable` in the file, and the project
  has exactly one `CMakeLists.txt`, so there is no sibling tree or earlier-added subdirectory to miss
  (B3, B4). It also propagates into FetchContent subprojects.
- **C++20 ≥ C++17** — ET's guard keys on `_MSVC_LANG`, which `/std:c++20` satisfies;
  `/Zc:__cplusplus` is not required (B5).

Remaining Stream B deliverables are therefore documentation and one measurement:

1. Run the B5 grep during the Phase 2 spike (item S4) to confirm empirically that `/std:c++20`
   reaches the ET-including TUs, rather than inferring it from the CMake source.
2. Add a line to `CLAUDE.md` and the build docs recording that `find_package(ExecuTorch)` exports
   **no** `INTERFACE_COMPILE_FEATURES` and therefore supplies no language standard — every consumer
   must state it. This is the durable artifact: it is what stops a future cleanup from deleting
   `set(CMAKE_CXX_STANDARD 20)` as apparently redundant.
3. Report the finding back to the producer repo, which holds a drafted upstream proposal. Do not file
   against `pytorch/executorch` from this repo.

If it ever resurfaces the fingerprint is Windows/MSVC-only: grep a failing build log for `C1189` or
`_MSVC_LANG`.

---

## 7. Done when

- [ ] `EtRuntimePin.cmake` is `v1.3.1-8`, comment header re-applied, Linux artifacts re-reviewed and
      Linux tests green
- [ ] Windows resolves `windows-x86_64-static`; the dynamic row remains explicitly selectable and
      tested
- [ ] No comparison, test, or CI grep still assumes the Windows pin row is `windows-x86_64`
- [ ] Every object linked into the DLL is `/MT`, Catch2 included
- [ ] CRT scan shows `LIBCMT`/`LIBCPMT`; import table shows no `VCRUNTIME140`/`MSVCP140`
- [ ] `./gradlew test` green on both platforms
- [ ] `/std:c++20` confirmed present on Windows ET-including TUs
- [ ] Build docs record that `find_package(ExecuTorch)` supplies no language standard
- [ ] Findings written back to the producer repo (both streams, including Stream B's "nothing to
      change" with its evidence)
