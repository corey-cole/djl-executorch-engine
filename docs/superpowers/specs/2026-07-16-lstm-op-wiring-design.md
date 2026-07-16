# Design: Wire the `etnp::lstm` custom op into the shim + golden-vector e2e test + third-party notice bundling

**Date:** 2026-07-16
**Status:** Approved (design), pending implementation plan

## Problem

The `v1.3.1-6` ExecuTorch runtime tarball ships a first-party custom LSTM op
(`libetnp_ops_lstm.a`, provider `etnp`), its Highway SIMD support lib
(`libhwy.a`), and a self-describing whole-archive CMake helper
(`lib/cmake/ETNPExtras/ETNPExtras.cmake`). Verified present in the shipped
`logging-linux-x86_64` tarball (SHA `e1c29f4f…542cd`, matches the pin), but the
engine does not link it: `native/CMakeLists.txt` never includes `ETNPExtras.cmake`,
so a `.pte` using `etnp::lstm` fails to load with "operator not found."

Separately, wiring the op in surfaces a pre-existing compliance gap: the shipped
`.so` already statically links third-party code with attribution requirements
(XNNPACK, cpuinfo, pthreadpool, FP16/FXdiv, flatbuffers/flatcc, abseil, plus
ExecuTorch's own BSD license), yet the native classifier jar bundles only the
binary — no notices. The LSTM op widens this by adding **Highway (Apache-2.0)**,
whose license requires retaining the NOTICE in binary redistributions.

## Goals

1. Link the `etnp::lstm` op into the shipped `executorch_djl` shim when the
   runtime provides it, so it self-registers at load.
2. Prove the full product path (shipped `.so` + JNI + DJL) end-to-end with a
   golden-vector integration test.
3. Bundle the runtime tarball's complete third-party notices into each native
   classifier jar, closing the pre-existing gap and covering Highway in one move.

## Non-goals

- No native (Catch2) test for the op — the JVM IT is the chosen proof surface.
- No fetch-at-build-time for the fixtures — they are committed (see §3).
- No change to the JNI shim C++ or the Java engine classes — the op registers
  via static init, and `EtSymbolBlock.forward()` already marshals N-in/N-out
  tensors.
- No aarch64 / Windows LSTM support — the op is Linux-only upstream.

## Key facts (authoritative, from `executorch-runtime-dist`)

Sourced from `extras/lstm/aot/lstm_case.py`, `extras/lstm/test/lstm_runner.cpp`,
and `extras/lstm/test/test_lstm_roundtrip.py` in the runtime dist repo:

- **Op:** registers as `etnp::lstm`, overload `out`.
- **Input arity/order:** exactly **3 inputs `(x, h0, c0)`**. The 4 LSTM weights
  (`w_ih`, `w_hh`, `b_ih`, `b_hh`) are plain module attributes that `torch.export`
  lifts as **constants baked into the `.pte`**, not graph inputs. Both the AOT
  case and the C++ runner assert `numInputs == 3`.
- **Dims:** `T=5, B=2, I=4, H=3`.
- **Shapes:** `x=[T,B,I]=[5,2,4]`, `h0=[B,H]=[2,3]`, `c0=[B,H]=[2,3]` (h0/c0 are
  zeros in the fixture but must still be passed). Output: single tensor
  `[T,B,H]=[5,2,3]`.
- **Fixture layout:** `in.bin` = `_flat(x, h0, c0)`, row-major little-endian
  float32 (52 floats). `out.bin` = eager `nn.LSTM` golden output (30 floats).
  `shape` = text `LSTM_T=5\nLSTM_B=2\nLSTM_I=4\nLSTM_H=3\n`.
- **Tolerance:** `np.allclose(got, ref, rtol=1e-4, atol=1e-4)`, i.e.
  `|got − ref| ≤ atol + rtol·|ref|`, both `1e-4`.

## Design

### 1. CMake wiring (`native/CMakeLists.txt`)

Inside the existing `if(NOT ET_BUILD_QA AND NOT ET_BUILD_BENCH)` block (the
shim-only path), after `target_link_libraries(executorch_djl PRIVATE et_runtime)`:

```cmake
# First-party custom ops (etnp LSTM) ship only in some runtime tarballs (Linux logging).
# The tarball's self-describing ETNPExtras.cmake declares the imported targets + the
# whole-archive helper. Auto-detect by file presence: Windows tarballs omit it, so the op
# is skipped there with no platform branch of our own.
if(EXISTS "${ET_INSTALL}/lib/cmake/ETNPExtras/ETNPExtras.cmake")
  include("${ET_INSTALL}/lib/cmake/ETNPExtras/ETNPExtras.cmake")
  etnp_extras_whole_archive(executorch_djl)   # force_load: the registration TU must survive GC
  message(STATUS "etnp extras: LSTM op whole-archived into executorch_djl")
else()
  message(STATUS "etnp extras: none in runtime (platform=${ET_PLATFORM}); LSTM op not linked")
endif()
```

- **Why whole-archive:** the op is a static-init registrar; a plain link GCs the
  registration out of the final `.so` (same failure mode the XNNPACK post-link
  guard defends against). `etnp_extras_whole_archive` applies the correct per-OS
  wrapping (`-Wl,--whole-archive` on Linux) and pulls `etnp_hwy` (Highway)
  transitively, plain-linked.
- **Gating:** auto-detect via file presence. `ET_INSTALL` is already set to the
  fetched runtime `SOURCE_DIR` (or the escape-hatch path) earlier in the file.
  Windows lacks the file → skipped; the `STATUS` messages make a Linux miss
  visible rather than silent.
- **Scope:** only the shim target. QA/bench targets link `et_runtime` only and
  are untouched (JVM-IT-only decision).
- **Cost:** the Linux shim grows ~180 KB (op + hwy). Acceptable — it is the feature.

### 2. Committed fixtures

`src/test/resources/lstm/` ← `lstm.pte`, `in.bin`, `out.bin`, `shape`, plus a
`PROVENANCE.txt` recording:

- Source: `https://github.com/measly-java-learning/executorch-runtime-dist/releases/download/v1.3.1-6/etnp-lstm-fixtures-1.3.1.tar.gz`
- `sha256 7bd2d3e822959de268daea3de0bffbe442aeebfd9a63bf86249ce2ad0e893c04`
- Adjacent `.sha256` in the same release; attested by the runtime-dist CI.

Mirrors the existing committed-fixture precedent (`native/spike/add.pte`). Total
1.5 KB; network-free tests.

### 3. The e2e test (`src/test/java/org/measly/executorch/LstmModelIT.java`)

Mirrors `extras/lstm/test/lstm_runner.cpp` (read blob → slice → 3 tensors →
forward → compare):

- **Guard:** a new `TestSupport.assumeLstmModelAvailable()` that (a) confirms the
  native lib loads (`Class.forName("org.measly.executorch.jni.EtNative")`), and
  (b) skips unless `LibUtils.platform()` is `linux-x86_64` — the op is Linux-only
  by design, so on Windows the shim legitimately lacks it (a skip, not a failure).
- **Read fixtures from classpath** (`/lstm/…`): parse `shape` → `T,B,I,H`; read
  `in.bin`, slice into three FLOAT32 `NDArray`s `x[T,B,I]`, `h0[B,H]`, `c0[B,H]`
  in that order.
- **Run:** load via `Criteria` (engine `ExecuTorch`, model path
  `src/test/resources/lstm`, model name `lstm`) with the existing
  `PassthroughTranslator` (`NDList → NDList`); execute through a `Predictor`.
- **Assert:** `meta.numInputs == 3`; one output tensor of shape `[5,2,3]`; every
  element within `allclose(rtol=1e-4, atol=1e-4)` of `out.bin`, i.e.
  `|got − ref| ≤ 1e-4 + 1e-4·|ref|`.
- **Registration proof:** loading `lstm.pte` *is* the proof the op is wired — if
  it were not whole-archived, load throws "operator etnp::lstm not found." No
  separate assertion needed.

### 4. Third-party notice bundling (comprehensive)

The runtime tarball carries `LICENSE` + `THIRD-PARTY-NOTICES/` (36 files incl.
`highway_LICENSE`) at `${ET_INSTALL}`. Bundle them into each **native classifier
jar** at `META-INF/licenses/executorch-runtime/`, per-platform (Linux and Windows
tarballs may curate different sets). Three touch points:

1. **`native/build.sh`** — after staging the `.so`, copy `${ET_INSTALL}/LICENSE`
   and `${ET_INSTALL}/THIRD-PARTY-NOTICES/` into
   `src/main/resources/native/<platform>/licenses/`. The existing EXIT-trap runs
   `chown -R` over `src/main/resources/native/linux*`, so the new `licenses/`
   subtree is already covered — no trap change needed. Apply the same staging to
   `native/build_desktop.sh` for local parity.
2. **`.github/workflows/native-build-job.yml`** — broaden the artifact upload
   globs (`**/*.so`, `**/*.dll`) to also include `**/licenses/**`, so the notices
   survive into the `executorch-libs-*` artifacts that `publish.yml` unpacks into
   `build/native-staging/`. **This is the easy-to-miss step** — without it the
   notices never reach the publish job.
3. **`build.gradle.kts`** — the classifier jar gets a second `from(...)` mapping
   `<platform>/licenses/` → `META-INF/licenses/executorch-runtime/`, and the
   existing binary copy `exclude("licenses/**")` so they do not double-land. Add a
   release guard mirroring the existing `require(resolvedLib.exists())`: fail the
   jar if the notices dir is empty, so a bare binary cannot ship.

Our own root `LICENSE` (Apache-2.0) is unaffected.

## Testing strategy

- **New:** one platform-gated golden-vector IT (`LstmModelIT`) + a
  `TestSupport.assumeLstmModelAvailable()` helper.
- **Local:** `./native/build.sh` (host fast path fetches the logging Linux tarball
  → op present, notices staged) then `./gradlew test` runs `LstmModelIT` green.
- **CI:** the existing `build-java-package` job runs on Linux and loads the Linux
  shim, so it exercises the IT with no workflow change beyond the upload-glob
  broadening in §4.2.
- **No** native C++ test; **no** new CI plumbing beyond the notice upload glob.

## Files touched

| File | Change |
|---|---|
| `native/CMakeLists.txt` | Auto-detect + whole-archive the LSTM op into the shim (§1) |
| `native/build.sh` | Stage `LICENSE` + `THIRD-PARTY-NOTICES/` next to the `.so` (§4.1) |
| `native/build_desktop.sh` | Same notice staging for local parity (§4.1) |
| `.github/workflows/native-build-job.yml` | Broaden upload globs to include `**/licenses/**` (§4.2) |
| `build.gradle.kts` | Classifier jar: map notices → `META-INF/licenses/…`, empty-guard (§4.3) |
| `src/test/resources/lstm/*` | Committed fixtures + `PROVENANCE.txt` (§2) |
| `src/test/java/org/measly/executorch/LstmModelIT.java` | New golden-vector IT (§3) |
| `src/test/java/org/measly/executorch/TestSupport.java` | New `assumeLstmModelAvailable()` (§3) |
| `README.md`, `CLAUDE.md` | Note the op is now linked when the runtime provides it |

## Open questions

None. Input order, arity, dims, fixture layout, and tolerance are all pinned to
the runtime-dist source of truth (`lstm_case.py` / `lstm_runner.cpp` /
`test_lstm_roundtrip.py`).
