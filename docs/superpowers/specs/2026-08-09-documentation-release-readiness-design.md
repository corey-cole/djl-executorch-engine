# Documentation release readiness

**Date:** 2026-08-09
**Status:** Approved, ready for planning
**Scope:** Make the project's documentation fit to publish — a landing page that explains what the
engine offers and how to use it, complete API reference, a documented native layer, and a `docs/`
directory where current guidance is distinguishable from historical research.

## Problem

The engine is functionally ready for a public release; its documentation is not.

- **The README never shows how to use the engine.** In 168 lines there is no example of loading a
  `.pte` and running a prediction. It opens with Docker prerequisites and the glibc floor — it
  answers "how do I compile this" before "what is it." It is simultaneously the GitHub landing page
  and the Maven Central description.
- **The javadoc jar already ships to Maven Central.** The `com.vanniktech.maven.publish` plugin
  builds and publishes it, so gaps are user-facing today: 25 warnings across 8 files in the
  published module, plus 7 in `example/`. They cluster on exactly the types a consumer touches —
  `EtNative` (6), `DType` (5), `EtTensor` (4), `MapTranslator` (3).
- **The native layer is no longer a thin shim.** `native/jni/executorch_djl_jni.cpp` is 308 lines
  at **6% comment density** and holds the subtlest code in the repository: cached global refs and
  method IDs, local-reference discipline, exception translation, and handle lifetime across the
  boundary. `native/jni/et_logging.cpp` is 8%. By contrast `native/core/et_runtime.h` is 40% and
  sets a good standard, so the gap is concentrated rather than systemic.
- **`docs/` mixes reference material with research artifacts.** Alongside current guidance sit two
  completed handover work orders, a 41K file with "wip" in its name, and cross-project research —
  with no signal telling a reader which is which. At least one handover document is known to have
  gone stale after a runtime pin bump.
- **The new observability surface is undocumented outside javadoc and CLAUDE.md.**
- **`.clangd` points at a README section that does not exist** ("Editor setup (clangd)").

## Design

### 1. Document map

**Stays in `docs/` — current reference:**

| File | Status |
|---|---|
| `benchmarking.md` | existing, current |
| `ci-native-build.md` | existing, current |
| `executorch-build-notes.md` | existing, current |
| `executorch-host-buffer-contract-brief.md` | existing, current — cited by code and CLAUDE.md |
| `building.md` | **new** — build-from-source, moved out of README |
| `native-architecture.md` | **new** — the C++ layer |
| `README.md` | **new** — index of the directory |

**Moves to `docs/research/` — point-in-time records:** `handover-to-engine.md`,
`handover-to-engine-2.md`, `handover-windows-static-cxx17.md`,
`handover-windows-static-cxx17-findings.md`, `host-buffer-contract-wip.md`,
`panama-research-sketch.md`, and the `iree-lessons-learned/` directory.

**The triage rule, stated so it is reproducible:** a document stays in `docs/` if something
*outside itself* cites it as current — a code comment, CLAUDE.md, or the README. Otherwise it is a
record of how we got here, and it moves.

Size is not the signal. `executorch-host-buffer-contract-brief.md` is 89K and stays, because
CLAUDE.md cites it for the input-copy contract and `native/harness/et_overread_harness.cpp:3` cites
it by section (`§3/W4`). `host-buffer-contract-wip.md` is 41K and moves, because nothing cites it
and it says of itself "read the code rather than this."

Nothing is deleted. Each moved file gains a header giving its original purpose, its date, and an
explicit statement that it is not current guidance.

**Inbound references must be updated in the same commit as the move.** Counted: one each for
`handover-to-engine-2`, `handover-windows-static-cxx17-findings`, and `iree-lessons-learned`, plus
the relative links *inside* `host-buffer-contract-wip.md`, which break when the file descends a
directory level.

### 2. README

Rewritten as the consumer landing page:

1. **What it is** — ExecuTorch models under DJL; why (DJL 0.36.0 supports only the deprecated
   TorchScript export API); what it is not (CPU-only, limited NDArray support).
2. **Supported platforms** — the existing table, kept.
3. **Add the dependency** — the existing Gradle capability and Maven classifier snippets, kept.
4. **Quickstart** — load a `.pte`, run a prediction. New.
5. **Configuration and tuning** — `ai.djl.executorch.num_threads`, `workspaceSharingMode`, and the
   threading guidance that more caller threads is usually *slower* under the default sharing mode,
   with the measured figures.
6. **Monitoring** — `EtEngineStats.snapshot()`, the JMX object name, `ai.djl.executorch.jmx_enabled`,
   and the `-1` (unavailable) versus `0` (genuinely zero) convention.
7. **Limitations** — one `Model`/`Predictor` per thread; no weight cache; XNNPACK delegate
   workspace not accounted for in the reported footprint.
8. **Building from source** — a link to `docs/building.md`.
9. **Third-party licenses** — kept verbatim; it is a legal notice tied to the runtime pin.

### 3. `docs/building.md`

Absorbs what leaves the README: prerequisites, the manylinux container and the glibc-2.28 floor,
the local fast path and why it must not ship, the Windows/MSVC path, native QA and benchmarking
harnesses, the container file-ownership workaround, and runtime provenance verification.

It also gains an **Editor setup (clangd)** section covering `./native/gen_clangd_db.sh` and the
staleness rule after a pin bump. `.clangd`'s comment header is repointed at it, resolving the
dangling reference.

### 4. The quickstart must not be invented

Documentation examples rot silently, and a broken snippet on the landing page is worse than no
snippet. The quickstart is therefore a real compiled class,
`example/src/main/java/org/measly/example/QuickStart.java`, which the README quotes. Being on the
build path, it breaks CI when the API changes rather than leaving the README quietly wrong.

It follows the project's DJL conventions: `Criteria` → `ZooModel` → `Predictor`, everything in
try-with-resources, one `Predictor` per thread.

### 5. Javadoc

Content only — **no build change**. Doclint already surfaces these warnings by default and they
stay non-fatal by decision. Clear all 25 in the published module and all 7 in `example/`; the
example module counts because it is the code users read to learn the API.

Target: `./gradlew javadoc` emits zero warnings. Beyond the immediate fix, a zero baseline is what
makes a future regression legible instead of hidden among existing noise.

### 6. Native comments

Raise the under-documented files to the standard `native/core/et_runtime.h` already sets, in the
same prose-explaining-*why* style. No Doxygen, no generated site, no comment-syntax conversion.

`native/jni/executorch_djl_jni.cpp` needs, specifically:

- The cached global class references and method IDs, and **the signature-string synchronisation
  hazard**: changing a Java constructor without updating the hardcoded signature literal fails at
  class-init runtime, not compile time, and Linux masks it because both sides are always rebuilt
  together. This trap was hit during the observability work and is the single most valuable comment
  in the file.
- Local-reference discipline in the marshalling loops.
- What `throwJava` and `throwIllegalArgument` guarantee, and the "an exception may already be
  pending" invariant that explains why one caches its class and the other null-checks `FindClass`.
- Handle lifetime: `jlong` ↔ `EtRuntime*`, who owns it, when it becomes invalid.
- The direct-`ByteBuffer` liveness requirement across `forward()`.

`native/jni/et_logging.cpp` needs the PAL sink contract and the level mapping. The four small
headers (`array_size_limits.h`, `et_log_level.h`, `staging.h`, `et_probes.h`) each need a sentence
on why the file exists, not what it contains.

### 7. `docs/native-architecture.md`

Covers what no single source file can:

- The three-layer split — core, JNI shim, harnesses and tests — and why the core is deliberately
  JNIEnv-free: tests and harnesses link it without a JDK.
- The ownership model: borrowed in, single-copy out, including the memory-planned caveat that makes
  "zero-copy in" false for every `.pte` exported with ExecuTorch's defaults.
- The staging-slot design: grow-only, sized at load from the declared bound, 64-byte aligned, with
  128 bytes of padding for XNNPACK's documented over-read allowance.
- Process-global state and its differing rules: the intra-op pool is write-once and sealed at the
  first model load; workspace sharing is per-model and order-independent.
- The USDT probes and how to observe them.
- Pointers to `building.md` for where the runtime comes from.

### 8. Untracked artifacts

Eight `docs/superpowers/` specs and plans from June–July are untracked while 29 siblings are
committed. They record decisions that shipped — core extraction, the logging bridge, the timing
harness, runtime-dist consumption. They are committed as-is, making the design history complete.

`.gitignore` gains a `.settings/` rule for the Eclipse/JDT directories generated in the repository
root and in `example/`.

## Verification

Documentation fails silently, so every claim gets a mechanical check.

- **`./gradlew javadoc` emits zero warnings.** The reason for clearing all 32 rather than most.
- **The quickstart compiles** — `:example:compileJava` proves it, because `QuickStart.java` is on
  the build path.
- **Every relative markdown link resolves.** The highest-risk defect in this cycle: eight files move
  and the README's internal links are rewritten. `tools/scripts/check_doc_links.sh` extracts
  relative `](…)` targets from tracked `.md` files and asserts each exists. Unlike a javadoc nit
  this is unambiguous and fast, so it is wired into `check`.
- **No stale inbound references after the move** — grep the tree for the old paths, including the
  source comment in `et_overread_harness.cpp` and the CLAUDE.md citation.
- **`./gradlew test` still green** — sanity. Nothing here should touch behaviour; if it does, that
  is a finding.
- **CLAUDE.md reflects the new layout**, since it cites `docs/` paths directly.

## Out of scope

- **No Doxygen** and no generated C++ documentation site. The repository's prose-explaining-*why*
  style is an asset; converting it to `@brief`/`@param` boilerplate would trade meaning for
  structure and add a toolchain to maintain.
- **No javadoc failure gate.** The cleanup is content-only; warnings stay non-fatal by decision.
- **No rewriting of `benchmarking.md`, `ci-native-build.md`, or `executorch-build-notes.md`.** They
  are current and cited, and they stay as they are.
- **Nothing is deleted.** Every moved document keeps its content and gains a status header.
- **No published documentation site or GitHub Pages.**
