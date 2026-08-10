# EtRuntime Core Extraction + Native QA — Design

> **Status:** design (2026-06-30). Extracts the `JNIEnv`-free core logic out of
> `native/jni/executorch_djl_jni.cpp` and stands up the two native QA layers that can only run
> against a JVM-free core. Realizes the "Couple the C++ refactor to this harness" note in
> [`docs/ci-native-build.md`](../../ci-native-build.md#native-memory-leak-gate-the-region-jvm-tools-cannot-see).
> Companion to the top-level [`djl-executorch-engine-design.md`](../../../djl-executorch-engine-design.md).

## Problem

`native/jni/executorch_djl_jni.cpp` interleaves two responsibilities: JNI translation (`JNIEnv`,
field/method-ID caching, Java object construction) and the pure inference logic (read input
`(data, shape, scalarType)` → `from_blob` → `Module::forward` → output descriptors). The pure logic
cannot be tested or leak-checked without a live JVM, yet the **pure-native ExecuTorch `Module`** is
the one memory region no JVM tool can see (per the leak-gate analysis in `ci-native-build.md`). The
fix is to disentangle the inference logic into a `JNIEnv`-free `EtRuntime` core, reduce JNI to a
translation shell, and apply native QA (leak harness + unit tests) directly to the core.

## Goal

A `JNIEnv`-free `EtRuntime` C++ core, consumed by three targets — the JNI shared library, an
ASan/LSan leak harness, and a Catch2 unit suite — with the existing JVM test suite proving the
extraction changed no observable behavior.

## Scope

**In scope:** the `EtRuntime` core extraction; the JNI shell rewrite over it; the leak harness; the
Catch2 unit suite; the `native/CMakeLists.txt` target restructure.

**Out of scope (explicit):**
- The libFuzzer load-path fuzzer — its own follow-on spec (per scoping decision).
- Any Java-side change. The JNI ABI and `EtTensor`/`EtMethodMeta`/`EtNative` signatures are
  unchanged; if any Java test needs editing, the seam leaked.
- New model fixtures beyond reusing `native/spike/add.pte`.
- The `build.sh`/`CMakeLists.txt` backend-flag TODO (multi-backend selection).
- `native/spike/` — left untouched as the historical artifact.

## Design decisions (settled during brainstorming)

1. **Output ownership = borrowed views, valid until next call.** `forward()` returns a `ForwardResult`
   that owns the ExecuTorch `EValue` vector and exposes per-output `OutputView{shape, scalarType,
   const void* data, nbytes}` pointing into ExecuTorch's host memory-planned arena. The view is valid
   until the next `forward()`/`destroy` on that instance. This preserves the Phase 2a **single-copy
   out** (the JNI shell copies the view into a Java `byte[]` synchronously inside the same native
   call, exactly where the copy lives today); core-owned copies were rejected because they add a
   second memcpy that only the two native consumers would ever benefit from, and neither needs it.
2. **The view contract binds only the non-JNI consumers.** Java never sees a view — by the time
   control returns to the JVM the bytes are already copied into a self-contained heap `EtTensor`. The
   "valid until next call" contract therefore constrains only the harness and the units, which read
   synchronously and discard. `ForwardResult` RAII scopes this: when it dies or is reassigned, the
   views die.
3. **Backend-agnostic seam.** The core operates at the `Module`/`EValue` layer; ExecuTorch presents
   method-level outputs as host tensors regardless of delegate (XNNPACK/CoreML/MPS/Vulkan/Qualcomm),
   so `const_data_ptr()` stays host-addressable. A future device-resident-output backend would be
   absorbed *inside* `EtRuntime` (a `synchronize()` + host copy in view materialization), invisible
   to the JNI ABI and Java. Adding backends is a build concern (`EXECUTORCH_BUILD_<backend>`), not a
   seam change.
4. **Error model = C++ exceptions translated at the JNI boundary.** The core throws
   `std::runtime_error` on `!ok()`; the JNI shell catches per entry point and `ThrowNew`s (replacing
   today's scattered inline throws); the harness lets exceptions propagate (abort = failure); Catch2
   uses `REQUIRE_THROWS`. Our TUs compile with exceptions enabled even though the ExecuTorch static
   libs are `-fno-exceptions` — we only throw from our own code after checking `.ok()`, never across
   an ExecuTorch frame. This matches DJL's own PyTorch engine, which translates C++ exceptions at the
   JNI boundary.

## Architecture

### How the core is shared across targets

The shipping `.so` builds **Release, no sanitizer**; the harness and units build
**`-fsanitize=address`**. The core therefore shares *source*, not compiled objects — the sanitizer is
selected per build directory (the `cmake -B native/asan` split `ci-native-build.md` already uses).

**`et_runtime` is one STATIC library target**, linked by all three consumers. In the Release build
dir it compiles clean; in the ASan build dir the whole graph (core + harness + units) compiles
instrumented. PIC is already on (required: the core's static deps get linked into the `.so`).

Rejected alternatives: an **OBJECT library** (its purpose is reusing compiled objects across targets
— exactly what the per-config sanitizer split forbids); a **header-only INTERFACE core** (forces
every consumer to recompile the heavy ExecuTorch headers and can't be unit-tested as a TU).

### File layout

```
native/
  core/    et_runtime.h  et_runtime.cpp     <- JNIEnv-free; ExecuTorch + std only
  jni/     executorch_djl_jni.cpp           <- thin shell: marshal <-> core, keeps JNI_OnLoad
  harness/ et_leak_harness.cpp              <- load -> forward -> read loop, ASan/LSan
  test/    et_runtime_test.cpp              <- Catch2 units against the core directly
  CMakeLists.txt                            <- et_runtime (STATIC) + 3 consumer targets
```

### `EtRuntime` API surface

`<jni.h>`-free; ExecuTorch + std only:

```cpp
namespace measly::et {

struct InputDesc {                 // borrowed, zero-copy in
  const void* data;
  std::vector<int64_t> shape;
  int8_t scalarType;
};

struct OutputView {                // borrowed, valid until next forward()/destroy
  std::vector<int64_t> shape;
  int8_t scalarType;
  const void* data;
  size_t nbytes;
};

struct MethodMeta {
  int numInputs;
  std::vector<int8_t> inputScalarTypes;          // -1 for non-tensor inputs (matches today's encoding)
  std::vector<std::vector<int64_t>> inputShapes; // per tensor input; empty for non-tensor (-1) inputs
};

class ForwardResult {              // RAII: owns the EValue vector that backs the views
public:
  std::span<const OutputView> outputs() const;
  ForwardResult(ForwardResult&&) noexcept;
  ForwardResult& operator=(ForwardResult&&) noexcept;
  ForwardResult(const ForwardResult&) = delete;
  ForwardResult& operator=(const ForwardResult&) = delete;
};

class EtRuntime {                  // owns the Module
public:
  explicit EtRuntime(const std::string& ptePath);   // load; throws on failure
  ~EtRuntime();                                       // destroy
  MethodMeta methodMeta() const;                      // method_meta("forward")
  ForwardResult forward(std::span<const InputDesc> inputs);   // throws on !ok()
  EtRuntime(const EtRuntime&) = delete;
  EtRuntime& operator=(const EtRuntime&) = delete;
};
}
```

### JNI shell after extraction

The Java-held handle now points to an `EtRuntime` (which owns the `Module`), not a bare `Module`.
`JNI_OnLoad` and all ID caching stay in the shell. The four entry points become pure translation,
each wrapped in `try/catch` → `ThrowNew`:

- `loadModule`: `jstring` → `std::string` → `new EtRuntime` → return handle.
- `methodMeta`: `core.methodMeta()` → build `EtMethodMeta` jobject from the struct's `numInputs` +
  `inputScalarTypes` only. `inputShapes` is harness-facing and unused Java-side (Java derives shapes
  from the caller's `NDList`/`model_spec`, not from meta), so `EtMethodMeta` stays `(int, int[])`.
- `forward`: marshal `EtTensor[]` → `vector<InputDesc>` (still `GetDirectBufferAddress`, zero-copy in)
  → `core.forward(...)` → copy each `OutputView` → `byte[]` → `EtTensor[]` (single-copy out, same as
  today).
- `destroy`: `delete` the `EtRuntime`.

The input marshal and output copy stay byte-for-byte where they are today; only their surrounding
logic moves into the core.

## QA layers

### Parity gate (extraction safety net)

The extraction is a pure refactor with an identical Java ABI, so the **existing JVM suite is the
parity gate**: `./gradlew test` (39 tests, incl. `NamedParamsIT` end-to-end and the dtype/forward
coverage) and the `leakTest` task must stay green after the `.so` is rebuilt from the shell+core
split, with **zero test changes**. Any required Java test edit means the seam leaked.

### Leak harness (`et_leak_harness`)

A `main(pte_path, iterations)` that loops: `EtRuntime{path}` → derive one host input buffer per tensor
input from `methodMeta()`'s `inputShapes` + `inputScalarTypes` (each buffer sized from the shape and
filled with 1s, so the harness is **model-agnostic**) → `forward(...)` →
read `outputs()` → let `ForwardResult` then `EtRuntime` destruct. Built ASan/LSan; **LSan's exit code
is the gate** (assertions optional). It exercises the two native-leak classes the JVM cannot see: the
load/destroy imbalance (a missing `~Module`) and per-`forward` native allocation. Runs against
`add.pte` today, any fixture later. This is the documented "first consumer" of the core.

### Catch2 units (`et_runtime_test`)

Assertion-level coverage the harness can't express (the harness only proves "doesn't leak/crash"):

- **load:** valid `.pte` constructs; missing/garbage path throws (`REQUIRE_THROWS`).
- **methodMeta:** `add.pte` reports input count `2` and both inputs' scalar-type codes (FLOAT32 = 6).
  The non-tensor `-1` encoding is preserved in the core but **not exercised here** — `add` has two
  tensor inputs; asserting `-1` would need a fixture with a non-tensor (scalar) input, which is out of
  scope (no new fixtures). Noted as a gap, not a test.
- **forward:** `add` of two known scalar inputs yields the expected output value, shape, and
  `scalarType` in the `OutputView`.
- **view lifetime (happy path):** after a second `forward()` on the same instance, the *fresh* result
  reads correctly. We assert the contract's happy path; we do not probe use-after-invalidation UB.

Both new targets link only `et_runtime` + ExecuTorch — **no JVM, no `JNIEnv`** — which is the entire
reason the extraction exists.

## Testing strategy

| Layer | What it proves | How it runs |
|---|---|---|
| JVM suite (existing) | Extraction changed no observable behavior | `./gradlew test` + `leakTest`, unchanged |
| Catch2 units | Core logic correct at assertion level | `et_runtime_test` (ASan build), per-commit-capable |
| Leak harness | No native leak across load/forward/destroy | `et_leak_harness` under LSan, nightly/manual (Stage A) |

## Success criteria

1. `native/jni/executorch_djl_jni.cpp` contains only JNI translation; all inference logic lives in
   `native/core/et_runtime.{h,cpp}` and `et_runtime.h` does not include `<jni.h>`.
2. The JVM suite and `leakTest` pass with **zero test-source changes**.
3. `et_runtime_test` passes under an ASan build.
4. `et_leak_harness` exits 0 (clean LSan) over `add.pte` and a non-trivial iteration count.
5. `et_runtime`, `executorch_djl`, `et_leak_harness`, `et_runtime_test` all build from one
   `native/CMakeLists.txt`; Release and ASan configurations select via separate build dirs.

## Follow-on (not this spec)

- libFuzzer load-path fuzzer over `EtRuntime` (own spec; shares this core + the ASan build).
- Examine DJL's PyTorch engine JNI for additional patterns/validation once this core is specced.
