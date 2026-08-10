> **Open research — no decision made, nothing implemented.**
> Written 2026-08. Explores a possible direction; it is not a plan and nothing here is committed to.
> For current guidance see [docs/README.md](../README.md).

# Panama research sketch: dual JNI + FFM front-ends

**Status**: research notes, not a plan. No code written, no decision made.
**Date**: 2026-07-25
**Question**: could `native/` support JNI (older JVMs) *and* Panama/FFM (Java 25+) from one
codebase? What blockers exist in the C++ as it stands today?

Java-side packaging concerns (multi-release JARs, module descriptors, `--enable-native-access`,
how DJL would select a front-end) are explicitly **out of scope** here.

## Verdict

Yes, and the C++ core is already most of the way there by design. Nothing in
`native/core/et_runtime.cpp` blocks it. The work is a new sibling facade next to `native/jni/`,
plus one real design change (lifetimes) and one genuinely JNI-shaped subsystem (logging).

## What already works in our favour

- `native/core/` is JNIEnv-free and pimpl'd (`et_runtime.h` — `struct ForwardState;` /
  `struct RuntimeState;`). This was a deliberate decision at extraction time and it is the
  reason this question has a cheap answer.
- The shim links `jni.h` for **headers only** and never links `libjvm`. So a single
  `.so`/`.dll` can export both a JNI surface and a C surface. A Panama consumer using
  `SymbolLookup.libraryLookup` never triggers `JNI_OnLoad`, so nothing JNI-specific
  initializes — the two surfaces don't interfere at load time.
- The Catch2 units (`native/test/`) and the leak harness (`native/harness/`) already consume the
  core through a non-JVM path. That is the existence proof that a second front-end is viable;
  those configures need no JDK at all.

## Blockers, in rough order of cost

### 1. The core's API is C++, not C ABI — Panama cannot bind it

`EtRuntime::forward(std::span<const InputDesc>)` is a mangled member function whose parameters
are `std::vector` / `std::span` / `std::string`. The FFM linker only speaks C.

Needs an `extern "C"` flat facade (say `native/capi/et_capi.{h,cpp}`) doing the same
translation-only job `native/jni/executorch_djl_jni.cpp` does today. Not a blocker in the
"can't be done" sense — just the same class of translation code, written a second time.

### 2. The boundary structs aren't describable to FFM

`InputDesc`, `OutputView`, and `MethodMeta` all embed `std::vector` (`et_runtime.h`). Panama
needs POD with a stable `MemoryLayout`: pointer + length pairs. The C facade therefore owns a
second, flat set of structs and converts.

Zero-copy-in survives (a raw pointer is a raw pointer either way); the shape arrays get
marshalled.

### 3. Exceptions must not cross the boundary — worse than under JNI

The core's contract is "throws `std::runtime_error` on load/forward/meta failure"
(`et_runtime.h`). A C++ exception unwinding through an FFM downcall stub is undefined
behaviour — typically a hard crash, **not** a Java exception.

Every `extern "C"` entry point needs a total catch-all converting to an error code plus an
out-param message buffer. The JNI shim gets away with its `throwJava` helper because JNI
defines that translation; Panama defines nothing.

### 4. `ForwardResult` is stack RAII, and Panama has no destructors

This is the one actual design change.

Today `forward()` returns `ForwardResult` by value, the JNI shim copies output bytes into a
Java `byte[]` via `SetByteArrayRegion` before the object drops at scope exit, and the
borrowed-view lifetime is enforced by C++ scoping inside a single native call.

Through a C ABI, `ForwardResult` becomes a heap handle with an explicit
`et_forward_result_free`, and the lifetime discipline moves into Java — an `Arena` /
try-with-resources, with a leak if a caller forgets.

Upside: Panama can then read the borrowed arena pointer directly as a `MemorySegment`, skipping
the copy JNI is forced into. So the FFM path is potentially *faster* — but only because it
takes on a lifetime obligation that JNI currently discharges for free.

### 5. Logging is the genuinely JNI-coupled subsystem

`native/jni/et_logging.cpp` holds a `JavaVM*`, calls `AttachCurrentThreadAsDaemon`, and invokes
`CallStaticVoidMethod`. None of that survives into Panama.

Refactor shape: parameterize the PAL sink on a plain `void(*)(int, const char*)` function
pointer. The JNI shim installs one from `JNI_OnLoad`; Panama installs an upcall stub. Two
sub-issues:

- **ET's PAL is process-global** and `register_pal` is effectively one-shot (the existing code
  already has to capture `g_defaultEmit` by value to avoid aliasing itself). If both front-ends
  were ever live in one process they would fight over it. Probably academic, but it is the only
  global state in play.
- **Upcalls from foreign threads.** `AttachCurrentThreadAsDaemon` exists precisely because ET
  may emit logs from a thread the JVM did not create. FFM upcall stubs are documented to support
  invocation from foreign threads, but this has **not** been verified against this codebase's
  threading. This is the first thing to spike before committing to a design.

### 6. Windows specifics (minor)

- C exports need `__declspec(dllexport)` or a `.def` file.
- The `/MT` static-CRT link (see `CLAUDE.md`) means anything crossing the boundary must not be
  freed by a different CRT. The handle-based design already guarantees this — only the DLL ever
  frees — but it's worth stating so nobody later "simplifies" by returning a `malloc`'d buffer
  for the caller to free.

### Non-issues

- **C++20 / `std::span`.** Purely internal to the core; invisible at a C boundary. The
  `CMAKE_CXX_STANDARD 20` requirement documented in `CLAUDE.md` is unaffected either way.
- **Thread-safety contract.** `EtSymbolBlock.forward()` being non-thread-safe per model is a
  property of the core, not the binding. Unchanged.
- **ExecuTorch runtime pin / XNNPACK registration.** Orthogonal — both front-ends link the same
  fetched runtime.

## Expected layout

```
native/core/     unchanged — JNIEnv-free C++, the single source of behaviour
native/capi/     new — extern "C" facade: flat PODs, error codes, heap handles
native/jni/      either rebased onto capi, or left as-is on core
```

**Open fork in the road**: does the JNI shim get rebased onto the C facade? One translation
layer instead of two is appealing, but it forces JNI to adopt the explicit-free lifetime model
it does not currently need — trading a real simplification for a real regression in safety on
the path that has all the users today. Undecided.

## If this is picked up

Suggested first spike, in order:

1. Prove an FFM upcall fires correctly from a thread ExecuTorch created (blocker 5). Cheapest
   way to invalidate the whole approach.
2. Sketch `et_capi.h` against `add.pte` only — load / meta / forward / free — and confirm the
   error-code translation shape (blocker 3) reads acceptably.
3. Decide the JNI-rebase question with those two answers in hand.
