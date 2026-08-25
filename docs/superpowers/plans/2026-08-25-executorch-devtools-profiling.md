# ExecuTorch devtools profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ExecuTorch's event tracer a per-model, load-time opt-in of this engine, so a caller can profile a model's forwards and pull the ETDump bytes for offline analysis with ExecuTorch's Python Inspector.

**Architecture:** The shipped `linux-x86_64` `.so` links a `devtools` ExecuTorch runtime. `EtRuntime` optionally owns an `ETDumpGen` handed to the `Module` constructor; `etDump()` finalizes and copies out the accumulated buffer. Capability is compile-time detected from the runtime tarball and cross-checked against BUILDINFO, surfaced to Java as `devtoolsAvailable()`, so a platform without a devtools runtime fails the load with a clear message instead of silently recording nothing.

**Tech Stack:** C++20 (JNIEnv-free core + JNI shim), CMake + Ninja/Make, Catch2, Java 17 / DJL 0.36.0, Gradle 9.6.1, ExecuTorch runtime 1.4.1 (pin `1.4.1-3`).

**Spec:** [docs/superpowers/specs/2026-08-24-executorch-devtools-profiling-design.md](../specs/2026-08-24-executorch-devtools-profiling-design.md)

## Global Constraints

- **ExecuTorch runtime pin is `1.4.1-3`.** Phase 0 (devtools headers, flatcc headers, BUILDINFO `event_tracer` key) is already delivered there and verified. Task 1 lands the pin.
- **Tasks 2–8 must build against a devtools runtime before Task 9 makes it the default.** Prefix every native build in those tasks with `ET_RUNTIME_VARIANT=devtools`. After Task 9 the prefix is unnecessary on Linux but harmless.
- **The blessed Linux native build is `./native/local_build_wrapper.sh`** — it runs the pinned `manylinux_2_28` image and holds the glibc-2.28 floor. Never stage a release artifact from a host build.
- **`native/CMakeLists.txt` must keep `CMAKE_CXX_STANDARD 20` + `CMAKE_CXX_STANDARD_REQUIRED ON`.** ExecuTorch's headers `#error` below C++17 and the installed package exports no `INTERFACE_COMPILE_FEATURES`.
- **Do not link `etdump` alone.** Its exported interface is `flatccrt;$<LINK_ONLY:executorch>`, and `$<LINK_ONLY:>` suppresses usage requirements — `C10_USING_CUSTOM_GENERATED_MACROS` would be lost and `etdump_flatcc.h` then fails on a `cmake_macros.h` no tarball installs. `et_runtime` already links the main ExecuTorch targets; keep it that way.
- **`EtSymbolBlock.forward()` is not thread-safe on one model**, and neither is `ETDumpGen`. One `Model`/`Predictor` per thread. This change adds no new constraint.
- **Byte-field conventions in the stats surface: `-1` = unavailable, `0` = genuinely zero.** The new field is a boolean and is exempt, but do not perturb the existing ones.
- **Tests assert behaviour, not diff shape.** No greps for current wording in `native/tests/*.sh`.
- **Comments state what is, not what was.** No history or task labels in shipped comments.
- **Commit style:** conventional prefix, imperative subject, body explaining why. End every commit message with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01EigyktjCWXFw6ckATPKu23
  ```

---

### Task 1: Bump the runtime pin to `1.4.1-3` — COMPLETE (`38a8c62`)

Proves `1.4.1-3` is a clean swap on the **existing** `logging` configuration before anything switches variant. A prepared pin file with the consumer-notes block already re-applied is at `/tmp/claude-1000/-home-corey-workspace-djl-executorch-engine/a07980c5-625d-4ef8-96ad-66462284baeb/scratchpad/EtRuntimePin.cmake.prepared`; if it is gone, regenerate with `gh release download v1.4.1-3 --repo measly-java-learning/executorch-runtime-dist -p EtRuntimePin.cmake` and re-apply the block by hand.

**Files:**
- Modify: `native/cmake/EtRuntimePin.cmake` (whole-file replacement + comment header)
- Modify: `native/tests/cmake_resolution.sh:30-32` (the release-tag assertion — the ET version is
  unchanged across pkgrevs, so this gate exists to make a pkgrev bump visible and must move with it)

**Interfaces:**
- Consumes: nothing.
- Produces: `ET_RUNTIME_VERSION "1.4.1-3"`; pin rows `ET_RUNTIME_URL_devtools_{linux-x86_64,linux-aarch64,windows-x86_64,windows-x86_64-static}` and their `SHA256` counterparts, resolvable through `et_runtime_dist_url(<variant> <row> url sha)`.

- [x] **Step 1: Replace the pin file with the release asset**

```bash
cp /tmp/claude-1000/-home-corey-workspace-djl-executorch-engine/a07980c5-625d-4ef8-96ad-66462284baeb/scratchpad/EtRuntimePin.cmake.prepared \
   native/cmake/EtRuntimePin.cmake
grep -n 'ET_RUNTIME_VERSION' native/cmake/EtRuntimePin.cmake
```

Expected: `set(ET_RUNTIME_VERSION "1.4.1-3")`.

- [x] **Step 2: Confirm the devtools rows resolve for a foreign platform without that hardware**

Run:
```bash
bash native/tests/cmake_resolution.sh
```

Expected: initially FAIL with `pin is not at release v1.4.1-2`. That assertion is the supply-chain
gate: the ET version string is identical across pkgrevs, so the tarball stem cannot distinguish
them and the release-tag path segment is asserted instead. Update both the comment and the `grep`
on lines 30-32 to `v1.4.1-3`, then re-run. Expected: `PASS: cmake resolution`.

- [x] **Step 3: Build the shim on the unchanged `logging` configuration**

Run:
```bash
./native/local_build_wrapper.sh
```

Expected: build succeeds and stages `src/main/resources/native/linux-x86_64/libexecutorch_djl.so`. This is the clean-swap proof: same variant, new pin.

- [x] **Step 4: Run the JVM suite against it**

Run:
```bash
./gradlew test
```

Expected: BUILD SUCCESSFUL. `LstmModelIT` passing here also confirms the `1.4.1-3` logging tarball still carries the `etnp::lstm` custom op.

- [x] **Step 5: Refresh the clangd database**

Run:
```bash
./native/gen_clangd_db.sh
```

Expected: completes. Skipping this leaves clangd resolving against the `1.4.1-2` headers, silently, with no warning — the new `devtools/` include would appear unresolvable in the editor while the build succeeds.

- [x] **Step 6: Commit**

```bash
git add native/cmake/EtRuntimePin.cmake
git commit -m "$(cat <<'EOF'
build(pin): bump the ExecuTorch runtime pin to v1.4.1-3

Adds devtools rows for every shipped platform, devtools-only installs of
executorch/devtools/** and flatcc/**, and an event_tracer key in BUILDINFO.
Verified as a clean swap on the unchanged logging configuration: shim build plus
the full JVM suite.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EigyktjCWXFw6ckATPKu23
EOF
)"
```

---

### Task 2: Detect the devtools capability in CMake and expose `devtoolsAvailable()`

**Files:**
- Modify: `native/CMakeLists.txt` (after `find_package(executorch ...)` at line 155; link block for `et_runtime` at line 162)
- Modify: `native/core/et_runtime.h` (declare next to `xnnpackWorkspaceBytes()`)
- Modify: `native/core/et_runtime.cpp`
- Test: `native/test/et_runtime_test.cpp`

**Interfaces:**
- Consumes: Task 1's pin.
- Produces: C++ free function `bool measly::et::devtoolsAvailable()`; CMake compile definition `ET_HAVE_DEVTOOLS=1` on `et_runtime` (absent, not 0, when unavailable).

- [ ] **Step 1: Write the failing Catch2 test**

Append to `native/test/et_runtime_test.cpp`:

```cpp
TEST_CASE("devtools: availability matches what this build linked") {
  // The build either linked the event tracer or it did not; the query must say which, and must
  // agree with the compile-time gate rather than guessing at runtime.
#ifdef ET_HAVE_DEVTOOLS
  REQUIRE(measly::et::devtoolsAvailable());
#else
  REQUIRE_FALSE(measly::et::devtoolsAvailable());
#endif
}
```

- [ ] **Step 2: Run it and watch it fail to compile**

Run:
```bash
ET_RUNTIME_VARIANT=devtools ./native/local_build_wrapper.sh native/build_qa.sh
```

Expected: compile error — `devtoolsAvailable` is not a member of `measly::et`.

- [ ] **Step 3: Add the CMake detection block**

In `native/CMakeLists.txt`, immediately after the `find_package(executorch ...)` line:

```cmake
# Devtools (ETDump event tracer). Two signals, deliberately, because neither is sufficient alone.
#
# `TARGET etdump` says the runtime tarball shipped the library. It is NOT a promise the tracer was
# compiled in: upstream's add_library(etdump)/install(TARGETS) are unguarded and the root
# CMakeLists adds devtools/ in both branches of if(EXECUTORCH_BUILD_DEVTOOLS), so a tarball could
# carry the target with EXECUTORCH_ENABLE_EVENT_TRACER=OFF. Tracing would then record nothing and
# report success -- the worst failure available.
#
# BUILDINFO's event_tracer key is the distribution's own statement of what it built. It is absent
# under the ET_INSTALL escape hatch, where the target is all we have.
set(ET_EVENT_TRACER_DECLARED "unknown")
if(EXISTS "${ET_INSTALL}/BUILDINFO")
  file(STRINGS "${ET_INSTALL}/BUILDINFO" _et_buildinfo REGEX "^event_tracer=")
  if(_et_buildinfo)
    string(REGEX REPLACE "^event_tracer=" "" ET_EVENT_TRACER_DECLARED "${_et_buildinfo}")
  endif()
endif()

if(TARGET etdump)
  if(ET_EVENT_TRACER_DECLARED STREQUAL "off")
    message(FATAL_ERROR
      "Runtime ships the etdump target but BUILDINFO says event_tracer=off. Linking it would "
      "produce a build that reports profiling support and records nothing. Refusing.")
  endif()
  set(ET_HAVE_DEVTOOLS ON)
  message(STATUS "devtools: event tracer available (BUILDINFO event_tracer=${ET_EVENT_TRACER_DECLARED})")
else()
  if(ET_EVENT_TRACER_DECLARED STREQUAL "on")
    message(FATAL_ERROR
      "BUILDINFO says event_tracer=on but no etdump target was exported. The runtime package is "
      "inconsistent; refusing rather than shipping a build whose capability is unclear.")
  endif()
  set(ET_HAVE_DEVTOOLS OFF)
  message(STATUS "devtools: not provisioned in this runtime (platform=${ET_PLATFORM})")
endif()
```

- [ ] **Step 4: Link `etdump` and define the macro on the core**

In the existing `target_link_libraries(et_runtime PUBLIC ...)` block at line 162, add nothing yet; after that block add:

```cmake
# PUBLIC so the Catch2 units and the harnesses inherit both the link and the macro -- they link
# et_runtime and compile code guarded on ET_HAVE_DEVTOOLS.
#
# etdump is linked IN ADDITION TO the ExecuTorch targets above, never instead of them: its exported
# interface is `flatccrt;$<LINK_ONLY:executorch>`, and $<LINK_ONLY:> suppresses usage requirements,
# so C10_USING_CUSTOM_GENERATED_MACROS does not arrive through it. Without that definition
# etdump_flatcc.h reaches a cmake_macros.h that no tarball installs.
if(ET_HAVE_DEVTOOLS)
  target_link_libraries(et_runtime PUBLIC etdump)
  target_compile_definitions(et_runtime PUBLIC ET_HAVE_DEVTOOLS=1)
endif()
```

- [ ] **Step 5: Implement the query**

In `native/core/et_runtime.h`, after the `xnnpackWorkspaceBytes()` declaration:

```cpp
// True when this build linked a runtime whose event tracer is compiled in, i.e. when profiling can
// actually record. Compile-time constant: the capability is a property of the linked runtime, not
// of anything discoverable at run time.
bool devtoolsAvailable();
```

In `native/core/et_runtime.cpp`, near the other free functions:

```cpp
bool devtoolsAvailable() {
#ifdef ET_HAVE_DEVTOOLS
  return true;
#else
  return false;
#endif
}
```

- [ ] **Step 6: Run the QA suite and verify the test passes**

Run:
```bash
ET_RUNTIME_VARIANT=devtools ./native/local_build_wrapper.sh native/build_qa.sh
```

Expected: PASS, with `devtools: event tracer available (BUILDINFO event_tracer=on)` in the configure output.

- [ ] **Step 7: Verify the negative arm on the logging runtime**

Run:
```bash
ET_RUNTIME_VARIANT=logging ./native/local_build_wrapper.sh native/build_qa.sh
```

Expected: PASS, with `devtools: not provisioned in this runtime`. Both arms of the test must pass — that is the point of it.

- [ ] **Step 8: Commit**

```bash
git add native/CMakeLists.txt native/core/et_runtime.h native/core/et_runtime.cpp native/test/et_runtime_test.cpp
git commit -m "$(cat <<'EOF'
feat(native): detect the devtools event tracer and expose devtoolsAvailable()

Gates on two signals and fails configure when they disagree. TARGET etdump alone
is not a promise the tracer was compiled in -- upstream's etdump install rule is
unguarded -- and BUILDINFO's event_tracer key is absent under ET_INSTALL, so
neither is sufficient by itself.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EigyktjCWXFw6ckATPKu23
EOF
)"
```

---

### Task 3: Attach the tracer and pull the ETDump

**Files:**
- Modify: `native/core/et_runtime.h` (`EtRuntime` class)
- Modify: `native/core/et_runtime.cpp` (`RuntimeState`, constructor at line 117, `forward()`)
- Test: `native/test/et_runtime_test.cpp`

**Interfaces:**
- Consumes: `ET_HAVE_DEVTOOLS`, `devtoolsAvailable()` from Task 2.
- Produces: `EtRuntime(const std::string&, int workspaceSharingMode = -1, bool traceEvents = false)`; `std::vector<uint8_t> EtRuntime::etDump()`.

- [ ] **Step 1: Write the failing Catch2 tests**

Append to `native/test/et_runtime_test.cpp`:

```cpp
#ifdef ET_HAVE_DEVTOOLS
TEST_CASE("etdump: an untraced runtime yields nothing") {
  EtRuntime rt("add.pte");
  REQUIRE(rt.etDump().empty());
}

TEST_CASE("etdump: tracing with no forward yet yields nothing") {
  EtRuntime rt("add.pte", -1, /*traceEvents=*/true);
  REQUIRE(rt.etDump().empty());
}

TEST_CASE("etdump: a forward produces a dump carrying the ED00 identifier") {
  EtRuntime rt("add.pte", -1, /*traceEvents=*/true);
  float a = 2.0f, b = 3.0f;
  std::vector<InputDesc> in{{&a, {1}, 6}, {&b, {1}, 6}};
  rt.forward(in);
  auto dump = rt.etDump();
  REQUIRE(dump.size() > 8);
  // Size-prefixed flatbuffer (start_as_root_with_size): 4-byte size, 4-byte root offset, then the
  // file identifier. Searching the first 16 bytes keeps the test honest about the exact layout.
  std::string head(reinterpret_cast<const char*>(dump.data()),
                   std::min<size_t>(dump.size(), 16));
  REQUIRE(head.find("ED00") != std::string::npos);
}

TEST_CASE("etdump: pulling twice without a forward returns the same bytes, not corruption") {
  EtRuntime rt("add.pte", -1, /*traceEvents=*/true);
  float a = 2.0f, b = 3.0f;
  std::vector<InputDesc> in{{&a, {1}, 6}, {&b, {1}, 6}};
  rt.forward(in);
  auto first = rt.etDump();
  auto second = rt.etDump();
  REQUIRE_FALSE(first.empty());
  REQUIRE(first == second);
}

TEST_CASE("etdump: a forward after a pull starts a fresh dump, not a cumulative one") {
  EtRuntime rt("add.pte", -1, /*traceEvents=*/true);
  float a = 2.0f, b = 3.0f;
  std::vector<InputDesc> in{{&a, {1}, 6}, {&b, {1}, 6}};
  for (int i = 0; i < 4; ++i) rt.forward(in);
  auto four = rt.etDump();
  rt.forward(in);
  auto one = rt.etDump();
  REQUIRE_FALSE(four.empty());
  REQUIRE_FALSE(one.empty());
  // Four Execute blocks against one. Upstream resets the generator on the first event block after
  // a finalize, so the second dump must be the smaller of the two.
  REQUIRE(one.size() < four.size());
}
#endif  // ET_HAVE_DEVTOOLS

TEST_CASE("etdump: requesting tracing without devtools fails the load") {
#ifndef ET_HAVE_DEVTOOLS
  REQUIRE_THROWS_AS(EtRuntime("add.pte", -1, /*traceEvents=*/true), std::runtime_error);
#endif
}
```

- [ ] **Step 2: Run and watch them fail**

Run:
```bash
ET_RUNTIME_VARIANT=devtools ./native/local_build_wrapper.sh native/build_qa.sh
```

Expected: compile error — `EtRuntime` takes no third argument and has no `etDump`.

- [ ] **Step 3: Declare the surface**

In `native/core/et_runtime.h`, replace the `EtRuntime` constructor declaration and add the pull:

```cpp
  // traceEvents attaches an ExecuTorch ETDump event tracer to this model. Costs are real and the
  // buffer is unbounded until pulled (see etDump), so this is a diagnostic, not a production mode.
  // Throws when the linked runtime has no event tracer -- profiling that silently records nothing
  // is worse than a failed load.
  explicit EtRuntime(const std::string& ptePath, int workspaceSharingMode = -1,
                     bool traceEvents = false);
```

```cpp
  // Finalized ETDump covering every forward since the last call, or empty when not tracing and
  // when no forward has run yet. Each forward appends one "Execute" block; the runtime resets the
  // generator on the first block after a finalize, so pulling IS the drain.
  //
  // Unlike OutputView, the returned bytes are owned by the caller and outlive this runtime.
  //
  // Calling twice with no forward in between returns a copy of the same bytes: upstream's
  // get_etdump_data() matches none of its guard branches in the finalized state and would run the
  // builder's end sequence a second time, so the cached copy is a correctness guard, not an
  // optimization.
  std::vector<uint8_t> etDump();
```

- [ ] **Step 4: Implement it**

In `native/core/et_runtime.cpp`, add the include near the top:

```cpp
#ifdef ET_HAVE_DEVTOOLS
#include <executorch/devtools/etdump/etdump_flatcc.h>
#endif
```

Extend `RuntimeState` (line 46):

```cpp
struct RuntimeState {
  Module module;
  MethodMeta meta;
  std::vector<std::unique_ptr<StagingSlot>> staging;
#ifdef ET_HAVE_DEVTOOLS
  // Non-owning: the Module owns the tracer, because its constructor takes the unique_ptr. Null
  // when this runtime is not tracing.
  executorch::etdump::ETDumpGen* tracer = nullptr;
#endif
  // Set by etDump(), cleared by forward(). While set, the cached copy is returned instead of
  // finalizing an already-finalized builder.
  bool dumpFinalized = false;
  std::vector<uint8_t> lastDump;

  RuntimeState(const std::string& path, std::unique_ptr<executorch::runtime::EventTracer> t)
      : module(path, Module::LoadMode::File, std::move(t)) {}
};
```

Replace the constructor's initializer (line 117-118):

```cpp
namespace {
// Builds the tracer the Module will own, or nullptr. Kept out of the ctor body so the throw for an
// unsupported runtime happens before any Module exists.
std::unique_ptr<executorch::runtime::EventTracer> makeTracer(bool traceEvents) {
  if (!traceEvents) return nullptr;
#ifdef ET_HAVE_DEVTOOLS
  return std::make_unique<executorch::etdump::ETDumpGen>();
#else
  throw std::runtime_error(
      "EtRuntime: profiling requested but this build links a runtime with no event tracer "
      "(devtools is not provisioned for this platform)");
#endif
}
}  // namespace

EtRuntime::EtRuntime(const std::string& ptePath, int workspaceSharingMode, bool traceEvents)
    : state_(std::make_unique<RuntimeState>(ptePath, makeTracer(traceEvents))) {
#ifdef ET_HAVE_DEVTOOLS
  if (traceEvents) {
    state_->tracer =
        static_cast<executorch::etdump::ETDumpGen*>(state_->module.event_tracer());
  }
#endif
```

The rest of the constructor body is unchanged. `Module::load_method` falls back to `this->event_tracer()` when its own argument is null, so the existing `load_forward()` picks the tracer up with no further change.

Add the pull:

```cpp
std::vector<uint8_t> EtRuntime::etDump() {
#ifdef ET_HAVE_DEVTOOLS
  if (state_->tracer == nullptr) return {};
  if (state_->dumpFinalized) return state_->lastDump;
  executorch::etdump::ETDumpResult result = state_->tracer->get_etdump_data();
  state_->dumpFinalized = true;
  state_->lastDump.clear();
  if (result.buf != nullptr && result.size > 0) {
    const auto* p = static_cast<const uint8_t*>(result.buf);
    state_->lastDump.assign(p, p + result.size);
    // Caller-owned: get_etdump_data() finalizes into a fresh allocation. free() is the idiom
    // upstream's own consumer uses (examples/devtools/example_runner) and is correct for flatcc's
    // aligned allocator on POSIX. A Windows devtools build must use flatcc_builder_aligned_free
    // instead, since flatcc allocates with _aligned_malloc there.
    std::free(result.buf);
  }
  return state_->lastDump;
#else
  return {};
#endif
}
```

In `EtRuntime::forward`, at the top of the function body:

```cpp
  // A new forward re-opens the dump: upstream resets the generator on the first event block after
  // a finalize, so the cached copy is stale from here on.
  state_->dumpFinalized = false;
```

- [ ] **Step 5: Run the QA suite under ASan and UBSan**

Run:
```bash
ET_RUNTIME_VARIANT=devtools ./native/local_build_wrapper.sh native/build_qa.sh
```

Expected: all Catch2 tests PASS with no ASan or UBSan findings. A `runtime error:` line is a failure even when no assertion fails.

- [ ] **Step 6: Verify the no-devtools arm still builds and passes**

Run:
```bash
ET_RUNTIME_VARIANT=logging ./native/local_build_wrapper.sh native/build_qa.sh
```

Expected: PASS. The `ET_HAVE_DEVTOOLS`-guarded tests compile out; the throw test runs.

- [ ] **Step 7: Commit**

```bash
git add native/core/et_runtime.h native/core/et_runtime.cpp native/test/et_runtime_test.cpp
git commit -m "$(cat <<'EOF'
feat(native): attach an ETDump event tracer per model and pull the buffer

The Module owns the tracer; EtRuntime keeps a non-owning pointer for the pull.
get_etdump_data() hands back a caller-owned allocation and leaves the builder
finalized, where a second call would run the end sequence again -- so the copy is
cached until the next forward, which is also when upstream resets the generator.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EigyktjCWXFw6ckATPKu23
EOF
)"
```

---

### Task 4: Cover the ETDump allocation in the leak harness

The manual `free()` of a third-party allocation is the most likely defect in this change, and LSan catches it for a few lines.

**Files:**
- Modify: `native/harness/et_leak_harness.cpp`

**Interfaces:**
- Consumes: `EtRuntime(path, mode, traceEvents)` and `etDump()` from Task 3.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Add the profiling arm**

In `native/harness/et_leak_harness.cpp`, inside the outer loop, construct the runtime with tracing on every other iteration and pull between forwards. Replace the `EtRuntime rt(pte);` line (line 57) with:

```cpp
    // Alternate so one run covers both the traced and untraced load/destroy paths. The pull is
    // what this arm exists for: get_etdump_data() returns a caller-owned buffer that etDump()
    // copies and frees, and a mistake there leaks the whole dump per iteration.
    const bool trace = (it % 2) == 1;
    EtRuntime rt(pte, -1, trace);
```

After the inner forward loop (following line 85's block), add:

```cpp
    if (trace) {
      std::vector<uint8_t> dump = rt.etDump();
      if (dump.empty()) {
        std::fprintf(stderr, "et_leak: traced run produced an empty dump\n");
        return 3;
      }
      // Pull again with no intervening forward: exercises the cached-copy guard under LSan.
      if (rt.etDump() != dump) {
        std::fprintf(stderr, "et_leak: second pull disagreed with the first\n");
        return 4;
      }
    }
```

Add `#include <vector>` and `#include <cstdint>` to the includes if not already present.

- [ ] **Step 2: Run the harness under ASan/LSan**

Run:
```bash
ET_RUNTIME_VARIANT=devtools ./native/local_build_wrapper.sh native/build_qa.sh
```

Expected: the leak harness completes with no LSan report. A leak here prints `ERROR: LeakSanitizer: detected memory leaks` with the allocation stack.

- [ ] **Step 3: Commit**

```bash
git add native/harness/et_leak_harness.cpp
git commit -m "$(cat <<'EOF'
test(native): exercise the ETDump pull in the leak harness

etDump() copies out of a caller-owned flatcc allocation and frees it. That free
is the most likely defect in the profiling path and LSan proves it for a few
lines of harness code.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EigyktjCWXFw6ckATPKu23
EOF
)"
```

---

### Task 5: JNI surface

**Files:**
- Modify: `native/jni/executorch_djl_jni.cpp` (`loadModule` at line 186; add two entry points near `backendRegistered` at line 402)
- Modify: `src/main/java/org/measly/executorch/jni/EtNative.java`
- Modify: `src/main/java/org/measly/executorch/engine/EtModel.java:70` (call-site arity only)

**Interfaces:**
- Consumes: `EtRuntime(path, mode, traceEvents)`, `etDump()`, `devtoolsAvailable()`.
- Produces: `EtNative.loadModule(String, int, boolean)`, `EtNative.etDump(long)` returning `byte[]`, `EtNative.devtoolsAvailable()` returning `boolean`.

- [ ] **Step 1: Change the `loadModule` signature**

In `native/jni/executorch_djl_jni.cpp`, extend the entry point at line 186 with a trailing `jboolean profiling` parameter and pass it through:

```cpp
extern "C" JNIEXPORT jlong JNICALL
Java_org_measly_executorch_jni_EtNative_loadModule(
    JNIEnv* env, jclass, jstring ptePath, jint workspaceSharingMode, jboolean profiling) {
```

and forward `profiling == JNI_TRUE` as the third `EtRuntime` argument at the construction site inside that function.

- [ ] **Step 2: Add the two new entry points**

Next to `backendRegistered`:

```cpp
extern "C" JNIEXPORT jbyteArray JNICALL
Java_org_measly_executorch_jni_EtNative_etDump(JNIEnv* env, jclass, jlong handle) {
  auto* rt = reinterpret_cast<measly::et::EtRuntime*>(handle);
  if (rt == nullptr) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                  "etDump: model is closed");
    return nullptr;
  }
  std::vector<uint8_t> dump;
  try {
    dump = rt->etDump();
  } catch (const std::exception& e) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    return nullptr;
  }
  // Empty array, never null: an unprofiled model has no dump, which is an answer, not an error.
  jbyteArray out = env->NewByteArray(static_cast<jsize>(dump.size()));
  if (out == nullptr) return nullptr;  // OOM already pending
  if (!dump.empty()) {
    env->SetByteArrayRegion(out, 0, static_cast<jsize>(dump.size()),
                            reinterpret_cast<const jbyte*>(dump.data()));
  }
  return out;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_org_measly_executorch_jni_EtNative_devtoolsAvailable(JNIEnv*, jclass) {
  return measly::et::devtoolsAvailable() ? JNI_TRUE : JNI_FALSE;
}
```

- [ ] **Step 3: Update `EtNative`**

Change the declaration and add the two natives:

```java
    /**
     * Loads a .pte.
     *
     * @param ptePath path to the model file
     * @param workspaceSharingMode XNNPACK workspace sharing for this model: 0=Disabled, 1=PerModel,
     *     2=Global, -1 to send no spec and leave the runtime default in force
     * @param profiling attach an ETDump event tracer to this model
     * @return the native handle
     */
    public static native long loadModule(String ptePath, int workspaceSharingMode, boolean profiling);

    /**
     * Finalized ETDump covering every forward since the last call.
     *
     * @param handle the native handle
     * @return the ETDump bytes, or an empty array when the model is not profiled or has not run
     * @throws IllegalStateException if {@code handle} is 0, i.e. the model has been closed
     */
    public static native byte[] etDump(long handle);

    /** @return true when this build links a runtime whose event tracer is compiled in */
    public static native boolean devtoolsAvailable();
```

- [ ] **Step 4: Update the one call site**

`EtModel.java:70` becomes:

```java
        long handle = EtNative.loadModule(modelFile.toString(), workspaceSharingMode, false);
```

Task 6 replaces the literal `false` with the resolved option.

- [ ] **Step 5: Build and run the JVM suite**

Run:
```bash
ET_RUNTIME_VARIANT=devtools ./native/local_build_wrapper.sh
./gradlew test
```

Expected: BUILD SUCCESSFUL, and no `WARNING in native method:` lines in the output — `-Xcheck:jni` is attached to every `Test` task and a signature or array mistake surfaces there rather than as a test failure.

- [ ] **Step 6: Commit**

```bash
git add native/jni/executorch_djl_jni.cpp src/main/java/org/measly/executorch/jni/EtNative.java src/main/java/org/measly/executorch/engine/EtModel.java
git commit -m "$(cat <<'EOF'
feat(jni): carry the profiling flag into loadModule and return the ETDump

etDump returns an empty array rather than null for an unprofiled model: having no
dump is an answer, not an error.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EigyktjCWXFw6ckATPKu23
EOF
)"
```

---

### Task 6: Java option plumbing and the model surface

**Files:**
- Create: `src/main/java/org/measly/executorch/engine/EtProfiling.java`
- Modify: `src/main/java/org/measly/executorch/engine/EtEngine.java`
- Modify: `src/main/java/org/measly/executorch/engine/EtModel.java`
- Create: `src/test/java/org/measly/executorch/engine/EtProfilingTest.java`

**Interfaces:**
- Consumes: `EtNative.loadModule(String, int, boolean)`, `EtNative.etDump(long)`, `EtNative.devtoolsAvailable()`.
- Produces: `EtEngine.PROFILING_OPTION` (`"profiling"`); `EtProfiling.resolve(Map<String, ?>)` returning `boolean`; `EtModel.etDump()` returning `byte[]`.

- [ ] **Step 1: Write the failing unit test**

Create `src/test/java/org/measly/executorch/engine/EtProfilingTest.java`:

```java
package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

class EtProfilingTest {

    @Test
    void absentOptionMeansOff() {
        assertFalse(EtProfiling.resolve(null));
        assertFalse(EtProfiling.resolve(Map.of()));
    }

    @Test
    void parsesBothValuesCaseInsensitivelyAndTrimmed() {
        assertTrue(EtProfiling.resolve(Map.of("profiling", "true")));
        assertTrue(EtProfiling.resolve(Map.of("profiling", "  TRUE ")));
        assertFalse(EtProfiling.resolve(Map.of("profiling", "False")));
    }

    @Test
    void rejectsAnythingElse() {
        // Boolean.parseBoolean would silently read "yes" and "1" as false. An option whose typo
        // disables the feature it names is worse than one that fails.
        assertThrows(IllegalArgumentException.class, () -> EtProfiling.resolve(Map.of("profiling", "yes")));
        assertThrows(IllegalArgumentException.class, () -> EtProfiling.resolve(Map.of("profiling", "1")));
        assertThrows(IllegalArgumentException.class, () -> EtProfiling.resolve(Map.of("profiling", "")));
    }
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `./gradlew test --tests 'org.measly.executorch.engine.EtProfilingTest'`
Expected: compilation failure — `EtProfiling` does not exist.

- [ ] **Step 3: Implement `EtProfiling`**

Create `src/main/java/org/measly/executorch/engine/EtProfiling.java`:

```java
package org.measly.executorch.engine;

import java.util.Locale;
import java.util.Map;

/**
 * Resolves the per-model profiling opt-in from a DJL model option.
 *
 * <p>There is deliberately <b>no</b> JVM-wide property counterpart, unlike {@link
 * EtWorkspaceSharing}. A property would let one JVM flag attach an event tracer to every model in
 * the process, including models whose owner never pulls the dump — and an ETDump grows across every
 * forward until it is pulled. Profiling is a diagnostic with a real memory cost, so enabling it is
 * a decision at the load site and nowhere else. This absence is the design, not an omission.
 */
final class EtProfiling {

    /** DJL per-model option key, e.g. {@code Criteria.optOption("profiling", "true")}. */
    static final String OPTION_KEY = "profiling";

    private EtProfiling() {}

    /**
     * Resolves the option.
     *
     * @param options the model's DJL options; may be null
     * @return whether to attach an event tracer to this model
     * @throws IllegalArgumentException if the value is neither {@code true} nor {@code false};
     *     case-insensitive and trimmed. Bare truthy spellings are rejected rather than coerced.
     */
    static boolean resolve(Map<String, ?> options) {
        Object raw = options == null ? null : options.get(OPTION_KEY);
        if (raw == null) {
            return false;
        }
        String v = String.valueOf(raw).trim().toLowerCase(Locale.ROOT);
        switch (v) {
            case "true":
                return true;
            case "false":
                return false;
            default:
                throw new IllegalArgumentException(
                        OPTION_KEY + ": unrecognized value '" + raw + "'; expected true|false");
        }
    }
}
```

- [ ] **Step 4: Run the unit test**

Run: `./gradlew test --tests 'org.measly.executorch.engine.EtProfilingTest'`
Expected: PASS.

- [ ] **Step 5: Publish the option constant**

In `EtEngine.java`, beside `WORKSPACE_SHARING_MODE_OPTION`:

```java
    /**
     * DJL per-model option enabling ExecuTorch event tracing for one model, e.g. {@code
     * Criteria.optOption(EtEngine.PROFILING_OPTION, "true")}. Accepted values are {@code true} and
     * {@code false}; anything else fails the load.
     *
     * <p>There is no JVM-wide property counterpart by design — see {@code EtProfiling}.
     *
     * <p>Requires a runtime whose event tracer is compiled in; {@link #devtoolsAvailable()} reports
     * whether this platform has one, and requesting profiling without it fails the load.
     */
    public static final String PROFILING_OPTION = EtProfiling.OPTION_KEY;

    /**
     * Whether this platform's shipped native library can record ExecuTorch profiling data.
     *
     * @return true when the linked runtime has the event tracer compiled in
     */
    public static boolean devtoolsAvailable() {
        return EtNative.devtoolsAvailable();
    }
```

Add `import org.measly.executorch.jni.EtNative;` if absent.

- [ ] **Step 6: Wire it into the load path**

In `EtModel.load`, immediately after the `EtWorkspaceSharing.resolve(...)` line:

```java
        // Per-model, like the sharing mode: resolved fresh on every load, nothing sealed. Throws
        // IllegalArgumentException for a bad value, before anything irreversible happens.
        boolean profiling = EtProfiling.resolve(options);
        if (profiling && !EtNative.devtoolsAvailable()) {
            throw new UnsupportedOperationException(
                    "profiling requested but this platform's ExecuTorch runtime has no event tracer"
                            + " compiled in; profiling is not provisioned here");
        }
```

Extend the existing log line and the `loadModule` call:

```java
        logger.info(
                "model {} workspaceSharingMode={} profiling={}",
                getName(), EtWorkspaceSharing.name(workspaceSharingMode), profiling);
```

```java
        long handle = EtNative.loadModule(modelFile.toString(), workspaceSharingMode, profiling);
```

Store the flag on the model for Task 7 and add the accessor:

```java
    private boolean profiling;
```

```java
    /**
     * Finalized ETDump covering every forward since the last call, for offline analysis with
     * ExecuTorch's Inspector.
     *
     * <p>Empty when this model was not loaded with {@link EtEngine#PROFILING_OPTION}, or when no
     * forward has run since the last call. The dump grows across every forward until pulled, so a
     * long-running profiled model should be drained periodically.
     *
     * @return the ETDump bytes, never null
     */
    public byte[] etDump() {
        EtSymbolBlock etBlock = (EtSymbolBlock) block;
        if (etBlock == null || etBlock.isClosed()) {
            return new byte[0];
        }
        return etBlock.etDump();
    }
```

`EtSymbolBlock.handle` is `private volatile long` with no accessor, and the read must be guarded the
way `statsSnapshot()` guards it at line 174. Add the package-private method to `EtSymbolBlock`
alongside `isClosed()`:

```java
    /**
     * Finalized ETDump for this model, or empty once closed.
     *
     * @return the ETDump bytes, never null
     */
    byte[] etDump() {
        final long h = handle;  // one volatile read, like statsSnapshot()
        return h == 0 ? new byte[0] : EtNative.etDump(h);
    }
```

- [ ] **Step 7: Run the full suite**

Run: `./gradlew test`
Expected: BUILD SUCCESSFUL.

- [ ] **Step 8: Commit**

```bash
git add src/main/java/org/measly/executorch/engine/EtProfiling.java src/main/java/org/measly/executorch/engine/EtEngine.java src/main/java/org/measly/executorch/engine/EtModel.java src/main/java/org/measly/executorch/engine/EtSymbolBlock.java src/test/java/org/measly/executorch/engine/EtProfilingTest.java
git commit -m "$(cat <<'EOF'
feat(engine): add the per-model profiling option and EtModel.etDump()

No JVM property counterpart, deliberately: one flag attaching a tracer to every
model in a process would grow an unbounded buffer per model whose owner never
pulls. The reasoning is stated in EtProfiling so it does not read as an oversight.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EigyktjCWXFw6ckATPKu23
EOF
)"
```

---

### Task 7: Surface profiling in the stats snapshot

**Files:**
- Modify: `src/main/java/org/measly/executorch/engine/EtModelCounters.java`
- Modify: `src/main/java/org/measly/executorch/engine/EtModelStats.java`
- Modify: `src/main/java/org/measly/executorch/engine/EtModel.java` (counters construction)
- Modify: `src/test/java/org/measly/executorch/engine/EtModelCountersTest.java`

**Interfaces:**
- Consumes: the `profiling` boolean resolved in Task 6.
- Produces: `EtModelStats.isProfiling()` returning `boolean`.

- [ ] **Step 1: Write the failing test**

Add to `EtModelCountersTest.java`:

```java
    @Test
    void carriesTheProfilingFlagFromLoad() {
        assertTrue(new EtModelCounters("add", "global", 4096L, 1_000_000L, true).profiling());
        assertFalse(new EtModelCounters("add", "global", 4096L, 1_000_000L, false).profiling());
    }
```

Add the imports `assertFalse` and `assertTrue`, and update the existing `counters()` helper at the
top of the file to pass a fifth argument (`false`) so the other four tests keep compiling.

- [ ] **Step 2: Run and watch it fail**

Run: `./gradlew test --tests 'org.measly.executorch.engine.EtModelCountersTest'`
Expected: compilation failure — the 5-argument constructor and `isProfiling()` do not exist.

- [ ] **Step 3: Thread the flag through**

Add `private final boolean profiling;` to `EtModelCounters`, take it as the fifth and final
constructor parameter, and add the package-private accessor `boolean profiling()` beside
`workspaceSharingMode()`.

`EtModelStats` is built in `EtSymbolBlock.java:183`, not in the counters class. Add `c.profiling()`
as a new argument immediately after `c.workspaceSharingMode()` — third of nine — so the load-time
metadata stays grouped. In `EtModelStats` add the field, the matching constructor
parameter, and:

```java
    /**
     * Whether this model was loaded with an ExecuTorch event tracer attached.
     *
     * <p>A profiled model accumulates an ETDump across every forward until the owner pulls it, so a
     * {@code true} here on a long-lived production model is worth investigating.
     *
     * @return true when profiling is enabled for this model
     */
    public boolean isProfiling() {
        return profiling;
    }
```

In `EtModel.load`, pass `profiling` as the new final argument to the `EtModelCounters` constructor.

- [ ] **Step 4: Run the test**

Run: `./gradlew test --tests 'org.measly.executorch.engine.EtModelCountersTest'`
Expected: PASS.

- [ ] **Step 5: Run the stats suites, which assert snapshot shape**

Run:
```bash
./gradlew test --tests 'org.measly.executorch.engine.EtEngineStats*'
./gradlew jmxDisabledTest statsDegradedTest
```

Expected: PASS. The MXBean serializes `EtModelStats`, so a new getter must not break the open-type mapping.

- [ ] **Step 6: Commit**

```bash
git add src/main/java/org/measly/executorch/engine/EtModelCounters.java src/main/java/org/measly/executorch/engine/EtModelStats.java src/main/java/org/measly/executorch/engine/EtModel.java src/test/java/org/measly/executorch/engine/EtModelCountersTest.java
git commit -m "$(cat <<'EOF'
feat(stats): report per-model profiling state in the snapshot

A model quietly accumulating an ETDump in production is exactly what the
monitoring surface should show.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EigyktjCWXFw6ckATPKu23
EOF
)"
```

---

### Task 8: End-to-end integration test

**Files:**
- Create: `src/test/java/org/measly/executorch/ProfilingIT.java`
- Possibly modify: `src/test/java/org/measly/executorch/TestSupport.java` (a `messageChain` helper, if absent)

**Interfaces:**
- Consumes: everything from Tasks 2–7.
- Produces: nothing.

Read `src/test/java/org/measly/executorch/WorkspaceSharingIT.java` first — it is the model for loading a fixture through `Criteria` with an option, and `TestSupport.java` holds the fixture helpers. Wrap models and predictors in try-with-resources, as it does.

- [ ] **Step 1: Write the test**

`WorkspaceSharingIT` lives at `src/test/java/org/measly/executorch/WorkspaceSharingIT.java` (package
`org.measly.executorch`, not `...engine`), and uses `AddTranslator` plus
`TestSupport.assumeNativeAvailable()`. Put the new test beside it, same package.

Create `src/test/java/org/measly/executorch/ProfilingIT.java`:

```java
package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import org.measly.executorch.engine.EtEngine;
import org.measly.executorch.engine.EtModel;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * End-to-end profiling. Both capability arms are real coverage: where devtools is provisioned the
 * dump must be well formed and must drain on pull; where it is not, the load must fail loudly
 * rather than record nothing.
 */
class ProfilingIT {

    private static final Logger logger = LoggerFactory.getLogger(ProfilingIT.class);

    private static Criteria<float[], Float> criteria(String profiling) {
        Criteria.Builder<float[], Float> b =
                Criteria.builder()
                        .setTypes(float[].class, Float.class)
                        .optEngine("ExecuTorch")
                        .optModelPath(Paths.get("native/spike"))
                        .optModelName("add")
                        .optTranslator(new AddTranslator());
        if (profiling != null) {
            b.optOption(EtEngine.PROFILING_OPTION, profiling);
        }
        return b.build();
    }

    /** The ETDump is a size-prefixed flatbuffer; its identifier sits within the first 16 bytes. */
    private static boolean carriesEtDumpIdentifier(byte[] dump) {
        int n = Math.min(dump.length, 16);
        return new String(dump, 0, n, StandardCharsets.ISO_8859_1).contains("ED00");
    }

    @Test
    void profiledModelYieldsAWellFormedDumpThatDrainsOnPull() throws Exception {
        TestSupport.assumeNativeAvailable();
        assumeTrue(
                EtEngine.devtoolsAvailable(),
                "devtools not provisioned on this platform; skipping the devtools-present arm");
        logger.info("ProfilingIT: running the devtools-present arm");
        try (ZooModel<float[], Float> model = criteria("true").loadModel();
                Predictor<float[], Float> predictor = model.newPredictor()) {
            EtModel etModel = (EtModel) model.getWrappedModel();

            assertEquals(0, etModel.etDump().length, "no forward has run yet");

            for (int i = 0; i < 4; i++) {
                assertEquals(5f, predictor.predict(new float[] {2f, 3f}), 1e-6);
            }
            byte[] four = etModel.etDump();
            assertTrue(four.length > 8, "four forwards must produce a dump");
            assertTrue(carriesEtDumpIdentifier(four), "dump must carry the ED00 identifier");

            // Pulling again with no forward in between must not corrupt the builder. Upstream's
            // get_etdump_data() matches none of its guard branches once finalized, so the cached
            // copy is what makes this safe.
            assertEquals(four.length, etModel.etDump().length, "second pull must repeat the first");

            // The forward after a pull starts a fresh dump: one Execute block, not five.
            assertEquals(5f, predictor.predict(new float[] {2f, 3f}), 1e-6);
            byte[] one = etModel.etDump();
            assertTrue(one.length > 8, "the fresh dump must still be well formed");
            assertTrue(one.length < four.length, "pulling drains; the dump must not accumulate");
        }
    }

    @Test
    void unprofiledModelYieldsAnEmptyDump() throws Exception {
        // Runs on every platform: having no dump is an answer, not an error.
        TestSupport.assumeNativeAvailable();
        try (ZooModel<float[], Float> model = criteria(null).loadModel();
                Predictor<float[], Float> predictor = model.newPredictor()) {
            assertEquals(5f, predictor.predict(new float[] {2f, 3f}), 1e-6);
            assertEquals(0, ((EtModel) model.getWrappedModel()).etDump().length);
        }
    }

    @Test
    void requestingProfilingWithoutDevtoolsFailsTheLoad() throws Exception {
        TestSupport.assumeNativeAvailable();
        assumeFalse(
                EtEngine.devtoolsAvailable(),
                "devtools is provisioned here; skipping the devtools-absent arm");
        logger.info("ProfilingIT: running the devtools-absent arm");
        Throwable t = assertThrows(Throwable.class, () -> criteria("true").loadModel());
        // DJL wraps load failures; the message must still name the provisioning, not fail generically.
        String messages = TestSupport.messageChain(t);
        assertTrue(
                messages.contains("not provisioned"),
                "load failure must explain the platform has no event tracer, got: " + messages);
    }

    @Test
    void unrecognizedOptionValueFailsTheLoad() throws Exception {
        // Runs on every platform: "yes" must fail rather than silently disable the feature it names.
        TestSupport.assumeNativeAvailable();
        Throwable t = assertThrows(Throwable.class, () -> criteria("yes").loadModel());
        String messages = TestSupport.messageChain(t);
        assertTrue(
                messages.contains(EtEngine.PROFILING_OPTION),
                "the failure must name the option key, got: " + messages);
    }
}
```

If `TestSupport` has no `messageChain` helper, add one — it walks `getCause()` concatenating
messages, and the OpenVINO ITs need the same thing, so check there first before writing a second
copy.

- [ ] **Step 2: Log which arm ran**

Each `assume`-gated test logs at INFO which arm it took, and the skip reason is the `assumeTrue`/`assumeFalse` message. A capability-gated test that silently no-ops is indistinguishable from a passing one; `a999b64` closed exactly this hole for `openvinoTest`.

- [ ] **Step 3: Run the suite**

Run:
```bash
ET_RUNTIME_VARIANT=devtools ./native/local_build_wrapper.sh
./gradlew test --tests 'org.measly.executorch.ProfilingIT' --info
```

Expected: the two universal tests plus the positive arm PASS; the negative arm reports as skipped with its reason visible.

- [ ] **Step 4: Commit**

```bash
git add src/test/java/org/measly/executorch/ProfilingIT.java src/test/java/org/measly/executorch/TestSupport.java
git commit -m "$(cat <<'EOF'
test(profiling): cover the dump end to end on both capability arms

Where devtools is provisioned the dump must be well formed and must drain on
pull; where it is not, the load must fail loudly. Each arm names itself in the
log so a skip is never mistaken for a pass.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EigyktjCWXFw6ckATPKu23
EOF
)"
```

---

### Task 9: Ship the devtools runtime on `linux-x86_64`

**Files:**
- Modify: `native/build.sh` (variant default near line 87; new supported-platform list near line 32)
- Modify: `native/build_qa.sh:34`
- Test: `native/tests/build_config.sh`

**Interfaces:**
- Consumes: everything above.
- Produces: `ET_DEVTOOLS_SUPPORTED_PLATFORMS` (default `linux-x86_64`), consulted by `build.sh` to choose the runtime variant.

- [ ] **Step 1: Write the failing shell test**

`native/tests/build_config.sh` already drives `build.sh` through its `PRINT_BUILD_CONFIG=1` seam.
Change the existing default-variant assertion and add the list mechanism cases. Replace this line:

```bash
grep -q 'ET_RUNTIME_VARIANT=logging' <<<"${out}" || fail "default variant not logging"
```

with:

```bash
# The shipped runtime variant is an engine-side decision keyed on the platform, not a mirror of what
# the pin publishes. This test host is linux-x86_64, the one platform provisioned for profiling.
grep -q 'ET_RUNTIME_VARIANT=devtools' <<<"${out}" || fail "linux-x86_64 must default to devtools"
```

and add after the existing override block:

```bash
# Emptying the list is how a platform leaves it, so the list is what decides -- not a hardcoded
# platform name buried in the variant default. Host-independent: it removes the host's own entry
# rather than depending on which platform this runs on.
out="$(ET_DEVTOOLS_SUPPORTED_PLATFORMS= PRINT_BUILD_CONFIG=1 bash native/build.sh)"
grep -q 'ET_RUNTIME_VARIANT=logging' <<<"${out}" \
  || fail "a platform absent from ET_DEVTOOLS_SUPPORTED_PLATFORMS must fall back to logging"

# An explicit variant still wins over the list, so benchmarking is unaffected.
out="$(ET_RUNTIME_VARIANT=bare PRINT_BUILD_CONFIG=1 bash native/build.sh)"
grep -q 'ET_RUNTIME_VARIANT=bare' <<<"${out}" || fail "explicit variant must beat the platform list"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `bash native/tests/build_config.sh`
Expected: FAIL with `linux-x86_64 must default to devtools` — the resolution does not exist yet.

- [ ] **Step 3: Add the platform list and variant resolution to `build.sh`**

Beside `ET_OPENVINO_SUPPORTED_PLATFORMS` (line 32):

```bash
# Platforms whose shipped artifact links a devtools runtime, enabling the per-model profiling
# option. An engine-side decision, not a mirror of the pin: the pin publishes devtools for every
# platform as of 1.4.1-3, and a platform joins this list once a test proves profiling works there.
# The cost of carrying it is +138 KB of .so and steady-state latency bounded under 0.35%.
ET_DEVTOOLS_SUPPORTED_PLATFORMS="${ET_DEVTOOLS_SUPPORTED_PLATFORMS:-linux-x86_64}"
```

Replace the variant default (line 87) with a platform-keyed resolution that an explicit env override still beats:

```bash
if [ -z "${ET_RUNTIME_VARIANT:-}" ]; then
  case " ${ET_DEVTOOLS_SUPPORTED_PLATFORMS} " in
    *" ${ET_PLATFORM} "*) ET_RUNTIME_VARIANT=devtools ;;
    *)                    ET_RUNTIME_VARIANT=logging ;;
  esac
fi
```

Place it after `ET_PLATFORM` is determined.

- [ ] **Step 4: Make the QA tree follow the shipped variant**

`native/build_qa.sh:34` currently hardcodes `logging`. The ASan/UBSan gate must exercise the runtime that ships, or the sanitizers cover a configuration nobody runs — and the ETDump pull's manual `free()` would go unchecked. Apply the same resolution, or source it from `build.sh` if the file structure allows a shared helper.

- [ ] **Step 5: Run the shell test and the full gate**

Run:
```bash
bash native/tests/build_config.sh
./native/local_build_wrapper.sh
./native/local_build_wrapper.sh native/build_qa.sh
./gradlew test
```

Expected: all PASS, with no `ET_RUNTIME_VARIANT` prefix needed now. Confirm the staged `.so` grew by roughly 138 KB:

```bash
ls -l src/main/resources/native/linux-x86_64/libexecutorch_djl.so
```

- [ ] **Step 6: Confirm the LSTM op and OpenVINO survived the variant switch**

Run:
```bash
./gradlew test --tests 'org.measly.executorch.engine.LstmModelIT'
./gradlew openvinoTest
```

Expected: PASS. The devtools `linux-x86_64` tarball ships `ETNPExtras` and `libopenvino_backend.a`, verified by listing the archive, but these tests are the behavioural proof.

- [ ] **Step 7: Commit**

```bash
git add native/build.sh native/build_qa.sh native/tests/build_config.sh
git commit -m "$(cat <<'EOF'
build: ship the devtools runtime on linux-x86_64

Measured cost of a devtools runtime with no tracer attached is +138 KB of .so and
under 0.35% steady state, so one artifact carries profiling rather than a second
build-matrix row. The QA tree follows the shipped variant: a sanitizer gate
against a runtime nobody ships would leave the ETDump free() unchecked.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EigyktjCWXFw6ckATPKu23
EOF
)"
```

---

### Task 10: Emit an ETRecord from the mobilenet export

**Files:**
- Modify: `tools/scripts/export_mobilenet.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `mobilenet_v2.etrecord` when `--etrecord` is passed.

- [ ] **Step 1: Add the flag and emit the record**

Add `import argparse` and `from executorch.devtools import generate_etrecord`, then hoist the program manager out of the write:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--etrecord",
        action="store_true",
        help="also emit mobilenet_v2.etrecord, which the ExecuTorch Inspector needs to attribute "
        "runtime events to graph ops. Off by default: an ETRecord embeds the program buffer and "
        "the graph modules, and the common case for this script is producing a demo model.",
    )
    args = parser.parse_args()
```

```python
    program = lowered.to_executorch()
    with open("mobilenet_v2.pte", "wb") as f:
        f.write(program.buffer)
    if args.etrecord:
        generate_etrecord("mobilenet_v2.etrecord", lowered, program)
```

Leave the `_unplanned` and TorchScript artifacts untouched.

- [ ] **Step 2: Verify both modes run**

Run:
```bash
./gradlew :example:exportModels
ls -l example/build/models/mobilenet_v2.pte
cd example/build/models && uv run --with executorch --with torchvision python ../../../tools/scripts/export_mobilenet.py --etrecord && ls -l mobilenet_v2.etrecord
```

Expected: the default run is unchanged and emits no `.etrecord`; the flagged run emits one. Match the `uv` invocation the Gradle task uses if it differs.

- [ ] **Step 3: Commit**

```bash
git add tools/scripts/export_mobilenet.py
git commit -m "$(cat <<'EOF'
feat(tools): optionally emit an ETRecord from the mobilenet export

The Inspector needs it to attribute runtime events to graph ops. Off by default:
an ETRecord embeds the program buffer and the graph modules, and this script's
common case is producing a demo model.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EigyktjCWXFw6ckATPKu23
EOF
)"
```

---

### Task 11: Documentation

**Files:**
- Create: `docs/profiling.md`
- Modify: `docs/README.md`
- Modify: `docs/benchmarking.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: the shipped behaviour from Tasks 1–10.
- Produces: nothing.

- [ ] **Step 1: Write `docs/profiling.md`**

Cover, as durable statements of what is:
- Enabling: `Criteria.optOption(EtEngine.PROFILING_OPTION, "true")`, and that there is no JVM property, with the reason.
- Pulling: `((EtModel) zooModel.getWrappedModel()).etDump()`, and that the buffer grows across forwards until pulled.
- Platforms: `EtEngine.devtoolsAvailable()` is the contract; `linux-x86_64` is provisioned, the others are not provisioned *yet* — never "unsupported".
- Exporting an ETRecord: `export_mobilenet.py --etrecord`.
- The manual Inspector procedure, with exact commands, stated as manual and as a per-pin-bump check.
- The cost: +138 KB of `.so`, steady state bounded under 0.35% with no tracer attached, and that attaching a tracer costs real per-forward time not measured here.

- [ ] **Step 2: Update the other three**

- `docs/README.md`: index `docs/profiling.md` under the current reference material.
- `docs/benchmarking.md`: replace the open gating question in the profiling section with the answer and the measured numbers from the spec's §1.
- `CLAUDE.md`: platforms no longer "all ship the `logging` runtime variant" — `linux-x86_64` ships `devtools`; `bare`/`devtools` are no longer solely benchmarking builds; and correct the `etnp::lstm` claim, which is in the devtools `linux-x86_64` tarball too, not the `logging` one only. Add a bullet for the profiling option next to the workspace-sharing bullet.

- [ ] **Step 3: Verify links and doc gates**

Run:
```bash
bash tools/scripts/check_doc_links.sh
bash native/tests/docs_present.sh
```

Expected: both PASS.

- [ ] **Step 4: Commit**

```bash
git add docs/profiling.md docs/README.md docs/benchmarking.md CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(profiling): document the option, the pull, and the Inspector procedure

Also answers benchmarking.md's open gating question with the measured numbers,
and corrects two platform claims that the devtools switch makes wrong.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EigyktjCWXFw6ckATPKu23
EOF
)"
```

---

### Task 12: Windows and aarch64 verification gate

Neither platform ships profiling, but both are affected: `EtNative.loadModule` changed arity, and Catch2 links the core only, so it cannot catch a JNI signature mismatch. This task ships no code unless a defect turns up.

**Files:**
- Modify (only if a defect is found): whichever layer broke.

**Interfaces:**
- Consumes: the full change set.
- Produces: nothing.

- [ ] **Step 1: Build and test on the winbox**

On the Windows host, with the MSVC dev shell active, run the native build and then:

```
gradlew.bat test
```

Expected: BUILD SUCCESSFUL. This is the only configuration that executes the JNI signature against a real JVM on Windows, and it also runs `ProfilingIT`'s devtools-absent arm for real. Drive the remote host in short chunks with `</dev/null` rather than one long-timeout script.

- [ ] **Step 2: Verify the CRT gate still holds**

Run:
```bash
bash native/tests/check_windows_crt.sh
```

Expected: PASS. MSVC does not reliably diagnose a CRT mismatch, so this script is the real gate.

- [ ] **Step 3: Verify aarch64**

On the radxa host, run the container build and the JVM suite. Expected: PASS on the `logging` runtime, with `ProfilingIT`'s devtools-absent arm executing.

- [ ] **Step 4: Record the devtools aarch64 tarball parity finding**

Extract the `devtools_linux-aarch64` tarball and confirm it ships `lib/cmake/ETNPExtras/`, `lib/libetnp_ops_lstm.a`, and `lib/libopenvino_backend.a`, the way the `linux-x86_64` one does:

```bash
tar tzf executorch-runtime-1.4.1-devtools-linux-aarch64.tar.gz \
  | grep -E 'ETNPExtras|libetnp_ops_lstm|libopenvino_backend'
```

Expected: all three present. If so, note in `docs/profiling.md` that aarch64 is ready to be provisioned and needs only the list edit plus a test run on the radxa host. If any is missing, record which, and that aarch64 stays on `logging` until the distribution provides it.

- [ ] **Step 5: Commit any documentation outcome**

```bash
git add docs/profiling.md
git commit -m "$(cat <<'EOF'
docs(profiling): record the aarch64 and Windows verification outcome

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01EigyktjCWXFw6ckATPKu23
EOF
)"
```

---

## Notes for the executor

- **Spec §4 says "`EtStatsSnapshot`'s per-model counters".** The concrete type is `EtModelStats`, built from `EtModelCounters`; `EtStatsSnapshot` holds the collection. Task 7 uses the real names.
- **`native/build_variants.sh` keeps working** and remains the way to compare `bare`/`logging`/`devtools`. `bare` is the only logging-free comparison point, because devtools is built with logging. Pass `MODEL=` to point it at a model big enough to resolve a delta; `add.pte` cannot.
- **If a native build fails to find `etdump_flatcc.h`** after Task 1, the pin did not land or the clangd/CMake tree is stale. Re-run `native/clean_stale_tree.sh` for the affected tree and re-run `./native/gen_clangd_db.sh`.
