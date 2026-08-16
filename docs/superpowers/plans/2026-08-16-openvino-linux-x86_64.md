# OpenVINO Delegate Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the OpenVINO delegate a supported feature of this engine — an opt-in qualified jar carrying the vendored OpenVINO runtime, `OPENVINO_LIB_PATH` resolved from JNI, a committed fixture, and a test suite that executes a delegated model.

**Architecture:** The delegate `dlopen`s the OpenVINO C API once, under `std::call_once`, with no retry, so every detectable failure must be raised before ExecuTorch is entered. A C++ guard inside `EtRuntime`'s constructor — between `Module` construction and `load_forward()` — raises all four error cases at zero extra cost. A conditional Java probe before `loadModule` extracts the vendored bundle to a content-addressed cache and sets `OPENVINO_LIB_PATH` via JNI, which is the only mechanism available to a JVM.

**Tech Stack:** C++20, ExecuTorch 1.3.1 backend/MethodMeta APIs, JNI, DJL 0.36.0, Gradle 9.6.1 / JDK 17, JUnit 5, OpenVINO 2025.4.1 runtime (vendored, no export tooling ever).

**Spec:** `docs/superpowers/specs/2026-08-16-openvino-linux-x86_64-design.md`

## Global Constraints

- Backend id is exactly **`OpenvinoBackend`** (lowercase `v`). Hardcoded as a string in C++ and Java.
- OpenVINO pin is **`2025.4.1`, exact**, and must agree across three files: `native/cmake/EtRuntimePin.cmake` (`ET_RUNTIME_OPENVINO_VERSION`), the bundle `MANIFEST` in the jar, and `src/test/resources/models/openvino/MANIFEST`.
- **ABI suffix is `2541`**, but never hardcode it — the bundle ships `BUILDINFO` carrying `ov_abi=2541`. Read it from there, in the extractor and in the shell tests alike. It tracks the version (`2025.4.1` → `2541`), so a literal turns every OpenVINO bump into an edit here and fails as "missing library" rather than "you bumped OpenVINO".
- **`docs/openvino-version-bump.md` is the checklist of what an OpenVINO bump must touch** (Task 7). Anything this work makes version-coupled belongs on that list, and any test that fails because of a bump should name it in the failure message.
- **No symlink is ever created.** Measured: with `libopenvino_c.so` absent, `dlopen("<dir>/libopenvino_c.so.2541")` resolves the whole graph through `$ORIGIN`. `OPENVINO_LIB_PATH` points at the versioned file.
- All bundle libraries extract into **one flat directory**. `RPATH=$ORIGIN` is what resolves the graph; splitting them breaks it.
- **Never load out of the staging directory.** Publish by atomic directory rename first, load second.
- Platform variance is expressed as **capability** — `if(TARGET openvino_backend)`, `ET_RUNTIME_OPENVINO_PLATFORM`, `LibUtils.platform()` — never as a platform name. Upstream is exploring a Windows delegate.
- **Do not add `openvino_backend` to the post-link XNNPACK registration guard's required set.** That guard catches XNNPACK being GC'd out of the `.so`; a target that legitimately does not exist on most platforms would turn it into a platform conditional.
- **An already-set `OPENVINO_LIB_PATH` always wins, and that is decided natively, never in Java.** `System.getenv` is a snapshot from JVM startup and does not observe a `setenv` issued afterwards, so a Java-side check cannot see a value installed natively after start; native `getenv` can. The JNI primitive is therefore set-if-absent and returns the value in force — callers must use the return value, not their own argument. Task 5 Step 9 verifies the snapshot behaviour rather than assuming it.
- **Every JNI entry point copies its strings and releases immediately, before any `try`.** The idiom is `GetStringUTFChars` → null-check → copy to `std::string` → `ReleaseStringUTFChars` → then do the work. `GetStringUTFChars` returns null with an OOM pending, and it is *not* among the JNI functions legal to call in that state, so a second unchecked `Get*` after a failed first is itself a violation — and a `std::string` built from null is UB the shim's UBSan gate would abort on. Releasing before the `try` also removes any question about whether cleanup runs on the throwing path. See `executorch_djl_jni.cpp:188-190` for the established shape.
- Parity tolerance is **`atol=1e-2`** and must not be tightened. bf16 hardware lands ~2.5e-3 from the f32 eager golden, f32 hardware ~6e-8. Both correct. A tighter bound asserts which machine CI allocated.
- **Every OpenVINO test runs in its own JVM.** `OPENVINO_LIB_PATH` is process env and the `dlopen` is once-only.
- Nothing in this work touches export tooling. No torch, no partitioner, no quantizer.
- After any C++/CMake edit, rebuild the shim: `./native/local_build_wrapper.sh`.

## Facts already verified (do not re-derive)

- Bundle asset: `https://github.com/measly-java-learning/executorch-runtime-dist/releases/download/v1.3.1-10/openvino-runtime-2025.4.1-linux-x86_64.tar.gz`, SHA256 `066084d23d1e70395929f840368c6ce1a2e43f5969989d9f5bd595265d10ce7b`. 20.6 MB compressed, ~72 MB extracted.
- Bundle layout: one top-level dir `openvino-runtime-2025.4.1-linux-x86_64/` containing `BUILDINFO`, `licenses/` (5 files), and `lib/` with **7 real libraries + 1 symlink**:
  `libopenvino.so.2541` (17.2M), `libopenvino_intel_cpu_plugin.so` (49.9M), `libopenvino_ir_frontend.so.2541` (509K), `libopenvino_c.so.2541` (324K), `libtbb.so.12` (359K), `libtbbbind_2_5.so.3` (31K), `libhwloc.so.15` (461K), and `libopenvino_c.so -> libopenvino_c.so.2541`.
- `BUILDINFO` contents: `ov_version=2025.4.1`, `ov_abi=2541`, `platform=linux-x86_64`, `hwloc_version=2.8.0`, `source_wheel=openvino-2025.4.1-20426-cp312-cp312-manylinux2014_x86_64.whl`, `source_wheel_sha256=…`, `build_utc=…`.
- Fixture asset: `etnp-openvino-fixtures-1.3.1-2025.4.1.tar.gz`, SHA256 in the release's `.sha256` sidecar. Members, flat with no top-level dir: `openvino_tiny.pte` (4793 B), `in.bin` (32 B), `out.bin` (32 B), `shape` (17 B).
- `shape` contents are exactly `OV_IN=8\nOV_OUT=8\n`. `in.bin`/`out.bin` are float32 — 32 bytes / 8 values.
- The C API call sequence works and returns `f32` on a non-bf16 host: `ov_core_create(&core)` → `ov_core_get_property(core, "CPU", "INFERENCE_PRECISION_HINT", &char_ptr)`, both returning `0` for OK.
- `EtModel.load` calls `EtNative.loadModule` at `EtModel.java:66`; `EtRuntime`'s ctor calls `load_forward()`, so detection must precede `loadModule`.
- Existing tag-filtered test tasks are registered at `build.gradle.kts:56-97` and excluded from `tasks.test` at line 33. `nativePlatforms` is at line 214; `nativeJarTasks` at 222; `nativeVariants` at 252.

---

### Task 1: `pteUsesBackend` — metadata-only backend detection

**Files:**
- Modify: `native/core/et_runtime.h` (declare beside `xnnpackWorkspaceBytes()`)
- Modify: `native/core/et_runtime.cpp` (implement beside `xnnpackWorkspaceBytes()`)
- Modify: `native/jni/executorch_djl_jni.cpp` (new entry point beside `Java_..._xnnpackWorkspaceBytes`)
- Modify: `src/main/java/org/measly/executorch/jni/EtNative.java`
- Modify: `native/CMakeLists.txt` (add `OPENVINO_TINY_PTE_PATH` beside `CONV_PTE_PATH`)
- Test: `native/test/et_runtime_test.cpp`

**Interfaces:**
- Produces: `bool measly::et::pteUsesBackend(const std::string& ptePath, const std::string& backend)` in C++, and `EtNative.pteUsesBackend(String ptePath, String backend) -> boolean` in Java. Tasks 4 and 5 consume both.

This must not construct an `EtRuntime`: that ctor calls `load_forward()`, which is delegate init, which is the thing we are trying to get ahead of. It builds a bare `Module` and asks `method_meta` instead.

- [ ] **Step 1: Stage the fixture so the test has something to assert against**

The fixture proper is Task 7's job, but Task 1's test needs a `.pte` that uses the backend. Fetch just the one file now:

```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p src/test/resources/models/openvino
gh release download v1.3.1-10 --repo measly-java-learning/executorch-runtime-dist \
  -p 'etnp-openvino-fixtures-1.3.1-2025.4.1.tar.gz' -D /tmp --clobber
tar xzf /tmp/etnp-openvino-fixtures-1.3.1-2025.4.1.tar.gz -C src/test/resources/models/openvino
ls -la src/test/resources/models/openvino
```

Expect four files: `openvino_tiny.pte`, `in.bin`, `out.bin`, `shape`.

- [ ] **Step 2: Wire the fixture path into the Catch2 build**

In `native/CMakeLists.txt`, extend the `target_compile_definitions` block that already defines `ADD_PTE_PATH` / `CONV_PTE_PATH`:

```cmake
    CONV_PTE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/spike/conv.pte"
    OPENVINO_TINY_PTE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../src/test/resources/models/openvino/openvino_tiny.pte")
```

- [ ] **Step 3: Write the failing test**

Append to `native/test/et_runtime_test.cpp`:

```cpp
// Detection must work on EVERY platform, including ones where the delegate is not linked -- that
// is precisely the case that needs a good error message. So this asserts metadata reading, not
// delegate availability, and is NOT gated on the backend being present.
TEST_CASE("backend detection: reports which delegate a .pte needs, without loading the method") {
  REQUIRE(pteUsesBackend(OPENVINO_TINY_PTE_PATH, "OpenvinoBackend"));
  REQUIRE_FALSE(pteUsesBackend(OPENVINO_TINY_PTE_PATH, "XnnpackBackend"));
  REQUIRE(pteUsesBackend(CONV_PTE_PATH, "XnnpackBackend"));
  REQUIRE_FALSE(pteUsesBackend(CONV_PTE_PATH, "OpenvinoBackend"));
}

TEST_CASE("backend detection: a missing file throws rather than reporting false") {
  // Reporting false would be indistinguishable from "this model needs no delegate", sending the
  // caller down the non-OpenVINO path and losing the real error until much later.
  REQUIRE_THROWS([] { pteUsesBackend("/nonexistent/definitely-not-here.pte", "OpenvinoBackend"); }());
}
```

- [ ] **Step 4: Run to verify it fails**

Run: `./native/local_build_wrapper.sh native/build_qa.sh 2>&1 | grep -E "error:|assertions:"`

Expected: FAIL to compile with `'pteUsesBackend' was not declared in this scope`.

- [ ] **Step 5: Declare it**

In `native/core/et_runtime.h`, after the `xnnpackWorkspaceBytes()` declaration:

```cpp
// True if `ptePath`'s "forward" method is delegated to `backend` (e.g. "OpenvinoBackend").
//
// Reads METADATA ONLY: it builds a bare Module and asks method_meta, never load_forward(). That
// distinction is the whole point. EtRuntime's ctor calls load_forward() unconditionally, and for a
// delegated model that IS delegate init -- which for OpenVINO means a dlopen under std::call_once
// with no retry. Anything that needs to act before delegate init must ask through here.
//
// Throws std::runtime_error if the file cannot be opened or its program cannot be read. Reporting
// false there would be indistinguishable from "needs no delegate" and would hide the real error.
bool pteUsesBackend(const std::string& ptePath, const std::string& backend);
```

- [ ] **Step 6: Implement it**

In `native/core/et_runtime.cpp`, beside `xnnpackWorkspaceBytes()`:

`Module::method_meta` is program-level — it calls `load()` and then `program_->method_meta()`, never `load_method`. That is already documented in this file at lines 128-129, where the ctor explains why it must call `load_forward()` separately. It is the property this whole function depends on.

```cpp
bool pteUsesBackend(const std::string& ptePath, const std::string& backend) {
  Module probe(ptePath);
  const auto meta = probe.method_meta("forward");
  if (!meta.ok()) {
    throw std::runtime_error(
        "pteUsesBackend: cannot read method metadata from " + ptePath + " (error " +
        std::to_string(static_cast<int>(meta.error())) + ")");
  }
  return meta->uses_backend(backend.c_str());
}
```

- [ ] **Step 7: Run to verify it passes**

Run: `./native/local_build_wrapper.sh native/build_qa.sh 2>&1 | grep -E "error:|All tests passed|assertions:"`

Expected: `All tests passed`. If `uses_backend` is not found, check `runtime/executor/method_meta.h:252-273` for the exact spelling in the pinned runtime and adjust.

- [ ] **Step 8: Expose it through JNI**

In `native/jni/executorch_djl_jni.cpp`, after `Java_..._xnnpackWorkspaceBytes`:

```cpp
// Metadata-only backend probe. Unlike loadModule this does NOT construct an EtRuntime, so it
// cannot trigger delegate init -- which is the only reason it exists as a separate entry point.
extern "C" JNIEXPORT jboolean JNICALL
Java_org_measly_executorch_jni_EtNative_pteUsesBackend(
    JNIEnv* env, jclass, jstring ptePath, jstring backend) {
  const char* path = env->GetStringUTFChars(ptePath, nullptr);
  if (path == nullptr) {
    return JNI_FALSE;  // OOM already pending; do not call another JNI function that could fail
  }
  std::string p(path);
  env->ReleaseStringUTFChars(ptePath, path);

  const char* name = env->GetStringUTFChars(backend, nullptr);
  if (name == nullptr) {
    return JNI_FALSE;
  }
  std::string b(name);
  env->ReleaseStringUTFChars(backend, name);

  try {
    return measly::et::pteUsesBackend(p, b) ? JNI_TRUE : JNI_FALSE;
  } catch (const std::exception& e) {
    throwJava(env, "Backend detection failed", &e);
    return JNI_FALSE;
  }
}
```

**Copy, release immediately, then enter the try** — the same shape `loadModule` uses at lines 188-190. This is not stylistic. Holding the UTF chars across the `try` raises the question of whether the release runs on the throwing path (it would, since `throwJava` schedules an exception rather than unwinding, and `ReleaseStringUTFChars` is one of the few JNI functions legal to call with an exception pending) — but releasing first means the question never arises.

The null checks matter for a second reason. `GetStringUTFChars` returns null with an OOM pending, and `GetStringUTFChars` is *not* on the list of functions safe to call in that state — so calling the second one after the first failed is itself a violation. Constructing a `std::string` from a null pointer would also be UB, which the shim's UBSan gate would rightly abort on. This file already reasons about the same hazard for `FindClass` at lines 58-63.

- [ ] **Step 9: Declare the Java side**

In `src/main/java/org/measly/executorch/jni/EtNative.java`, after `xnnpackWorkspaceBytes()`:

```java
    /**
     * Reports whether a {@code .pte}'s {@code forward} method is delegated to a given backend,
     * reading metadata only.
     *
     * <p>Deliberately separate from {@link #loadModule}: that call constructs the native runtime,
     * whose constructor calls {@code load_forward()} — which for a delegated model is delegate
     * init. OpenVINO's delegate init resolves its C API with a {@code dlopen} under
     * {@code std::call_once} and never retries, so a caller that must configure something first has
     * to ask before {@code loadModule}, not after.
     *
     * @param ptePath absolute path to the model file
     * @param backend backend id, e.g. {@code OpenvinoBackend} (lowercase {@code v})
     * @return true if {@code forward} is delegated to that backend
     */
    public static native boolean pteUsesBackend(String ptePath, String backend);
```

- [ ] **Step 10: Rebuild the shim and run the JVM suite**

```bash
./native/local_build_wrapper.sh
./gradlew test
```

Expected: BUILD SUCCESSFUL. Nothing calls the new method from Java yet; this proves the shim still loads and the new symbol did not break `-Xcheck:jni`.

- [ ] **Step 11: Commit**

```bash
git add native/core/et_runtime.h native/core/et_runtime.cpp native/jni/executorch_djl_jni.cpp \
  native/CMakeLists.txt native/test/et_runtime_test.cpp \
  src/main/java/org/measly/executorch/jni/EtNative.java \
  src/test/resources/models/openvino
git commit -m "feat: add metadata-only backend detection

pteUsesBackend reads a .pte's method_meta and reports whether forward is
delegated to a named backend, without constructing an EtRuntime. That
distinction is the point: EtRuntime's ctor calls load_forward(), which for a
delegated model is delegate init, and OpenVINO's delegate init is a dlopen
under std::call_once with no retry. Anything that must act first has to ask
before loadModule.

A missing file throws rather than reporting false, which would otherwise be
indistinguishable from 'needs no delegate'."
```

---

### Task 2: Link the delegate, capability-keyed

**Files:**
- Modify: `native/CMakeLists.txt` (link block near the existing `xnnpack` linkage)
- Test: `native/tests/openvino_linkage.sh` (new)

**Interfaces:**
- Consumes: nothing.
- Produces: `libopenvino_backend.a` linked into `executorch_djl` wherever the target exists, and `OpenvinoBackend` registered at runtime. Task 4's guard depends on the registration being real.

- [ ] **Step 1: Write the failing test**

Create `native/tests/openvino_linkage.sh`:

```bash
#!/usr/bin/env bash
# The OpenVINO delegate must be linked into the shim wherever the runtime tarball provides it, and
# must be ABSENT wherever it does not -- both are correctness, not just presence. Asserted against
# the built artifact rather than the CMake source, so a link line that silently GC's the
# registration fails here too.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

SO="src/main/resources/native/linux-x86_64/libexecutorch_djl.so"
[ -f "${SO}" ] || { echo "SKIP: no staged linux-x86_64 shim"; exit 0; }

RUNTIME_LIB="native/build/_deps/et_runtime-src/lib/libopenvino_backend.a"
if [ -f "${RUNTIME_LIB}" ]; then
  nm -C --defined-only "${SO}" 2>/dev/null | grep -q 'OpenvinoBackend' \
    || fail "runtime ships libopenvino_backend.a but the shim carries no OpenvinoBackend symbols"
  echo "PASS: OpenVINO delegate linked"
else
  echo "PASS: runtime ships no OpenVINO delegate on this platform; nothing to link"
fi
```

```bash
chmod +x native/tests/openvino_linkage.sh
```

- [ ] **Step 2: Run to verify it fails**

Run: `./native/tests/openvino_linkage.sh`

Expected: `FAIL: runtime ships libopenvino_backend.a but the shim carries no OpenvinoBackend symbols`.

- [ ] **Step 3: Add the link block**

In `native/CMakeLists.txt`, after the existing `target_link_libraries` for the shim:

```cmake
# The OpenVINO delegate ships only in tarballs that were built with it -- linux-x86_64 today.
# Keyed on the TARGET rather than on ET_PLATFORM so a platform that later gains the delegate works
# with no edit here; upstream is actively exploring Windows.
#
# Deliberately NOT added to the post-link XNNPACK registration assertion: that guard exists to catch
# the XNNPACK registration being garbage-collected out of the .so, and folding in a target that
# legitimately does not exist on most platforms would convert a real guard into a platform
# conditional and weaken it.
#
# The archive is static and resolves the OpenVINO C API by dlopen at first use, so this adds no
# DT_NEEDED and drags in no OpenVINO shared object at link time.
if(TARGET openvino_backend)
  target_link_libraries(executorch_djl PRIVATE
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,openvino_backend>")
  message(STATUS "OpenVINO delegate: linked")
else()
  message(STATUS "OpenVINO delegate: not present in this runtime (platform=${ET_PLATFORM})")
endif()
```

Whole-archive is required for the same reason XNNPACK needs it: backend registration happens in a static initializer with no referenced symbol, so a normal link drops the object file.

- [ ] **Step 4: Rebuild and verify the test passes**

```bash
./native/local_build_wrapper.sh
./native/tests/openvino_linkage.sh
```

Expected: `PASS: OpenVINO delegate linked`.

- [ ] **Step 5: Measure the size cost (spec verify-item 2)**

```bash
ls -l src/main/resources/native/linux-x86_64/libexecutorch_djl.so
```

Record the size against the pre-change 12 MB. If the growth is large enough to be objectionable for consumers who never use OpenVINO, **stop and escalate** — the spec's single-shim decision assumed it was small, and a second shim variant reopens as a design question rather than an implementation detail.

- [ ] **Step 6: Confirm the whole suite still passes**

```bash
./gradlew test
./native/local_build_wrapper.sh native/build_qa.sh 2>&1 | grep -E "All tests passed|assertions:"
```

Expected: BUILD SUCCESSFUL and `All tests passed`. Linking a delegate nothing yet references must be inert.

- [ ] **Step 7: Commit**

```bash
git add native/CMakeLists.txt native/tests/openvino_linkage.sh
git commit -m "build: link the OpenVINO delegate where the runtime provides it

Keyed on if(TARGET openvino_backend) rather than on a platform name, so a
platform that later gains the delegate needs no edit. Whole-archived for the
same reason XNNPACK is: registration happens in a static initializer with no
referenced symbol, so a plain link drops the object.

Deliberately not added to the post-link XNNPACK registration guard -- that
guard catches a GC'd registration, and a target absent on most platforms would
turn it into a platform conditional.

openvino_linkage.sh asserts against the built artifact in both directions:
linked where the runtime ships the archive, absent where it does not."
```

---

### Task 3: The bundle jar variant

**Files:**
- Modify: `native/build.sh` (stage the bundle beside the shim)
- Modify: `build.gradle.kts:214-280` (new jar task + GMM variant)
- Test: `native/tests/openvino_bundle_staging.sh` (new)

**Interfaces:**
- Consumes: `ET_RUNTIME_OPENVINO_URL` / `_SHA256` / `_VERSION` / `_PLATFORM` from `native/cmake/EtRuntimePin.cmake`.
- Produces: `build/native-staging/<platform>/openvino/` containing the 7 libraries, `BUILDINFO`, `MANIFEST`, and `licenses/`; and a `nativeJar-<platform>-openvino` task publishing capability `org.measly:djl-executorch-engine-<platform>-openvino`. Task 5 reads `/native/<platform>/openvino/MANIFEST` from the classpath.

- [ ] **Step 1: Write the failing test**

Create `native/tests/openvino_bundle_staging.sh`:

```bash
#!/usr/bin/env bash
# The staged bundle must be usable as-is: one flat directory, every library present, no symlink
# required, and a MANIFEST whose version agrees with the pin. A bundle that is merely "downloaded"
# but split across directories or missing a library fails at model load with an import error that
# names none of these causes.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

DIR="build/native-staging/linux-x86_64/openvino"
[ -d "${DIR}" ] || { echo "SKIP: bundle not staged"; exit 0; }

# The ABI suffix is DERIVED from BUILDINFO, never hardcoded: it tracks the OpenVINO version
# (2025.4.1 -> 2541), so a hardcoded literal would make this test a thing to edit on every bump,
# and a stale one would fail with "missing library" rather than "you bumped OpenVINO".
abi="$(grep -oP '^ov_abi=\K.*' "${DIR}/BUILDINFO")"
[ -n "${abi}" ] || fail "BUILDINFO carries no ov_abi"

# The SET of libraries is the part worth reviewing on a version bump -- an OpenVINO release can add
# or drop a transitive dependency, and a missing one fails at model load with an error naming none
# of this. Keep in sync with OpenVinoRuntime.LIBS; docs/openvino-version-bump.md is the checklist.
for f in "libopenvino_c.so.${abi}" "libopenvino.so.${abi}" libopenvino_intel_cpu_plugin.so \
         "libopenvino_ir_frontend.so.${abi}" libtbb.so.12 libtbbbind_2_5.so.3 libhwloc.so.15; do
  [ -f "${DIR}/lib/${f}" ] || fail "missing library: ${f} (see docs/openvino-version-bump.md)"
done

# Nothing may be shipped that no one enumerated: an unlisted library means the bundle grew and
# OpenVinoRuntime.LIBS will not extract it, which fails at dlopen rather than here.
count="$(find "${DIR}/lib" -maxdepth 1 -type f | wc -l)"
[ "${count}" -eq 7 ] \
  || fail "expected 7 libraries, found ${count} -- the bundle changed; see docs/openvino-version-bump.md"

# Flat, not nested: RPATH=$ORIGIN is what resolves the graph.
find "${DIR}/lib" -mindepth 1 -type d | grep -q . && fail "lib/ must be flat, found a subdirectory"

[ -f "${DIR}/BUILDINFO" ] || fail "missing BUILDINFO"
[ -f "${DIR}/MANIFEST" ]  || fail "missing MANIFEST"
[ -d "${DIR}/licenses" ]  || fail "missing licenses/"

pin_ver="$(grep -oP 'set\(ET_RUNTIME_OPENVINO_VERSION "\K[^"]+' native/cmake/EtRuntimePin.cmake)"
man_ver="$(grep -oP '^openvino_version=\K.*' "${DIR}/MANIFEST")"
[ "${pin_ver}" = "${man_ver}" ] || fail "MANIFEST openvino_version=${man_ver} != pin ${pin_ver}"

echo "PASS: openvino bundle staging"
```

```bash
chmod +x native/tests/openvino_bundle_staging.sh
```

- [ ] **Step 2: Run to verify it skips (nothing staged yet)**

Run: `./native/tests/openvino_bundle_staging.sh`

Expected: `SKIP: bundle not staged`. That is the correct red state here — the assertions cannot run until Step 3 produces something.

- [ ] **Step 3: Stage the bundle in `native/build.sh`**

Add near the existing license-staging block, after the shim is copied:

```bash
# --- OpenVINO runtime bundle (optional, published as a separate opt-in jar) ---
# Fetched here rather than by CMake because nothing links against it: the delegate dlopens the C
# API at runtime. Guarded on the pin declaring a bundle for THIS row, so a release that published
# none, or a platform the bundle does not cover, stages nothing and the jar task skips.
PIN="native/cmake/EtRuntimePin.cmake"
OV_PLATFORM="$(grep -oP 'set\(ET_RUNTIME_OPENVINO_PLATFORM "\K[^"]+' "${PIN}" || true)"
if [ -n "${OV_PLATFORM}" ] && [ "${OV_PLATFORM}" = "${OUT_PLATFORM}" ]; then
  OV_URL="$(grep -oPz 'set\(ET_RUNTIME_OPENVINO_URL\s*\n?\s*"\K[^"]+' "${PIN}" | tr -d '\0')"
  OV_SHA="$(grep -oP 'set\(ET_RUNTIME_OPENVINO_SHA256 "\K[^"]+' "${PIN}")"
  OV_VER="$(grep -oP 'set\(ET_RUNTIME_OPENVINO_VERSION "\K[^"]+' "${PIN}")"
  OV_OUT="${OUT}/openvino"
  TARBALL="native/build/openvino-runtime.tar.gz"

  curl -fsSL -o "${TARBALL}" "${OV_URL}"
  echo "${OV_SHA}  ${TARBALL}" | sha256sum -c - \
    || { echo "OpenVINO bundle SHA256 mismatch -- refusing to stage"; exit 1; }

  rm -rf "${OV_OUT}"
  mkdir -p "${OV_OUT}"
  # --strip-components=1 drops the single top-level dir, keeping lib/, licenses/ and BUILDINFO.
  tar xzf "${TARBALL}" --strip-components=1 -C "${OV_OUT}"
  # The symlink is deliberately not shipped: jars do not preserve symlinks, and it is unnecessary --
  # OPENVINO_LIB_PATH names the versioned file directly and $ORIGIN resolves the rest. Verified
  # against this exact bundle.
  rm -f "${OV_OUT}/lib/libopenvino_c.so"

  {
    echo "openvino_version=${OV_VER}"
    echo "tarball_sha256=${OV_SHA}"
    echo "tarball_url=${OV_URL}"
  } > "${OV_OUT}/MANIFEST"

  echo "OpenVINO bundle staged: ${OV_OUT} ($(du -sh "${OV_OUT}" | cut -f1))"
else
  echo "OpenVINO bundle: pin declares none for ${OUT_PLATFORM}; skipping"
fi
```

- [ ] **Step 4: Rebuild and verify the staging test passes**

```bash
./native/local_build_wrapper.sh
./native/tests/openvino_bundle_staging.sh
```

Expected: `PASS: openvino bundle staging`.

- [ ] **Step 5: Add the jar task and GMM variant**

In `build.gradle.kts`, after the `nativeJarTasks` block:

```kotlin
// The OpenVINO runtime ships as a SEPARATE opt-in variant, never folded into the platform jar:
// it is ~21 MB compressed and ~72 MB extracted, for a delegate most consumers never load. A
// consumer opts in by requesting the capability.
//
// Registered for every platform but only produced where build.sh staged a bundle -- the pin decides
// which platforms have one, so this needs no platform name.
val openvinoJarTasks = nativePlatforms.map { platform ->
  tasks.register<Jar>("nativeJar-${platform}-openvino") {
    archiveClassifier.set("${platform}-openvino")
    from(nativeStaging.map { it.dir("${platform}/openvino") }) {
      exclude("licenses/**")
      into("native/${platform}/openvino")
    }
    from(nativeStaging.map { it.dir("${platform}/openvino/licenses") }) {
      into("META-INF/licenses/openvino-runtime")
    }
    val bundleDir = nativeStaging.get().dir(platform).dir("openvino").asFile
    onlyIf { bundleDir.isDirectory }
    doFirst { // A jar with the libraries but no notices is not shippable
      require(File(bundleDir, "licenses").isDirectory) {
        "Missing OpenVINO third-party notices for ${platform}: ${bundleDir}/licenses"
      }
    }
  }
}
```

And after the existing `nativeVariants` block:

```kotlin
val openvinoVariants = nativePlatforms.map { platform ->
  val osFamily = platform.substringBefore("-")
  val arch = platform.substringAfter("-")
  configurations.consumable("openvinoRuntimeElements-${platform}") {
    attributes {
      attribute(Category.CATEGORY_ATTRIBUTE, objects.named(Category.LIBRARY))
      attribute(Usage.USAGE_ATTRIBUTE, objects.named(Usage.JAVA_RUNTIME))
      attribute(Bundling.BUNDLING_ATTRIBUTE, objects.named(Bundling.EXTERNAL))
      attribute(LibraryElements.LIBRARY_ELEMENTS_ATTRIBUTE, objects.named(LibraryElements.JAR))
      attribute(OperatingSystemFamily.OPERATING_SYSTEM_ATTRIBUTE, objects.named(osFamily))
      attribute(
        MachineArchitecture.ARCHITECTURE_ATTRIBUTE,
        objects.named(if (arch == "aarch64") MachineArchitecture.ARM64 else MachineArchitecture.X86_64)
      )
    }
    outgoing {
      capability("${project.group}:djl-executorch-engine-${platform}-openvino:${project.version}")
      artifact(tasks.named("nativeJar-${platform}-openvino"))
    }
  }
}
```

Register these with the java component exactly as the existing `nativeVariants.forEach` block does, in the same `AdhocComponentWithVariants` `apply` block.

- [ ] **Step 6: Verify the jar builds and contains what it should**

```bash
./gradlew nativeJar-linux-x86_64-openvino
unzip -l build/libs/*linux-x86_64-openvino.jar | head -20
unzip -l build/libs/*linux-x86_64-openvino.jar | grep -c 'META-INF/licenses/openvino-runtime'
```

Expected: 7 libraries plus `BUILDINFO` and `MANIFEST` under `native/linux-x86_64/openvino/`, no `libopenvino_c.so` symlink entry, and 5 license files. Jar size ~21 MB.

- [ ] **Step 7: Verify the standard jar did NOT grow**

```bash
./gradlew nativeJar-linux-x86_64
unzip -l build/libs/*linux-x86_64.jar | grep -c openvino
```

Expected: `0`. If any OpenVINO file appears in the standard jar, the opt-in property is broken and the whole packaging decision is void.

- [ ] **Step 8: Commit**

```bash
git add native/build.sh build.gradle.kts native/tests/openvino_bundle_staging.sh
git commit -m "build: publish the OpenVINO runtime as an opt-in jar variant

The bundle is ~21 MB compressed and ~72 MB extracted for a delegate most
consumers never load, so it ships as its own GMM variant with its own
capability rather than inside the platform jar. Registered for every platform
but produced only where the pin declares a bundle, so it carries no platform
name.

build.sh fetches and SHA256-verifies the tarball, flattens it, and drops the
libopenvino_c.so symlink: jars do not preserve symlinks and it is unnecessary --
OPENVINO_LIB_PATH names the versioned file and \$ORIGIN resolves the rest."
```

---

### Task 4: The C++ guard

**Files:**
- Modify: `native/core/et_runtime.cpp` (guard inside the `EtRuntime` ctor, before `load_forward()`)
- Test: `native/test/et_runtime_test.cpp`

**Interfaces:**
- Consumes: `pteUsesBackend` (Task 1), the linked delegate (Task 2).
- Produces: an `EtRuntime` ctor that throws before delegate init when OpenVINO cannot possibly work. Task 5's Java layer relies on this as its backstop.

- [ ] **Step 1: Write the failing test**

Append to `native/test/et_runtime_test.cpp`:

```cpp
// This is the test that protects the process. Without the guard, constructing an EtRuntime over an
// OpenVINO model with OPENVINO_LIB_PATH unset reaches OpenvinoBackend::init, whose dlopen runs
// under std::call_once and never retries -- so the FIRST bad attempt poisons every later attempt in
// this process, including correct ones. Catch2 runs all cases in one process, which is exactly the
// blast radius this prevents.
TEST_CASE("openvino: an unconfigured OPENVINO_LIB_PATH is refused before delegate init") {
  unsetenv("OPENVINO_LIB_PATH");
  REQUIRE_THROWS([] { EtRuntime rt(OPENVINO_TINY_PTE_PATH); }());
}

TEST_CASE("openvino: OPENVINO_LIB_PATH pointing at a directory is refused") {
  // Upstream's documented top mistake. The error the delegate would otherwise produce mentions
  // LD_LIBRARY_PATH, which reads like it wants a directory. It does not -- it wants the file.
  setenv("OPENVINO_LIB_PATH", "/tmp", 1);
  REQUIRE_THROWS([] { EtRuntime rt(OPENVINO_TINY_PTE_PATH); }());
  unsetenv("OPENVINO_LIB_PATH");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `./native/local_build_wrapper.sh native/build_qa.sh 2>&1 | grep -E "FAILED|assertions:"`

Expected: both FAIL — no throw, because the load reaches the delegate. Note the run may also emit OpenVINO's own error output; that is the unguarded behaviour being replaced.

- [ ] **Step 3: Add the backend-availability helper**

In `native/core/et_runtime.cpp`, as a file-local helper in the existing anonymous namespace, above the constructor (it must be defined before the guard in Step 4 uses it):

```cpp
// Whether a backend is registered in this build. Registration is link-time, so this answers "was
// the delegate compiled in", not "is it configured". Signatures per
// runtime/backend/interface.h:179,184 in the pinned runtime -- note both are size_t-indexed.
bool isBackendAvailable(const char* name) {
  const size_t n = executorch::ET_RUNTIME_NAMESPACE::get_num_registered_backends();
  for (size_t i = 0; i < n; ++i) {
    const auto backendName = executorch::ET_RUNTIME_NAMESPACE::get_backend_name(i);
    if (backendName.ok() && std::strcmp(*backendName, name) == 0) {
      return true;
    }
  }
  return false;
}
```

- [ ] **Step 4: Add the guard**

In `native/core/et_runtime.cpp`, inside the `EtRuntime` constructor, immediately **before** the `state_->module.load_forward()` call at line ~138.

Note this calls `state_->module.method_meta("forward")` directly rather than reading `state_->meta`: that field is *our own* `measly::et::MethodMeta` struct and it is not built until after `load_forward()` returns. Reading ExecuTorch's own metadata here is safe and is already documented in this file at lines 128-129 — `Module::method_meta()` is program-level, calling `load()` and then `program_->method_meta()`, never `load_method`. That documented property is exactly what lets the guard run before delegate init.

```cpp
  // Refuse an OpenVINO-delegated model that cannot possibly succeed, BEFORE load_forward() -- which
  // is delegate init. This matters more than a typical precondition check: OpenvinoBackend resolves
  // the OpenVINO C API with dlopen under std::call_once and never retries, so a failure that
  // reaches it leaves the whole process broken until restart. Raising here keeps the failure an
  // ordinary exception and the process usable.
  //
  // Duplicated by the Java layer deliberately: EtNative is public and bypasses EtModel, and our own
  // tests call it directly.
  auto etMeta = state_->module.method_meta("forward");
  if (etMeta.ok() && etMeta->uses_backend("OpenvinoBackend")) {
    if (!isBackendAvailable("OpenvinoBackend")) {
      throw std::runtime_error(
          "This .pte uses the OpenvinoBackend delegate, which this build does not provide. "
          "The OpenVINO delegate ships only where the runtime tarball was built with it. "
          "Re-export without the OpenVINO partitioner to run here.");
    }
    const char* lib = std::getenv("OPENVINO_LIB_PATH");
    if (lib == nullptr || *lib == '\0') {
      throw std::runtime_error(
          "This .pte uses the OpenvinoBackend delegate, but OPENVINO_LIB_PATH is not set. "
          "Set it to the FULL PATH OF THE LIBRARY FILE (not a directory) before the first "
          "inference, or add the djl-executorch-engine <platform>-openvino artifact and load "
          "through EtModel, which resolves it for you.");
    }
    struct stat st {};
    if (stat(lib, &st) != 0 || !S_ISREG(st.st_mode)) {
      throw std::runtime_error(
          std::string("OPENVINO_LIB_PATH does not name a readable file: '") + lib +
          "'. It must be the full path to the library FILE, not the directory containing it.");
    }
  }
```

Add `#include <sys/stat.h>`, `#include <cstdlib>` and `#include <cstring>` to the includes.

- [ ] **Step 5: Run to verify the tests pass**

Run: `./native/local_build_wrapper.sh native/build_qa.sh 2>&1 | grep -E "All tests passed|FAILED|assertions:"`

Expected: `All tests passed`.

- [ ] **Step 6: Commit**

```bash
git add native/core/et_runtime.cpp native/test/et_runtime_test.cpp
git commit -m "feat: refuse misconfigured OpenVINO loads before delegate init

OpenvinoBackend resolves its C API with dlopen under std::call_once and never
retries, so the first bad attempt poisons every later one in the process. The
guard sits inside EtRuntime's ctor between Module construction and
load_forward() -- the last point before delegate init, and one where method_meta
is already available so uses_backend costs nothing.

Three refusals: the delegate is not linked in this build; OPENVINO_LIB_PATH is
unset; OPENVINO_LIB_PATH does not name a regular file. The last is upstream's
documented top mistake, because the error otherwise produced mentions
LD_LIBRARY_PATH and reads like it wants a directory."
```

---

### Task 5: Java bundle extraction and `OPENVINO_LIB_PATH` resolution

**Files:**
- Create: `src/main/java/org/measly/executorch/engine/OpenVinoRuntime.java`
- Modify: `native/jni/executorch_djl_jni.cpp` (a `setOpenVinoLibPathIfAbsent` entry point)
- Modify: `src/main/java/org/measly/executorch/jni/EtNative.java`
- Modify: `src/main/java/org/measly/executorch/engine/EtModel.java:64-66` (probe before `loadModule`)
- Modify: `src/main/java/org/measly/executorch/engine/LibUtils.java` (expose `cacheRoot()` to the package)
- Test: `src/test/java/org/measly/executorch/engine/OpenVinoRuntimeTest.java`

**Interfaces:**
- Consumes: `EtNative.pteUsesBackend` (Task 1), the staged bundle resource `/native/<platform>/openvino/` (Task 3).
- Produces: `OpenVinoRuntime.ensureReady(Path ptePath)`, called from `EtModel.load`; and `OpenVinoRuntime.bundleAvailable() -> boolean` and `OpenVinoRuntime.resolvedLibPath() -> String` for Task 8's diagnostics.

- [ ] **Step 1: Write the failing test**

Create `src/test/java/org/measly/executorch/engine/OpenVinoRuntimeTest.java`:

```java
package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.measly.executorch.TestSupport;

@Tag("openvino")
class OpenVinoRuntimeTest {

    @Test
    void extractsTheBundleToAFlatDirectoryAndResolvesTheVersionedLibrary() throws Exception {
        TestSupport.assumeOpenVinoBundleAvailable();

        Path dir = OpenVinoRuntime.ensureExtracted();
        assertNotNull(dir);
        assertTrue(Files.isDirectory(dir), "bundle must extract to a directory: " + dir);

        // Flat, not nested: RPATH=$ORIGIN only resolves siblings.
        try (var entries = Files.list(dir)) {
            assertTrue(
                    entries.noneMatch(Files::isDirectory),
                    "the library directory must be flat; $ORIGIN does not search subdirectories");
        }

        String lib = OpenVinoRuntime.resolvedLibPath();
        assertNotNull(lib);
        assertTrue(Files.isRegularFile(Paths.get(lib)), "must name a file, not a directory: " + lib);
        // The versioned file, never an unversioned symlink: jars do not carry symlinks, so the
        // extraction never creates one and the resolved path must not depend on one existing.
        assertTrue(lib.contains(".so."), "must resolve the versioned library: " + lib);
    }

    @Test
    void repeatedExtractionIsIdempotentAndReturnsTheSameDirectory() throws Exception {
        TestSupport.assumeOpenVinoBundleAvailable();
        assertEquals(OpenVinoRuntime.ensureExtracted(), OpenVinoRuntime.ensureExtracted());
    }
}
```

Add to `src/test/java/org/measly/executorch/TestSupport.java`:

```java
    /** Skips when the OpenVINO bundle jar is not on the classpath. */
    public static void assumeOpenVinoBundleAvailable() {
        assumeNativeLibraryAvailable();
        org.junit.jupiter.api.Assumptions.assumeTrue(
                org.measly.executorch.engine.OpenVinoRuntime.bundleAvailable(),
                "OpenVINO bundle jar not on the classpath");
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `./gradlew openvinoTest --tests '*OpenVinoRuntimeTest*'`

Expected: compilation failure — `OpenVinoRuntime` does not exist. (The `openvinoTest` task arrives in Step 6; until then use `./gradlew compileTestJava` to see the same failure.)

- [ ] **Step 3: Make `cacheRoot` reachable**

In `LibUtils.java`, change `static Path cacheRoot()` to keep package-private visibility (it already is) and add to its javadoc:

```java
     * <p>Shared with {@link OpenVinoRuntime}, which extracts the OpenVINO bundle into a sibling
     * subdirectory under the same root so one cache location covers every native payload.
```

- [ ] **Step 4: Implement `OpenVinoRuntime`**

Create `src/main/java/org/measly/executorch/engine/OpenVinoRuntime.java`:

```java
package org.measly.executorch.engine;

import ai.djl.engine.EngineException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import org.measly.executorch.jni.EtNative;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Extracts the vendored OpenVINO runtime and points the delegate at it.
 *
 * <p>Exists because a JVM cannot use {@code LD_LIBRARY_PATH}: glibc's loader reads it once at
 * process start, {@code System.getenv} is read-only, and {@code ProcessBuilder} affects only child
 * processes. {@code OPENVINO_LIB_PATH} is read at {@code dlopen} time, which makes it the only
 * mechanism available to us.
 *
 * <p>Everything here happens once per process and must complete before the first OpenVINO
 * inference: the delegate's {@code dlopen} runs under {@code std::call_once} with no retry, so a
 * late or failed configuration cannot be repaired without restarting the JVM.
 */
public final class OpenVinoRuntime {

    private static final Logger logger = LoggerFactory.getLogger(OpenVinoRuntime.class);

    /** Backend id as ExecuTorch spells it — lowercase {@code v}. */
    public static final String BACKEND = "OpenvinoBackend";

    private static final int BUF = 64 * 1024;
    private static final String MANIFEST = "MANIFEST";
    private static final String BUILDINFO = "BUILDINFO";

    // The libraries, in no particular order; all must land in one flat directory because each
    // carries RPATH=$ORIGIN and $ORIGIN does not search subdirectories.
    private static final List<String> LIBS = List.of(
            "libopenvino.so",
            "libopenvino_c.so",
            "libopenvino_intel_cpu_plugin.so",
            "libopenvino_ir_frontend.so",
            "libtbb.so.12",
            "libtbbbind_2_5.so.3",
            "libhwloc.so.15");

    private static Path extracted;
    private static String libPath;
    private static boolean configured;

    private OpenVinoRuntime() {}

    /** @return true if the OpenVINO bundle jar is on the classpath for this platform */
    public static boolean bundleAvailable() {
        return OpenVinoRuntime.class.getResource(resourceBase() + MANIFEST) != null;
    }

    /**
     * Ensures the delegate can load, if and only if this model needs it.
     *
     * <p>Called before {@code loadModule}, never after: that call constructs the native runtime,
     * whose constructor runs delegate init.
     *
     * @param ptePath the model about to be loaded
     */
    static synchronized void ensureReady(Path ptePath) {
        if (configured) {
            return; // one-shot per process; the delegate's dlopen is too
        }
        String existing = System.getenv("OPENVINO_LIB_PATH");
        boolean overridden = existing != null && !existing.isEmpty();
        if (!overridden && !bundleAvailable()) {
            return; // nothing to configure and nothing to check; the native guard reports it
        }
        // The probe comes BEFORE the override check, not after. Validating an override for every
        // model would fail a pure-XNNPACK workload that happens to carry a stale OPENVINO_LIB_PATH
        // in its environment -- punishing a caller for a variable their models never touch.
        if (!EtNative.pteUsesBackend(ptePath.toString(), BACKEND)) {
            return; // not an OpenVINO model; extract nothing, validate nothing
        }
        if (overridden) {
            // An operator override always wins -- but a wrong one is worth catching here rather
            // than letting it reach the delegate, whose dlopen is once-only. Checked explicitly
            // rather than with a set-if-absent idiom, whose eager default would extract 72 MB even
            // when the variable is already correct.
            validateOverride(existing);
            configured = true;
            libPath = existing;
            return;
        }
        try {
            Path dir = ensureExtracted();
            String ours = resolvedLibPath();
            // Use what the native side reports as in force, not what we asked for. If something
            // installed a path after JVM start -- invisible to the System.getenv check above --
            // that path wins and this is how we find out.
            String effective = EtNative.setOpenVinoLibPathIfAbsent(ours);
            if (effective != null && !effective.equals(ours)) {
                logger.info(
                        "OpenVINO runtime already configured elsewhere; honouring {} instead of the"
                                + " vendored {}", effective, ours);
            }
            libPath = (effective == null) ? ours : effective;
            configured = true;
            logger.info("OpenVINO runtime resolved: {}", libPath);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to extract the OpenVINO runtime bundle", e);
        }
    }

    /**
     * Rejects an {@code OPENVINO_LIB_PATH} that cannot work, before the delegate sees it.
     *
     * <p>Deliberately does <b>not</b> fall back to the vendored bundle. An operator who set this
     * variable meant to, and quietly substituting our runtime for theirs would turn a typo into a
     * silently different OpenVINO — which, because a {@code .pte} embeds a precompiled blob, could
     * surface much later as an import failure. Failing here names the value they actually set.
     *
     * @param value the environment variable's contents
     * @throws EngineException if it does not name a readable regular file
     */
    static void validateOverride(String value) {
        Path candidate;
        try {
            candidate = Paths.get(value);
        } catch (InvalidPathException e) {
            throw new EngineException(
                    "OPENVINO_LIB_PATH is not a usable path: '" + value + "'. It must be the full "
                            + "path to the OpenVINO C library FILE.", e);
        }
        if (Files.isDirectory(candidate)) {
            // Upstream's documented top mistake, and an easy one to make: the error the delegate
            // would otherwise produce mentions LD_LIBRARY_PATH, which reads like it wants a
            // directory. It does not.
            throw new EngineException(
                    "OPENVINO_LIB_PATH points at a directory: '" + value + "'. It must be the full "
                            + "path to the library FILE itself, e.g. <dir>/libopenvino_c.so."
                            + "<abi>.");
        }
        if (!Files.isRegularFile(candidate) || !Files.isReadable(candidate)) {
            throw new EngineException(
                    "OPENVINO_LIB_PATH does not name a readable file: '" + value + "'."
                            + (bundleAvailable()
                                    ? " Unset it to use the OpenVINO runtime vendored in this"
                                            + " engine's openvino artifact."
                                    : " Set it to the full path of the OpenVINO C library file."));
        }
    }

    /**
     * Extracts the bundle into the content-addressed cache, once.
     *
     * @return the flat directory holding the libraries
     * @throws IOException if extraction fails
     */
    static synchronized Path ensureExtracted() throws IOException {
        if (extracted != null) {
            return extracted;
        }
        String sha = manifest().getProperty("tarball_sha256");
        if (sha == null || sha.isEmpty()) {
            throw new IOException("OpenVINO bundle MANIFEST carries no tarball_sha256");
        }
        // Keyed on the upstream tarball hash rather than on a digest we compute: LibUtils hashes its
        // own resource, but that costs a full read on every JVM start, which is fine at 12 MB and
        // not at 72 MB on the model-load path. A cache hit here reads nothing.
        Path target = LibUtils.cacheRoot().resolve("openvino").resolve(sha);
        if (!Files.isDirectory(target)) {
            publish(target);
        }
        extracted = target;
        return target;
    }

    /** @return absolute path of the versioned OpenVINO C library, or null before extraction */
    public static synchronized String resolvedLibPath() {
        if (libPath != null) {
            return libPath;
        }
        if (extracted == null) {
            return null;
        }
        // The ABI suffix comes from BUILDINFO, never hardcoded: it changes with the OpenVINO
        // version and a stale literal would fail at dlopen with a confusing "file not found".
        String abi = buildInfo().getProperty("ov_abi");
        libPath = extracted.resolve("libopenvino_c.so." + abi).toAbsolutePath().toString();
        return libPath;
    }

    // Extract into a staging directory, then publish by atomic rename. Nothing is ever loaded out
    // of the staging directory, which is what lets a loser in a race delete its own work even on a
    // platform that refuses to delete a loaded library.
    private static void publish(Path target) throws IOException {
        Files.createDirectories(target.getParent());
        Path staging = Files.createTempDirectory(target.getParent(), "staging-");
        try {
            String abi = buildInfo().getProperty("ov_abi");
            for (String lib : LIBS) {
                // Versioned names carry the ABI; already-versioned ones (libtbb.so.12) do not.
                String name = lib.endsWith(".so") ? lib + "." + abi : lib;
                copy(resourceBase() + "lib/" + name, staging.resolve(name));
            }
            copy(resourceBase() + BUILDINFO, staging.resolve(BUILDINFO));
            try {
                Files.move(staging, target, StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException e) {
                // A concurrent JVM published first. The path is content-addressed, so its bytes are
                // ours byte-for-byte; adopt rather than overwrite a directory another process may
                // already have loaded from.
                if (!Files.isDirectory(target)) {
                    throw e;
                }
            }
        } finally {
            deleteRecursivelyIfPresent(staging);
        }
    }

    private static void copy(String resource, Path target) throws IOException {
        try (InputStream is = open(resource); OutputStream os = Files.newOutputStream(target)) {
            byte[] buf = new byte[BUF];
            int n;
            while ((n = is.read(buf)) != -1) {
                os.write(buf, 0, n);
            }
        }
    }

    private static void deleteRecursivelyIfPresent(Path dir) throws IOException {
        if (!Files.exists(dir)) {
            return;
        }
        List<Path> paths = new ArrayList<>();
        try (var walk = Files.walk(dir)) {
            walk.forEach(paths::add);
        }
        for (int i = paths.size() - 1; i >= 0; i--) {
            Files.deleteIfExists(paths.get(i));
        }
    }

    private static Properties manifest() {
        return readProperties(resourceBase() + MANIFEST);
    }

    private static Properties buildInfo() {
        return readProperties(resourceBase() + BUILDINFO);
    }

    private static Properties readProperties(String resource) {
        Properties props = new Properties();
        try (InputStream is = open(resource)) {
            props.load(new java.io.InputStreamReader(is, StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new IllegalStateException("Cannot read " + resource, e);
        }
        return props;
    }

    private static InputStream open(String resource) {
        InputStream is = OpenVinoRuntime.class.getResourceAsStream(resource);
        if (is == null) {
            throw new IllegalStateException("OpenVINO bundle resource missing: " + resource);
        }
        return is;
    }

    private static String resourceBase() {
        return "/native/" + LibUtils.platform() + "/openvino/";
    }
}
```

- [ ] **Step 5: Add the `setOpenVinoLibPathIfAbsent` JNI entry point**

In `native/jni/executorch_djl_jni.cpp`, beside the other free functions:

```cpp
// Sets OPENVINO_LIB_PATH only if it is not already set, and reports the value in force afterwards.
// This exists because a JVM has no other way to configure it: System.getenv is read-only and
// glibc's loader read LD_LIBRARY_PATH once, long ago. The delegate reads OPENVINO_LIB_PATH at
// dlopen time, so writing it here still lands -- provided it runs before the first OpenVINO
// inference, because that dlopen is once-only.
//
// SET-IF-ABSENT, not overwrite, and the decision is made HERE rather than in Java on purpose.
// std::getenv sees the live environment; Java's System.getenv is a snapshot taken when the JVM
// built its environment map and does not observe a setenv issued afterwards. So a value installed
// natively after JVM start -- by an agent, another library, or this engine under a different
// classloader -- is invisible to the Java check, and an overwrite there would silently replace a
// configuration someone deliberately installed. Only this frame can see the truth, so only this
// frame gets to decide.
//
// Returning the effective value makes that decision observable: the caller learns which path is
// actually in force rather than assuming its own argument won.
extern "C" JNIEXPORT jstring JNICALL
Java_org_measly_executorch_jni_EtNative_setOpenVinoLibPathIfAbsent(
    JNIEnv* env, jclass, jstring path) {
  const char* existing = std::getenv("OPENVINO_LIB_PATH");
  if (existing != nullptr && *existing != '\0') {
    return env->NewStringUTF(existing);  // someone got here first; they win
  }
  const char* value = env->GetStringUTFChars(path, nullptr);
  if (value == nullptr) {
    return nullptr;  // OOM already pending; setenv with a null value would be UB
  }
  const int rc = setenv("OPENVINO_LIB_PATH", value, 1);
  std::string effective(value);
  // Released before the throw, not after: nothing is held across a JNI call that can fail.
  env->ReleaseStringUTFChars(path, value);
  if (rc != 0) {
    throwJava(env, "setenv(OPENVINO_LIB_PATH) failed", nullptr);
    return nullptr;
  }
  return env->NewStringUTF(effective.c_str());
}
```

And in `EtNative.java`:

```java
    /**
     * Sets {@code OPENVINO_LIB_PATH} if it is not already set, and returns the value in force.
     *
     * <p>The only mechanism available to a JVM: {@code System.getenv} is read-only, and glibc's
     * loader read {@code LD_LIBRARY_PATH} once at process start. The delegate reads this variable
     * at {@code dlopen} time, so a write from here still lands — but only if it precedes the first
     * OpenVINO inference, because that {@code dlopen} runs once and never retries.
     *
     * <p><b>An already-set value always wins</b>, and that decision is made natively rather than in
     * Java. {@code System.getenv} is a snapshot taken when the JVM built its environment map and
     * does not observe a {@code setenv} issued afterwards, so a Java-side check cannot see a value
     * installed natively after startup. Native {@code getenv} can.
     *
     * <p>Callers must use the <b>returned</b> value rather than assuming their argument was
     * applied — it may be someone else's path.
     *
     * @param path absolute path to the OpenVINO C library file, used only if none is set
     * @return the path actually in force, or {@code null} if it could not be determined
     */
    public static native String setOpenVinoLibPathIfAbsent(String path);
```

- [ ] **Step 6: Register the `openvinoTest` task**

In `build.gradle.kts`, add `"openvino"` to the `excludeTags` list at line 33, then register beside the other tag tasks:

```kotlin
// Forked per class: OPENVINO_LIB_PATH is process environment and the delegate's dlopen is
// once-only, so cases sharing a JVM contaminate each other in ways that present as flakes.
tasks.register<Test>("openvinoTest") {
  group = "verification"
  description = "OpenVINO delegate tests (linux-x86_64 with the openvino bundle)"
  testClassesDirs = sourceSets["test"].output.classesDirs
  classpath = sourceSets["test"].runtimeClasspath
  useJUnitPlatform { includeTags("openvino") }
  forkEvery = 1
}
```

- [ ] **Step 7: Wire the probe into `EtModel.load`**

In `EtModel.java`, immediately before the `EtNative.loadModule` call at line 66 (and before `loadStartNanos`, so extraction time is not charged to load latency):

```java
        // Before loadModule, never after: that call constructs the native runtime, whose ctor calls
        // load_forward() -- delegate init. For OpenVINO that is a dlopen under std::call_once with
        // no retry, so anything we need to configure has to be configured by now. Cheap in the
        // common case: returns immediately once configured, and when no bundle is on the classpath.
        OpenVinoRuntime.ensureReady(modelFile);
```

- [ ] **Step 8: Rebuild and run the tests**

```bash
./native/local_build_wrapper.sh
./gradlew test
./gradlew openvinoTest
```

Expected: `test` BUILD SUCCESSFUL (unchanged behaviour for non-OpenVINO models); `openvinoTest` passes both cases, or skips cleanly if the bundle jar is not on the test classpath. If it skips, add the bundle jar output to the test runtime classpath before proceeding — a silently skipping suite proves nothing.

- [ ] **Step 9: Verify the `System.getenv` snapshot assumption**

The set-if-absent decision lives natively *because* Java's environment view is a startup snapshot. That claim is load-bearing, so confirm it on this JDK rather than trusting it — if `System.getenv` did observe a JNI `setenv`, the native check would be redundant and the design should be simplified rather than left with a rationale that is not true here.

Add a temporary assertion to `OpenVinoRuntimeTest` (delete it once observed):

```java
    @Test
    void javaEnvironmentViewDoesNotObserveANativeSetenv() throws Exception {
        TestSupport.assumeOpenVinoBundleAvailable();
        OpenVinoRuntime.ensureExtracted();
        String applied = EtNative.setOpenVinoLibPathIfAbsent(OpenVinoRuntime.resolvedLibPath());
        System.out.println("native reports: " + applied);
        System.out.println("System.getenv reports: " + System.getenv("OPENVINO_LIB_PATH"));
    }
```

Run: `./gradlew openvinoTest --tests '*OpenVinoRuntimeTest*' -i`

Expected: the native line shows a path; the `System.getenv` line shows `null`. That divergence is the whole reason the decision is native. **If both show the path**, stop and reconsider — the rationale in the JNI comment would be wrong on this JDK and must be corrected rather than left in place. Record what you observed in the commit message either way, then delete the test.

- [ ] **Step 10: Commit**

```bash
git add src/main/java/org/measly/executorch/engine/OpenVinoRuntime.java \
  src/main/java/org/measly/executorch/engine/EtModel.java \
  src/main/java/org/measly/executorch/engine/LibUtils.java \
  src/main/java/org/measly/executorch/jni/EtNative.java \
  src/test/java/org/measly/executorch/engine/OpenVinoRuntimeTest.java \
  src/test/java/org/measly/executorch/TestSupport.java \
  native/jni/executorch_djl_jni.cpp build.gradle.kts
git commit -m "feat: extract the vendored OpenVINO runtime and resolve OPENVINO_LIB_PATH

A JVM cannot use LD_LIBRARY_PATH -- glibc reads it once at process start and
System.getenv is read-only -- so OPENVINO_LIB_PATH set from JNI is the only
mechanism available. It is read at dlopen time, which is late enough to work
and early enough to matter, since that dlopen is once-only.

Extraction is content-addressed on the upstream tarball SHA and published by
atomic directory rename, adopting a concurrent winner. Nothing is ever loaded
out of the staging directory, so the loser can always delete its own work.

The probe runs before loadModule and only when the bundle is present and the
variable is unresolved, so platforms without the delegate and consumers without
the jar pay nothing. An operator-set OPENVINO_LIB_PATH always wins untouched."
```

---

### Task 6: Configuration edge cases and the no-poison guard

**Files:**
- Modify: `src/test/java/org/measly/executorch/engine/OpenVinoRuntimeTest.java`
- Create: `src/test/java/org/measly/executorch/engine/OpenVinoConcurrentExtractionTest.java`

**Interfaces:**
- Consumes: everything from Task 5. Adds no production code — these are the behaviours Task 5 claims but does not yet prove.

Four cases the spec's testing table requires and Task 5 leaves unproven. The first is the most important test in this plan: it is what stands between a correct implementation and one that silently poisons the process.

- [ ] **Step 1: The no-poison guard**

Append to `OpenVinoRuntimeTest.java`:

```java
    @Test
    void probingForABackendDoesNotBurnTheDelegatesOneShotDlopen() throws Exception {
        TestSupport.assumeOpenVinoBundleAvailable();

        Path pte = Paths.get("src/test/resources/models/openvino/openvino_tiny.pte");

        // Probe FIRST, with OPENVINO_LIB_PATH deliberately unresolved. If pteUsesBackend loaded the
        // method rather than just the program, this would run delegate init unconfigured -- and the
        // delegate's dlopen is std::call_once with no retry, so the load below would then fail
        // forever in this JVM no matter how correctly we configure afterwards.
        assertTrue(EtNative.pteUsesBackend(pte.toString(), OpenVinoRuntime.BACKEND));

        // Now configure and load for real. Success here proves the probe consumed nothing.
        try (Model model = Model.newInstance("openvino_tiny", "ExecuTorch")) {
            model.load(pte.getParent(), "openvino_tiny");
        }
    }
```

Add the imports it needs: `ai.djl.Model`, `org.measly.executorch.jni.EtNative`.

This replaces the weaker guard the sibling project uses. Theirs asserts the XNNPACK workspace is 0 after a metadata call, intending to prove the method did not load — but the workspace only grows on the first *execute*, so that assertion would also pass if the method had loaded eagerly. Asserting the behaviour we actually depend on is both stronger and cheaper.

- [ ] **Step 2: Run it**

Run: `./gradlew openvinoTest --tests '*OpenVinoRuntimeTest*'`

Expected: PASS. A failure here means `pteUsesBackend` is loading the method — stop and fix Task 1 rather than adjusting this test.

- [ ] **Step 3: The caller-override and bundle-absent cases**

Append to `OpenVinoRuntimeTest.java`:

```java
    @Test
    void anOperatorSetLibPathIsHonouredUntouched() throws Exception {
        // Cannot be asserted by mutating this JVM's environment -- Java cannot -- so this asserts
        // the decision function instead: given a non-empty existing value, resolution must return
        // it unchanged and must not extract anything.
        String existing = System.getenv("OPENVINO_LIB_PATH");
        Assumptions.assumeTrue(
                existing == null || existing.isEmpty(),
                "this asserts the default path; an inherited OPENVINO_LIB_PATH would mask it");
        // With no override set, a non-OpenVINO model must leave configuration untouched: the
        // bundle is not extracted and no lib path is resolved for a model that never needs one.
        OpenVinoRuntime.ensureReady(Paths.get(TestSupport.addPtePath()));
        assertNull(
                OpenVinoRuntime.resolvedLibPath(),
                "a non-OpenVINO model must not trigger bundle resolution");
    }

    @Test
    void anUnusableLibPathOverrideIsRejectedWithTheValueThatCausedIt() throws Exception {
        // Tested as a pure function because a JVM cannot set its own environment. The end-to-end
        // env path is covered natively in et_runtime_test.cpp, which can call setenv in-process.
        Path realFile = Files.createTempFile("not-a-library", ".so");
        Path dir = Files.createTempDirectory("openvino-dir");
        try {
            EngineException nonexistent = assertThrows(
                    EngineException.class, () -> OpenVinoRuntime.validateOverride("XXX"));
            assertTrue(
                    nonexistent.getMessage().contains("XXX"),
                    "the message must quote the offending value: " + nonexistent.getMessage());

            EngineException directory = assertThrows(
                    EngineException.class,
                    () -> OpenVinoRuntime.validateOverride(dir.toString()));
            assertTrue(
                    directory.getMessage().contains("directory"),
                    "a directory must be called out by name, because the error the delegate would "
                            + "otherwise give mentions LD_LIBRARY_PATH and misleads: "
                            + directory.getMessage());

            // Any readable regular file passes. Validation deliberately stops at "could this be
            // dlopen'd at all" -- proving it is really OpenVINO would mean loading it, which is
            // the once-only operation this check exists to protect.
            assertDoesNotThrow(() -> OpenVinoRuntime.validateOverride(realFile.toString()));
        } finally {
            Files.deleteIfExists(realFile);
            Files.deleteIfExists(dir);
        }
    }

    @Test
    void reportsBundleAvailabilityFromTheClasspathRatherThanThePlatform() {
        TestSupport.assumeNativeLibraryAvailable();
        // A boolean either way is correct -- what must NOT happen is a throw. This runs on every
        // platform, including ones with no bundle, because that is the case whose error path
        // matters most.
        assertDoesNotThrow(OpenVinoRuntime::bundleAvailable);
    }
```

Add imports: `org.junit.jupiter.api.Assumptions`, `ai.djl.engine.EngineException`, `java.nio.file.Files`, and the static imports `assertNull`, `assertDoesNotThrow`, `assertThrows`.

- [ ] **Step 4: Run them**

Run: `./gradlew openvinoTest --tests '*OpenVinoRuntimeTest*'`

Expected: PASS. If `resolvedLibPath()` is non-null after loading a non-OpenVINO model, `ensureReady` is extracting unconditionally — the whole point of the conditional probe is lost, and Task 5's `pteUsesBackend` check is misplaced.

- [ ] **Step 5: Concurrent extraction (spec verify-item 5)**

Create `src/test/java/org/measly/executorch/engine/OpenVinoConcurrentExtractionTest.java`:

```java
package org.measly.executorch.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.measly.executorch.TestSupport;

/**
 * Extraction publishes by atomic directory rename and adopts a concurrent winner. A unit test of
 * the rename would prove nothing — the interesting case is two extractions racing for the same
 * content-addressed path.
 */
@Tag("openvino")
class OpenVinoConcurrentExtractionTest {

    @Test
    void racingExtractionsConvergeOnOneDirectory() throws Exception {
        TestSupport.assumeOpenVinoBundleAvailable();

        int threads = 4;
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            List<Callable<Path>> tasks = new ArrayList<>();
            for (int i = 0; i < threads; i++) {
                tasks.add(OpenVinoRuntime::ensureExtracted);
            }
            List<Future<Path>> results = pool.invokeAll(tasks);
            Path first = results.get(0).get();
            for (Future<Path> f : results) {
                assertEquals(first, f.get(), "every racer must land on the same published directory");
            }
            assertTrue(Files.isDirectory(first));
            // No staging directory may survive. A leaked one means the loser could not clean up,
            // which is the failure mode that would matter on a platform refusing to delete a
            // loaded library.
            try (var siblings = Files.list(first.getParent())) {
                assertTrue(
                        siblings.noneMatch(p -> p.getFileName().toString().startsWith("staging-")),
                        "a staging directory leaked; the loser could not clean up after itself");
            }
        } finally {
            pool.shutdownNow();
        }
    }
}
```

- [ ] **Step 6: Run it**

Run: `./gradlew openvinoTest --tests '*OpenVinoConcurrentExtraction*'`

Expected: PASS. Note this races *threads*, which `ensureExtracted`'s `synchronized` already serialises — so it proves the adoption path is correct but not that two *processes* converge. For the cross-process case, run the task twice concurrently from two shells after clearing the cache:

```bash
rm -rf ~/.cache/executorch-djl/openvino
./gradlew openvinoTest --tests '*OpenVinoConcurrentExtraction*' &
./gradlew openvinoTest --tests '*OpenVinoConcurrentExtraction*' &
wait
```

- [ ] **Step 7: Commit**

```bash
git add src/test/java/org/measly/executorch/engine/OpenVinoRuntimeTest.java \
  src/test/java/org/measly/executorch/engine/OpenVinoConcurrentExtractionTest.java
git commit -m "test: OpenVINO configuration edge cases and the no-poison guard

The guard is the important one: it probes a .pte for the backend with
OPENVINO_LIB_PATH unresolved, then configures and loads the same model for
real. If the probe had loaded the method rather than the program, delegate init
would have run unconfigured and burned the once-only dlopen, making the second
step fail forever in that JVM. Success proves the probe consumed nothing.

This deliberately replaces the weaker guard the sibling project uses -- asserting
the XNNPACK workspace is still 0 after a metadata call, which also holds if the
method DID load, since the arena only grows on first execute.

Also covers: a non-OpenVINO model must not trigger extraction, bundle
availability must never throw, and racing extractions must converge on one
published directory leaving no staging directory behind."
```

---

### Task 7: Fixture parity and the version-coupling guard

**Files:**
- Create: `src/test/resources/models/openvino/MANIFEST`
- Create: `src/test/java/org/measly/executorch/OpenVinoModelIT.java`
- Create: `native/tests/openvino_version_coupling.sh`
- Create: `docs/openvino-version-bump.md`
- Modify: `docs/README.md` (add the runbook to its reference list)

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: end-to-end proof that a delegated model runs and returns correct numbers.

- [ ] **Step 1: Write the fixture manifest**

Get the asset SHA and write `src/test/resources/models/openvino/MANIFEST`:

```bash
gh release download v1.3.1-10 --repo measly-java-learning/executorch-runtime-dist \
  -p 'etnp-openvino-fixtures-1.3.1-2025.4.1.tar.gz.sha256' -D /tmp --clobber
cat /tmp/etnp-openvino-fixtures-1.3.1-2025.4.1.tar.gz.sha256
```

```
tarball_url=https://github.com/measly-java-learning/executorch-runtime-dist/releases/download/v1.3.1-10/etnp-openvino-fixtures-1.3.1-2025.4.1.tar.gz
tarball_sha256=<the sha from the command above>
openvino_version=2025.4.1
executorch_version=1.3.1
```

- [ ] **Step 2: Write the failing parity test**

Create `src/test/java/org/measly/executorch/OpenVinoModelIT.java`:

```java
package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.measly.executorch.engine.EtEngine;

@Tag("openvino")
class OpenVinoModelIT {

    private static final Path DIR = Paths.get("src/test/resources/models/openvino");

    private static float[] readFloats(Path p) throws Exception {
        ByteBuffer bb = ByteBuffer.wrap(Files.readAllBytes(p)).order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[bb.remaining() / Float.BYTES];
        bb.asFloatBuffer().get(out);
        return out;
    }

    @Test
    void delegatedModelMatchesTheEagerGolden() throws Exception {
        TestSupport.assumeOpenVinoBundleAvailable();

        float[] in = readFloats(DIR.resolve("in.bin"));
        float[] golden = readFloats(DIR.resolve("out.bin"));

        // Reported, not asserted: atol=1e-2 alone cannot distinguish "correct in bf16" from
        // "quietly degraded", so the run records which precision it actually got.
        System.out.println("OpenVINO inference precision: " + EtEngine.openVinoInferencePrecision());

        try (Model model = Model.newInstance("openvino_tiny", "ExecuTorch")) {
            model.load(DIR, "openvino_tiny");
            try (NDManager manager = NDManager.newBaseManager("ExecuTorch");
                    NDList inputs = new NDList(manager.create(in, new Shape(1, in.length)));
                    NDList outputs = model.getBlock().forward(null, inputs, false)) {
                float[] actual = outputs.singletonOrThrow().toFloatArray();
                assertEquals(golden.length, actual.length);
                for (int i = 0; i < golden.length; i++) {
                    // atol=1e-2 and DO NOT TIGHTEN. OpenVINO picks its inference precision from the
                    // CPU it lands on, at import time rather than blob-compile time: on
                    // avx512_bf16/AMX hardware it computes in bf16 and lands ~2.5e-3 from this f32
                    // eager golden; elsewhere ~6e-8. Both are correct OpenVINO results. A tolerance
                    // drawn between them asserts which machine CI allocated -- a property this
                    // project does not own -- and fails at random. Upstream hit exactly this
                    // (ea393da in executorch-runtime-dist) after a green and a red run on identical
                    // artifacts. The loose bound still catches zeros, garbage, or the wrong model,
                    // which are orders of magnitude out.
                    assertEquals(golden[i], actual[i], 1e-2, "element " + i);
                }
            }
        }
    }
}
```

- [ ] **Step 3: Run to verify it fails**

Run: `./gradlew openvinoTest --tests '*OpenVinoModelIT*'`

Expected: FAIL — `EtEngine.openVinoInferencePrecision()` does not exist yet (Task 8). To keep this task independently testable, comment out the precision line, confirm the parity assertions pass, then restore it and let Task 8 complete the picture. If parity itself fails, stop: that is a real defect in Tasks 1-5, not a missing accessor.

- [ ] **Step 4: Write the version-coupling guard**

Create `native/tests/openvino_version_coupling.sh`:

```bash
#!/usr/bin/env bash
# The OpenVINO version lives in three places. They must agree, or a rebuild can vendor a runtime
# that cannot import the fixture's precompiled blob -- which surfaces at model load rather than at
# build time. OpenVINO versions independently of ExecuTorch, so an OV re-roll can invalidate the
# committed fixture with no ET bump; this is what makes that a build failure.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
fail() { echo "FAIL: $1"; exit 1; }

pin="$(grep -oP 'set\(ET_RUNTIME_OPENVINO_VERSION "\K[^"]+' native/cmake/EtRuntimePin.cmake)"
[ -n "${pin}" ] || fail "no ET_RUNTIME_OPENVINO_VERSION in the pin"

fixture="$(grep -oP '^openvino_version=\K.*' src/test/resources/models/openvino/MANIFEST)"
[ "${pin}" = "${fixture}" ] \
  || fail "fixture MANIFEST openvino_version=${fixture} != pin ${pin}"

staged="build/native-staging/linux-x86_64/openvino/MANIFEST"
if [ -f "${staged}" ]; then
  bundle="$(grep -oP '^openvino_version=\K.*' "${staged}")"
  [ "${pin}" = "${bundle}" ] || fail "bundle MANIFEST openvino_version=${bundle} != pin ${pin}"
fi

echo "PASS: openvino version coupling (${pin})"
```

```bash
chmod +x native/tests/openvino_version_coupling.sh
./native/tests/openvino_version_coupling.sh
```

Expected: `PASS: openvino version coupling (2025.4.1)`.

- [ ] **Step 5: Write the version-bump runbook**

Create `docs/openvino-version-bump.md`. OpenVINO versions independently of ExecuTorch, so a bump can arrive with no pin bump and vice versa, and the touch points are spread across four directories. This is the list.

````markdown
# Bumping the vendored OpenVINO version

OpenVINO versions independently of ExecuTorch: a runtime-dist release can change
`ET_RUNTIME_OPENVINO_VERSION` without changing the ExecuTorch version, and an ExecuTorch bump can
leave OpenVINO untouched. So this is its own procedure, not a step inside a pin bump.

The failure this prevents is specific. A `.pte` embeds a **precompiled OpenVINO blob**, so vendoring
a runtime the committed fixture's blob cannot be imported by fails at *model load* with
`failed to import model for device 'CPU'` — a message naming none of the causes below.

## What must change together

1. **`native/cmake/EtRuntimePin.cmake`** — generated. Replace it wholesale with the asset from the
   new runtime-dist release; do not hand-edit. This carries
   `ET_RUNTIME_OPENVINO_{VERSION,PLATFORM,URL,SHA256}`.
2. **`src/test/resources/models/openvino/`** — the four fixture members **and** their `MANIFEST`.
   The fixture asset is OpenVINO-version-coupled by name
   (`etnp-openvino-fixtures-<etver>-<ovver>.tar.gz`), so a new OpenVINO means a new fixture. Update
   `openvino_version`, `tarball_url`, and `tarball_sha256` in the `MANIFEST` to match the asset you
   actually unpacked.
3. **`OpenVinoRuntime.LIBS`** — only if the bundle's library set changed. Compare against the new
   tarball's `lib/`. `native/tests/openvino_bundle_staging.sh` fails with a count mismatch when it
   does, which is the signal to come here.

## What must NOT change

- **The ABI suffix is never hardcoded anywhere.** It tracks the version (`2025.4.1` → `2541`) and is
  read from the bundle's `BUILDINFO` (`ov_abi`) by both the extractor and the staging test. If you
  find yourself editing a `2541` literal, something has regressed.
- **No symlink is ever created.** `OPENVINO_LIB_PATH` names the versioned file; `$ORIGIN` resolves
  the rest. Verified against the shipped bundle.
- **`atol=1e-2` in the parity test.** A new OpenVINO does not justify tightening it — the bound is
  about which CPU the test lands on, not which version it runs.

## Verifying the bump

```bash
./native/local_build_wrapper.sh                     # restages the bundle from the new pin
./native/tests/openvino_version_coupling.sh         # pin == fixture == staged bundle
./native/tests/openvino_bundle_staging.sh           # library set and ABI derivation
./gradlew openvinoTest                              # parity against the new fixture
```

If parity fails but everything else passes, the fixture and the runtime disagree — you almost
certainly updated one of items 1 and 2 without the other.
````

Add a pointer to it from `docs/README.md`'s reference list, and reference it from the coupling test's failure message:

```bash
  || fail "fixture MANIFEST openvino_version=${fixture} != pin ${pin} (see docs/openvino-version-bump.md)"
```

- [ ] **Step 6: Commit**

```bash
git add src/test/resources/models/openvino/MANIFEST \
  src/test/java/org/measly/executorch/OpenVinoModelIT.java \
  native/tests/openvino_version_coupling.sh docs/openvino-version-bump.md docs/README.md
git commit -m "test: OpenVINO fixture parity and version coupling

Runs the committed openvino_tiny fixture end to end and compares against the
eager golden at atol=1e-2. That bound must not be tightened: OpenVINO picks
bf16 or f32 from the CPU it lands on, landing ~2.5e-3 or ~6e-8 from the golden
respectively, and both are correct -- a tighter bound asserts which machine CI
allocated. It still catches zeros, garbage, or the wrong model.

openvino_version_coupling.sh asserts the pin, the fixture manifest, and the
staged bundle manifest agree. OpenVINO versions independently of ExecuTorch, so
an OV re-roll can otherwise invalidate the committed fixture with no ET bump."
```

---

### Task 8: The precision accessor

**Files:**
- Modify: `native/core/et_runtime.h`, `native/core/et_runtime.cpp`
- Modify: `native/jni/executorch_djl_jni.cpp`
- Modify: `src/main/java/org/measly/executorch/jni/EtNative.java`
- Modify: `src/main/java/org/measly/executorch/engine/EtEngine.java`
- Test: `src/test/java/org/measly/executorch/engine/OpenVinoRuntimeTest.java`

**Interfaces:**
- Consumes: `OpenVinoRuntime.resolvedLibPath()` (Task 5).
- Produces: `EtEngine.openVinoInferencePrecision() -> String`, consumed by Task 7's parity test.

- [ ] **Step 1: Write the failing test**

Append to `OpenVinoRuntimeTest.java`:

```java
    @Test
    void reportsTheInferencePrecisionOpenVinoWillUseOnThisHost() throws Exception {
        TestSupport.assumeOpenVinoBundleAvailable();
        OpenVinoRuntime.ensureExtracted();

        String precision = EtEngine.openVinoInferencePrecision();
        // The VALUE is not asserted -- it is a property of the CPU this happens to run on, and
        // asserting it would assert the hardware. What is asserted is that the read succeeded, i.e.
        // the vendored C API loaded and answered. "unavailable" means it did not.
        assertNotEquals("unavailable", precision, "the C API should have loaded and answered");
        assertTrue(
                precision.equals("f32") || precision.equals("bf16") || precision.equals("f16"),
                "unexpected precision: " + precision);
    }
```

Add `import static org.junit.jupiter.api.Assertions.assertNotEquals;`.

- [ ] **Step 2: Run to verify it fails**

Run: `./gradlew openvinoTest --tests '*OpenVinoRuntimeTest*'`

Expected: compilation failure — `openVinoInferencePrecision` does not exist.

- [ ] **Step 3: Implement the native read**

Declare in `native/core/et_runtime.h`:

```cpp
// The numeric type OpenVINO will use for CPU inference on this host ("f32", "bf16", ...), or
// "unavailable" if it cannot be determined.
//
// Reads through a FRESHLY CREATED ov::Core, not the Core the delegate built inside OpenvinoBackend.
// Those agree today because the choice derives from CPU capability alone; if per-model precision
// control is ever added they could diverge, and this would have to read through the delegate --
// which ExecuTorch exposes no way to do.
//
// Creating a Core loads the CPU plugin and is not cheap. This is an on-demand diagnostic: never
// call it on the hot path or during model load.
std::string openVinoInferencePrecision(const std::string& libPath);
```

Implement in `native/core/et_runtime.cpp` (add `#include <dlfcn.h>`):

```cpp
std::string openVinoInferencePrecision(const std::string& libPath) {
  // dlopen'd rather than linked: we have no OpenVINO at link time, and the delegate resolves the
  // same library the same way. Refcounted, so opening it here is safe alongside the delegate's own
  // handle. RTLD_LOCAL so nothing here perturbs the delegate's symbol resolution.
  void* handle = dlopen(libPath.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (handle == nullptr) {
    return "unavailable";
  }
  using CoreCreate = int (*)(void**);
  using CoreGetProperty = int (*)(void*, const char*, const char*, char**);
  using CoreFree = void (*)(void*);
  using Free = void (*)(const char*);

  auto create = reinterpret_cast<CoreCreate>(dlsym(handle, "ov_core_create"));
  auto getProperty = reinterpret_cast<CoreGetProperty>(dlsym(handle, "ov_core_get_property"));
  auto coreFree = reinterpret_cast<CoreFree>(dlsym(handle, "ov_core_free"));
  auto ovFree = reinterpret_cast<Free>(dlsym(handle, "ov_free"));
  if (create == nullptr || getProperty == nullptr || coreFree == nullptr) {
    dlclose(handle);
    return "unavailable";
  }

  void* core = nullptr;
  if (create(&core) != 0 || core == nullptr) {
    dlclose(handle);
    return "unavailable";
  }
  char* value = nullptr;
  std::string result = "unavailable";
  if (getProperty(core, "CPU", "INFERENCE_PRECISION_HINT", &value) == 0 && value != nullptr) {
    result = value;
    if (ovFree != nullptr) {
      ovFree(value);
    }
  }
  coreFree(core);
  // Not dlclose'd on the success path: the delegate may hold the same library, and OpenVINO
  // registers plugin state that does not expect to be torn down and rebuilt. The handle is
  // process-lifetime by design; this is a diagnostic called a handful of times at most.
  return result;
}
```

- [ ] **Step 4: Bridge it to Java**

In `native/jni/executorch_djl_jni.cpp`:

```cpp
extern "C" JNIEXPORT jstring JNICALL
Java_org_measly_executorch_jni_EtNative_openVinoInferencePrecision(
    JNIEnv* env, jclass, jstring libPath) {
  const char* path = env->GetStringUTFChars(libPath, nullptr);
  if (path == nullptr) {
    return nullptr;  // OOM pending; the Java wrapper degrades this to "unavailable"
  }
  std::string p(path);
  env->ReleaseStringUTFChars(libPath, path);
  // Same copy-then-release shape as the other entry points: no JNI resource is held across the
  // call, so no path through here can leak one.
  const std::string result = measly::et::openVinoInferencePrecision(p);
  return env->NewStringUTF(result.c_str());
}
```

In `EtNative.java`:

```java
    /**
     * Reads the numeric type OpenVINO will use for CPU inference on this host.
     *
     * @param libPath absolute path to the OpenVINO C library file
     * @return e.g. {@code f32} or {@code bf16}, or {@code unavailable}
     */
    public static native String openVinoInferencePrecision(String libPath);
```

- [ ] **Step 5: Add the public accessor**

In `EtEngine.java`:

```java
    /**
     * The numeric type OpenVINO will use for CPU inference on this host, e.g. {@code "f32"} or
     * {@code "bf16"}, or {@code "unavailable"} when it cannot be determined.
     *
     * <p>OpenVINO selects this from the CPU it lands on rather than from how the model was
     * compiled: on avx512_bf16/AMX hardware it computes in bf16, elsewhere in f32. Both are
     * correct, and the difference against an f32 golden is ~2.5e-3 versus ~6e-8 — which is why
     * OpenVINO parity tests use a loose tolerance. This exists so that looseness stays observable
     * instead of hiding a silent shift.
     *
     * <p>Reports what a <b>freshly created</b> Core would choose, not a reading from the Core the
     * delegate built. Those agree today because the choice derives from CPU capability alone.
     *
     * <p>Creating a Core loads the CPU plugin and is not cheap. This is an on-demand diagnostic:
     * do not call it on the hot path or during model load. It returns {@code "unavailable"} rather
     * than throwing, because a diagnostic that throws is a diagnostic people stop calling.
     *
     * @return the precision, or {@code "unavailable"}
     */
    public static String openVinoInferencePrecision() {
        String lib = OpenVinoRuntime.resolvedLibPath();
        if (lib == null) {
            return "unavailable";
        }
        try {
            // Null rather than a string means the native side could not even read its argument
            // (OOM pending). Fold it into the same sentinel: this is a diagnostic, and a caller
            // reading it should never have to distinguish degrees of unavailability.
            String precision = EtNative.openVinoInferencePrecision(lib);
            return (precision == null || precision.isEmpty()) ? "unavailable" : precision;
        } catch (RuntimeException | LinkageError e) {
            return "unavailable";
        }
    }
```

- [ ] **Step 6: Rebuild and run**

```bash
./native/local_build_wrapper.sh
./gradlew openvinoTest
```

Expected: all OpenVINO tests pass, and the parity test prints a precision line. Restore the precision line in `OpenVinoModelIT` if it was commented out in Task 7.

- [ ] **Step 7: Commit**

```bash
git add native/core/et_runtime.h native/core/et_runtime.cpp native/jni/executorch_djl_jni.cpp \
  src/main/java/org/measly/executorch/jni/EtNative.java \
  src/main/java/org/measly/executorch/engine/EtEngine.java \
  src/test/java/org/measly/executorch/engine/OpenVinoRuntimeTest.java
git commit -m "feat: report the inference precision OpenVINO picks for this host

OpenVINO chooses bf16 or f32 from the CPU it lands on, not from how the model
was compiled, and the two land ~2.5e-3 and ~6e-8 from an f32 golden. Parity
tests therefore need a loose tolerance, and a loose tolerance cannot tell
'correct in bf16' from 'quietly degraded' -- so the run records which it got.

Read through the vendored C API by dlopen, since there is no OpenVINO at link
time. The value reflects a freshly created Core, not the delegate's; those agree
while precision derives from CPU capability alone. Returns 'unavailable' rather
than throwing, because a diagnostic that throws stops being called."
```

---

### Task 9: The off-platform error, CI, and documentation

**Files:**
- Create: `src/test/java/org/measly/executorch/OpenVinoUnsupportedIT.java`
- Modify: `.github/workflows/native-build-job.yml` and `.github/workflows/native-build.yml`
- Modify: `CLAUDE.md`
- Modify: `docs/README.md`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the off-platform test**

Create `src/test/java/org/measly/executorch/OpenVinoUnsupportedIT.java`:

```java
package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.djl.Model;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.measly.executorch.engine.OpenVinoRuntime;

/**
 * The inverse of {@link OpenVinoModelIT}: this runs where the delegate is ABSENT, so both matrix
 * legs assert something real instead of one of them merely skipping.
 */
@Tag("openvino-unsupported")
class OpenVinoUnsupportedIT {

    private static final Path DIR = Paths.get("src/test/resources/models/openvino");

    @Test
    void anOpenVinoModelOnAPlatformWithoutTheDelegateNamesTheRealProblem() {
        TestSupport.assumeNativeAvailable();
        Assumptions.assumeFalse(
                OpenVinoRuntime.bundleAvailable(), "this asserts the UNSUPPORTED platform's message");

        Exception e = assertThrows(Exception.class, () -> {
            try (Model model = Model.newInstance("openvino_tiny", "ExecuTorch")) {
                model.load(DIR, "openvino_tiny");
            }
        });
        String message = String.valueOf(e.getMessage()) + String.valueOf(e.getCause());
        // Without the guard this falls through the generic load-failure path and reports a corrupt
        // or version-mismatched model -- which actively misdirects. The .pte is fine; the platform
        // cannot run it.
        assertTrue(message.contains("OpenvinoBackend"), "must name the backend: " + message);
        assertTrue(
                message.toLowerCase().contains("re-export") || message.contains("does not provide"),
                "must say what to do about it: " + message);
    }
}
```

Add `"openvino-unsupported"` to the `excludeTags` list at `build.gradle.kts:33` and register a task for it exactly like `openvinoTest` but with `includeTags("openvino-unsupported")`.

- [ ] **Step 2: Run it**

Run: `./gradlew openvinoUnsupportedTest`

Expected: on linux-x86_64 with the bundle present it skips by assumption; the aarch64 leg is where it asserts. To exercise it locally, run without the bundle jar on the classpath.

- [ ] **Step 3: Wire CI**

In `.github/workflows/native-build-job.yml`, in the `linux-x86_64` row after the existing test steps:

```yaml
      - name: OpenVINO delegate tests
        if: matrix.platform == 'linux-x86_64'
        run: ./gradlew openvinoTest
```

And in the aarch64 row:

```yaml
      - name: OpenVINO unsupported-platform error
        if: matrix.platform == 'linux-aarch64'
        run: ./gradlew openvinoUnsupportedTest
```

Also add the two new shell tests wherever `cmake_resolution.sh` is already invoked:

```yaml
      - run: ./native/tests/openvino_linkage.sh
      - run: ./native/tests/openvino_bundle_staging.sh
      - run: ./native/tests/openvino_version_coupling.sh
```

- [ ] **Step 4: Document it**

In `CLAUDE.md`, after the workspace-metric bullet:

```markdown
- **OpenVINO delegate** (`linux-x86_64` only today; upstream is exploring Windows). The delegate is
  compiled into the runtime tarball and linked whenever `TARGET openvino_backend` exists, but the
  OpenVINO *runtime* is a separate ~21 MB opt-in jar
  (capability `org.measly:djl-executorch-engine-<platform>-openvino`) — it is not in the standard
  platform jar. Loading an OpenVINO `.pte` without it fails with a message naming the missing
  artifact. The delegate resolves its C API by `dlopen` under `std::call_once` **with no retry**, so
  every check happens before delegate init: a C++ guard in `EtRuntime`'s ctor before
  `load_forward()`, and `OpenVinoRuntime.ensureReady` before `loadModule`. A JVM cannot use
  `LD_LIBRARY_PATH` (glibc reads it once at process start), so `OPENVINO_LIB_PATH` set from JNI is
  the only mechanism — and a caller-set value always wins. Bundle extraction is content-addressed on
  the upstream tarball SHA under the `LibUtils` cache root, published by atomic directory rename;
  nothing is ever loaded out of the staging directory. `EtEngine.openVinoInferencePrecision()`
  reports whether this host computes in `f32` or `bf16`, which is what keeps the parity test's
  `atol=1e-2` honest — **never tighten it**, both values are correct and the bound would then assert
  which machine CI allocated. Bumping the vendored OpenVINO version touches four places across the
  tree — `docs/openvino-version-bump.md` is the checklist, and the coupling tests name it when they
  fail. Design: `docs/superpowers/specs/2026-08-16-openvino-linux-x86_64-design.md`.
```

`docs/README.md` needs no edit: line 18 points at the `superpowers/` directory rather than listing individual specs, so the new spec is already covered. Confirm that is still true rather than assuming it.

- [ ] **Step 5: Full verification**

```bash
./native/local_build_wrapper.sh
./native/local_build_wrapper.sh native/build_qa.sh 2>&1 | grep -E "All tests passed|assertions:"
./native/tests/cmake_resolution.sh
./native/tests/openvino_linkage.sh
./native/tests/openvino_bundle_staging.sh
./native/tests/openvino_version_coupling.sh
./gradlew test
./gradlew openvinoTest
./native/local_build_wrapper.sh native/ubsan_gate.sh
ET_UBSAN_MODE=test ./native/ubsan_gate.sh
```

All must pass. The UBSan gate is not optional here: this work adds three JNI entry points, and that gate is the only configuration that instruments the shim.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows CLAUDE.md docs/README.md \
  src/test/java/org/measly/executorch/OpenVinoUnsupportedIT.java build.gradle.kts
git commit -m "ci: gate OpenVINO on both Linux legs, and document the feature

linux-x86_64 runs the delegate for real; linux-aarch64 runs the inverse test
that a .pte needing an unlinked delegate produces our platform error rather than
the generic corrupt-model message. Both legs assert something instead of one of
them skipping."
```

---

## Notes for the executor

- **Never tighten `atol=1e-2`.** On an f32 host the measured diff is ~6e-8 and the bound will look absurd. A bf16 host produces ~2.5e-3 and both are correct. See the test's comment.
- **Every OpenVINO test runs in its own JVM** (`forkEvery = 1`). `OPENVINO_LIB_PATH` is process env and the `dlopen` is once-only; cases sharing a JVM contaminate each other in ways that look like flakes.
- **Never create the `libopenvino_c.so` symlink.** It is measured to be unnecessary, jars cannot carry it, and Windows has no unprivileged equivalent. `OPENVINO_LIB_PATH` names the versioned file.
- **Never hardcode the ABI suffix `2541`.** It is in `BUILDINFO` as `ov_abi`.
- **Do not add `openvino_backend` to the post-link XNNPACK registration guard.** It is a guard against a GC'd registration, not an inventory.
- **If Task 2's size measurement is large, stop and escalate.** The single-shim decision assumed otherwise, and a second shim variant is a design change, not an implementation detail.
- A UB hit under the UBSan gate presents as a **JVM hard crash** mid-test, with the `runtime error:` line above the JVM's crash dump. That is the gate working.
