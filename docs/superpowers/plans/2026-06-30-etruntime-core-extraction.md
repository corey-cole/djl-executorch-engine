# EtRuntime Core Extraction + Native QA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the `JNIEnv`-free inference logic out of `native/jni/executorch_djl_jni.cpp` into an `EtRuntime` C++ core, reduce JNI to a translation shell over it, and stand up the two native QA layers (Catch2 units + ASan/LSan leak harness) that can only run against a JVM-free core.

**Architecture:** One `et_runtime` STATIC library holds the pure logic (`load → method_meta → forward → output views`, exceptions on failure). Three consumers link it: the `executorch_djl` shared library (JNI shell), `et_runtime_test` (Catch2), and `et_leak_harness` (ASan/LSan). The shipping `.so` builds Release; the QA targets build instrumented in a separate build dir (`-DET_BUILD_QA=ON`). The existing JVM suite is the parity gate.

**Tech Stack:** C++20 (the core's public API uses `std::span`), ExecuTorch v1.3.1 extension API (`Module`, `from_blob`, `method_meta`), CMake (STATIC lib + per-config build dirs), Catch2 v3 (FetchContent), AddressSanitizer/LeakSanitizer, Gradle (existing JVM tests).

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-06-30-etruntime-core-extraction-design.md`.
- **Error model:** core throws `std::runtime_error` on `!ok()`; JNI shell catches per entry point and `ThrowNew`s. Never throw across an ExecuTorch frame — only from our code after checking `.ok()`.
- **Output model:** borrowed `OutputView`s into ExecuTorch's host arena, valid until the next `forward()`/`destroy`. The JNI shell copies the view into a Java `byte[]` synchronously (single-copy out). No core-owned output copies.
- **Zero Java-side change:** `EtNative`, `EtMethodMeta(int, int[])`, `EtTensor` signatures unchanged. The existing JVM suite (39 tests) + `leakTest` must pass with **zero test-source edits** — that is the parity gate.
- **ScalarType codes (c10 canonical):** FLOAT32 = 6, FLOAT64 = 7, INT32 = 3, INT64 = 4, UINT8 = 0, INT8 = 1, BOOL = 11. Non-tensor input encodes `-1`.
- **`et_runtime.h` must not include `<jni.h>`.** Header stays ExecuTorch-free via pimpl; only `core/et_runtime.cpp` includes ExecuTorch.
- **Naming:** new code lives under `native/core/`, `native/test/`, `native/harness/`. `native/spike/` is untouched (historical artifact); reuse `native/spike/add.pte` as the test fixture.
- **PIC required** on `et_runtime` (its static deps get linked into the `.so`).
- **Commit trailer (every commit):** `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Extract the `EtRuntime` core + Catch2 unit suite

Stand up the JNIEnv-free core and prove it with assertion-level units. At the end of this task the shipping `.so` still builds from the **old** JNI code (unchanged) — the core exists and is independently unit-tested but not yet wired into the shell. That is the reviewable boundary.

**Files:**
- Create: `native/core/et_runtime.h`
- Create: `native/core/et_runtime.cpp`
- Create: `native/test/et_runtime_test.cpp`
- Modify: `native/CMakeLists.txt` (add `et_runtime` STATIC target + `ET_BUILD_QA` block with `et_runtime_test`)

**Interfaces:**
- Produces: `namespace measly::et` with `struct InputDesc { const void* data; std::vector<int64_t> shape; int8_t scalarType; }`, `struct OutputView { std::vector<int64_t> shape; int8_t scalarType; const void* data; size_t nbytes; }`, `struct MethodMeta { int numInputs; std::vector<int8_t> inputScalarTypes; std::vector<std::vector<int64_t>> inputShapes; }`, `class ForwardResult { std::span<const OutputView> outputs() const; }` (movable, non-copyable), `class EtRuntime { explicit EtRuntime(const std::string&); MethodMeta methodMeta() const; ForwardResult forward(std::span<const InputDesc>); }` (non-copyable).

- [ ] **Step 1: Write the core header**

Create `native/core/et_runtime.h` (no ExecuTorch, no JNI — pimpl keeps the header clean):

```cpp
#ifndef MEASLY_ET_RUNTIME_H
#define MEASLY_ET_RUNTIME_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace measly::et {

// Borrowed input: data is a host pointer the caller keeps valid across forward(). Zero-copy in.
struct InputDesc {
  const void* data;
  std::vector<int64_t> shape;
  int8_t scalarType;  // ExecuTorch ScalarType code
};

// Borrowed output: data points into ExecuTorch's host arena, valid until the next
// forward()/destroy on the originating EtRuntime. Single-copy out happens in the consumer.
struct OutputView {
  std::vector<int64_t> shape;
  int8_t scalarType;
  const void* data;
  size_t nbytes;
};

// Static metadata for the "forward" method.
struct MethodMeta {
  int numInputs;
  std::vector<int8_t> inputScalarTypes;           // -1 for a non-tensor input
  std::vector<std::vector<int64_t>> inputShapes;   // per tensor input; empty for non-tensor (-1)
};

struct ForwardState;  // pimpl
struct RuntimeState;  // pimpl

// Owns the ExecuTorch EValue vector backing the views. RAII: dropping it ends the view lifetime.
class ForwardResult {
 public:
  explicit ForwardResult(std::unique_ptr<ForwardState> state);
  ~ForwardResult();
  ForwardResult(ForwardResult&&) noexcept;
  ForwardResult& operator=(ForwardResult&&) noexcept;
  ForwardResult(const ForwardResult&) = delete;
  ForwardResult& operator=(const ForwardResult&) = delete;
  std::span<const OutputView> outputs() const;

 private:
  std::unique_ptr<ForwardState> state_;
};

// Owns the ExecuTorch Module. Throws std::runtime_error on load/forward/meta failure.
class EtRuntime {
 public:
  explicit EtRuntime(const std::string& ptePath);
  ~EtRuntime();
  EtRuntime(const EtRuntime&) = delete;
  EtRuntime& operator=(const EtRuntime&) = delete;
  MethodMeta methodMeta() const;
  ForwardResult forward(std::span<const InputDesc> inputs);

 private:
  std::unique_ptr<RuntimeState> state_;
};

}  // namespace measly::et
#endif  // MEASLY_ET_RUNTIME_H
```

- [ ] **Step 2: Write the failing unit test**

Create `native/test/et_runtime_test.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

#include "et_runtime.h"

using namespace measly::et;

#ifndef ADD_PTE_PATH
#define ADD_PTE_PATH "add.pte"
#endif

TEST_CASE("load: missing path throws") {
  REQUIRE_THROWS([] { EtRuntime rt("/nonexistent/definitely-not-here.pte"); }());
}

TEST_CASE("load: valid pte constructs") {
  REQUIRE_NOTHROW([] { EtRuntime rt(ADD_PTE_PATH); }());
}

TEST_CASE("methodMeta: add has two float32 tensor inputs of shape [1]") {
  EtRuntime rt(ADD_PTE_PATH);
  MethodMeta meta = rt.methodMeta();
  REQUIRE(meta.numInputs == 2);
  REQUIRE(meta.inputScalarTypes.size() == 2);
  REQUIRE(meta.inputScalarTypes[0] == 6);  // FLOAT32
  REQUIRE(meta.inputScalarTypes[1] == 6);
  REQUIRE(meta.inputShapes.size() == 2);
  REQUIRE(meta.inputShapes[0] == std::vector<int64_t>{1});
  REQUIRE(meta.inputShapes[1] == std::vector<int64_t>{1});
}

TEST_CASE("forward: add(2,3) == 5 with correct view metadata") {
  EtRuntime rt(ADD_PTE_PATH);
  float a = 2.0f, b = 3.0f;
  std::vector<InputDesc> inputs = {{&a, {1}, 6}, {&b, {1}, 6}};
  ForwardResult result = rt.forward(inputs);
  auto outs = result.outputs();
  REQUIRE(outs.size() == 1);
  REQUIRE(outs[0].scalarType == 6);
  REQUIRE(outs[0].nbytes == sizeof(float));
  REQUIRE(outs[0].shape == std::vector<int64_t>{1});
  REQUIRE(*static_cast<const float*>(outs[0].data) == 5.0f);
}

TEST_CASE("forward: a second call yields a fresh correct result (view-lifetime happy path)") {
  EtRuntime rt(ADD_PTE_PATH);
  float a1 = 2.0f, b1 = 3.0f;
  std::vector<InputDesc> in1 = {{&a1, {1}, 6}, {&b1, {1}, 6}};
  ForwardResult r1 = rt.forward(in1);
  REQUIRE(*static_cast<const float*>(r1.outputs()[0].data) == 5.0f);

  float a2 = 10.0f, b2 = 7.0f;
  std::vector<InputDesc> in2 = {{&a2, {1}, 6}, {&b2, {1}, 6}};
  ForwardResult r2 = rt.forward(in2);
  REQUIRE(*static_cast<const float*>(r2.outputs()[0].data) == 17.0f);
}
```

- [ ] **Step 3: Add the `et_runtime` STATIC target and the QA block to CMake**

Replace `native/CMakeLists.txt` entirely with:

```cmake
cmake_minimum_required(VERSION 3.24)
project(executorch_djl LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)  # the core's public API uses std::span (C++20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ET_BUILD_QA "Build native QA targets (Catch2 units + leak harness)" OFF)

# TODO: ET_INSTALL needs to cover the entire path and should default to something that
# is compatible with local and GHA builds
set(ET_INSTALL "$ENV{HOME}/workspace/executorch/cmake-out" CACHE PATH "ExecuTorch install prefix")
list(PREPEND CMAKE_PREFIX_PATH "${ET_INSTALL}")
set(tokenizers_DIR "${ET_INSTALL}/lib/cmake/tokenizers" CACHE PATH "")
find_package(executorch CONFIG REQUIRED PATHS "${ET_INSTALL}/lib/cmake/ExecuTorch")

# --- JNIEnv-free core: shared by the .so, the units, and the leak harness ---
add_library(et_runtime STATIC ${CMAKE_CURRENT_SOURCE_DIR}/core/et_runtime.cpp)
target_include_directories(et_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core)
set_target_properties(et_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Bare target names; backend/ops self-whole-archive — link plainly. PUBLIC so consumers inherit.
target_link_libraries(et_runtime PUBLIC
  extension_module_static
  extension_tensor
  xnnpack_backend
  portable_ops_lib
)

# --- JNI shell shared library (loaded BY the JVM: JNI headers only, never libjvm/libjawt) ---
if(DEFINED ENV{JAVA_HOME})
  set(JAVA_HOME "$ENV{JAVA_HOME}")
else()
  find_program(JAVAC_EXECUTABLE javac REQUIRED)
  get_filename_component(JAVA_HOME "${JAVAC_EXECUTABLE}" DIRECTORY)
  get_filename_component(JAVA_HOME "${JAVA_HOME}" DIRECTORY)
endif()

add_library(executorch_djl SHARED ${CMAKE_CURRENT_SOURCE_DIR}/jni/executorch_djl_jni.cpp)
target_include_directories(executorch_djl PRIVATE
  "${JAVA_HOME}/include" "${JAVA_HOME}/include/linux")
target_link_libraries(executorch_djl PRIVATE et_runtime)

# --- Native QA targets (Catch2 units + ASan/LSan leak harness). OFF for the shipping build. ---
if(ET_BUILD_QA)
  include(FetchContent)
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.15.1)
  FetchContent_MakeAvailable(Catch2)

  add_executable(et_runtime_test ${CMAKE_CURRENT_SOURCE_DIR}/test/et_runtime_test.cpp)
  target_link_libraries(et_runtime_test PRIVATE et_runtime Catch2::Catch2WithMain)
  target_compile_definitions(et_runtime_test PRIVATE
    ADD_PTE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/spike/add.pte")
endif()
```

- [ ] **Step 4: Create an empty core implementation (so CMake configures; the link is what fails)**

CMake errors at *configure* if a listed source file is missing, so create `native/core/et_runtime.cpp` with no definitions yet — this turns the red into a genuine *link* error:

```cpp
#include "et_runtime.h"

namespace measly::et {
// Intentionally empty — implemented in the next step. Compiles to an empty object so the
// test link fails with "undefined reference", the TDD red for a compiled language.
}  // namespace measly::et
```

- [ ] **Step 5: Configure + build the test to verify it fails at link**

```bash
cd /home/corey/workspace/djl-executorch-engine
export ET_INSTALL="${ET_INSTALL:-$HOME/workspace/executorch/cmake-out}"
cmake -B native/asan -S native -DET_INSTALL="$ET_INSTALL" -DET_BUILD_QA=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"
cmake --build native/asan --target et_runtime_test
```

Expected: configure succeeds; build fails at **link** — `undefined reference to measly::et::EtRuntime::EtRuntime(std::string const&)` (and the other members). Not a configure/compile error.

- [ ] **Step 6: Implement the core**

Replace `native/core/et_runtime.cpp` with the full implementation:

```cpp
#include "et_runtime.h"

#include <stdexcept>
#include <utility>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/executor/method_meta.h>

namespace measly::et {

using executorch::extension::Module;
using executorch::extension::from_blob;
using executorch::extension::TensorPtr;
using executorch::runtime::EValue;

struct RuntimeState {
  Module module;
  explicit RuntimeState(const std::string& path) : module(path) {}
};

struct ForwardState {
  std::vector<EValue> outputs;    // owns the result EValues
  std::vector<OutputView> views;  // descriptors into the host arena
};

EtRuntime::EtRuntime(const std::string& ptePath)
    : state_(std::make_unique<RuntimeState>(ptePath)) {
  // Force-load now so a bad path/file throws at construction (the "load throws" contract).
  if (state_->module.load() != executorch::runtime::Error::Ok) {
    throw std::runtime_error("EtRuntime: failed to load .pte: " + ptePath);
  }
}

EtRuntime::~EtRuntime() = default;

MethodMeta EtRuntime::methodMeta() const {
  auto meta = state_->module.method_meta("forward");
  if (!meta.ok()) {
    throw std::runtime_error("EtRuntime: method_meta(\"forward\") failed");
  }
  const int n = static_cast<int>(meta->num_inputs());
  MethodMeta out;
  out.numInputs = n;
  out.inputScalarTypes.resize(n);
  out.inputShapes.resize(n);
  for (int i = 0; i < n; ++i) {
    auto info = meta->input_tensor_meta(i);
    if (info.ok()) {
      out.inputScalarTypes[i] = static_cast<int8_t>(info->scalar_type());
      auto sizes = info->sizes();  // Span<const int32_t>
      out.inputShapes[i].assign(sizes.begin(), sizes.end());
    } else {
      out.inputScalarTypes[i] = -1;  // non-tensor input; inputShapes[i] left empty
    }
  }
  return out;
}

ForwardResult EtRuntime::forward(std::span<const InputDesc> inputs) {
  // from_blob does not copy: each InputDesc.data must stay valid through module.forward().
  std::vector<std::vector<executorch::aten::SizesType>> shapes(inputs.size());
  std::vector<TensorPtr> tensors;
  std::vector<EValue> evalues;
  tensors.reserve(inputs.size());
  evalues.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& in = inputs[i];
    shapes[i].assign(in.shape.begin(), in.shape.end());
    tensors.push_back(from_blob(
        const_cast<void*>(in.data), shapes[i],
        static_cast<executorch::aten::ScalarType>(in.scalarType)));
    evalues.emplace_back(tensors[i]);
  }

  auto result = state_->module.forward(evalues);
  if (!result.ok()) {
    throw std::runtime_error("EtRuntime: forward() failed");
  }

  auto fs = std::make_unique<ForwardState>();
  fs->outputs = std::move(*result);
  fs->views.reserve(fs->outputs.size());
  for (auto& ev : fs->outputs) {
    auto t = ev.toTensor();
    OutputView v;
    v.scalarType = static_cast<int8_t>(t.scalar_type());
    v.data = t.const_data_ptr();
    v.nbytes = t.nbytes();
    const auto ndim = t.dim();
    v.shape.resize(ndim);
    for (auto k = 0; k < ndim; ++k) {
      v.shape[k] = static_cast<int64_t>(t.size(k));
    }
    fs->views.push_back(std::move(v));
  }
  return ForwardResult(std::move(fs));
}

ForwardResult::ForwardResult(std::unique_ptr<ForwardState> state)
    : state_(std::move(state)) {}
ForwardResult::~ForwardResult() = default;
ForwardResult::ForwardResult(ForwardResult&&) noexcept = default;
ForwardResult& ForwardResult::operator=(ForwardResult&&) noexcept = default;

std::span<const OutputView> ForwardResult::outputs() const {
  return {state_->views.data(), state_->views.size()};
}

}  // namespace measly::et
```

- [ ] **Step 7: Build and run the units to verify they pass**

```bash
cd /home/corey/workspace/djl-executorch-engine
cmake --build native/asan --target et_runtime_test
./native/asan/et_runtime_test
```

Expected: `All tests passed (N assertions in 5 test cases)` and **clean ASan exit** (no leak report, exit 0).

- [ ] **Step 8: Commit**

```bash
cd /home/corey/workspace/djl-executorch-engine
git add native/core/et_runtime.h native/core/et_runtime.cpp native/test/et_runtime_test.cpp native/CMakeLists.txt
git commit -m "$(cat <<'EOF'
feat(native): extract JNIEnv-free EtRuntime core + Catch2 unit suite

Pure load/methodMeta/forward logic moves into native/core/et_runtime.{h,cpp}
(pimpl, exceptions on failure, borrowed output views). Unit-tested directly
against add.pte under ASan. JNI shell not yet migrated.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Rewrite the JNI shell over the core + parity gate

Replace the inline `Module` logic in the shell with calls to `EtRuntime`. The Java-held handle now points to an `EtRuntime`. Each entry point becomes pure translation wrapped in `try/catch` → `ThrowNew`. The existing JVM suite proves behavior is unchanged.

**Files:**
- Modify: `native/jni/executorch_djl_jni.cpp` (replace the bodies of `loadModule`/`methodMeta`/`forward`/`destroy`; keep `JNI_OnLoad` + ID caching)

**Interfaces:**
- Consumes: `measly::et::EtRuntime`, `MethodMeta`, `InputDesc`, `OutputView`, `ForwardResult` from Task 1.
- Produces: unchanged JNI ABI (`Java_org_measly_executorch_jni_EtNative_{loadModule,methodMeta,forward,destroy}`).

- [ ] **Step 1: Rewrite the shell**

Replace `native/jni/executorch_djl_jni.cpp` entirely with (JNI translation only; all inference logic is in the core):

```cpp
// Thin JNI shell over measly::et::EtRuntime. Raw JNI, no fbjni. Translation only.
#include <jni.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "et_runtime.h"

using measly::et::EtRuntime;
using measly::et::InputDesc;
using measly::et::MethodMeta;

static jclass g_etTensorClass = nullptr;
static jfieldID g_fShape = nullptr;
static jfieldID g_fScalarType = nullptr;
static jfieldID g_fData = nullptr;
static jmethodID g_ctor = nullptr;

static jclass g_etMethodMetaClass = nullptr;
static jmethodID g_metaCtor = nullptr;

static jclass g_byteBufferClass = nullptr;
static jmethodID g_byteBufferWrap = nullptr;

// Translate a C++ exception into a Java RuntimeException. Call from a catch block.
static void throwJava(JNIEnv* env, const char* fallback, const std::exception* e) {
  jclass cls = env->FindClass("java/lang/RuntimeException");
  env->ThrowNew(cls, e ? e->what() : fallback);
}

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  JNIEnv* env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }
  jclass local = env->FindClass("org/measly/executorch/jni/EtTensor");
  if (local == nullptr) {
    return JNI_ERR;  // class not found -> System.load fails clearly
  }
  g_etTensorClass = static_cast<jclass>(env->NewGlobalRef(local));
  env->DeleteLocalRef(local);
  g_fShape = env->GetFieldID(g_etTensorClass, "shape", "[J");
  g_fScalarType = env->GetFieldID(g_etTensorClass, "scalarType", "I");
  g_fData = env->GetFieldID(g_etTensorClass, "data", "Ljava/nio/ByteBuffer;");
  g_ctor = env->GetMethodID(g_etTensorClass, "<init>", "([JILjava/nio/ByteBuffer;)V");
  if (g_fShape == nullptr || g_fScalarType == nullptr || g_fData == nullptr || g_ctor == nullptr) {
    return JNI_ERR;
  }
  jclass mlocal = env->FindClass("org/measly/executorch/jni/EtMethodMeta");
  if (mlocal == nullptr) {
    return JNI_ERR;
  }
  g_etMethodMetaClass = static_cast<jclass>(env->NewGlobalRef(mlocal));
  env->DeleteLocalRef(mlocal);
  g_metaCtor = env->GetMethodID(g_etMethodMetaClass, "<init>", "(I[I)V");
  if (g_metaCtor == nullptr) {
    return JNI_ERR;
  }
  jclass bblocal = env->FindClass("java/nio/ByteBuffer");
  if (bblocal == nullptr) {
    return JNI_ERR;
  }
  g_byteBufferClass = static_cast<jclass>(env->NewGlobalRef(bblocal));
  env->DeleteLocalRef(bblocal);
  g_byteBufferWrap = env->GetStaticMethodID(g_byteBufferClass, "wrap", "([B)Ljava/nio/ByteBuffer;");
  if (g_byteBufferWrap == nullptr) {
    return JNI_ERR;
  }
  return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jlong JNICALL
Java_org_measly_executorch_jni_EtNative_loadModule(JNIEnv* env, jclass, jstring jpath) {
  const char* path = env->GetStringUTFChars(jpath, nullptr);
  std::string p(path);
  env->ReleaseStringUTFChars(jpath, path);
  try {
    return reinterpret_cast<jlong>(new EtRuntime(p));
  } catch (const std::exception& e) {
    throwJava(env, "EtRuntime load failed", &e);
    return 0;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_org_measly_executorch_jni_EtNative_methodMeta(JNIEnv* env, jclass, jlong handle) {
  auto* rt = reinterpret_cast<EtRuntime*>(handle);
  MethodMeta meta;
  try {
    meta = rt->methodMeta();
  } catch (const std::exception& e) {
    throwJava(env, "methodMeta failed", &e);
    return nullptr;
  }
  const jsize n = static_cast<jsize>(meta.numInputs);
  jintArray types = env->NewIntArray(n);
  if (types == nullptr) {
    return nullptr;  // OOM: exception already pending
  }
  std::vector<jint> tmp(n);
  for (jsize i = 0; i < n; ++i) {
    tmp[i] = static_cast<jint>(meta.inputScalarTypes[i]);
  }
  env->SetIntArrayRegion(types, 0, n, tmp.data());
  return env->NewObject(g_etMethodMetaClass, g_metaCtor, static_cast<jint>(n), types);
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_org_measly_executorch_jni_EtNative_forward(JNIEnv* env, jclass, jlong handle,
                                                jobjectArray jinputs) {
  auto* rt = reinterpret_cast<EtRuntime*>(handle);

  jsize nIn = env->GetArrayLength(jinputs);
  std::vector<InputDesc> inputs(nIn);
  // The direct ByteBuffers (jinputs elements) stay live for the whole call, so the
  // addresses below remain valid through rt->forward().
  for (jsize i = 0; i < nIn; ++i) {
    jobject jt = env->GetObjectArrayElement(jinputs, i);
    auto jshape = static_cast<jlongArray>(env->GetObjectField(jt, g_fShape));
    jint st = env->GetIntField(jt, g_fScalarType);
    jobject jbuf = env->GetObjectField(jt, g_fData);

    jsize nd = env->GetArrayLength(jshape);
    std::vector<jlong> sh(nd);
    env->GetLongArrayRegion(jshape, 0, nd, sh.data());
    inputs[i].shape.assign(sh.begin(), sh.end());
    inputs[i].scalarType = static_cast<int8_t>(st);

    void* addr = env->GetDirectBufferAddress(jbuf);
    if (addr == nullptr) {
      env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                    "EtTensor.data must be a direct ByteBuffer");
      return nullptr;
    }
    inputs[i].data = addr;

    env->DeleteLocalRef(jshape);
    env->DeleteLocalRef(jbuf);
    env->DeleteLocalRef(jt);
  }

  try {
    auto result = rt->forward(inputs);
    auto outs = result.outputs();
    jsize nOut = static_cast<jsize>(outs.size());
    jobjectArray jout = env->NewObjectArray(nOut, g_etTensorClass, nullptr);

    for (jsize i = 0; i < nOut; ++i) {
      const auto& v = outs[i];
      jsize ndim = static_cast<jsize>(v.shape.size());
      jlongArray jshape = env->NewLongArray(ndim);
      {
        std::vector<jlong> sh(ndim);
        for (jsize k = 0; k < ndim; ++k) {
          sh[k] = static_cast<jlong>(v.shape[k]);
        }
        env->SetLongArrayRegion(jshape, 0, ndim, sh.data());
      }
      jsize nbytes = static_cast<jsize>(v.nbytes);
      jbyteArray jbytes = env->NewByteArray(nbytes);
      env->SetByteArrayRegion(jbytes, 0, nbytes, reinterpret_cast<const jbyte*>(v.data));
      jobject jbuf = env->CallStaticObjectMethod(g_byteBufferClass, g_byteBufferWrap, jbytes);

      jobject obj = env->NewObject(g_etTensorClass, g_ctor, jshape,
                                   static_cast<jint>(v.scalarType), jbuf);
      env->SetObjectArrayElement(jout, i, obj);

      env->DeleteLocalRef(jshape);
      env->DeleteLocalRef(jbytes);
      env->DeleteLocalRef(jbuf);
      env->DeleteLocalRef(obj);
    }
    return jout;
  } catch (const std::exception& e) {
    throwJava(env, "ExecuTorch forward() failed", &e);
    return nullptr;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_org_measly_executorch_jni_EtNative_destroy(JNIEnv*, jclass, jlong handle) {
  delete reinterpret_cast<EtRuntime*>(handle);
}
```

- [ ] **Step 2: Rebuild the shipping `.so` and stage it**

```bash
cd /home/corey/workspace/djl-executorch-engine
ET_INSTALL="${ET_INSTALL:-$HOME/workspace/executorch/cmake-out}" bash native/build_desktop.sh
```

Expected: `Artifact: src/main/resources/native/linux-x86_64/libexecutorch_djl.so` and a non-empty `ls -lh`. (`build_desktop.sh` builds with `ET_BUILD_QA` OFF, so no Catch2 fetch.)

- [ ] **Step 3: Run the JVM suite as the parity gate**

```bash
cd /home/corey/workspace/djl-executorch-engine
./gradlew test
```

Expected: **BUILD SUCCESSFUL**, all 39 tests pass (including `NamedParamsIT`), with **no edits to any test source**. If a test needs changing to pass, the seam leaked — stop and reconcile against the spec.

- [ ] **Step 4: Run the JVM leak gate**

```bash
cd /home/corey/workspace/djl-executorch-engine
./gradlew leakTest
```

Expected: **BUILD SUCCESSFUL** (no `OutOfMemoryError` from the constrained-memory stress test).

- [ ] **Step 5: Commit**

```bash
cd /home/corey/workspace/djl-executorch-engine
git add native/jni/executorch_djl_jni.cpp
git commit -m "$(cat <<'EOF'
refactor(native): reduce JNI shim to a translation shell over EtRuntime

loadModule/methodMeta/forward/destroy now delegate to measly::et::EtRuntime;
the Java handle owns an EtRuntime. C++ exceptions translated to Java at each
entry point. Zero-copy in / single-copy out preserved. JVM suite + leakTest
green with no test changes (parity gate).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Native leak harness (`et_leak_harness`)

A model-agnostic `load → forward → read` loop over the core, built under ASan/LSan. LSan's exit code is the gate.

**Files:**
- Create: `native/harness/et_leak_harness.cpp`
- Modify: `native/CMakeLists.txt` (add `et_leak_harness` inside the existing `ET_BUILD_QA` block)

**Interfaces:**
- Consumes: `measly::et::EtRuntime`, `MethodMeta`, `InputDesc`, `ForwardResult` from Task 1.

- [ ] **Step 1: Write the harness**

Create `native/harness/et_leak_harness.cpp`:

```cpp
// JNI-free leak harness: load -> forward loop over EtRuntime, built under ASan/LSan.
// LSan reports unfreed allocations at process exit; a leak -> non-zero exit. Model-agnostic:
// tensor inputs are derived from methodMeta() and backed by 1-filled host buffers.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "et_runtime.h"

using namespace measly::et;

static size_t dtypeSize(int8_t st) {
  switch (st) {
    case 6:           // FLOAT32
    case 3: return 4;  // INT32
    case 7:           // FLOAT64
    case 4: return 8;  // INT64
    case 0:           // UINT8
    case 1:           // INT8
    case 11: return 1;  // BOOL
    default: return 4;
  }
}

int main(int argc, char** argv) {
  const char* pte = (argc > 1) ? argv[1] : "add.pte";
  const int outerIters = (argc > 2) ? std::atoi(argv[2]) : 1000;
  const int forwardsPerLoad = 4;

  for (int it = 0; it < outerIters; ++it) {
    EtRuntime rt(pte);  // exercises load/destroy balance across iterations
    MethodMeta meta = rt.methodMeta();

    std::vector<std::vector<uint8_t>> buffers(meta.numInputs);
    std::vector<InputDesc> inputs;
    inputs.reserve(meta.numInputs);
    for (int i = 0; i < meta.numInputs; ++i) {
      if (meta.inputScalarTypes[i] < 0) {
        continue;  // non-tensor input: forward() only consumes tensor inputs
      }
      size_t count = 1;
      for (int64_t d : meta.inputShapes[i]) {
        count *= static_cast<size_t>(d);
      }
      size_t bytes = count * dtypeSize(meta.inputScalarTypes[i]);
      buffers[i].assign(bytes, 0);
      if (meta.inputScalarTypes[i] == 6) {  // fill float32 with 1.0f
        float one = 1.0f;
        for (size_t b = 0; b + sizeof(float) <= bytes; b += sizeof(float)) {
          std::memcpy(buffers[i].data() + b, &one, sizeof(float));
        }
      } else {
        std::memset(buffers[i].data(), 1, bytes);
      }
      inputs.push_back(InputDesc{buffers[i].data(), meta.inputShapes[i],
                                 meta.inputScalarTypes[i]});
    }

    for (int f = 0; f < forwardsPerLoad; ++f) {  // exercises per-forward allocations
      ForwardResult result = rt.forward(inputs);
      auto outs = result.outputs();
      if (!outs.empty()) {
        volatile const unsigned char first =
            *static_cast<const unsigned char*>(outs[0].data);  // touch the view
        (void)first;
      }
    }
  }

  std::printf("OK: %d loads x %d forwards over %s\n", outerIters, forwardsPerLoad, pte);
  return 0;
}
```

- [ ] **Step 2: Add the harness target to the `ET_BUILD_QA` block in CMake**

In `native/CMakeLists.txt`, inside `if(ET_BUILD_QA)` (after the `et_runtime_test` target, before the closing `endif()`), add:

```cmake
  add_executable(et_leak_harness ${CMAKE_CURRENT_SOURCE_DIR}/harness/et_leak_harness.cpp)
  target_link_libraries(et_leak_harness PRIVATE et_runtime)
```

- [ ] **Step 3: Build the harness under ASan/LSan**

```bash
cd /home/corey/workspace/djl-executorch-engine
cmake -B native/asan -S native -DET_INSTALL="${ET_INSTALL:-$HOME/workspace/executorch/cmake-out}" \
  -DET_BUILD_QA=ON -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"
cmake --build native/asan --target et_leak_harness
```

Expected: builds `native/asan/et_leak_harness`.

- [ ] **Step 4: Run the harness and verify a clean LSan exit**

```bash
cd /home/corey/workspace/djl-executorch-engine
./native/asan/et_leak_harness native/spike/add.pte 1000; echo "exit=$?"
```

Expected: `OK: 1000 loads x 4 forwards over native/spike/add.pte`, **no `ERROR: LeakSanitizer` block**, and `exit=0`. (An un-instrumented ExecuTorch runtime may surface interceptor noise; a genuine leak in our code yields a non-zero exit with our frames in the report.)

- [ ] **Step 5: Commit**

```bash
cd /home/corey/workspace/djl-executorch-engine
git add native/harness/et_leak_harness.cpp native/CMakeLists.txt
git commit -m "$(cat <<'EOF'
test(native): add model-agnostic ASan/LSan leak harness over EtRuntime

load -> forward loop deriving tensor inputs from methodMeta(); LSan exit code
is the gate. The first consumer of the JNIEnv-free core; covers the native
memory region no JVM tool can see.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Notes for the implementer

- **ExecuTorch API exactness:** `Module::load()` returns `executorch::runtime::Error`; `method_meta("forward")` returns a `Result` with `num_inputs()` / `input_tensor_meta(i)`; `TensorInfo::sizes()` is a `Span<const int32_t>`; `from_blob(void*, sizes, ScalarType)` does not copy. These match the pre-extraction `executorch_djl_jni.cpp` and `native/spike/cpp_smoke.cpp` — cross-check against them if a signature differs in the pinned ET version.
- **`ADD_PTE_PATH`** is injected as a compile definition pointing at `native/spike/add.pte`; the unit binary therefore runs from any CWD.
- **Two build dirs, by design:** `native/build` (Release `.so`, via `build_desktop.sh`, `ET_BUILD_QA` OFF) and `native/asan` (instrumented QA, `ET_BUILD_QA` ON). Never run the JVM suite against an ASan `.so`.
- **If `./gradlew test` requires editing a test to pass, stop** — that violates the parity gate (spec success criterion #2) and means the extraction changed observable behavior.
```
