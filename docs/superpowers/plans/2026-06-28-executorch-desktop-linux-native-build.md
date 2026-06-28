# ExecuTorch Desktop-Linux Native Build — Feasibility & Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a desktop-Linux (`linux-x86_64`) native shared library that lets a standard (non-Android) JVM load a `.pte` file and run ExecuTorch inference — proving that the first item in the DJL ExecuTorch engine plan is not a show-stopper, and delivering a reproducible build script for it.

**Architecture:** The ExecuTorch C++ runtime + XNNPACK build for desktop is a **first-class, upstream-documented target** — this plan does not re-derive it. The **Prerequisite** has you build it per the official docs (with a few flags that make it linkable) and hand over `$ET_INSTALL`. The plan's actual content is the part nothing upstream covers: a **thin custom JNI shim** over `executorch::extension::Module` that bridges a desktop JVM to that runtime — chosen because ExecuTorch's stock Android JNI (`extension/android`) is hard-gated to Android and pulls in a prebuilt Android `fbjni` AAR. Each task ends in an explicit go/no-go gate.

**Tech Stack:** ExecuTorch (CMake, C++17), XNNPACK CPU backend, host JDK 17 (`jni.h` via `find_package(JNI)`), Python 3.10–3.13 (AOT export), GCC 13 / CMake 3.28 (already present on this machine).

---

## Why this is a spike plan, not a red-green TDD plan

This deliverable is a **native build + linkage** problem, not application logic. The "test" at each step is *"does the artifact build and produce the right number?"* So each task ends in a **GATE**: an exact command and the exact output that means pass. A failed gate is a stop-and-diagnose point, not a refactor. Where there is genuine logic (the JNI shim, the export), real assertions are included.

## Feasibility verdict (read first)

| Concern | Finding | Risk |
|---|---|---|
| C++ runtime on desktop Linux | First-class & upstream-documented: `cmake --preset linux`, produces `libexecutorch.a` + extensions + XNNPACK | **Low** |
| Running a `.pte` from C++ on the host | Supported via `executorch::extension::Module` | **Low** |
| Reusing ExecuTorch's Android JNI `.so` as-is | Not viable: `extension/android/CMakeLists.txt` does `if(NOT ANDROID) → fatal error`, downloads an **Android** `fbjni` AAR, relies on NDK `jni.h` | **High** (avoid) |
| Custom thin JNI shim → host JVM | Standard JNI against the C++ `Module` API; no `fbjni`, no Android. **Not covered by upstream docs** | **Medium** |
| Symbol conflict with DJL's libtorch | ExecuTorch runtime shares no symbols with libtorch | **Low** |

**Conclusion to validate:** building the native libraries for desktop Linux is practical. The runtime half is documented and low-risk (Prerequisite). The original design's "Option A" (recompile the stock Android JNI) is the wrong path; the thin-shim path (design "Option B") is the lower-risk route and is what this plan drives toward. **Tasks 1 and 2 are the real go/no-go gates.**

## Path decision for the JVM bridge

Resolve this in **Task 2**. Default recommendation is **Path B**.

- **Path B — thin custom JNI shim (recommended, primary).** One `.cpp` using `<jni.h>` + `executorch::extension::Module`. No `fbjni`, no `extension/android`. We control the `.so`. Cost: we do **not** reuse the upstream `org.pytorch.executorch.{Module,Tensor,EValue}` Java classes; we write our own small native-method surface (which the DJL `EtModel`/`EtNDArray` layer will call anyway).
- **Path A — patch `extension/android` for host (fallback, appendix).** Remove the `ANDROID` guard, swap NDK `jni.h` for `find_package(JNI)`, build `fbjni` from source for the host. Pro: lets you reuse the upstream Java classes verbatim. Con: `fbjni` host build + AAR assumptions are extra moving parts. Documented in Appendix A; attempt only if reusing upstream Java classes proves worth it.

---

## File / artifact map

External, provided by the Prerequisite (not built by this plan's tasks):
- `$ET_INSTALL` — an installed ExecuTorch runtime tree (static libs + headers + `lib/cmake/executorch/executorch-config.cmake`). Default location: `~/workspace/executorch/cmake-out`.
- `~/workspace/executorch/.venv` — the Python env from `install_executorch.sh`, used only to export the test `.pte`.

Inside this repo (`~/workspace/djl-executorch-engine/`):
- Create: `native/spike/export_add.py` — Python AOT export of a trivial 2-input add model → `add.pte`
- Create: `native/spike/cpp_smoke.cpp` — pure-C++ loader/runner (Task 1)
- Create: `native/spike/CMakeLists.txt` — builds the C++ smoke binary and the JNI shim
- Create: `native/spike/EtNative.java` — minimal Java class with `native` methods (Task 2)
- Create: `native/spike/EtSmoke.java` — Java `main` that loads the `.so`, runs `add.pte`, asserts
- Create: `native/jni/executorch_djl_jni.cpp` — the thin JNI shim (promoted from spike)
- Create: `native/CMakeLists.txt` — production CMake for `libexecutorch_djl.so`
- Create: `native/build_desktop.sh` — builds the shim against `$ET_INSTALL` and packages the `.so`
- Create: `docs/superpowers/reports/executorch-desktop-feasibility.md` — final feasibility report (Task 3)

---

## Prerequisite: a linkable ExecuTorch runtime (follow upstream docs)

The desktop C++ runtime build is first-class and documented upstream — **do not reinvent it.** Follow the official guides:
- Building from source: <https://docs.pytorch.org/executorch/main/using-executorch-building-from-source.html>
- Desktop overview: <https://docs.pytorch.org/executorch/main/desktop-section.html>

The **only** things this plan adds to the upstream instructions are the CMake flags that make the runtime *linkable into a shared object*. After this section you hand over `$ET_INSTALL`; everything from Task 1 on is the novel work.

- [x] **Step 1: Install OS build prerequisites**

Run:
```bash
sudo apt-get update && sudo apt-get install -y build-essential cmake ninja-build libssl-dev python3-venv git
```
Expected: completes 0. (GCC 13 / CMake 3.28 already present; this ensures `ninja`, `venv`, headers.)

- [ ] **Step 2: Clone and pin the version**

`v1.3.1` is the newest stable ExecuTorch release — pin it, do not track `main`.
```bash
export ET_VERSION=v1.3.1
git clone --depth 1 --branch "$ET_VERSION" --recurse-submodules --shallow-submodules \
  https://github.com/pytorch/executorch.git "$HOME/workspace/executorch"
```
Expected: clone completes; `~/workspace/executorch/backends/xnnpack/third-party/XNNPACK` is non-empty.
Note: if Task 2 ever falls back to Path A (reusing upstream Java classes), this tag must match the `org.pytorch:executorch-android` AAR version you mirror.

- [ ] **Step 3: Build, per the upstream docs, with the linkability flags this plan requires**

The general procedure (venv, `install_executorch.sh`, presets) is in the upstream "Building from source" guide. Run it with these specific flags — they are the delta this plan needs:
```bash
cd ~/workspace/executorch
python3 -m venv .venv && source .venv/bin/activate
./install_executorch.sh          # AOT/python side, used later to export add.pte

cmake -B cmake-out -S . -G Ninja --preset linux \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=cmake-out \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \      # REQUIRED: static libs get linked into a .so
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON
cmake --build cmake-out -j"$(nproc)"
cmake --install cmake-out --prefix cmake-out  # REQUIRED: emits executorch-config.cmake for find_package
```
Expected: build completes 0 (first build is slow — XNNPACK). If `--preset linux` is rejected by this tag, drop it and add `-DEXECUTORCH_BUILD_HOST_TARGETS=ON` (see the upstream guide).

- [ ] **Step 4: GATE — the runtime is installed and linkable; export `$ET_INSTALL`**

Run:
```bash
export ET_INSTALL="$HOME/workspace/executorch/cmake-out"
test -f "$ET_INSTALL/lib/cmake/ExecuTorch/executorch-config.cmake" \
  && echo "executorch CMake package present" \
  && find "$ET_INSTALL" -name 'libexecutorch*.a' -o -name 'libextension_module*.a' \
       -o -name 'libxnnpack_backend*.a' -o -name 'libextension_tensor*.a' | sort
```
Expected: prints `executorch CMake package present` and lists at least `libexecutorch.a`, an `extension_module` lib, an `extension_tensor` lib, and `libxnnpack_backend.a`. **This clears "can the runtime build on desktop Linux?"** If the config file is missing, the `cmake --install` in Step 3 didn't run; if libs are missing, check `cmake-out/CMakeCache.txt` for which `EXECUTORCH_BUILD_*` flags took.

Validation output from step 4
```
executorch CMake package present
/home/corey/workspace/executorch/cmake-out/backends/xnnpack/libxnnpack_backend.a
/home/corey/workspace/executorch/cmake-out/extension/module/libextension_module.a
/home/corey/workspace/executorch/cmake-out/extension/module/libextension_module_static.a
/home/corey/workspace/executorch/cmake-out/extension/tensor/libextension_tensor.a
/home/corey/workspace/executorch/cmake-out/libexecutorch.a
/home/corey/workspace/executorch/cmake-out/libexecutorch_core.a
/home/corey/workspace/executorch/cmake-out/lib/libexecutorch.a
/home/corey/workspace/executorch/cmake-out/lib/libexecutorch_core.a
/home/corey/workspace/executorch/cmake-out/lib/libextension_module.a
/home/corey/workspace/executorch/cmake-out/lib/libextension_module_static.a
/home/corey/workspace/executorch/cmake-out/lib/libextension_tensor.a
/home/corey/workspace/executorch/cmake-out/lib/libxnnpack_backend.a
```

`$ET_INSTALL` is the only output the rest of the plan consumes. Keep it exported (or pass `-DET_INSTALL=...` to the CMake steps below).

---

## Task 1: Prove pure-C++ inference on the host (no JVM) — runtime feasibility gate

Cheap sanity check that *this specific* install actually loads a `.pte` and runs XNNPACK before any JVM is involved. Isolates runtime problems from bridge problems.

**Files:**
- Create: `native/spike/export_add.py`
- Create: `native/spike/cpp_smoke.cpp`
- Create: `native/spike/CMakeLists.txt`

- [ ] **Step 1: Write the AOT export for a trivial 2-input add model**

Create `native/spike/export_add.py`:
```python
"""Export a trivial (a + b) model to add.pte for desktop runtime smoke testing."""
import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


class Add(torch.nn.Module):
    def forward(self, a, b):
        return a + b


def main() -> None:
    model = Add().eval()
    example_inputs = (torch.ones(1), torch.ones(1))
    exported = export(model, example_inputs)
    lowered = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()]
    ).to_executorch()
    with open("add.pte", "wb") as f:
        f.write(lowered.buffer)
    print("wrote add.pte")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the export**

Run:
```bash
cd ~/workspace/djl-executorch-engine/native/spike && \
source ~/workspace/executorch/.venv/bin/activate && \
python3 export_add.py
```
Expected: prints `wrote add.pte`; `add.pte` exists and is non-empty. If `XnnpackPartitioner` import path differs in the pinned tag, drop the `partitioner=` arg (runs on the portable CPU kernels instead — still a valid gate).

- [ ] **Step 3: Write the pure-C++ runner**

Create `native/spike/cpp_smoke.cpp`:
```cpp
// Loads add.pte, runs forward(2.0, 3.0), expects 5.0. No JVM involved.
#include <cstdio>
#include <vector>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

using executorch::extension::Module;
using executorch::extension::from_blob;

int main(int argc, char** argv) {
  const char* pte = (argc > 1) ? argv[1] : "add.pte";
  Module module(pte);

  std::vector<float> a{2.0f};
  std::vector<float> b{3.0f};
  auto ta = from_blob(a.data(), {1});
  auto tb = from_blob(b.data(), {1});

  auto result = module.forward({ta, tb});
  if (!result.ok()) {
    std::fprintf(stderr, "forward failed, error=%d\n",
                 static_cast<int>(result.error()));
    return 1;
  }
  float out = result->at(0).toTensor().const_data_ptr<float>()[0];
  std::printf("RESULT=%.1f\n", out);
  return (out == 5.0f) ? 0 : 2;
}
```

- [ ] **Step 4: Write the CMake for the smoke binary**

Create `native/spike/CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.24)
project(et_desktop_spike LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point at the runtime install tree from the Prerequisite ($ET_INSTALL).
set(ET_INSTALL "$ENV{HOME}/workspace/executorch/cmake-out" CACHE PATH "ExecuTorch install prefix")
# executorch-config.cmake internally calls find_package(tokenizers CONFIG) with no hint;
# tokenizers is a SEPARATE package at lib/cmake/tokenizers. If it isn't discoverable, its
# imported targets go missing ("missing target tokenizers::tokenizers"). Put the prefix on
# the search path and pin tokenizers_DIR so the inner lookup resolves.
list(PREPEND CMAKE_PREFIX_PATH "${ET_INSTALL}")
set(tokenizers_DIR "${ET_INSTALL}/lib/cmake/tokenizers" CACHE PATH "")
# NB: config dir is lib/cmake/ExecuTorch (capital E) on a case-sensitive FS.
find_package(executorch CONFIG REQUIRED PATHS "${ET_INSTALL}/lib/cmake/ExecuTorch")

add_executable(cpp_smoke cpp_smoke.cpp)
# Targets are BARE (no executorch:: namespace), and each backend/ops imported target
# ALREADY embeds its own --whole-archive in INTERFACE_LINK_OPTIONS. Link plainly — wrapping
# again double-includes the archive and triggers "multiple definition" link errors.
target_link_libraries(cpp_smoke PRIVATE
  extension_module_static
  extension_tensor
  xnnpack_backend
  portable_ops_lib
)
```
Note: target names come from `ExecuTorchTargets.cmake` in the install. If a name is rejected, list the real ones with:
`grep -oE 'add_library\([A-Za-z_]+' "$ET_INSTALL"/lib/cmake/ExecuTorch/ExecuTorchTargets.cmake | sort -u`.

- [ ] **Step 5: Build and run — GATE**

Run:
```bash
cd ~/workspace/djl-executorch-engine/native/spike && \
cmake -B build -G Ninja -DET_INSTALL="$ET_INSTALL" && cmake --build build && \
./build/cpp_smoke add.pte
```
Expected: prints `RESULT=5.0` and exits 0.
**Decisive native-runtime gate.** If green: this install runs `.pte` inference on desktop Linux; remaining work is purely the JVM bridge. If the link fails on missing op/backend symbols at runtime, the whole-archive block in Step 4 is the cause.

- [ ] **Step 6: Commit**

```bash
git add native/spike/export_add.py native/spike/cpp_smoke.cpp native/spike/CMakeLists.txt
git commit -m "test(native): pure-C++ desktop inference smoke (add.pte -> 5.0)"
```

---

## Task 2: Bridge to the JVM via a thin custom JNI shim (Path B) — integration feasibility gate

**Files:**
- Create: `native/jni/executorch_djl_jni.cpp`
- Modify: `native/spike/CMakeLists.txt` (add the shim target)
- Create: `native/spike/EtNative.java`
- Create: `native/spike/EtSmoke.java`

- [ ] **Step 1: Write the Java native-method surface**

Create `native/spike/EtNative.java`:
```java
package org.measly.executorch.jni;

/** Minimal native surface for the desktop ExecuTorch JNI shim (spike). */
public final class EtNative {
    static {
        String lib = System.getenv("EXECUTORCH_LIBRARY_PATH");
        if (lib == null) {
            throw new IllegalStateException("set EXECUTORCH_LIBRARY_PATH to libexecutorch_djl.so");
        }
        System.load(lib);
    }

    private EtNative() {}

    /** Loads a .pte and returns an opaque native handle. */
    public static native long loadModule(String ptePath);

    /** Runs forward(a, b) on a 2-input float model, returns the output as a float[]. */
    public static native float[] forwardFloat(long handle, float[] a, float[] b);

    /** Frees the native module. */
    public static native void destroy(long handle);
}
```

- [ ] **Step 2: Write the JNI shim (no fbjni, raw JNI over `Module`)**

Create `native/jni/executorch_djl_jni.cpp`:
```cpp
// Thin JNI shim over executorch::extension::Module. Raw JNI only — no fbjni,
// no Android. Method names match org.measly.executorch.jni.EtNative.
#include <jni.h>
#include <vector>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

using executorch::extension::Module;
using executorch::extension::from_blob;

extern "C" JNIEXPORT jlong JNICALL
Java_org_measly_executorch_jni_EtNative_loadModule(
    JNIEnv* env, jclass, jstring jpath) {
  const char* path = env->GetStringUTFChars(jpath, nullptr);
  auto* module = new Module(path);
  env->ReleaseStringUTFChars(jpath, path);
  return reinterpret_cast<jlong>(module);
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_org_measly_executorch_jni_EtNative_forwardFloat(
    JNIEnv* env, jclass, jlong handle, jfloatArray ja, jfloatArray jb) {
  auto* module = reinterpret_cast<Module*>(handle);

  const jsize na = env->GetArrayLength(ja);
  const jsize nb = env->GetArrayLength(jb);
  std::vector<float> a(na), b(nb);
  env->GetFloatArrayRegion(ja, 0, na, a.data());
  env->GetFloatArrayRegion(jb, 0, nb, b.data());

  auto ta = from_blob(a.data(), {na});
  auto tb = from_blob(b.data(), {nb});

  auto result = module->forward({ta, tb});
  if (!result.ok()) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                  "ExecuTorch forward() failed");
    return nullptr;
  }
  auto out = result->at(0).toTensor();
  const jsize no = static_cast<jsize>(out.numel());
  jfloatArray jout = env->NewFloatArray(no);
  env->SetFloatArrayRegion(jout, 0, no, out.const_data_ptr<float>());
  return jout;
}

extern "C" JNIEXPORT void JNICALL
Java_org_measly_executorch_jni_EtNative_destroy(
    JNIEnv*, jclass, jlong handle) {
  delete reinterpret_cast<Module*>(handle);
}
```

- [ ] **Step 3: Add the shared-library target to CMake (host JNI via `find_package(JNI)`)**

Append to `native/spike/CMakeLists.txt`:
```cmake
# JNI headers ONLY — the .so is loaded BY the JVM, so it must not link libjvm/libjawt.
# find_package(JNI) insists on those libs and fails ("missing AWT/JVM"); derive jni.h from
# JAVA_HOME, falling back to the JDK that owns `javac`.
if(DEFINED ENV{JAVA_HOME})
  set(JAVA_HOME "$ENV{JAVA_HOME}")
else()
  find_program(JAVAC_EXECUTABLE javac REQUIRED)
  get_filename_component(JAVA_HOME "${JAVAC_EXECUTABLE}" DIRECTORY)   # .../bin
  get_filename_component(JAVA_HOME "${JAVA_HOME}" DIRECTORY)          # JDK root
endif()
if(NOT EXISTS "${JAVA_HOME}/include/jni.h")
  message(FATAL_ERROR "jni.h not found under ${JAVA_HOME}/include — set JAVA_HOME to a JDK")
endif()

add_library(executorch_djl SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/../jni/executorch_djl_jni.cpp)
target_include_directories(executorch_djl PRIVATE
  "${JAVA_HOME}/include" "${JAVA_HOME}/include/linux")
# Bare target names; backend/ops targets self-whole-archive — link plainly (see Task 1 Step 4).
target_link_libraries(executorch_djl PRIVATE
  extension_module_static
  extension_tensor
  xnnpack_backend
  portable_ops_lib
)
```

- [ ] **Step 4: GATE — the `.so` builds and resolves all symbols**

Run:
```bash
cd ~/workspace/djl-executorch-engine/native/spike && \
cmake -B build -G Ninja -DET_INSTALL="$ET_INSTALL" && \
cmake --build build --target executorch_djl && \
ldd -r build/libexecutorch_djl.so 2>&1 | grep -i 'undefined symbol' || echo "NO UNDEFINED SYMBOLS"
```
Expected: builds; prints `NO UNDEFINED SYMBOLS`. Undefined `Java_...` symbols are fine to ignore (resolved by the JVM), but there must be **no undefined ExecuTorch/XNNPACK symbols** — those would mean static libs weren't linked (revisit the Prerequisite PIC flag / Step 3 link list).

- [ ] **Step 5: Write the Java smoke `main` and run end-to-end — INTEGRATION GATE**

Create `native/spike/EtSmoke.java`:
```java
import org.measly.executorch.jni.EtNative;

public final class EtSmoke {
    public static void main(String[] args) {
        String pte = args.length > 0 ? args[0] : "add.pte";
        long h = EtNative.loadModule(pte);
        try {
            float[] out = EtNative.forwardFloat(h, new float[]{2.0f}, new float[]{3.0f});
            float v = out[0];
            System.out.println("RESULT=" + v);
            if (v != 5.0f) {
                throw new AssertionError("expected 5.0 but got " + v);
            }
            System.out.println("JVM<->ExecuTorch desktop bridge OK");
        } finally {
            EtNative.destroy(h);
        }
    }
}
```
Run:
```bash
cd ~/workspace/djl-executorch-engine/native/spike && \
export EXECUTORCH_LIBRARY_PATH="$PWD/build/libexecutorch_djl.so" && \
javac -d build EtNative.java EtSmoke.java && \
java -cp build EtSmoke add.pte
```
Expected:
```
RESULT=5.0
JVM<->ExecuTorch desktop bridge OK
```
**This is the decisive integration gate.** Green here means the show-stopper is cleared end-to-end: a standard desktop JVM loads a `.pte` and runs ExecuTorch inference. A `java.lang.UnsatisfiedLinkError` for a `Java_...` symbol means a name mismatch between `EtNative` (package/method) and the `extern "C"` function names — fix the function name, not the Java.

- [ ] **Step 6: Commit**

```bash
git add native/jni/executorch_djl_jni.cpp native/spike/CMakeLists.txt \
        native/spike/EtNative.java native/spike/EtSmoke.java
git commit -m "feat(native): thin JNI shim bridges desktop JVM to ExecuTorch (add.pte -> 5.0)"
```

---

## Task 3: Reproducible production build + feasibility report

> **Scope note:** this task produces the artifact **locally**. Distributing it across platforms
> (old-glibc/manylinux build env, two-stage runtime+shim pipeline, portability gate, license
> bundling) is deferred to the release pipeline — see [`docs/ci-native-build.md`](../../ci-native-build.md).
> Key constraint from there: the build host's glibc becomes the user's minimum, so a dev-box
> build is not distributable.

**Files:**
- Create: `native/CMakeLists.txt`
- Create: `native/build_desktop.sh`
- Create: `docs/superpowers/reports/executorch-desktop-feasibility.md`

- [ ] **Step 1: Promote the shim CMake to a standalone production file**

Create `native/CMakeLists.txt` (same content as the `executorch_djl` portion of `native/spike/CMakeLists.txt`, parameterized on `ET_INSTALL`, building only `executorch_djl` from `native/jni/executorch_djl_jni.cpp`). It must not depend on anything under `native/spike/`. Skeleton:
```cmake
cmake_minimum_required(VERSION 3.24)
project(executorch_djl LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ET_INSTALL "$ENV{HOME}/workspace/executorch/cmake-out" CACHE PATH "ExecuTorch install prefix")
list(PREPEND CMAKE_PREFIX_PATH "${ET_INSTALL}")
set(tokenizers_DIR "${ET_INSTALL}/lib/cmake/tokenizers" CACHE PATH "")
find_package(executorch CONFIG REQUIRED PATHS "${ET_INSTALL}/lib/cmake/ExecuTorch")

# JNI headers only (see Task 2 Step 3 rationale) — never link libjvm/libjawt.
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
# Bare target names; backend/ops targets self-whole-archive — link plainly (see Task 1 Step 4).
target_link_libraries(executorch_djl PRIVATE
  extension_module_static
  extension_tensor
  xnnpack_backend
  portable_ops_lib
)
```

- [ ] **Step 2: Write `build_desktop.sh` so one command builds + packages the shim**

Create `native/build_desktop.sh`:
```bash
#!/usr/bin/env bash
# Build the desktop-Linux ExecuTorch JNI shim against a prebuilt runtime ($ET_INSTALL)
# and stage the .so for JAR bundling. The runtime itself is built per the Prerequisite.
# See docs/superpowers/plans/2026-06-28-executorch-desktop-linux-native-build.md
set -euo pipefail

# Run from the repo root regardless of invocation CWD (relative paths below assume it).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ET_INSTALL="${ET_INSTALL:-$HOME/workspace/executorch/cmake-out}"
JOBS="${JOBS:-$(nproc)}"

test -f "${ET_INSTALL}/lib/cmake/ExecuTorch/executorch-config.cmake" \
  || { echo "ET_INSTALL=${ET_INSTALL} has no executorch-config.cmake; build the runtime first (see Prerequisite)"; exit 1; }

cmake -B native/build -S native -G Ninja -DET_INSTALL="${ET_INSTALL}"
cmake --build native/build -j"${JOBS}"

OUT="src/main/resources/native/linux-x86_64"
mkdir -p "${OUT}"
cp native/build/libexecutorch_djl.so "${OUT}/"
echo "Artifact: ${OUT}/libexecutorch_djl.so"
ls -lh "${OUT}/libexecutorch_djl.so"
```
Then:
```bash
chmod +x native/build_desktop.sh
```

- [ ] **Step 3: GATE — clean reproducible build from scratch**

Run:
```bash
cd ~/workspace/djl-executorch-engine && rm -rf native/build && ./native/build_desktop.sh
```
Expected: ends by printing the artifact path and its size; `src/main/resources/native/linux-x86_64/libexecutorch_djl.so` exists. Record the size (expected on the order of single-digit MB with XNNPACK statically linked — small vs. libtorch's hundreds of MB, confirming the footprint claim).

- [ ] **Step 4: Re-run the Java smoke against the bundled artifact**

Run:
```bash
cd ~/workspace/djl-executorch-engine/native/spike && \
export EXECUTORCH_LIBRARY_PATH="$PWD/../../src/main/resources/native/linux-x86_64/libexecutorch_djl.so" && \
java -cp build EtSmoke add.pte
```
Expected: `RESULT=5.0` / `JVM<->ExecuTorch desktop bridge OK`. Confirms the *bundled* `.so` (not just the spike build tree) works.

- [ ] **Step 5: Write the feasibility report**

Create `docs/superpowers/reports/executorch-desktop-feasibility.md` capturing, with evidence: pinned `ET_VERSION`; the exact gate outputs from the Prerequisite and Tasks 1/2; the chosen bridge path (B vs A) and why; the final `.so` size and `ldd` dependency list (`ldd build/libexecutorch_djl.so`); and a verdict line — **"Desktop-Linux native build is practical: GO"** or the specific blocker. This report is the input the rest of the DJL engine implementation (EtModel/EtNDArray) builds on.

- [ ] **Step 6: Commit**

```bash
git add native/CMakeLists.txt native/build_desktop.sh docs/superpowers/reports/executorch-desktop-feasibility.md
git commit -m "build(native): one-command reproducible desktop .so + feasibility report"
```

---

## Self-review against the task

- **Show-stopper question answered?** Yes — Task 1 (runtime runs a `.pte` on the host) and Task 2 (JVM bridge) are explicit go/no-go gates that together prove `.pte` inference from a desktop JVM. The runtime build itself is deferred to upstream-documented steps (Prerequisite).
- **Original design's Option A risk surfaced?** Yes — the `if(NOT ANDROID)` guard + Android `fbjni` AAR are documented; the plan routes around them via the thin shim (Path B) and keeps Path A as Appendix A.
- **No placeholders?** Concrete code for export, C++ runner, JNI shim, Java surface, and CMake; exact commands with expected output at every gate. Target names that depend on the pinned tag include an exact discovery command rather than a guess.
- **Type/name consistency?** `org.measly.executorch.jni.EtNative.{loadModule,forwardFloat,destroy}` ↔ `Java_org_measly_executorch_jni_EtNative_{loadModule,forwardFloat,destroy}` match across Tasks 2–3; `EXECUTORCH_LIBRARY_PATH` env var consistent; `ET_INSTALL` threaded from Prerequisite through every CMake invocation; `executorch_djl` target / `libexecutorch_djl.so` consistent.

## What this plan deliberately does NOT cover

Per the writing-plans scope check, the rest of the DJL engine (`EtEngine`, `EtModel`, `EtNDArray`, `MapTranslator`, SPI registration, hybrid mode, multi-platform natives for macOS/Windows, CDN/caching, dtype coverage beyond float32) is **out of scope** — those are separate plans that begin only after this one's GO verdict. Multi-platform builds reuse Task 3's `build_desktop.sh` structure with a different toolchain/runtime build.

---

## Appendix A: Path A fallback — patch `extension/android` for the host

Use only if reusing the upstream `org.pytorch.executorch.{Module,Tensor,EValue}` Java classes is judged worth the extra moving parts. Sketch (not a committed step list):

1. **Build `fbjni` for the host.** Clone `facebookincubator/fbjni`, `cmake -B build -DFBJNI_SKIP_TESTS=ON`, build `libfbjni.so` for `linux-x86_64`. (PyTorch's desktop tooling already builds `libfbjni.dylib`/`.so` this way, so it is feasible — just undocumented for non-Android.)
2. **Neutralize the Android guard.** In `extension/android/CMakeLists.txt`, the `if(NOT ANDROID) → FATAL_ERROR "This directory is for Android build only."` block must be bypassed for the host.
3. **Replace NDK `jni.h` with the host JDK.** Add `find_package(JNI REQUIRED)` and `target_include_directories(... ${JNI_INCLUDE_DIRS})`; the stock build assumes the NDK supplies JNI headers implicitly.
4. **Replace the fbjni AAR download** (the build fetches a prebuilt **Android** AAR from Maven and imports its `.so`) with the host `libfbjni.so` from step 1.
5. Build the `executorch_jni` target, then verify the upstream Java `Module.load(...).forward(...)` path against `add.pte`.

Each of these is a place Path A can fail that Path B simply does not have — which is why Path B is primary.
