# ExecuTorch → slf4j Logging Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route ExecuTorch's internal `ET_LOG` diagnostics into the Java slf4j framework via a custom PAL log sink installed in the JNI shell, leaving the `EtRuntime` core JVM-free.

**Architecture:** A jni-free level-map header (shared contract), a single Java helper `EtNative.nativeLog(int, String)` on a fixed logger, and a JNI-shell PAL sink installed at `JNI_OnLoad` via `register_pal` that forwards `ET_LOG` → `nativeLog`. Logging is non-essential: any failure in the path degrades to stderr and never affects inference.

**Tech Stack:** C++20, ExecuTorch v1.3.1 PAL (`register_pal`/`PalImpl`/`get_pal_impl`, `runtime/platform/platform.h`), raw JNI, slf4j-api 2.0.17 (compileOnly), logback-classic (test only), Catch2 v3, Gradle/JUnit 5.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-06-30-executorch-logging-bridge-design.md`.
- **Carry only `ET_LOG`** — no shell/engine logging; the bridge surfaces ExecuTorch's own diagnostics only.
- **Level int contract (native ↔ Java):** `0=DEBUG, 1=INFO, 2=WARN, 3=ERROR`. ET PAL char → code: `'D'→0, 'I'→1, 'E'→3, 'F'→3, '?'→2`, unknown → `1`.
- **Install via `register_pal` at `JNI_OnLoad`**, overriding only `emit_log_message` (null entries keep ET defaults). Capture the default emit **function pointer by value before** `register_pal` for fallback (the `get_pal_impl()` table is mutated by `register_pal`).
- **Single Java helper** `EtNative.nativeLog(int level, String message)` (package-private), logger `org.measly.executorch.native`. JNI caches one method ID `(ILjava/lang/String;)V`.
- **JNI cache minimal:** add the `EtNative` class + `nativeLog` ID; extract `cacheGlobalClass(env, name)` for the `FindClass→NewGlobalRef→DeleteLocalRef` block (class pattern only; leave `GetMethodID`/`GetFieldID` inline). No cache struct.
- **Logging is non-essential:** if the `EtNative`/`nativeLog` lookup fails, **skip** `register_pal` and return success — never fail the load. Inference IDs (EtTensor/EtMethodMeta/ByteBuffer) remain load-critical (`JNI_ERR` on miss).
- **Sink is exception-transparent:** never call into Java with a pending exception; never leave one pending; on any failure (no VM, attach fail, pending exception, OOM jstring) delegate to the captured default emit (stderr), never drop, never crash. Worker threads: `AttachCurrentThreadAsDaemon` (auto-detach).
- **No message-length cap** — ET bounds messages at `kMaxLogMessageLength = 256` (`runtime/platform/log.cpp:118`) before the PAL; document this where the message is marshalled.
- **`EtRuntime` core untouched:** `et_runtime.h` stays `<jni.h>`-free; harness/units keep ET's default PAL and build JVM-free.
- **Dependencies:** `slf4j-api:2.0.17` as **`compileOnly`** (matches `ai.djl:api` — the host provides the DJL+slf4j stack); `slf4j-api:2.0.17` + `ch.qos.logback:logback-classic:1.5.18` as `testImplementation`.
- **Build env:** `JAVA_HOME=/usr/lib/jvm/zulu-17-amd64`; `ET_INSTALL=$HOME/workspace/executorch/cmake-out`; ASan QA build via `-DET_BUILD_QA=ON`.
- **Commit trailer (every commit):** `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Level-map contract (jni-free) + Catch2 unit

A header-only, ExecuTorch-and-JNI-free mapping from the ET PAL log-level char to the slf4j level int, unit-tested in the existing Catch2 suite. Establishes the int contract that Task 2 (Java) and Task 3 (native sink) both depend on.

**Files:**
- Create: `native/jni/et_log_level.h`
- Modify: `native/test/et_runtime_test.cpp` (add mapping cases)
- Modify: `native/CMakeLists.txt` (add `native/jni` to `et_runtime_test` includes)

**Interfaces:**
- Produces: `namespace measly::et` — `enum Slf4jLevel : int { kSlf4jDebug=0, kSlf4jInfo=1, kSlf4jWarn=2, kSlf4jError=3 }` and `constexpr int et_djl_level_to_slf4j(char level)`.

- [ ] **Step 1: Write the failing test cases**

Append to `native/test/et_runtime_test.cpp` (add the include near the top with the others):

```cpp
#include "et_log_level.h"
```

and add these cases at the end of the file:

```cpp
TEST_CASE("level map: ET PAL chars -> slf4j level codes") {
  using namespace measly::et;
  REQUIRE(et_djl_level_to_slf4j('D') == kSlf4jDebug);
  REQUIRE(et_djl_level_to_slf4j('I') == kSlf4jInfo);
  REQUIRE(et_djl_level_to_slf4j('E') == kSlf4jError);
  REQUIRE(et_djl_level_to_slf4j('F') == kSlf4jError);  // slf4j has no FATAL
  REQUIRE(et_djl_level_to_slf4j('?') == kSlf4jWarn);
  REQUIRE(et_djl_level_to_slf4j('X') == kSlf4jInfo);   // unknown -> INFO default
}
```

- [ ] **Step 2: Add the include dir to the test target**

In `native/CMakeLists.txt`, inside the `if(ET_BUILD_QA)` block, immediately after the
`target_compile_definitions(et_runtime_test ...)` line, add:

```cmake
  target_include_directories(et_runtime_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/jni)
```

- [ ] **Step 3: Build to verify it fails (header missing)**

```bash
cd /home/corey/workspace/djl-executorch-engine
export JAVA_HOME=/usr/lib/jvm/zulu-17-amd64 ET_INSTALL="$HOME/workspace/executorch/cmake-out"
cmake -B native/asan -S native -DET_INSTALL="$ET_INSTALL" -DET_BUILD_QA=ON \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"
cmake --build native/asan --target et_runtime_test
```

Expected: **compile error** — `et_log_level.h: No such file or directory`.

- [ ] **Step 4: Create the header**

Create `native/jni/et_log_level.h`:

```cpp
#ifndef MEASLY_ET_LOG_LEVEL_H
#define MEASLY_ET_LOG_LEVEL_H

namespace measly::et {

// slf4j level codes shared across the JNI boundary with EtNative.nativeLog.
enum Slf4jLevel : int {
  kSlf4jDebug = 0,
  kSlf4jInfo = 1,
  kSlf4jWarn = 2,
  kSlf4jError = 3,
};

// Map an ExecuTorch PAL log-level char ('D','I','E','F','?') to an slf4j level code.
// Char-based (not et_pal_log_level_t) so this header is free of ExecuTorch AND JNI — the
// Catch2 unit and the JNI sink both include it.
constexpr int et_djl_level_to_slf4j(char level) {
  switch (level) {
    case 'D': return kSlf4jDebug;
    case 'I': return kSlf4jInfo;
    case 'E': return kSlf4jError;
    case 'F': return kSlf4jError;  // slf4j has no FATAL
    case '?': return kSlf4jWarn;
    default: return kSlf4jInfo;
  }
}

}  // namespace measly::et
#endif  // MEASLY_ET_LOG_LEVEL_H
```

- [ ] **Step 5: Build + run to verify pass**

```bash
cd /home/corey/workspace/djl-executorch-engine
cmake --build native/asan --target et_runtime_test
./native/asan/et_runtime_test
```

Expected: `All tests passed` (now 6 test cases — the original 5 plus the level-map case), clean ASan exit.

- [ ] **Step 6: Commit**

```bash
cd /home/corey/workspace/djl-executorch-engine
git add native/jni/et_log_level.h native/test/et_runtime_test.cpp native/CMakeLists.txt
git commit -m "$(cat <<'EOF'
feat(logging): jni-free ET PAL log-level -> slf4j level-code map + unit

Shared contract header (char-based, ExecuTorch/JNI-free) consumed by both the
Java helper and the native sink; unit-tested in the Catch2 suite.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Java helper `EtNative.nativeLog` + logger + slf4j/logback deps + Java unit

The Java side of the bridge: a package-private `nativeLog(int, String)` that routes to slf4j on a fixed logger, plus the dependency wiring and a logback-captured unit test. Independent of the native sink (testable by calling `nativeLog` directly).

**Files:**
- Modify: `src/main/java/org/measly/executorch/jni/EtNative.java`
- Modify: `build.gradle.kts` (deps)
- Create: `src/test/java/org/measly/executorch/jni/EtNativeLogTest.java`

**Interfaces:**
- Consumes: the int level contract from Task 1 (`0=debug,1=info,2=warn,3=error`).
- Produces: `static void EtNative.nativeLog(int level, String message)` (package-private), the native sink's Java target.

- [ ] **Step 1: Add the dependencies**

In `build.gradle.kts`, replace the `dependencies { ... }` block body with (keep existing lines, add the four new ones):

```kotlin
dependencies {
    compileOnly("ai.djl:api:$djlVersion")
    compileOnly("org.slf4j:slf4j-api:2.0.17")

    testImplementation("ai.djl:api:$djlVersion")
    testImplementation("org.slf4j:slf4j-api:2.0.17")
    testImplementation("ch.qos.logback:logback-classic:1.5.18")
    testImplementation(platform("org.junit:junit-bom:5.10.2"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}
```

- [ ] **Step 2: Write the failing Java unit test**

Create `src/test/java/org/measly/executorch/jni/EtNativeLogTest.java`:

```java
package org.measly.executorch.jni;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

class EtNativeLogTest {

    private Logger logger;
    private ListAppender<ILoggingEvent> appender;

    @BeforeEach
    void attach() {
        logger = (Logger) LoggerFactory.getLogger("org.measly.executorch.native");
        logger.setLevel(Level.TRACE);
        appender = new ListAppender<>();
        appender.start();
        logger.addAppender(appender);
    }

    @AfterEach
    void detach() {
        logger.detachAppender(appender);
    }

    @Test
    void routesEachLevel() {
        EtNative.nativeLog(0, "dbg");
        EtNative.nativeLog(1, "inf");
        EtNative.nativeLog(2, "wrn");
        EtNative.nativeLog(3, "err");
        assertEquals(4, appender.list.size());
        assertEquals(Level.DEBUG, appender.list.get(0).getLevel());
        assertEquals("dbg", appender.list.get(0).getMessage());
        assertEquals(Level.INFO, appender.list.get(1).getLevel());
        assertEquals(Level.WARN, appender.list.get(2).getLevel());
        assertEquals(Level.ERROR, appender.list.get(3).getLevel());
    }

    @Test
    void unknownLevelDefaultsToInfo() {
        EtNative.nativeLog(99, "huh");
        assertEquals(1, appender.list.size());
        assertEquals(Level.INFO, appender.list.get(0).getLevel());
    }
}
```

- [ ] **Step 3: Run to verify it fails (no nativeLog yet)**

```bash
cd /home/corey/workspace/djl-executorch-engine
JAVA_HOME=/usr/lib/jvm/zulu-17-amd64 ./gradlew test --tests 'org.measly.executorch.jni.EtNativeLogTest'
```

Expected: **compilation failure** — `cannot find symbol: method nativeLog(int,String)`.

- [ ] **Step 4: Add the logger + nativeLog to EtNative**

Replace `src/main/java/org/measly/executorch/jni/EtNative.java` with:

```java
package org.measly.executorch.jni;

import org.measly.executorch.engine.LibUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** JNI surface to the ExecuTorch native library. Loads the .so on class init. */
public final class EtNative {

    /** Sink for ExecuTorch's native ET_LOG output, forwarded by the JNI PAL bridge. */
    private static final Logger NATIVE_LOG = LoggerFactory.getLogger("org.measly.executorch.native");

    static {
        LibUtils.loadLibrary();
    }

    private EtNative() {}

    public static native long loadModule(String ptePath);

    public static native EtMethodMeta methodMeta(long handle);

    public static native EtTensor[] forward(long handle, EtTensor[] inputs);

    public static native void destroy(long handle);

    /**
     * Called from native code (the ExecuTorch PAL sink) to route an ET_LOG message to slf4j.
     * Level codes match {@code measly::et::Slf4jLevel}: 0=debug, 1=info, 2=warn, 3=error
     * (unknown → info).
     */
    static void nativeLog(int level, String message) {
        switch (level) {
            case 0:
                NATIVE_LOG.debug(message);
                break;
            case 2:
                NATIVE_LOG.warn(message);
                break;
            case 3:
                NATIVE_LOG.error(message);
                break;
            case 1:
            default:
                NATIVE_LOG.info(message);
                break;
        }
    }
}
```

- [ ] **Step 5: Run to verify the new test passes and nothing regressed**

```bash
cd /home/corey/workspace/djl-executorch-engine
JAVA_HOME=/usr/lib/jvm/zulu-17-amd64 ./gradlew test
```

Expected: **BUILD SUCCESSFUL** — the existing 39 tests plus the 2 new `EtNativeLogTest` cases all pass. (The committed `.so` is unchanged; this task is pure Java.)

- [ ] **Step 6: Commit**

```bash
cd /home/corey/workspace/djl-executorch-engine
git add src/main/java/org/measly/executorch/jni/EtNative.java build.gradle.kts \
  src/test/java/org/measly/executorch/jni/EtNativeLogTest.java
git commit -m "$(cat <<'EOF'
feat(logging): EtNative.nativeLog slf4j helper + fixed native logger

Package-private nativeLog(int,String) routes ET log levels to slf4j on logger
'org.measly.executorch.native'. slf4j-api added compileOnly (host provides it,
like ai.djl:api); logback-classic test-only for capture.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Native PAL sink + install + JNI_OnLoad wiring

The native bridge: a PAL log sink that forwards to `EtNative.nativeLog`, installed at `JNI_OnLoad` via `register_pal`, with the `cacheGlobalClass` refactor and graceful degradation. Rebuilds and re-stages the `.so`; parity proven by the unchanged JVM suite.

**Files:**
- Create: `native/jni/et_logging.h`
- Create: `native/jni/et_logging.cpp`
- Modify: `native/jni/executorch_djl_jni.cpp` (cacheGlobalClass; cache EtNative + nativeLog; call install)
- Modify: `native/CMakeLists.txt` (add `et_logging.cpp` to `executorch_djl`)

**Interfaces:**
- Consumes: `EtNative.nativeLog` (Task 2); `et_djl_level_to_slf4j` (Task 1); ET `register_pal`/`PalImpl`/`get_pal_impl`.
- Produces: `bool measly::et::installLoggingBridge(JavaVM*, jclass etNativeClass, jmethodID nativeLogMethod)`.

- [ ] **Step 1: Create the install header**

Create `native/jni/et_logging.h`:

```cpp
#ifndef MEASLY_ET_LOGGING_H
#define MEASLY_ET_LOGGING_H

#include <jni.h>

namespace measly::et {

// Install the ExecuTorch PAL log sink that forwards ET_LOG -> EtNative.nativeLog (slf4j).
// Call once from JNI_OnLoad AFTER the JavaVM* and the nativeLog method ID are available.
// The caller passes a process-lifetime global ref for etNativeClass. Returns false (non-fatal)
// if arguments are null or registration fails; the engine then keeps ET's default PAL.
bool installLoggingBridge(JavaVM* vm, jclass etNativeClass, jmethodID nativeLogMethod);

}  // namespace measly::et
#endif  // MEASLY_ET_LOGGING_H
```

- [ ] **Step 2: Create the sink implementation**

Create `native/jni/et_logging.cpp`:

```cpp
#include "et_logging.h"

#include <cstdio>
#include <string>

#include <executorch/runtime/platform/platform.h>

#include "et_log_level.h"

namespace measly::et {
namespace {

JavaVM* g_vm = nullptr;
jclass g_etNativeClass = nullptr;
jmethodID g_nativeLogMethod = nullptr;
// Captured BY VALUE before register_pal — capturing the get_pal_impl() table pointer would alias
// our own emitter after registration and recurse infinitely.
pal_emit_log_message_method g_defaultEmit = nullptr;

void emitFallback(et_timestamp_t ts, et_pal_log_level_t level, const char* file,
                  const char* func, size_t line, const char* msg, size_t len) {
  if (g_defaultEmit != nullptr) {
    g_defaultEmit(ts, level, file, func, line, msg, len);
  } else {
    std::fprintf(stderr, "%c executorch: %.*s\n", static_cast<char>(level),
                 static_cast<int>(len), msg);
  }
}

// The PAL sink. Must be exception-transparent: never call into Java with a pending exception,
// never leave one pending, never drop a message, never crash.
void etDjlEmitLog(et_timestamp_t ts, et_pal_log_level_t level, const char* file,
                  const char* func, size_t line, const char* msg, size_t len) {
  if (g_vm == nullptr) {
    emitFallback(ts, level, file, func, line, msg, len);
    return;
  }
  JNIEnv* env = nullptr;
  jint rc = g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
  if (rc == JNI_EDETACHED) {
    // Daemon attach auto-detaches at thread exit — no explicit DetachCurrentThread needed.
    if (g_vm->AttachCurrentThreadAsDaemon(reinterpret_cast<void**>(&env), nullptr) != JNI_OK) {
      emitFallback(ts, level, file, func, line, msg, len);
      return;
    }
  } else if (rc != JNI_OK) {
    emitFallback(ts, level, file, func, line, msg, len);
    return;
  }
  if (env->ExceptionCheck()) {
    // A Java exception is in flight; calling into Java is illegal. Leave it untouched.
    emitFallback(ts, level, file, func, line, msg, len);
    return;
  }
  // ET caps messages at kMaxLogMessageLength (256) before the PAL, so length is safely bounded;
  // no truncation guard is needed (see spec design decision 6).
  std::string text(msg, len);
  jstring jmsg = env->NewStringUTF(text.c_str());
  if (jmsg == nullptr) {
    if (env->ExceptionCheck()) {
      env->ExceptionClear();
    }
    emitFallback(ts, level, file, func, line, msg, len);
    return;
  }
  env->CallStaticVoidMethod(g_etNativeClass, g_nativeLogMethod,
                            static_cast<jint>(et_djl_level_to_slf4j(static_cast<char>(level))),
                            jmsg);
  env->DeleteLocalRef(jmsg);
  if (env->ExceptionCheck()) {
    env->ExceptionClear();  // the sink must not perturb the caller's exception state
  }
}

}  // namespace

bool installLoggingBridge(JavaVM* vm, jclass etNativeClass, jmethodID nativeLogMethod) {
  if (vm == nullptr || etNativeClass == nullptr || nativeLogMethod == nullptr) {
    return false;
  }
  g_vm = vm;
  g_etNativeClass = etNativeClass;  // process-lifetime global ref owned by the caller
  g_nativeLogMethod = nativeLogMethod;

  const executorch::runtime::PalImpl* current = executorch::runtime::get_pal_impl();
  g_defaultEmit = (current != nullptr) ? current->emit_log_message : nullptr;

  executorch::runtime::PalImpl impl =
      executorch::runtime::PalImpl::create(etDjlEmitLog, __FILE__);
  return executorch::runtime::register_pal(impl);
}

}  // namespace measly::et
```

- [ ] **Step 3: Wire JNI_OnLoad — cacheGlobalClass + install**

In `native/jni/executorch_djl_jni.cpp`:

(a) Add the include after the existing `#include "et_runtime.h"`:

```cpp
#include "et_logging.h"
```

(b) Add the `cacheGlobalClass` helper immediately after the `throwJava` helper definition:

```cpp
// FindClass -> NewGlobalRef -> DeleteLocalRef. Returns a process-lifetime global ref, or nullptr
// (pending exception) so the caller can fail JNI_OnLoad.
static jclass cacheGlobalClass(JNIEnv* env, const char* name) {
  jclass local = env->FindClass(name);
  if (local == nullptr) {
    return nullptr;
  }
  jclass global = static_cast<jclass>(env->NewGlobalRef(local));
  env->DeleteLocalRef(local);
  return global;
}
```

(c) Replace the body of `JNI_OnLoad` (from the first `FindClass` through the final `return JNI_VERSION_1_6;`) with the version below — the three existing class lookups now use `cacheGlobalClass`, and the logging bridge is installed last (non-fatally):

```cpp
extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  JNIEnv* env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }

  g_etTensorClass = cacheGlobalClass(env, "org/measly/executorch/jni/EtTensor");
  if (g_etTensorClass == nullptr) {
    return JNI_ERR;  // class not found -> System.load fails clearly
  }
  g_fShape = env->GetFieldID(g_etTensorClass, "shape", "[J");
  g_fScalarType = env->GetFieldID(g_etTensorClass, "scalarType", "I");
  g_fData = env->GetFieldID(g_etTensorClass, "data", "Ljava/nio/ByteBuffer;");
  g_ctor = env->GetMethodID(g_etTensorClass, "<init>", "([JILjava/nio/ByteBuffer;)V");
  if (g_fShape == nullptr || g_fScalarType == nullptr || g_fData == nullptr || g_ctor == nullptr) {
    return JNI_ERR;
  }

  g_etMethodMetaClass = cacheGlobalClass(env, "org/measly/executorch/jni/EtMethodMeta");
  if (g_etMethodMetaClass == nullptr) {
    return JNI_ERR;
  }
  g_metaCtor = env->GetMethodID(g_etMethodMetaClass, "<init>", "(I[I)V");
  if (g_metaCtor == nullptr) {
    return JNI_ERR;
  }

  g_byteBufferClass = cacheGlobalClass(env, "java/nio/ByteBuffer");
  if (g_byteBufferClass == nullptr) {
    return JNI_ERR;
  }
  g_byteBufferWrap = env->GetStaticMethodID(g_byteBufferClass, "wrap", "([B)Ljava/nio/ByteBuffer;");
  if (g_byteBufferWrap == nullptr) {
    return JNI_ERR;
  }

  // Logging bridge is non-essential: if the hooks aren't found, skip it and keep ET's default
  // PAL — never fail the load over logging.
  jclass etNativeClass = cacheGlobalClass(env, "org/measly/executorch/jni/EtNative");
  if (etNativeClass != nullptr) {
    jmethodID nativeLog =
        env->GetStaticMethodID(etNativeClass, "nativeLog", "(ILjava/lang/String;)V");
    if (nativeLog != nullptr) {
      measly::et::installLoggingBridge(vm, etNativeClass, nativeLog);
    }
  }

  return JNI_VERSION_1_6;
}
```

- [ ] **Step 4: Add the sink to the CMake source list**

In `native/CMakeLists.txt`, change the `executorch_djl` library definition to include the new source:

```cmake
add_library(executorch_djl SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/jni/executorch_djl_jni.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/jni/et_logging.cpp)
```

- [ ] **Step 5: Rebuild + stage the `.so`**

```bash
cd /home/corey/workspace/djl-executorch-engine
JAVA_HOME=/usr/lib/jvm/zulu-17-amd64 ET_INSTALL="$HOME/workspace/executorch/cmake-out" \
  bash native/build_desktop.sh
```

Expected: `Artifact: src/main/resources/native/linux-x86_64/libexecutorch_djl.so` and a non-empty `ls -lh`.

- [ ] **Step 6: Parity gate — existing suite stays green**

```bash
cd /home/corey/workspace/djl-executorch-engine
JAVA_HOME=/usr/lib/jvm/zulu-17-amd64 ./gradlew test --rerun-tasks
```

Expected: **BUILD SUCCESSFUL** — all 39 original tests plus Task 2's `EtNativeLogTest` pass. The installed PAL bridge must not break inference. If any test fails, stop and diagnose — the bridge install is regressing the engine.

- [ ] **Step 7: Commit**

```bash
cd /home/corey/workspace/djl-executorch-engine
git add native/jni/et_logging.h native/jni/et_logging.cpp native/jni/executorch_djl_jni.cpp \
  native/CMakeLists.txt src/main/resources/native/linux-x86_64/libexecutorch_djl.so
git commit -m "$(cat <<'EOF'
feat(logging): install ExecuTorch PAL sink forwarding ET_LOG to slf4j

JNI_OnLoad registers a PAL emit override (via register_pal) that marshals ET_LOG
to EtNative.nativeLog; exception-transparent, daemon-attaches worker threads, and
degrades to the captured default emit. Adds cacheGlobalClass (4th caller). Logging
is non-essential: a missing nativeLog skips install, never fails the load.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: End-to-end integration test (ET_LOG → slf4j)

Prove the full path: a corrupt `.pte` makes ExecuTorch emit an ERROR during load, which the PAL sink routes to the native slf4j logger, captured by a logback appender.

**Files:**
- Create: `src/test/resources/models/corrupt.pte`
- Create: `src/test/java/org/measly/executorch/LoggingBridgeIT.java`

**Interfaces:**
- Consumes: the installed bridge from Task 3; `EtNative.loadModule` (throws on bad load).

- [ ] **Step 1: Create the corrupt fixture**

```bash
cd /home/corey/workspace/djl-executorch-engine
mkdir -p src/test/resources/models
printf 'NOT_A_PTE\x00\x01\x02\x03garbage-bytes-not-a-flatbuffer' \
  > src/test/resources/models/corrupt.pte
```

- [ ] **Step 2: Write the integration test**

Create `src/test/java/org/measly/executorch/LoggingBridgeIT.java`:

```java
package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.measly.executorch.jni.EtNative;
import org.slf4j.LoggerFactory;

class LoggingBridgeIT {

    private Logger logger;
    private ListAppender<ILoggingEvent> appender;

    @BeforeEach
    void attach() {
        logger = (Logger) LoggerFactory.getLogger("org.measly.executorch.native");
        logger.setLevel(Level.TRACE);
        appender = new ListAppender<>();
        appender.start();
        logger.addAppender(appender);
    }

    @AfterEach
    void detach() {
        logger.detachAppender(appender);
    }

    @Test
    void corruptModelLoadEmitsNativeErrorLogThroughSlf4j() throws Exception {
        Path corrupt =
                Paths.get(getClass().getResource("/models/corrupt.pte").toURI());
        // The bad load throws; before that, ExecuTorch ET_LOGs the failure, which the PAL bridge
        // routes to the native slf4j logger.
        assertThrows(RuntimeException.class,
                () -> EtNative.loadModule(corrupt.toAbsolutePath().toString()));
        assertTrue(
                appender.list.stream().anyMatch(e -> e.getLevel() == Level.ERROR),
                "expected a native ERROR log from the failed ExecuTorch load");
    }
}
```

- [ ] **Step 3: Run the integration test**

```bash
cd /home/corey/workspace/djl-executorch-engine
JAVA_HOME=/usr/lib/jvm/zulu-17-amd64 ./gradlew test --tests 'org.measly.executorch.LoggingBridgeIT'
```

Expected: **PASS** — the corrupt load throws and an ERROR event is captured.

**Verify-during-impl note:** this assumes ExecuTorch emits an `ET_LOG` at ERROR on a corrupt load (it normally logs the program-parse failure). If the appender captures nothing, confirm via a quick stderr check that ET logged at all; if a corrupt file proves silent, switch the trigger to a guaranteed-logging path — e.g. call `EtNative.methodMeta` for a method that does not exist, or load a file whose header parses but whose body is truncated — and adjust the assertion's expected level to match. Do not weaken the test to assert nothing; it must assert a real captured event.

- [ ] **Step 4: Full suite green**

```bash
cd /home/corey/workspace/djl-executorch-engine
JAVA_HOME=/usr/lib/jvm/zulu-17-amd64 ./gradlew test
```

Expected: **BUILD SUCCESSFUL** — all prior tests plus `LoggingBridgeIT`.

- [ ] **Step 5: Commit**

```bash
cd /home/corey/workspace/djl-executorch-engine
git add src/test/resources/models/corrupt.pte src/test/java/org/measly/executorch/LoggingBridgeIT.java
git commit -m "$(cat <<'EOF'
test(logging): end-to-end ET_LOG -> slf4j via corrupt-model load

Loading a corrupt .pte makes ExecuTorch ET_LOG an error during load; the PAL
bridge routes it to the native slf4j logger, captured by a logback appender.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Notes for the implementer

- **ExecuTorch PAL API (verified against the installed v1.3.1 headers):** `runtime/platform/platform.h` —
  `PalImpl::create(pal_emit_log_message_method, const char* source_filename)` (single-emit overload;
  null fields keep defaults), `bool register_pal(PalImpl)`, `const PalImpl* get_pal_impl()` with field
  `emit_log_message`. The sink/level types (`et_timestamp_t`, `et_pal_log_level_t` with
  `kDebug='D'/kInfo='I'/kError='E'/kFatal='F'/kUnknown='?'`, `pal_emit_log_message_method`) are global
  (declared in that header). Cross-check there if a signature differs.
- **Two build dirs, by design:** `native/build` (Release `.so`, via `build_desktop.sh`) and
  `native/asan` (instrumented QA, `-DET_BUILD_QA=ON`). Task 1's Catch2 unit uses the ASan dir; Task 3
  ships the Release `.so`.
- **The rebuilt `.so` is tracked and bundled** — Task 3 commits it alongside the source so the
  artifact matches the code (same convention as the EtRuntime branch). It is not byte-reproducible
  incremental-vs-clean; that's a known documented gap, not a defect.
- **`JAVA_HOME=/usr/lib/jvm/zulu-17-amd64`** is required for every `build_desktop.sh` and `./gradlew`
  invocation (JDK 17 with `jni.h`).
```
