# ExecuTorch → slf4j Logging Bridge — Design

> **Status:** design (2026-06-30). Routes ExecuTorch's internal `ET_LOG` diagnostics into the Java
> slf4j logging framework via a custom PAL sink installed in the JNI shell. Item 1 (with item 2's
> JNI-cache touch-up folded in) of the DJL-PyTorch-JNI study; profiling (item 3) is parked in
> `docs/benchmarking.md`. Companion to [`djl-executorch-engine-design.md`](../../../djl-executorch-engine-design.md).

## Problem

ExecuTorch emits diagnostics (backend/partitioner selection, op-kernel warnings, loader/runtime
failure reasons) through `ET_LOG`, which by default goes to the platform's stderr sink. In a DJL
deployment that output is **invisible** to the application's logging configuration — a user running
under log4j2/logback sees nothing, and a failed load/inference gives no native context beyond the
Java exception. We want ET's logs surfaced through slf4j, the facade DJL already uses.

## Goal

ExecuTorch `ET_LOG` output appears through slf4j (logger `org.measly.executorch.native`) at a mapped
severity, installed entirely in the JNI shell so the `EtRuntime` core stays JVM-free, and built so
that any failure in the logging path never affects inference.

## Scope

**In scope:** a PAL log sink in the JNI shell forwarding `ET_LOG` → slf4j; the `EtNative.nativeLog`
Java helper + fixed logger; level mapping; the `cacheGlobalClass` JNI helper (item 2, minimal);
slf4j-api as an explicit dependency; tests.

**Out of scope (explicit):**
- **Shell/engine logging.** Our shell has no native-only signal worth logging — failures already
  surface as Java exceptions, and lifecycle/context logging belongs in the Java engine layer with
  direct slf4j calls (no bridge). The bridge carries **only** ExecuTorch's `ET_LOG`.
- **Level gating** (skipping the JNI hop when the slf4j level is disabled). `ET_LOG` volume is low;
  deferred. A "measure the saved JNI hop" note goes in `docs/benchmarking.md`.
- **`EtRuntime` core changes.** The core is untouched and stays JVM-free; the harness/units keep
  ExecuTorch's default stderr PAL.
- **A message-length cap.** ExecuTorch bounds messages upstream (see Design decision 6); no guard
  needed.
- The JNI-cache *struct/registry* consolidation (item 2's larger form) — minimal additions only.

## Design decisions (settled during brainstorming)

1. **Carry only `ET_LOG`.** The native bridge's sole justification is surfacing ExecuTorch's
   otherwise-invisible internal diagnostics. Engine/shell messages are out of scope (above).
2. **Level-mapped, not gated.** Map `et_pal_log_level_t` → slf4j level so severity is correct; no
   `isXxxEnabled` gating (deferred to a benchmarking measurement).
3. **Install via `register_pal`, not weak-symbol override.** ExecuTorch 1.3 exposes
   `register_pal(PalImpl)` / `PalImpl::create(...)` / `get_pal_impl()`
   (`runtime/platform/platform.h:163-207`). Registration at `JNI_OnLoad` is deterministic (no
   link-order ambiguity with the whole-archived static runtime) and runs *after* the `JavaVM*` is
   cached, so ordering is clean. We override **only** the log emit and keep ET's default
   `allocate`/`free`; the default emit is captured via `get_pal_impl()` for fallback.
4. **Single Java helper + one fixed logger.** `EtNative.nativeLog(int level, String message)` logs
   to `LoggerFactory.getLogger("org.measly.executorch.native")`. All slf4j logic lives in Java; the
   JNI side caches one method ID and passes a mapped level int. (No per-ET-file logger names.)
5. **JNI cache: minimal.** Add `g_javaVM` and `g_nativeLog` to the existing file-scope statics; do
   not restructure into a cache struct. Extract a `cacheGlobalClass(env, name)` helper for the
   `FindClass → NewGlobalRef → DeleteLocalRef` block (the logging bridge is its 4th caller —
   EtTensor, EtMethodMeta, ByteBuffer, EtNative), applied to the class-caching pattern only;
   `GetMethodID`/`GetFieldID` calls stay inline.
6. **No message-length cap (documented).** `ET_LOG` formats into a 256-byte stack buffer and passes
   the already-bounded length to the PAL (`runtime/platform/log.cpp:118`,
   `kMaxLogMessageLength = 256`). `std::string(message, length)` therefore safely trusts `length`;
   a `MAX_LOG_LENGTH` guard would be dead code against an upstream-enforced bound. If ET ever raises
   or removes that cap, revisit.

## Architecture

### Placement

The bridge is a **JNI-shell-layer** concern (it calls into Java, needs a `JNIEnv`). It is process-
global — one PAL, one engine `.so`, one bridge installed once at `JNI_OnLoad`. The `EtRuntime` core
(`native/core/`) is not touched and gains no JVM dependency.

### File structure

```
native/jni/
  et_log_level.h            <- jni-free, header-only: constexpr level map (shared with the unit test)
  et_logging.h              <- install entry point declaration
  et_logging.cpp            <- PAL sink + register_pal install + stderr fallback (includes jni.h)
  executorch_djl_jni.cpp    <- JNI_OnLoad: cacheGlobalClass helper, cache g_javaVM/g_nativeLog/EtNative,
                               call et_logging install
src/main/java/org/measly/executorch/jni/EtNative.java   <- nativeLog(int, String) + fixed logger
native/CMakeLists.txt       <- add et_logging.cpp to the executorch_djl sources; et_runtime_test
                               includes et_log_level.h for the mapping unit
build.gradle.kts            <- declare slf4j-api (implementation); logback-classic (testImplementation)
```

### Level mapping (the native↔Java contract)

A stable int contract, defined once and shared by name:

| `et_pal_log_level_t` | int | slf4j call |
|---|---|---|
| `kDebug` ('D') | 0 | `logger.debug` |
| `kInfo` ('I')  | 1 | `logger.info`  |
| `kError` ('E') | 3 | `logger.error` |
| `kFatal` ('F') | 3 | `logger.error` (slf4j has no FATAL) |
| `kUnknown` ('?') | 2 | `logger.warn` |

`et_log_level.h` exposes `constexpr int et_djl_level_to_slf4j(et_pal_log_level_t)` returning the int
above (default → 1/INFO). It is **jni-free and header-only** so the Catch2 unit can test the full
table without a JVM. `EtNative.nativeLog` switches the int to the slf4j call (default → `info`).

### Data flow

```
ET_LOG(...) [native]
  -> et_djl_emit_log(timestamp, level, file, func, line, msg, len)   (our PAL sink)
       level int = et_djl_level_to_slf4j(level)
       jstring = NewStringUTF(std::string(msg, len).c_str())
  -> env->CallStaticVoidMethod(EtNative, nativeLog, levelInt, jstring)
  -> EtNative.nativeLog(level, message) [Java]
  -> slf4j logger "org.measly.executorch.native"
  -> user's binding (log4j2 / logback / …)
```

### Installation (`JNI_OnLoad`)

1. Cache `JavaVM*` (`env->GetJavaVM(&g_javaVM)`).
2. Cache the `EtNative` class (via `cacheGlobalClass`) and `g_nativeLog =
   GetStaticMethodID(EtNative, "nativeLog", "(ILjava/lang/String;)V")`.
3. **If either lookup fails: skip logging install** (return success for the inference path). Logging
   is non-essential; ET's default stderr PAL stays in place.
4. Otherwise capture `get_pal_impl()` (default, for fallback), build a `PalImpl` overriding only the
   emit with `et_djl_emit_log`, and `register_pal(impl)`.

The inference IDs (EtTensor/EtMethodMeta/ByteBuffer) remain load-critical (missing → `JNI_ERR`, as
today). Only the logging additions degrade gracefully.

## Error handling — the sink is exception-transparent

The sink runs inside live JNI calls and on possibly-unattached native threads. Invariants:

- **Threading.** `g_javaVM->GetEnv(&env, JNI_VERSION_1_6)`. If the thread isn't attached,
  `AttachCurrentThreadAsDaemon` (daemon → auto-detaches at thread exit, no leak). If `g_javaVM` is
  null (a log before `JNI_OnLoad` finished) or attach fails → fallback.
- **Pending-exception transparency.** If `env->ExceptionCheck()` is true on entry, do **not** call
  into Java (illegal with a pending exception) → fallback, leaving the pending exception untouched.
  After the `nativeLog` call, if anything left an exception pending, `ExceptionClear()` it (the sink
  must never alter the in-flight call's exception state).
- **Marshalling.** `std::string(message, length)` then `NewStringUTF`; if it returns null (OOM) →
  fallback. `DeleteLocalRef(jstring)` after the call.
- **Fallback** = delegate to the captured default ET emit impl (stderr) — never drop, never crash.
- **Fatal.** `kFatal` → slf4j `error`; ET's own abort/`ET_CHECK` behavior is unchanged.

## Testing strategy

| Layer | Verifies | How |
|---|---|---|
| Catch2 (JVM-free) | level map table | `et_runtime_test` includes `et_log_level.h`; assert `et_djl_level_to_slf4j` for D/I/E/F/? |
| Java unit | `nativeLog` int→slf4j routing | call `EtNative.nativeLog(level, msg)` per level; capture via a logback `ListAppender` on `org.measly.executorch.native`; assert level + text |
| Java integration | full `ET_LOG`→slf4j path | load a corrupt `.pte` fixture (garbage bytes) → ET logs ERROR during the failed load *and* `loadModule` throws; assert the appender captured an ERROR from the native logger |

- **Test deps:** add `ch.qos.logback:logback-classic` as `testImplementation` (production stays
  facade-only on slf4j-api). Capture with a `ListAppender` attached to the named logger.
- **Fixture:** `src/test/resources/models/corrupt.pte` — a small file of non-`.pte` bytes.
- **Verify-during-impl risk:** the integration test assumes ExecuTorch reliably `ET_LOG`s on a bad
  load (it normally does). If a corrupt load proves silent, swap to another deterministic trigger
  (e.g. `method_meta` on a missing method) — flagged in the plan.
- Worker-thread `AttachCurrentThreadAsDaemon` isn't cheaply assertable; exercised opportunistically,
  not unit-tested.

## Build changes

- Declare `org.slf4j:slf4j-api` explicitly as **`compileOnly`** (version `2.0.17`, the version
  `ai.djl:api:0.36.0` resolves). `EtNative` now uses `LoggerFactory` directly, but the engine is a
  plugin and the host application provides the DJL + slf4j stack at runtime — so `compileOnly` mirrors
  the existing `compileOnly("ai.djl:api")` and avoids bundling a second slf4j-api. Add `slf4j-api` +
  `ch.qos.logback:logback-classic` as `testImplementation` for the test runtime/capture.
- Add `ch.qos.logback:logback-classic` as `testImplementation` only.
- `native/CMakeLists.txt`: add `et_logging.cpp` to the `executorch_djl` target sources; the QA
  `et_runtime_test` target gains the `et_log_level.h` include (no new link deps — header-only).

## Success criteria

1. `ET_LOG` emitted during load/forward appears via slf4j on `org.measly.executorch.native` at the
   mapped level.
2. `EtRuntime` core unchanged: `et_runtime.h` still `<jni.h>`-free; the harness and units still
   build and run JVM-free against ET's default PAL.
3. Logging failure never breaks inference: a missing `nativeLog` skips `register_pal` (engine still
   loads/infers); the sink degrades to stderr and is exception-transparent.
4. The Catch2 mapping unit passes JVM-free; the Java unit and integration tests pass.
5. The existing 39 JVM tests + `leakTest` stay green; `slf4j-api` is an explicit dependency.

## Follow-on (not this spec)

- Add a "measure the JNI-hop cost to decide level-gating" note to `docs/benchmarking.md`.
- Profiling (devtools/ETDump) — parked in `docs/benchmarking.md` pending the overhead spike.
