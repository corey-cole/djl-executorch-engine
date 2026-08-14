// Thin JNI shell over measly::et::EtRuntime. Raw JNI, no fbjni. Translation only.
#include <jni.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "et_runtime.h"
#include "et_logging.h"
#include "array_size_limits.h"

using measly::et::EtRuntime;
using measly::et::InputDesc;
using measly::et::MethodMeta;

// Class refs, field IDs and method IDs cached once in JNI_OnLoad. Lookups are relatively expensive
// and FindClass is unsafe with an exception pending, so nothing here is resolved per call. The
// jclass values are global refs (see cacheGlobalClass) because a local ref would not survive the
// return from JNI_OnLoad.
//
// DANGER: the descriptor strings passed to GetFieldID/GetMethodID below are string literals and no
// compiler on either side checks them. Change a Java field or constructor without updating its
// descriptor here and the lookup returns null. For the six IDs cached into the globals below --
// g_fShape, g_fScalarType, g_fData, g_ctor, g_metaCtor, g_byteBufferWrap -- null fails JNI_OnLoad
// with JNI_ERR, which surfaces as an UnsatisfiedLinkError when EtNative's static initializer runs
// System.load: a RUNTIME failure at class init, not a build failure.
//
// The seventh descriptor in JNI_OnLoad is the dangerous one. The "nativeLog" lookup for the logging
// bridge is null-checked and then deliberately IGNORED -- the pending exception is cleared and the
// load is allowed to succeed, because logging must never fail a model load. So a descriptor drift
// on EtNative.nativeLog produces no error anywhere: the bridge is simply never installed and native
// ET_LOG output goes silently dead while everything else keeps working. If native logging has
// vanished for no apparent reason, suspect that literal first. A local
// `./gradlew test` hides it only because Java and the shim get rebuilt from the same tree; the real
// exposure is a staged per-platform binary (the .dll, or a resource .so someone did not rebuild)
// that is a revision behind the Java classes. Treat the Java declaration and the literal here as a
// single edit. Not hypothetical: adding EtMethodMeta.plannedArenaBytes required changing
// "(I[I[Z)V" to "(I[I[ZJ)V" (commit 717eda2).
static jclass g_etTensorClass = nullptr;
static jfieldID g_fShape = nullptr;
static jfieldID g_fScalarType = nullptr;
static jfieldID g_fData = nullptr;
static jmethodID g_ctor = nullptr;

static jclass g_etMethodMetaClass = nullptr;
static jmethodID g_metaCtor = nullptr;

static jclass g_byteBufferClass = nullptr;
static jmethodID g_byteBufferWrap = nullptr;

static jclass g_runtimeExceptionClass = nullptr;
static jclass g_illegalArgumentExceptionClass = nullptr;

// Exception translation. Both helpers only SCHEDULE a Java exception; neither returns control to
// the JVM, so every caller must return immediately afterwards. Continuing to make JNI calls with an
// exception pending is undefined behaviour.
//
// The two obtain their class differently, on purpose. throwJava is called from catch blocks where a
// Java exception may already be pending, and FindClass would then itself fail and hand a null
// jclass to ThrowNew -- hence the global ref cached at JNI_OnLoad. throwIllegalArgument is called
// from argument checks before anything of ours can have thrown, so a per-call FindClass is
// affordable there. (forward() also throws IllegalArgumentException directly off the cached
// g_illegalArgumentExceptionClass for the direct-buffer check; the two spellings are equivalent.)

// Translate a C++ exception into a Java RuntimeException. Call from a catch block.
// The class is cached at JNI_OnLoad: a per-call FindClass here would itself be UB when an
// exception is already pending (FindClass fails -> null passed to ThrowNew).
static void throwJava(JNIEnv* env, const char* fallback, const std::exception* e) {
  env->ThrowNew(g_runtimeExceptionClass, e ? e->what() : fallback);
}

// Throw IllegalArgumentException from a JNI input check. FindClass is null-checked: it can
// fail only when an exception is already pending, and that pending exception propagates instead.
static void throwIllegalArgument(JNIEnv* env, const char* msg) {
  jclass cls = env->FindClass("java/lang/IllegalArgumentException");
  if (cls != nullptr) {
    env->ThrowNew(cls, msg);
    env->DeleteLocalRef(cls);
  }
}

// Throw IllegalStateException, the zero-handle rejection. Same FindClass-before-anything-throws
// reasoning as throwIllegalArgument above.
static void throwIllegalState(JNIEnv* env, const char* msg) {
  jclass cls = env->FindClass("java/lang/IllegalStateException");
  if (cls != nullptr) {
    env->ThrowNew(cls, msg);
    env->DeleteLocalRef(cls);
  }
}

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
  g_metaCtor = env->GetMethodID(g_etMethodMetaClass, "<init>", "(I[I[ZJ)V");
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

  g_runtimeExceptionClass = cacheGlobalClass(env, "java/lang/RuntimeException");
  if (g_runtimeExceptionClass == nullptr) {
    return JNI_ERR;
  }
  g_illegalArgumentExceptionClass = cacheGlobalClass(env, "java/lang/IllegalArgumentException");
  if (g_illegalArgumentExceptionClass == nullptr) {
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

  // Logging is optional: a failed EtNative/nativeLog lookup leaves a pending exception
  // (NoClassDefFoundError / NoSuchMethodError). Clear it so it never leaks past JNI_OnLoad.
  if (env->ExceptionCheck()) {
    env->ExceptionClear();
  }

  return JNI_VERSION_1_6;
}

// Handle convention for every entry point below. A `jlong handle` is a reinterpret_cast of an
// EtRuntime* whose ownership lives on the Java side: loadModule news it and hands the pointer over,
// destroy deletes it. Exactly one value is validated, 0 -- the value EtSymbolBlock.close() writes
// once it has destroyed the module -- and it is rejected with IllegalStateException. There is no
// registry, so nothing else is checkable: a handle that was already destroyed or was never returned
// by loadModule is dereferenced blind, a use-after-free inside native code rather than a Java
// exception. destroy() is the one exception to the check, because `delete nullptr` is already a
// well-defined no-op.
//
// Java-side discipline therefore still carries most of the safety story, and it is partial by
// design: EtSymbolBlock.close() zeroes its handle field under statsLock and toStats() re-reads it
// under the same monitor, so the stats poll can never race a destroy. forwardInternal reads the
// handle once and rejects 0 itself, but deliberately stays off that monitor to keep the hot path
// lock-free -- which is why the zero check closes the *ordered* use-after-close only, and "do not
// close a model with a forward in flight" remains a documented caller contract rather than
// something enforced.
extern "C" JNIEXPORT jlong JNICALL
Java_org_measly_executorch_jni_EtNative_loadModule(
    JNIEnv* env, jclass, jstring jpath, jint jworkspaceSharingMode) {
  const char* path = env->GetStringUTFChars(jpath, nullptr);
  std::string p(path);
  env->ReleaseStringUTFChars(jpath, path);
  try {
    // No range check here: the Java layer emits only -1/0/1/2, and any other value is deliberately
    // passed through so ExecuTorch rejects it at delegate init. et_runtime_test.cpp relies on that
    // to prove the runtime spec reaches the XNNPACK backend.
    return reinterpret_cast<jlong>(new EtRuntime(p, static_cast<int>(jworkspaceSharingMode)));
  } catch (const std::exception& e) {
    throwJava(env, "EtRuntime load failed", &e);
    return 0;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_org_measly_executorch_jni_EtNative_methodMeta(JNIEnv* env, jclass, jlong handle) {
  if (handle == 0) {
    throwIllegalState(env, "methodMeta() on a closed ExecuTorch model (native handle is 0)");
    return nullptr;
  }
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
  jbooleanArray planned = env->NewBooleanArray(n);
  if (planned == nullptr) {
    return nullptr;  // OOM: exception already pending
  }
  std::vector<jboolean> p(n);
  for (jsize i = 0; i < n; ++i) {
    p[i] = meta.inputMemoryPlanned[i] ? JNI_TRUE : JNI_FALSE;
  }
  env->SetBooleanArrayRegion(planned, 0, n, p.data());
  return env->NewObject(g_etMethodMetaClass, g_metaCtor, static_cast<jint>(n), types, planned,
                        static_cast<jlong>(meta.plannedArenaBytes));
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_org_measly_executorch_jni_EtNative_forward(JNIEnv* env, jclass, jlong handle,
                                                jobjectArray jinputs) {
  if (handle == 0) {
    throwIllegalState(env, "forward() on a closed ExecuTorch model (native handle is 0)");
    return nullptr;
  }
  auto* rt = reinterpret_cast<EtRuntime*>(handle);

  jsize nIn = env->GetArrayLength(jinputs);
  std::vector<InputDesc> inputs(nIn);
  // The direct ByteBuffers reached through jinputs must stay live for the whole call:
  // GetDirectBufferAddress hands back a raw pointer into JVM-managed off-heap memory, and that
  // memory is only reachable through the DirectByteBuffer object. Both consumers read it inside
  // rt->forward() -- a memory-planned input is memcpy'd by ExecuTorch at set_input (the export
  // default), an unplanned one is memcpy'd into our staging slot first -- so the addresses must stay
  // valid until forward() returns. They do, because jinputs is a parameter local ref alive for the
  // whole frame and its elements are reachable from it. That reachability is what makes the
  // DeleteLocalRef calls at the bottom of this loop safe.
  //
  // Each GetObjectArrayElement / GetObjectField below mints a LOCAL reference, and a frame is only
  // guaranteed 16 free local slots by the JNI spec (real JVMs grant far more, which is what lets an
  // omission here go unnoticed). Three refs per input would exhaust that guarantee at six inputs, so
  // the loop releases all three before iterating. Keep it that way if you add another ref.
  for (jsize i = 0; i < nIn; ++i) {
    jobject jt = env->GetObjectArrayElement(jinputs, i);
    if (jt == nullptr) {
      throwIllegalArgument(env, ("EtTensor[" + std::to_string(i) + "] is null").c_str());
      return nullptr;
    }
    auto jshape = static_cast<jlongArray>(env->GetObjectField(jt, g_fShape));
    if (jshape == nullptr) {
      throwIllegalArgument(env, "EtTensor.shape is null");
      return nullptr;
    }
    jint st = env->GetIntField(jt, g_fScalarType);
    jobject jbuf = env->GetObjectField(jt, g_fData);

    jsize nd = env->GetArrayLength(jshape);
    std::vector<jlong> sh(nd);
    env->GetLongArrayRegion(jshape, 0, nd, sh.data());
    inputs[i].shape.assign(sh.begin(), sh.end());
    inputs[i].scalarType = static_cast<int8_t>(st);

    void* addr = env->GetDirectBufferAddress(jbuf);
    if (addr == nullptr) {
      env->ThrowNew(g_illegalArgumentExceptionClass,
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
    if (jout == nullptr) {
      return nullptr;  // OOM: exception already pending
    }

    for (jsize i = 0; i < nOut; ++i) {
      const auto& v = outs[i];
      if (measly::et::exceedsJniByteArrayLimit(v.nbytes)) {
        throwJava(env, "ExecuTorch output exceeds the 2GB JNI array limit", nullptr);
        return nullptr;
      }
      jsize ndim = static_cast<jsize>(v.shape.size());
      jlongArray jshape = env->NewLongArray(ndim);
      if (jshape == nullptr) {
        return nullptr;  // OOM: exception already pending
      }
      {
        std::vector<jlong> sh(ndim);
        for (jsize k = 0; k < ndim; ++k) {
          sh[k] = static_cast<jlong>(v.shape[k]);
        }
        env->SetLongArrayRegion(jshape, 0, ndim, sh.data());
      }
      jsize nbytes = static_cast<jsize>(v.nbytes);
      jbyteArray jbytes = env->NewByteArray(nbytes);
      if (jbytes == nullptr) {
        return nullptr;  // OOM: exception already pending
      }
      env->SetByteArrayRegion(jbytes, 0, nbytes, reinterpret_cast<const jbyte*>(v.data));
      jobject jbuf = env->CallStaticObjectMethod(g_byteBufferClass, g_byteBufferWrap, jbytes);
      if (env->ExceptionCheck()) {
        return nullptr;  // ByteBuffer.wrap failed; exception pending
      }

      jobject obj = env->NewObject(g_etTensorClass, g_ctor, jshape,
                                   static_cast<jint>(v.scalarType), jbuf);
      if (obj == nullptr) {
        return nullptr;  // OOM: exception already pending
      }
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

// Frees the runtime and its arena. A handle of 0 is safe -- `delete` on a null pointer is a defined
// no-op -- so the Java-side `if (handle != 0)` guard is about not double-freeing, not about 0
// itself. Nothing here makes destroy idempotent for a NON-zero handle: calling it twice is a double
// free, which is why EtSymbolBlock.close() zeroes the field inside the same synchronized block.
// The catch below is defensive only, and does NOT make this entry point safe against a failing
// teardown. ~EtRuntime is `= default` over a pimpl whose members all have non-throwing destructors,
// so it is implicitly noexcept: an exception escaping ExecuTorch teardown calls std::terminate and
// never reaches the handler. Nor does the catch help against a stale or wild handle, which is
// undefined behaviour rather than a throw. Do not read it as a recovery path.
extern "C" JNIEXPORT void JNICALL
Java_org_measly_executorch_jni_EtNative_destroy(JNIEnv* env, jclass, jlong handle) {
  try {
    delete reinterpret_cast<EtRuntime*>(handle);
  } catch (const std::exception& e) {
    throwJava(env, "EtRuntime destroy failed", &e);
  }
}

extern "C" JNIEXPORT jint JNICALL
Java_org_measly_executorch_jni_EtNative_setIntraOpThreads(JNIEnv* env, jclass, jint n) {
  if (n < 1) {
    // jint is signed: -1 (or 0) must never reach the pool allocator as a huge uint32_t. No-op
    // and report the current count -- the core's "count in effect AFTER the attempt" contract
    // (issue #24). The Java gate already rejects n < 1, but EtNative is a public entry point.
    return static_cast<jint>(measly::et::intraOpThreads());
  }
  return static_cast<jint>(measly::et::setIntraOpThreads(static_cast<uint32_t>(n)));
}

extern "C" JNIEXPORT jint JNICALL
Java_org_measly_executorch_jni_EtNative_intraOpThreads(JNIEnv* env, jclass) {
  return static_cast<jint>(measly::et::intraOpThreads());
}

// Total capacity of the input staging slots, for the stats path. Two distinct zero-ish results the
// caller must not conflate: a genuine 0 means every input is memory-planned (the export default)
// and nothing is ever staged, whereas -1 is the error return after an exception was scheduled.
// EtSymbolBlock.toStats() also reports -1 for a closed block, so -1 uniformly means "unavailable".
extern "C" JNIEXPORT jlong JNICALL
Java_org_measly_executorch_jni_EtNative_stagingBytes(JNIEnv* env, jclass, jlong handle) {
  if (handle == 0) {
    throwIllegalState(env, "stagingBytes() on a closed ExecuTorch model (native handle is 0)");
    return -1;
  }
  auto* rt = reinterpret_cast<EtRuntime*>(handle);
  try {
    return static_cast<jlong>(rt->stagingBytes());
  } catch (const std::exception& e) {
    throwJava(env, "stagingBytes failed", &e);
    return -1;
  }
}
