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
