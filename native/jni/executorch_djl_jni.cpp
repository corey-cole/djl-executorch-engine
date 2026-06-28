// Thin JNI shim over executorch::extension::Module. Raw JNI, no fbjni.
#include <jni.h>
#include <vector>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/executor/method_meta.h>

using executorch::extension::Module;
using executorch::extension::from_blob;

static jclass g_etTensorClass = nullptr;
static jfieldID g_fShape = nullptr;
static jfieldID g_fData = nullptr;
static jmethodID g_ctor = nullptr;

static jclass g_etMethodMetaClass = nullptr;
static jmethodID g_metaCtor = nullptr;

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  JNIEnv* env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }
  jclass local = env->FindClass("org/measly/executorch/jni/EtTensor");
  if (local == nullptr) {
    return JNI_ERR; // class not found -> System.load fails clearly
  }
  g_etTensorClass = static_cast<jclass>(env->NewGlobalRef(local));
  env->DeleteLocalRef(local);
  g_fShape = env->GetFieldID(g_etTensorClass, "shape", "[J");
  g_fData = env->GetFieldID(g_etTensorClass, "data", "[F");
  g_ctor = env->GetMethodID(g_etTensorClass, "<init>", "([J[F)V");
  if (g_fShape == nullptr || g_fData == nullptr || g_ctor == nullptr) {
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
  return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jlong JNICALL
Java_org_measly_executorch_jni_EtNative_loadModule(
    JNIEnv* env, jclass, jstring jpath) {
  const char* path = env->GetStringUTFChars(jpath, nullptr);
  auto* module = new Module(path);
  env->ReleaseStringUTFChars(jpath, path);
  return reinterpret_cast<jlong>(module);
}

extern "C" JNIEXPORT jobject JNICALL
Java_org_measly_executorch_jni_EtNative_methodMeta(
    JNIEnv* env, jclass, jlong handle) {
  auto* module = reinterpret_cast<Module*>(handle);
  auto meta = module->method_meta("forward");
  if (!meta.ok()) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                  "method_meta(\"forward\") failed");
    return nullptr;
  }
  const auto n = static_cast<jsize>(meta->num_inputs());
  jintArray types = env->NewIntArray(n);
  if (types == nullptr) {
    return nullptr; // OOM: exception already pending
  }
  std::vector<jint> tmp(n);
  // Non-tensor inputs (scalars/optional args) have no tensor meta -> encoded as -1.
  for (jsize i = 0; i < n; ++i) {
    auto info = meta->input_tensor_meta(i);
    tmp[i] = info.ok() ? static_cast<jint>(info->scalar_type()) : -1;
  }
  env->SetIntArrayRegion(types, 0, n, tmp.data());
  return env->NewObject(g_etMethodMetaClass, g_metaCtor, static_cast<jint>(n), types);
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_org_measly_executorch_jni_EtNative_forward(
    JNIEnv* env, jclass, jlong handle, jobjectArray jinputs) {
  auto* module = reinterpret_cast<Module*>(handle);

  jsize nIn = env->GetArrayLength(jinputs);

  // Backing storage must outlive forward(); from_blob does not copy.
  std::vector<std::vector<float>> inData(nIn);
  std::vector<std::vector<executorch::aten::SizesType>> inShape(nIn);
  std::vector<executorch::extension::TensorPtr> tensors;
  std::vector<executorch::runtime::EValue> evalues;
  tensors.reserve(nIn);
  evalues.reserve(nIn);

  for (jsize i = 0; i < nIn; ++i) {
    jobject jt = env->GetObjectArrayElement(jinputs, i);
    auto jshape = static_cast<jlongArray>(env->GetObjectField(jt, g_fShape));
    auto jdata = static_cast<jfloatArray>(env->GetObjectField(jt, g_fData));

    jsize nd = env->GetArrayLength(jshape);
    inShape[i].resize(nd);
    {
      std::vector<jlong> tmp(nd);
      env->GetLongArrayRegion(jshape, 0, nd, tmp.data());
      for (jsize k = 0; k < nd; ++k) {
        inShape[i][k] = static_cast<executorch::aten::SizesType>(tmp[k]);
      }
    }
    jsize nElem = env->GetArrayLength(jdata);
    inData[i].resize(nElem);
    env->GetFloatArrayRegion(jdata, 0, nElem, inData[i].data());

    tensors.push_back(from_blob(inData[i].data(), inShape[i]));
    evalues.emplace_back(tensors[i]);
    env->DeleteLocalRef(jshape);
    env->DeleteLocalRef(jdata);
    env->DeleteLocalRef(jt);
  }

  auto result = module->forward(evalues);
  if (!result.ok()) {
    env->ThrowNew(env->FindClass("java/lang/RuntimeException"),
                  "ExecuTorch forward() failed");
    return nullptr;
  }

  const auto& outputs = *result;
  jsize nOut = static_cast<jsize>(outputs.size());
  jobjectArray jout = env->NewObjectArray(nOut, g_etTensorClass, nullptr);

  for (jsize i = 0; i < nOut; ++i) {
    auto t = outputs[i].toTensor();
    jsize ndim = static_cast<jsize>(t.dim());
    jlongArray jshape = env->NewLongArray(ndim);
    {
      std::vector<jlong> sh(ndim);
      for (jsize k = 0; k < ndim; ++k) {
        sh[k] = static_cast<jlong>(t.size(k));
      }
      env->SetLongArrayRegion(jshape, 0, ndim, sh.data());
    }
    jsize nElem = static_cast<jsize>(t.numel());
    jfloatArray jdata = env->NewFloatArray(nElem);
    env->SetFloatArrayRegion(jdata, 0, nElem, t.const_data_ptr<float>());

    jobject obj = env->NewObject(g_etTensorClass, g_ctor, jshape, jdata);
    env->SetObjectArrayElement(jout, i, obj);
    env->DeleteLocalRef(jshape);
    env->DeleteLocalRef(jdata);
    env->DeleteLocalRef(obj);
  }
  return jout;
}

extern "C" JNIEXPORT void JNICALL
Java_org_measly_executorch_jni_EtNative_destroy(
    JNIEnv*, jclass, jlong handle) {
  delete reinterpret_cast<Module*>(handle);
}
