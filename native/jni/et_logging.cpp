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
