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
