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
