#pragma once

#include <cstddef>
#include <cstdint>

// The jsize boundary, in one place. An ExecuTorch output larger than INT32_MAX bytes cannot be
// represented as a Java array at all, so the JNI shim rejects it explicitly instead of letting the
// cast to jsize truncate and produce a silently short byte[]. Split out of the shim so the Catch2
// units can pin the boundary without a JNIEnv -- this header is free of <jni.h> for that reason,
// which is also why the limit is spelled INT32_MAX rather than the jsize type itself.
namespace measly::et {

// A JNI jbyteArray length is a signed 32-bit jsize (jint). Outputs larger than INT32_MAX
// cannot be marshalled through byte[]; forward() must reject them before the truncating cast.
inline constexpr size_t kJniByteArrayMaxBytes = static_cast<size_t>(INT32_MAX);

inline bool exceedsJniByteArrayLimit(size_t nbytes) {
  return nbytes > kJniByteArrayMaxBytes;
}

}  // namespace measly::et
