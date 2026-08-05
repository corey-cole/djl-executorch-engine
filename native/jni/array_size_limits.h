#pragma once

#include <cstddef>
#include <cstdint>

namespace measly::et {

// A JNI jbyteArray length is a signed 32-bit jsize (jint). Outputs larger than INT32_MAX
// cannot be marshalled through byte[]; forward() must reject them before the truncating cast.
inline constexpr size_t kJniByteArrayMaxBytes = static_cast<size_t>(INT32_MAX);

inline bool exceedsJniByteArrayLimit(size_t nbytes) {
  return nbytes > kJniByteArrayMaxBytes;
}

}  // namespace measly::et
