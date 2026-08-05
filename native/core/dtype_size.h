#ifndef MEASLY_ET_DTYPE_SIZE_H
#define MEASLY_ET_DTYPE_SIZE_H

#include <cstddef>
#include <cstdint>

namespace measly::et {

// Byte width of an ExecuTorch ScalarType code, for the subset the JNI-free harnesses build
// 1-filled host buffers for. Shared by et_leak_harness.cpp and et_timing_harness.cpp.
inline size_t dtypeSize(int8_t st) {
  switch (st) {
    case 6:            // FLOAT32
    case 3: return 4;  // INT32
    case 7:            // FLOAT64
    case 4: return 8;  // INT64
    case 0:            // UINT8
    case 1:            // INT8
    case 11: return 1;  // BOOL
    default: return 4;
  }
}

}  // namespace measly::et

#endif  // MEASLY_ET_DTYPE_SIZE_H
