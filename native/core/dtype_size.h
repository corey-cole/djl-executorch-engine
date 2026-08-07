#ifndef MEASLY_ET_DTYPE_SIZE_H
#define MEASLY_ET_DTYPE_SIZE_H

#include <cstddef>
#include <cstdint>

namespace measly::et {

// Byte width of an ExecuTorch ScalarType code, over the dtypes this engine supports — the same set
// EtDataTypes maps on the Java side. Shared by et_runtime.cpp (where it sizes the staging copy),
// et_leak_harness.cpp, and et_timing_harness.cpp.
//
// Returns 0 for anything else, which callers must treat as "unsupported", never as a size. It used
// to return 4 as a catch-all; that was harmless while this only sized harness scratch buffers, but
// it became a memcpy length when staging landed, where guessing 4 for a 2-byte FLOAT16 would read
// twice the source. EtRuntime's constructor rejects a model declaring any code that returns 0, so
// the harnesses -- which only ever pass metadata from a loaded model -- cannot observe it.
inline size_t dtypeSize(int8_t st) {
  switch (st) {
    case 6:            // FLOAT32
    case 3: return 4;  // INT32
    case 7:            // FLOAT64
    case 4: return 8;  // INT64
    case 0:            // UINT8
    case 1:            // INT8
    case 11: return 1;  // BOOL
    default: return 0;  // unsupported: INT16 (2), FLOAT16 (5), complex, ...
  }
}

}  // namespace measly::et

#endif  // MEASLY_ET_DTYPE_SIZE_H
