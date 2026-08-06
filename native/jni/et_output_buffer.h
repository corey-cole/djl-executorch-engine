#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include "et_probes.h"

namespace measly::et::jni {
// JNI-allocated output buffers are NOT counted against -XX:MaxDirectMemorySize (the W6 OOM
// mechanism), so the live count -- not direct-memory pressure -- is the leak signal. Incremented
// after a successful allocation, decremented on free; free(nullptr) is a no-op (idempotent by
// contract, so a mis-registration can never double-free).
inline std::atomic<int64_t> g_alive_output_buffers{0};

inline void* allocOutputBuffer(size_t nbytes) {
  void* p = ::operator new(nbytes);  // throws std::bad_alloc on OOM; counter not incremented
  ++g_alive_output_buffers;
  ET_PROBE_OUTPUT_ALLOC(nbytes, reinterpret_cast<uint64_t>(p));
  return p;
}

inline void freeOutputBuffer(void* p) {
  if (p == nullptr) return;
  ::operator delete(p);
  --g_alive_output_buffers;
  ET_PROBE_OUTPUT_FREE(reinterpret_cast<uint64_t>(p));
}

inline int64_t aliveOutputBuffers() {
  return g_alive_output_buffers.load(std::memory_order_relaxed);
}
}  // namespace measly::et::jni
