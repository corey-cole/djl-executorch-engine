#pragma once
#include <atomic>
#include <cstdint>
#if defined(__linux__) && defined(__GNUC__)
#include <sys/sdt.h>
#else
#define DTRACE_PROBE4(provider, name, a, b, c, d) ((void)0)
#define DTRACE_PROBE5(provider, name, a, b, c, d, e) ((void)0)
#endif

namespace measly::et {
enum EtProbeId : uint32_t { kProbeStagingGrow = 0, kProbeStagingInput = 1 };
using EtProbeFn = void (*)(uint32_t id, uint64_t a, uint64_t b, uint64_t c, uint64_t d);

// One handler slot shared by the dispatch and the setter/clearer: function-local statics in the
// inline functions below would each get their own instance, silently breaking set-then-dispatch.
inline std::atomic<EtProbeFn> g_probe_handler{nullptr};

inline void probe_dispatch(uint32_t id, uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
  EtProbeFn fn = g_probe_handler.load(std::memory_order_relaxed);
  if (fn) fn(id, a, b, c, d);
}
inline void et_probe_set_handler(EtProbeFn fn) {
  g_probe_handler.store(fn, std::memory_order_relaxed);
}
inline void et_probe_clear_handler() { et_probe_set_handler(nullptr); }
}  // namespace measly::et

// SDT macro first (matches the dist's etnp::lstm probes: provider "etnp", Semaphore: 0x0 — plain
// DTRACE_PROBE*, no _ENABLED/semaphore — so one set of bpftrace/perf tooling covers both), then
// the in-process dispatch.
// Both expand to two statements, so they are wrapped in do/while(0): without it an unbraced
// `if (cond) ET_PROBE_...;` would fire the dispatch unconditionally, which is the classic form of
// this bug and would be invisible here (a spurious probe fire, not a compile error).
#define ET_PROBE_STAGING_GROW(slot, old_bytes, new_bytes)                                            \
  do {                                                                                               \
    DTRACE_PROBE4(measly, staging_grow, (slot), (old_bytes), (new_bytes), 0);                        \
    ::measly::et::probe_dispatch(::measly::et::kProbeStagingGrow, (slot), (old_bytes), (new_bytes),  \
                                 0);                                                                 \
  } while (0)

#define ET_PROBE_STAGING_INPUT(slot, nbytes, planned, staged)                                        \
  do {                                                                                               \
    DTRACE_PROBE5(measly, staging_input, (slot), (nbytes), (planned), (staged), 0);                  \
    ::measly::et::probe_dispatch(::measly::et::kProbeStagingInput, (slot), (nbytes), (planned),      \
                                 (staged));                                                          \
  } while (0)
