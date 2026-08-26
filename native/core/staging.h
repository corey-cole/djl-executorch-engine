#ifndef MEASLY_ET_STAGING_H
#define MEASLY_ET_STAGING_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

#if defined(_WIN32)
#include <malloc.h>
#endif

#include <xnnpack.h>

namespace measly::et {

// XNN_EXTRA_BYTES (16 x86/ARM, 128 Hexagon) is XNNPACK's documented max out-of-bounds read past a
// tensor's declared end; doubled here per ExecuTorch's own vendored runtime.c, which sizes its
// XNNPACK arena with "mem_arena_size += 2 * XNN_EXTRA_BYTES" and comments "sparse microkernels can
// read up to 2 * XNN_EXTRA_BYTES beyond array bounds". This shim never defines XNN_ARCH_HEXAGON, so
// XNN_EXTRA_BYTES always resolves to 16 here -- the static_assert below proves that rather than
// assuming it, and re-fires if a future pin ever grows the constant past what 128 covers.
inline constexpr size_t kStagingPadding = 128;
static_assert(kStagingPadding >= 2 * XNN_EXTRA_BYTES,
              "kStagingPadding no longer covers XNNPACK's documented over-read "
              "(XNN_EXTRA_BYTES, doubled for sparse microkernels per ExecuTorch's runtime.c) -- "
              "bump kStagingPadding before shipping this pin");

namespace detail {
inline void* stagingAlloc(size_t size) {
#if defined(_WIN32)
  return _aligned_malloc(size, 64);
#else
  // size is always a multiple of 64 by construction (see StagingSlot::ensure), satisfying C11's
  // aligned_alloc size/alignment multiple requirement.
  return std::aligned_alloc(64, size);
#endif
}

inline void stagingFree(void* p) {
#if defined(_WIN32)
  _aligned_free(p);
#else
  std::free(p);
#endif
}
}  // namespace detail

// Grow-only, 64-byte-aligned slot buffer. Ensure() never shrinks; steady state is zero allocations.
//
// The slot knows nothing about kStagingPadding: ensure() sizes exactly what it is asked for, and it
// is every caller's job to ask for tensor bytes PLUS the padding so XNNPACK's over-read lands in
// slack. Both call sites in et_runtime.cpp do (load-time sizing and the forward() regrow guard);
// a new caller that forgets is an out-of-bounds read that no assertion here will catch.
//
// The 64-byte alignment is not about XNNPACK -- it is what makes the rounding legal: POSIX
// aligned_alloc requires the size be a multiple of the alignment, which is why ensure() rounds up
// to a multiple of 64 rather than allocating `needed` directly.
class StagingSlot {
 public:
  StagingSlot() = default;
  ~StagingSlot() { detail::stagingFree(data_); }
  StagingSlot(const StagingSlot&) = delete;
  StagingSlot& operator=(const StagingSlot&) = delete;
  StagingSlot(StagingSlot&& other) noexcept
      : data_(other.data_), capacity_(other.capacity_) {
    other.data_ = nullptr;
    other.capacity_ = 0;
  }
  StagingSlot& operator=(StagingSlot&& other) noexcept {
    if (this != &other) {
      detail::stagingFree(data_);
      data_ = other.data_;
      capacity_ = other.capacity_;
      other.data_ = nullptr;
      other.capacity_ = 0;
    }
    return *this;
  }
  // Returns the buffer, allocating if capacity_ < needed. New capacity = ceil(needed / 64) * 64.
  //
  // Contents are NOT preserved across a growing call: the only caller overwrites the whole slot
  // with the incoming input immediately afterwards, so copying the old bytes forward was work whose
  // result was always discarded. Callers that need the old contents must save them first.
  // Throws std::bad_alloc if the allocation fails.
  void* ensure(size_t needed) {
    if (needed <= capacity_) {
      return data_;
    }
    const size_t rounded = ((needed + 63) / 64) * 64;
    uint8_t* fresh = static_cast<uint8_t*>(detail::stagingAlloc(rounded));
    if (fresh == nullptr) {
      // Slots are sized at load from the model's declared bound, so this is reachable with a real
      // allocation request. Throw rather than memcpy through null: EtRuntime's constructor turns it
      // into the same load failure as any other, and forward() into a clean exception.
      throw std::bad_alloc();
    }
    detail::stagingFree(data_);  // no-op on nullptr; old contents are intentionally dropped
    data_ = fresh;
    capacity_ = rounded;
    return data_;
  }
  void* data() const { return data_; }
  size_t capacity() const { return capacity_; }

 private:
  uint8_t* data_ = nullptr;
  size_t capacity_ = 0;
};

}  // namespace measly::et

#endif  // MEASLY_ET_STAGING_H
