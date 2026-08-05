// W4 guard-page over-read harness: proves (or disproves) XNNPACK's documented out-of-bounds read
// against a borrowed, exact-sized host buffer (config a) and against ExecuTorch's own memory-
// planned arena (config b). See docs/executorch-host-buffer-contract-brief.md §3/W4.
//
// CLI: et_overread_harness <borrowed|arena> <pte>
//   borrowed — input 0 is placed in a GuardedRegion sized exactly nbytes, with the page after it
//              PROT_NONE'd, and handed to ExecuTorch via from_blob -> set_input. For an artifact
//              exported with alloc_graph_input=False, set_input takes share_tensor_data: XNNPACK
//              reads the caller's raw buffer, and an annotated kernel's tail load past the end
//              faults (exit 139 / SIGSEGV) regardless of uarch micro-architecture.
//   arena    — config (b): the memory-planned arena is replaced with a HierarchicalAllocator over
//              GuardedRegions sized exactly memory_planned_buffer_size(i). Prints the per-buffer
//              sizes and whether input 0 is placed last in its buffer (the precondition for a
//              stock-ExecuTorch arena-end over-read), then runs one forward from host buffers.
//
// Deliberately NOT wired into any default test target (brief §3/W4): reaching an annotated kernel
// needs CPUID masking under qemu-x86_64 or AMD Zen 1-3 hardware, neither of which belongs in CI.
// It is also why borrowed mode bypasses EtRuntime::forward — W7 staging pads the slot, which
// absorbs the over-read; the harness must exercise the raw share_tensor_data path the brief names.
//
// Outcome contract: exit 0 = clean run; SIGSEGV ($? == 139) = the over-read faulted on a guard page.
//
// Run recipe (evidence goes in brief §8): see §3/W4 "Run recipe" block.
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include <cpuinfo.h>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>

using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::extension::from_blob;
using executorch::runtime::EValue;
using executorch::runtime::Error;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::Span;

namespace {

[[noreturn]] void die(const char* what, const char* detail) {
  std::fprintf(stderr, "et_overread_harness: %s: %s\n", what, detail);
  std::exit(2);
}

// mmap region of size + 2 pages; data pointer placed so data + size lands exactly on a page
// boundary; everything from that boundary to the end is mprotect(PROT_NONE). A read of even one
// byte past data + size faults on the hardware guard page — the only detector ASan cannot suppress
// (XNN_OOB_READS expands to no_sanitize("address"); brief §3/W4).
class GuardedRegion {
 public:
  explicit GuardedRegion(size_t size) : size_(size) {
    page_ = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    map_len_ = size_ + 2 * page_;
    void* base = mmap(nullptr, map_len_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) {
      die("mmap failed", std::strerror(errno));
    }
    base_ = static_cast<uint8_t*>(base);
    data_ = (size_ % page_ != 0) ? base_ + page_ - (size_ % page_) : base_ + page_;
    uint8_t* guard_start = data_ + size_;  // page-aligned by construction
    const size_t guard_len = base_ + map_len_ - guard_start;
    if (mprotect(guard_start, guard_len, PROT_NONE) != 0) {
      die("mprotect failed", std::strerror(errno));
    }
  }
  ~GuardedRegion() { munmap(base_, map_len_); }
  GuardedRegion(const GuardedRegion&) = delete;
  GuardedRegion& operator=(const GuardedRegion&) = delete;
  GuardedRegion(GuardedRegion&& other) noexcept
      : size_(other.size_),
        page_(other.page_),
        map_len_(other.map_len_),
        base_(other.base_),
        data_(other.data_) {
    other.base_ = nullptr;
    other.map_len_ = 0;
  }
  GuardedRegion& operator=(GuardedRegion&& other) noexcept {
    if (this != &other) {
      munmap(base_, map_len_);
      size_ = other.size_;
      page_ = other.page_;
      map_len_ = other.map_len_;
      base_ = other.base_;
      data_ = other.data_;
      other.base_ = nullptr;
      other.map_len_ = 0;
    }
    return *this;
  }
  void* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  size_t size_ = 0;
  size_t page_ = 0;
  size_t map_len_ = 0;
  uint8_t* base_ = nullptr;
  uint8_t* data_ = nullptr;
};

void printCpuInfo() {
  if (!cpuinfo_initialize()) {
    std::printf("cpuinfo: initialize failed\n");
    return;
  }
  const uint32_t idx = cpuinfo_get_current_uarch_index();
  const struct cpuinfo_uarch_info* ui = cpuinfo_get_uarch(idx);
  std::printf("uarch_id=0x%08x cores=%u\n", ui ? ui->uarch : 0, ui ? ui->core_count : 0);
#if defined(CPUINFO_ARCH_X86_64) && CPUINFO_ARCH_X86_64
  // sse2 is implicit on x86_64 (not exposed as a member); the flags below are what distinguish
  // the kernels XNNPACK may select.
  std::printf("isa sse3=%d sse4_1=%d avx=%d fma3=%d avx2=%d avx512f=%d\n", cpuinfo_isa.sse3,
              cpuinfo_isa.sse4_1, cpuinfo_isa.avx, cpuinfo_isa.fma3, cpuinfo_isa.avx2,
              cpuinfo_isa.avx512f);
#endif
}

// Host-side input buffers, one per declared input; 1-filled (f32 gets 1.0f, else memset 1).
std::vector<std::vector<uint8_t>> makeHostBuffers(const executorch::runtime::MethodMeta& meta) {
  std::vector<std::vector<uint8_t>> buffers(static_cast<size_t>(meta.num_inputs()));
  for (int i = 0; i < meta.num_inputs(); ++i) {
    auto info = meta.input_tensor_meta(i);
    if (!info.ok()) {
      continue;  // non-tensor input: no host buffer
    }
    const size_t bytes = info->nbytes();
    buffers[static_cast<size_t>(i)].assign(bytes, 0);
    if (info->scalar_type() == executorch::aten::ScalarType::Float) {
      const float one = 1.0f;
      for (size_t b = 0; b + sizeof(float) <= bytes; b += sizeof(float)) {
        std::memcpy(buffers[static_cast<size_t>(i)].data() + b, &one, sizeof(float));
      }
    } else {
      std::memset(buffers[static_cast<size_t>(i)].data(), 1, bytes);
    }
  }
  return buffers;
}

// Builds EValues over the given per-input data pointers (from_blob borrows; data must stay valid
// through the forward). The EValue's Tensor is a raw TensorImpl* into the Storage owned by the
// aliasing shared_ptr, so keepalive MUST outlive the forward() that consumes evalues — the same
// lifetime rule EtRuntime::forward obeys (its `tensors` vector lives through module.forward()).
struct HostTensors {
  std::vector<TensorPtr> keepalive;
  std::vector<EValue> evalues;
};

HostTensors makeHostTensors(const executorch::runtime::MethodMeta& meta,
                            const std::vector<const void*>& data) {
  HostTensors out;
  for (int i = 0; i < meta.num_inputs(); ++i) {
    auto info = meta.input_tensor_meta(i);
    if (!info.ok()) {
      continue;  // non-tensor input: forward() only consumes tensor inputs
    }
    auto sizes = info->sizes();
    std::vector<executorch::aten::SizesType> shape(sizes.begin(), sizes.end());
    TensorPtr t = from_blob(const_cast<void*>(data[static_cast<size_t>(i)]), shape,
                            info->scalar_type());
    out.evalues.emplace_back(t);  // moves the Tensor out of *t (template deref ctor)
    out.keepalive.push_back(std::move(t));  // keep the Storage/TensorImpl alive
  }
  return out;
}

int runBorrowed(const char* pte) {
  Module mm(pte);
  if (mm.load() != Error::Ok) {
    die("load failed", pte);
  }
  auto meta_res = mm.method_meta("forward");
  if (!meta_res.ok()) {
    die("method_meta failed", pte);
  }
  const executorch::runtime::MethodMeta& meta = meta_res.get();
  auto info_res = meta.input_tensor_meta(0);
  if (!info_res.ok()) {
    die("input 0 is not a tensor", pte);
  }
  const auto& info = info_res.get();
  std::printf("model=%s input0_nbytes=%zu planned=%d\n", pte, info.nbytes(),
              info.is_memory_planned() ? 1 : 0);
  std::fflush(stdout);

  // Input 0 abuts a guard page; any kernel tail load past its exact size faults.
  GuardedRegion region(info.nbytes());
  std::vector<std::vector<uint8_t>> buffers = makeHostBuffers(meta);
  std::vector<const void*> data(static_cast<size_t>(meta.num_inputs()), nullptr);
  data[0] = region.data();
  for (int i = 1; i < meta.num_inputs(); ++i) {
    if (!buffers[static_cast<size_t>(i)].empty()) {
      data[static_cast<size_t>(i)] = buffers[static_cast<size_t>(i)].data();
    }
  }
  // Borrowed path, deliberately raw: from_blob -> Module::execute -> set_input ->
  // share_tensor_data (the artifact is alloc_graph_input=False). EtRuntime::forward would stage
  // into a padded slot and absorb the over-read, which is W7, not W4.
  HostTensors host = makeHostTensors(meta, data);
  auto result = mm.forward(host.evalues);
  if (!result.ok()) {
    std::fprintf(stderr, "forward failed: 0x%x\n", static_cast<unsigned>(result.error()));
    return 3;
  }
  std::printf("clean: forward returned Ok, no fault on input0 guard page\n");
  return 0;
}

int runArena(const char* pte) {
  Module mm(pte);
  if (mm.load() != Error::Ok) {
    die("load failed", pte);
  }
  auto meta_res = mm.method_meta("forward");
  if (!meta_res.ok()) {
    die("method_meta failed", pte);
  }
  const executorch::runtime::MethodMeta& meta = meta_res.get();
  const size_t nbuf = meta.num_memory_planned_buffers();

  // Arena spans sized exactly (each abuts its own guard page), mirroring Module's default
  // std::vector<uint8_t>(size) arena minus malloc slack — the instrumented arena.
  std::vector<GuardedRegion> regions;
  regions.reserve(nbuf);
  std::vector<Span<uint8_t>> spans;
  spans.reserve(nbuf);
  std::printf("model=%s planned_buffers=%zu", pte, nbuf);
  for (size_t i = 0; i < nbuf; ++i) {
    auto sz = meta.memory_planned_buffer_size(i);
    if (!sz.ok()) {
      die("memory_planned_buffer_size failed", pte);
    }
    const size_t size = static_cast<size_t>(sz.get());
    std::printf(" buffer[%zu]=%zu", i, size);
    regions.emplace_back(size);
    spans.emplace_back(static_cast<uint8_t*>(regions.back().data()), size);
  }
  std::printf("\n");
  std::fflush(stdout);

  HierarchicalAllocator ha(Span<Span<uint8_t>>(spans.data(), spans.size()));
  if (mm.load_method("forward", &ha) != Error::Ok) {
    die("load_method failed", pte);
  }

  // Placement report: does the planner put an XNNPACK external input last in its buffer? That is
  // the UNVERIFIED precondition for a stock-ExecuTorch arena-end over-read (brief §3/W4).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"  // Module::method / Method::get_inputs
  auto method_res = mm.method("forward");
  if (!method_res.ok()) {
    die("method() failed", pte);
  }
  std::vector<EValue> evals(static_cast<size_t>(meta.num_inputs()));
  if (method_res.get()->get_inputs(evals.data(), evals.size()) != Error::Ok) {
    die("get_inputs failed", pte);
  }
#pragma GCC diagnostic pop

  bool input0_last = false;
  for (size_t i = 0; i < evals.size(); ++i) {
    if (!evals[i].isTensor()) {
      continue;
    }
    const auto t = evals[i].toTensor();
    const uint8_t* ptr = static_cast<const uint8_t*>(t.const_data_ptr());
    const size_t nbytes = t.nbytes();
    for (size_t b = 0; b < nbuf; ++b) {
      const uint8_t* begin = static_cast<const uint8_t*>(regions[b].data());
      const uint8_t* end = begin + regions[b].size();
      if (ptr >= begin && ptr < end) {
        const bool last = (ptr + nbytes == end);
        if (i == 0) {
          input0_last = last;
        }
        std::printf("input %zu in buffer %zu%s\n", i, b, last ? " LAST" : "");
      }
    }
  }
  std::printf("input placed last in its buffer: %s\n", input0_last ? "yes" : "no");
  std::fflush(stdout);

  // One forward from host buffers; set_input copies into the guarded arena, then the delegate
  // executes. Fault (if the planner placed input 0 last) or clean exit.
  std::vector<std::vector<uint8_t>> buffers = makeHostBuffers(meta);
  std::vector<const void*> data(static_cast<size_t>(meta.num_inputs()), nullptr);
  for (int i = 0; i < meta.num_inputs(); ++i) {
    if (!buffers[static_cast<size_t>(i)].empty()) {
      data[static_cast<size_t>(i)] = buffers[static_cast<size_t>(i)].data();
    }
  }
  HostTensors host = makeHostTensors(meta, data);
  auto result = mm.forward(host.evalues);
  if (!result.ok()) {
    std::fprintf(stderr, "forward failed: 0x%x\n", static_cast<unsigned>(result.error()));
    return 3;
  }
  std::printf("clean: forward returned Ok, no fault on arena guard pages\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);  // uarch/placement lines must survive a SIGSEGV
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <borrowed|arena> <pte>\n", argv[0]);
    return 2;
  }
  const std::string mode = argv[1];
  const char* pte = argv[2];
  printCpuInfo();
  if (mode == "borrowed") {
    return runBorrowed(pte);
  }
  if (mode == "arena") {
    return runArena(pte);
  }
  std::fprintf(stderr, "usage: %s <borrowed|arena> <pte>\n", argv[0]);
  return 2;
}
