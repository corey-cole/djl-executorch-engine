#include "et_runtime.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/threadpool/threadpool.h>
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/executor/method_meta.h>

#include "dtype_size.h"
#include "et_probes.h"
#include "staging.h"

namespace measly::et {

using executorch::extension::Module;
using executorch::extension::from_blob;
using executorch::extension::TensorPtr;
using executorch::runtime::BackendOptions;
using executorch::runtime::EValue;
using executorch::runtime::LoadBackendOptionsMap;

struct RuntimeState {
  Module module;
  MethodMeta meta;
  std::vector<std::unique_ptr<StagingSlot>> staging;
  explicit RuntimeState(const std::string& path) : module(path) {}
};

struct ForwardState {
  std::vector<EValue> outputs;    // owns the result EValues
  std::vector<OutputView> views;  // descriptors into the host arena
};

namespace {

// Set once any EtRuntime has been constructed (even by a throwing ctor); read by the intra-op
// reset guard. Guards the XNNPACK pthreadpool_t capture, so it must outlive all runtimes.
std::atomic<bool> g_etRuntimeConstructed{false};

// Builds the MethodMeta snapshot for the "forward" method. Same throws / -1 / 0 conventions as the
// old EtRuntime::methodMeta() body: a non-tensor input keeps -1 / empty shape / 0.
MethodMeta buildMethodMeta(Module& module) {
  auto meta = module.method_meta("forward");
  if (!meta.ok()) {
    throw std::runtime_error("EtRuntime: method_meta(\"forward\") failed");
  }
  const int n = static_cast<int>(meta->num_inputs());
  MethodMeta out;
  out.numInputs = n;
  out.inputScalarTypes.resize(n, -1);  // non-tensor inputs keep -1
  out.inputShapes.resize(n);           // non-tensor inputs keep empty
  out.inputMemoryPlanned.resize(n, 0); // non-tensor inputs keep 0 (no TensorInfo exists)
  out.inputNbytes.resize(n, 0);        // non-tensor inputs keep 0
  for (int i = 0; i < n; ++i) {
    auto info = meta->input_tensor_meta(i);
    if (info.ok()) {
      out.inputScalarTypes[i] = static_cast<int8_t>(info->scalar_type());
      auto sizes = info->sizes();  // Span<const int32_t>
      out.inputShapes[i].assign(sizes.begin(), sizes.end());
      out.inputMemoryPlanned[i] = info->is_memory_planned() ? 1 : 0;
      out.inputNbytes[i] = info->nbytes();
    }
  }
  return out;
}

}  // namespace

EtRuntime::EtRuntime(const std::string& ptePath, int workspaceSharingMode)
    : state_(std::make_unique<RuntimeState>(ptePath)) {
  // Set even when this ctor later throws: the pool is captured by XNNPACK at runtime creation,
  // so "has ever been constructed" is the safe boundary for the intra-op reset guard.
  g_etRuntimeConstructed.store(true);
  // Force-load now so a bad path/file throws at construction (the "load throws" contract).
  //
  // The options map and its BackendOptions storage are stack-local, which the non-owning-span
  // caveat on LoadBackendOptionsMap would normally forbid. It is correct here because
  // Module::load deep-copies into Module-owned storage before returning, and the load_forward()
  // call below in this constructor consumes that copy -- see the doc comment on
  // Module::load(const LoadBackendOptionsMap&, Verification).
  executorch::runtime::Error loadErr;
  if (workspaceSharingMode >= 0) {
    BackendOptions<1> xnnOpts;
    // Key from backends/xnnpack/runtime/XNNPACKBackend.h (workspace_sharing_mode_option_key).
    if (xnnOpts.set_option("workspace_sharing_mode", workspaceSharingMode) !=
        executorch::runtime::Error::Ok) {
      throw std::runtime_error(
          "EtRuntime: failed to set workspace_sharing_mode option (mode=" +
          std::to_string(workspaceSharingMode) + ")");
    }
    LoadBackendOptionsMap optionsMap;
    // Backend id from the same header (xnnpack_backend_key). Spelled EXACTLY "XnnpackBackend": the
    // id match happens during delegate init, which Method::load drives (surfacing through
    // BackendInitContext::get_runtime_spec inside XnnpackBackendOptions::resolve_sharing_mode), and
    // a mismatch there is a SILENT no-op, not an error.
    if (optionsMap.set_options("XnnpackBackend", xnnOpts.view()) !=
        executorch::runtime::Error::Ok) {
      throw std::runtime_error(
          "EtRuntime: failed to register XnnpackBackend options in LoadBackendOptionsMap");
    }
    loadErr = state_->module.load(optionsMap);
  } else {
    loadErr = state_->module.load();
  }
  if (loadErr != executorch::runtime::Error::Ok) {
    throw std::runtime_error("EtRuntime: failed to load .pte: " + ptePath);
  }
  // Force the "forward" Method to load too. Module::load() and Module::method_meta() are both
  // PROGRAM-level: method_meta() calls load() and then program_->method_meta(), never load_method.
  // Delegate init -- the only place XnnpackBackendOptions::resolve_sharing_mode runs -- happens in
  // load_method, which is otherwise triggered lazily by the first forward(). Without this call the
  // runtime spec above would sit unused until first inference, so an invalid mode would surface at
  // predict() rather than at load, breaking this codebase's "load throws" contract.
  //
  // Side effect, intended: the XNNPACK subgraph compile now happens at construction instead of on
  // the first forward(). In the timing harness that shifts cost from cold_ms into load_ms; warmup
  // is discarded there, so steady-state numbers are unaffected.
  if (state_->module.load_forward() != executorch::runtime::Error::Ok) {
    throw std::runtime_error("EtRuntime: failed to load \"forward\" from .pte: " + ptePath);
  }
  state_->meta = buildMethodMeta(state_->module);

  // This engine builds a tensor for every input (forward() has only from_blob), so a method with a
  // non-tensor input -- a prim int/double/bool, or None -- cannot be driven correctly through it:
  // InputDesc has no way to express the traced prim value that Method::set_input demands. Reject at
  // load rather than at first inference, where it surfaced as a garbage from_blob over ScalarType
  // -1 followed by an opaque tag-mismatch from ExecuTorch.
  //
  // Rejecting here is also what makes inputMemoryPlanned unambiguous downstream. The flag is 0 both
  // for a genuinely borrowed input and for a non-tensor one (no TensorInfo exists, so nothing sets
  // it), and forward() branches on it to decide whether to stage. With this check the second case
  // cannot occur, so "planned == 0" means "borrowed tensor" everywhere below.
  for (int i = 0; i < state_->meta.numInputs; ++i) {
    if (state_->meta.inputScalarTypes[i] < 0) {
      throw std::invalid_argument(
          "EtRuntime: input " + std::to_string(i) + " of \"forward\" is not a tensor; this engine "
          "supports only methods whose inputs are all tensors: " + ptePath);
    }
    // Same reasoning one step further: a dtype we cannot size is a dtype we cannot stage, and
    // dtypeSize() is what turns a shape into a memcpy length. Rejecting here is what lets every
    // later use of dtypeSize() assume a nonzero result. This matches EtDataTypes on the Java side,
    // which already refuses these codes in both directions.
    if (dtypeSize(state_->meta.inputScalarTypes[i]) == 0) {
      throw std::invalid_argument(
          "EtRuntime: input " + std::to_string(i) + " of \"forward\" has unsupported ScalarType " +
          std::to_string(static_cast<int>(state_->meta.inputScalarTypes[i])) + ": " + ptePath);
    }
  }

  // One slot per input position; resize() would default-construct null unique_ptrs, so create each
  // slot explicitly. Slots for staged inputs are sized *here*, from the bound the .pte declares —
  // TensorInfo::nbytes() is available at load for planned and unplanned inputs alike, so "grow-only"
  // degenerates to "allocate once". Planned slots stay at capacity 0 (never staged, never
  // allocated). Every input is a tensor by the check above, so every unplanned slot has a bound.
  state_->staging.reserve(static_cast<size_t>(state_->meta.numInputs));
  for (int i = 0; i < state_->meta.numInputs; ++i) {
    state_->staging.push_back(std::make_unique<StagingSlot>());
    if (state_->meta.inputMemoryPlanned[i] == 0) {
      state_->staging.back()->ensure(state_->meta.inputNbytes[i] + kStagingPadding);
    }
  }
}

EtRuntime::~EtRuntime() = default;

MethodMeta EtRuntime::methodMeta() const { return state_->meta; }

ForwardResult EtRuntime::forward(std::span<const InputDesc> inputs) {
  // from_blob does not copy: for memory-planned inputs each InputDesc.data must stay valid through
  // module.forward(); for unplanned inputs the data is memcpy'd into the engine-owned staging slot
  // first, so the borrowed pointer lives as long as the RuntimeState, not the caller's buffer.
  std::vector<std::vector<executorch::aten::SizesType>> shapes(inputs.size());
  std::vector<TensorPtr> tensors;
  std::vector<EValue> evalues;
  tensors.reserve(inputs.size());
  evalues.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& in = inputs[i];
    shapes[i].assign(in.shape.begin(), in.shape.end());

    // The dtype has to match the model's before it can be trusted to size anything: `actual` below
    // is dtypeSize(caller's code) x shape product, so a caller claiming FLOAT32 over a FLOAT16
    // buffer would compute twice the bytes that are really there. ExecuTorch checks this too
    // (set_input, method.cpp:1203) but only inside module.forward(), after the staging copy.
    // EtSymbolBlock performs the same check in Java; this is the core owning it for every consumer.
    if (i < state_->meta.inputScalarTypes.size() &&
        in.scalarType != state_->meta.inputScalarTypes[i]) {
      throw std::invalid_argument(
          "EtRuntime: input " + std::to_string(i) + " has ScalarType " +
          std::to_string(static_cast<int>(in.scalarType)) + " but the model declares " +
          std::to_string(static_cast<int>(state_->meta.inputScalarTypes[i])));
    }

    // Byte count of this input; product of an empty shape is 1. Matches dtypeSize's conventions
    // (the subset the harnesses build buffers for), so planned/unplanned classification is exact.
    size_t actual = dtypeSize(in.scalarType);
    for (int64_t d : in.shape) {
      actual *= static_cast<size_t>(d);
    }

    // The declared bound is the only thing that makes `actual` safe to act on: it is derived from
    // the caller's shape, and nothing upstream cross-checks that shape against the buffer behind
    // in.data. ExecuTorch does validate (resize_tensor, method.cpp:1240) — but only inside
    // module.forward(), which is after the staging memcpy below, so a shape larger than the source
    // buffer would over-read it before ExecuTorch ever saw the input. Checked for every tensor
    // input, not just staged ones, so the diagnostic is the same on both paths. nbytes() is exact
    // for a static shape and an upper bound for a dynamic one, so `>` is the right comparison in
    // both cases. Every declared input has a bound: the constructor rejects non-tensor inputs.
    if (i < state_->meta.inputNbytes.size() && actual > state_->meta.inputNbytes[i]) {
      throw std::invalid_argument(
          "EtRuntime: input " + std::to_string(i) + " is " + std::to_string(actual) +
          " bytes but the model declares at most " + std::to_string(state_->meta.inputNbytes[i]));
    }

    const void* blob = in.data;
    // A caller passing more inputs than the model declares gets the same module.forward() rejection
    // as before; guard the meta/staging index so that path stays a clean error, not an OOB.
    if (i < state_->meta.inputMemoryPlanned.size() && state_->meta.inputMemoryPlanned[i] == 0) {
      // Unplanned (borrowed) input: stage into the grow-only, 64-byte-aligned, kStagingPadding-padded
      // slot. XNNPACK's documented over-read lands in our slack on every microarchitecture, and the
      // pointer ExecuTorch retains is engine-owned for the lifetime of the Method (§4 lifetime hazard
      // closed by construction). The memcpy is the accepted safety cost of the borrow path.
      auto& slot = *state_->staging[i];
      const size_t needed = actual + kStagingPadding;
      void* dst = slot.data();
      if (needed > slot.capacity()) {
        // Unreachable as the code stands, and deliberately kept. The slot was sized at load from
        // this input's declared bound and the check above rejects anything past that bound, so
        // `needed` cannot exceed `capacity()`. What remains is a regression guard on those two
        // invariants: if slot sizing or the bound check ever stops agreeing, staging_grow fires and
        // et_leak_harness's `grow == 0` assertion fails instead of the mismatch going unnoticed.
        const size_t old = slot.capacity();
        dst = slot.ensure(needed);
        ET_PROBE_STAGING_GROW(i, old, slot.capacity());
      }
      std::memcpy(dst, in.data, actual);
      ET_PROBE_STAGING_INPUT(i, actual, 0, 1);
      blob = dst;
    } else {
      ET_PROBE_STAGING_INPUT(i, actual, 1, 0);
    }

    tensors.push_back(from_blob(
        const_cast<void*>(blob), shapes[i],
        static_cast<executorch::aten::ScalarType>(in.scalarType)));
    evalues.emplace_back(tensors[i]);
  }

  auto result = state_->module.forward(evalues);
  if (!result.ok()) {
    throw std::runtime_error("EtRuntime: forward() failed");
  }

  auto fs = std::make_unique<ForwardState>();
  fs->outputs = std::move(*result);
  fs->views.reserve(fs->outputs.size());
  for (auto& ev : fs->outputs) {
    auto t = ev.toTensor();
    OutputView v;
    v.scalarType = static_cast<int8_t>(t.scalar_type());
    v.data = t.const_data_ptr();
    v.nbytes = t.nbytes();
    const auto ndim = t.dim();
    v.shape.resize(ndim);
    for (auto k = 0; k < ndim; ++k) {
      v.shape[k] = static_cast<int64_t>(t.size(k));
    }
    fs->views.push_back(std::move(v));
  }
  return ForwardResult(std::move(fs));
}

ForwardResult::ForwardResult(std::unique_ptr<ForwardState> state)
    : state_(std::move(state)) {}
ForwardResult::~ForwardResult() = default;
ForwardResult::ForwardResult(ForwardResult&&) noexcept = default;
ForwardResult& ForwardResult::operator=(ForwardResult&&) noexcept = default;

std::span<const OutputView> ForwardResult::outputs() const {
  return {state_->views.data(), state_->views.size()};
}

uint32_t setIntraOpThreads(uint32_t n) {
  if (n < 1) {
    // uint32_t can only observe 0 here (a negative jint is absorbed by the shim's guard);
    // keep the no-op-and-report contract for native callers too (issue #24).
    return intraOpThreads();
  }
  if (g_etRuntimeConstructed.load()) {
    // XNNPACK captures the pthreadpool_t at runtime creation (xnn_create_runtime_v2) and a
    // reset destroys the old pool object, so a reset after any EtRuntime exists is a
    // use-after-free on the next forward(), not merely a race (issue #26). Refuse it.
    const uint32_t cur = intraOpThreads();
    std::fprintf(stderr,
        "measly::et: setIntraOpThreads(%u) ignored: an EtRuntime already exists; the shared "
        "pool is fixed at %u threads\n", n, cur);
    return cur;
  }
  executorch::extension::threadpool::ThreadPool* pool =
      executorch::extension::threadpool::get_threadpool();
  pool->_unsafe_reset_threadpool(n);  // documented to always return true; no-ops for 0/unchanged
  return static_cast<uint32_t>(pool->get_thread_count());
}

uint32_t intraOpThreads() {
  return static_cast<uint32_t>(
      executorch::extension::threadpool::get_threadpool()->get_thread_count());
}

}  // namespace measly::et
