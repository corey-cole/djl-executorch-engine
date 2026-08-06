#include "et_runtime.h"

#include <cstring>
#include <stdexcept>
#include <utility>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/executor/method_meta.h>

#include "dtype_size.h"
#include "et_probes.h"
#include "staging.h"

namespace measly::et {

using executorch::extension::Module;
using executorch::extension::from_blob;
using executorch::extension::TensorPtr;
using executorch::runtime::EValue;

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

EtRuntime::EtRuntime(const std::string& ptePath)
    : state_(std::make_unique<RuntimeState>(ptePath)) {
  // Force-load now so a bad path/file throws at construction (the "load throws" contract).
  if (state_->module.load() != executorch::runtime::Error::Ok) {
    throw std::runtime_error("EtRuntime: failed to load .pte: " + ptePath);
  }
  state_->meta = buildMethodMeta(state_->module);
  // One slot per input position; resize() would default-construct null unique_ptrs, so create each
  // slot explicitly. Slots for staged inputs are sized *here*, from the bound the .pte declares —
  // TensorInfo::nbytes() is available at load for planned and unplanned inputs alike, so "grow-only"
  // degenerates to "allocate once" and staging_grow goes back to being the anomaly detector W8
  // specified. Planned slots stay at capacity 0 (never staged, never allocated).
  //
  // The tensor-ness test matters: inputMemoryPlanned is 0 both for a genuinely borrowed input and
  // for a non-tensor one (no TensorInfo exists), and only the former has a declared bound to size
  // from. forward() still stages on the planned flag alone, so a non-tensor input would grow its
  // slot on first use rather than being rejected — see F3 in docs/host-buffer-contract-wip.md.
  state_->staging.reserve(static_cast<size_t>(state_->meta.numInputs));
  for (int i = 0; i < state_->meta.numInputs; ++i) {
    state_->staging.push_back(std::make_unique<StagingSlot>());
    const bool staged =
        state_->meta.inputMemoryPlanned[i] == 0 && state_->meta.inputScalarTypes[i] >= 0;
    if (staged) {
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

    // Byte count of this input; product of an empty shape is 1. Matches dtypeSize's conventions
    // (the subset the harnesses build buffers for), so planned/unplanned classification is exact.
    size_t actual = dtypeSize(in.scalarType);
    for (int64_t d : in.shape) {
      actual *= static_cast<size_t>(d);
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
        // Anomaly path. The slot was sized at load from the .pte's declared bound, so reaching here
        // means this input exceeded it — a dynamic shape past its upper bound, or a caller shape
        // that does not match the model. staging_grow fires only here: in steady state it never
        // fires at all, and a fire is the signal W8 designed it to be rather than routine
        // first-touch noise.
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

}  // namespace measly::et
