#include "et_runtime.h"

#include <stdexcept>
#include <utility>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/executor/method_meta.h>

namespace measly::et {

using executorch::extension::Module;
using executorch::extension::from_blob;
using executorch::extension::TensorPtr;
using executorch::runtime::EValue;

struct RuntimeState {
  Module module;
  explicit RuntimeState(const std::string& path) : module(path) {}
};

struct ForwardState {
  std::vector<EValue> outputs;    // owns the result EValues
  std::vector<OutputView> views;  // descriptors into the host arena
};

EtRuntime::EtRuntime(const std::string& ptePath)
    : state_(std::make_unique<RuntimeState>(ptePath)) {
  // Force-load now so a bad path/file throws at construction (the "load throws" contract).
  if (state_->module.load() != executorch::runtime::Error::Ok) {
    throw std::runtime_error("EtRuntime: failed to load .pte: " + ptePath);
  }
}

EtRuntime::~EtRuntime() = default;

MethodMeta EtRuntime::methodMeta() const {
  auto meta = state_->module.method_meta("forward");
  if (!meta.ok()) {
    throw std::runtime_error("EtRuntime: method_meta(\"forward\") failed");
  }
  const int n = static_cast<int>(meta->num_inputs());
  MethodMeta out;
  out.numInputs = n;
  out.inputScalarTypes.resize(n);
  out.inputShapes.resize(n);
  for (int i = 0; i < n; ++i) {
    auto info = meta->input_tensor_meta(i);
    if (info.ok()) {
      out.inputScalarTypes[i] = static_cast<int8_t>(info->scalar_type());
      auto sizes = info->sizes();  // Span<const int32_t>
      out.inputShapes[i].assign(sizes.begin(), sizes.end());
    } else {
      out.inputScalarTypes[i] = -1;  // non-tensor input; inputShapes[i] left empty
    }
  }
  return out;
}

ForwardResult EtRuntime::forward(std::span<const InputDesc> inputs) {
  // from_blob does not copy: each InputDesc.data must stay valid through module.forward().
  std::vector<std::vector<executorch::aten::SizesType>> shapes(inputs.size());
  std::vector<TensorPtr> tensors;
  std::vector<EValue> evalues;
  tensors.reserve(inputs.size());
  evalues.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& in = inputs[i];
    shapes[i].assign(in.shape.begin(), in.shape.end());
    tensors.push_back(from_blob(
        const_cast<void*>(in.data), shapes[i],
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
