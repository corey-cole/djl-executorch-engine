#ifndef MEASLY_ET_RUNTIME_H
#define MEASLY_ET_RUNTIME_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace measly::et {

// Borrowed input: data is a host pointer the caller keeps valid across forward(). Zero-copy in.
struct InputDesc {
  const void* data;
  std::vector<int64_t> shape;
  int8_t scalarType;  // ExecuTorch ScalarType code
};

// Borrowed output: data points into ExecuTorch's host arena, valid until the next
// forward()/destroy on the originating EtRuntime. Single-copy out happens in the consumer.
struct OutputView {
  std::vector<int64_t> shape;
  int8_t scalarType;
  const void* data;
  size_t nbytes;
};

// Static metadata for the "forward" method.
struct MethodMeta {
  int numInputs;
  std::vector<int8_t> inputScalarTypes;           // -1 for a non-tensor input
  std::vector<std::vector<int64_t>> inputShapes;   // per tensor input; empty for non-tensor (-1)
};

struct ForwardState;  // pimpl
struct RuntimeState;  // pimpl

// Owns the ExecuTorch EValue vector backing the views. RAII: dropping it ends the view lifetime.
class ForwardResult {
 public:
  explicit ForwardResult(std::unique_ptr<ForwardState> state);
  ~ForwardResult();
  ForwardResult(ForwardResult&&) noexcept;
  ForwardResult& operator=(ForwardResult&&) noexcept;
  ForwardResult(const ForwardResult&) = delete;
  ForwardResult& operator=(const ForwardResult&) = delete;
  std::span<const OutputView> outputs() const;

 private:
  std::unique_ptr<ForwardState> state_;
};

// Owns the ExecuTorch Module. Throws std::runtime_error on load/forward/meta failure.
class EtRuntime {
 public:
  explicit EtRuntime(const std::string& ptePath);
  ~EtRuntime();
  EtRuntime(const EtRuntime&) = delete;
  EtRuntime& operator=(const EtRuntime&) = delete;
  MethodMeta methodMeta() const;
  ForwardResult forward(std::span<const InputDesc> inputs);

 private:
  std::unique_ptr<RuntimeState> state_;
};

}  // namespace measly::et
#endif  // MEASLY_ET_RUNTIME_H
