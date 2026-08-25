#ifndef MEASLY_ET_RUNTIME_H
#define MEASLY_ET_RUNTIME_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace measly::et {

// Borrowed input: data is a host pointer the caller keeps valid across forward().
// NOT zero-copy end to end. ExecuTorch's Method::set_input copies this into its arena for any
// input where MethodMeta::input_tensor_meta(i)->is_memory_planned() is true -- the export
// default (MemoryPlanningPass(alloc_graph_input=True)) -- and borrows the pointer only when
// the .pte was exported with alloc_graph_input=False.
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
  // 1 = memory-planned: ExecuTorch copies this input into its arena at set_input; 0 = borrowed
  // pointer. Non-tensor inputs are 0 (no TensorInfo exists).
  std::vector<uint8_t> inputMemoryPlanned;
  // Declared byte count per input, from TensorInfo::nbytes() at load. Exact for a static shape,
  // an upper bound for a dynamic one; 0 for a non-tensor input. Staging slots are sized from this
  // at construction, so a slot only ever grows when an input exceeds its declared bound.
  std::vector<size_t> inputNbytes;
  // Sum of MethodMeta::memory_planned_buffer_size(i) over num_memory_planned_buffers(), captured
  // once at load. This is ExecuTorch's planned activation arena for "forward". It does NOT include
  // the XNNPACK delegate workspace: xnn_workspace_t is opaque in the shipped xnnpack.h (create and
  // release only, no size accessor), and under the default `global` sharing mode that workspace is
  // not per-model in any case. Treat this as an exact lower bound on native footprint, not a total.
  size_t plannedArenaBytes = 0;
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
  // workspaceSharingMode: XNNPACK workspace sharing for THIS model, supplied as a per-load backend
  // runtime spec (Module::load(LoadBackendOptionsMap)). 0=Disabled, 1=PerModel, 2=Global, matching
  // executorch::backends::xnnpack::WorkspaceSharingMode in backends/xnnpack/runtime/XNNPACKBackend.h
  // -- a header the runtime tarball does NOT install, hence the hardcoded values.
  //
  // -1 omits the spec entirely, leaving whatever default the runtime was compiled with (Global for
  // our pin: EXECUTORCH_XNNPACK_SHARED_WORKSPACE=ON). Omitting is deliberately not the same as
  // passing 2 -- it follows the pin rather than pinning a value we would have to keep in sync.
  //
  // Any other value is passed through to ExecuTorch, which rejects it at delegate init and fails
  // the load. et_runtime_test.cpp depends on that to prove the spec reaches the backend.
  //
  // traceEvents attaches an ExecuTorch ETDump event tracer to this model. Costs are real and the
  // buffer is unbounded until pulled (see etDump), so this is a diagnostic, not a production mode.
  // Throws when the linked runtime has no event tracer -- profiling that silently records nothing
  // is worse than a failed load.
  explicit EtRuntime(const std::string& ptePath, int workspaceSharingMode = -1,
                     bool traceEvents = false);
  ~EtRuntime();
  EtRuntime(const EtRuntime&) = delete;
  EtRuntime& operator=(const EtRuntime&) = delete;
  MethodMeta methodMeta() const;

  // Total bytes currently held by this runtime's input staging slots. Returns 0 when every input
  // is memory-planned (the ExecuTorch export default) -- planned inputs are never staged, so their
  // slots stay at capacity 0. Cold path: O(numInputs), intended for a monitoring poll, not the
  // forward path.
  size_t stagingBytes() const;

  ForwardResult forward(std::span<const InputDesc> inputs);

  // Finalized ETDump covering every forward since the last call, or empty when not tracing and
  // when no forward has run yet. Each forward appends one "Execute" block; the runtime resets the
  // generator on the first block after a finalize, so pulling IS the drain.
  //
  // Unlike OutputView, the returned bytes are owned by the caller and outlive this runtime.
  //
  // Calling twice with no forward in between returns a copy of the same bytes: upstream's
  // get_etdump_data() matches none of its guard branches in the finalized state and would run the
  // builder's end sequence a second time, so the cached copy is a correctness guard, not an
  // optimization.
  std::vector<uint8_t> etDump();

 private:
  std::unique_ptr<RuntimeState> state_;
};

// Intra-op (XNNPACK) thread pool size, backed by ExecuTorch's process-global
// extension::threadpool singleton. The pool sizes itself to the performance-core count by
// default and nothing reads an env var (verified for v1.3.1: no getenv in extension/threadpool,
// the vendored pthreadpool, XNNPACK init, or cpuinfo), so this is the ONLY control surface.
//
// setIntraOpThreads returns the count in effect AFTER the attempt. Upstream's
// _unsafe_reset_threadpool always returns true (it early-returns for n == 0 and for
// n == get_thread_count()), so a bool status would be meaningless -- callers compare instead.
//
// Enforced, not just documented: the core refuses a reset once any EtRuntime has been
// constructed -- a logged no-op returning the current count. XNNPACK captures the
// pthreadpool_t when it creates a runtime, and _unsafe_reset_threadpool destroys the old pool
// object, so a late reset is a use-after-free on the next forward(), not merely a race.
uint32_t setIntraOpThreads(uint32_t n);
uint32_t intraOpThreads();

// Size in bytes of the XNNPACK delegate's memory arena, or -1 when it cannot be read.
//
// Read through the backend-options API by string: XNNPACKBackend.h is not an installed header,
// so the backend name and option key cannot be included and are spelled out in the .cpp.
//
// Three properties a caller must not paper over:
//   * Process-wide. The runtime pins EXECUTORCH_XNNPACK_SHARED_WORKSPACE=ON, so every delegate
//     instance shares one arena and this figure is already the total. Never sum it per model.
//   * A high-water mark, including allocator alignment padding. The arena grows on demand and is
//     never shrunk, so it is the peak, not the live footprint, and slightly over-states tensor data.
//   * 0 before the first XNNPACK-delegated method loads. The arena is created lazily during
//     delegate init, so a 0 there is the correct answer, not a broken accessor -- which is why
//     failure is reported as -1 and never folded into 0.
//
// The option is a vendored patch in the pinned distribution, not upstream ExecuTorch. Against a
// stock runtime this returns -1 rather than failing to build, since it names the key by string.
int64_t xnnpackWorkspaceBytes();

// True when this build linked a runtime whose event tracer is compiled in, i.e. when profiling can
// actually record. Compile-time constant: the capability is a property of the linked runtime, not
// of anything discoverable at run time.
bool devtoolsAvailable();

// True if `ptePath`'s "forward" method is delegated to `backend` (e.g. "OpenvinoBackend").
//
// Reads METADATA ONLY: it builds a bare Module and asks method_meta, never load_forward(). That
// distinction is the whole point. EtRuntime's ctor calls load_forward() unconditionally, and for a
// delegated model that IS delegate init -- which for OpenVINO means a dlopen under std::call_once
// with no retry. Anything that needs to act before delegate init must ask through here.
//
// Throws std::runtime_error if the file cannot be opened or its program cannot be read. Reporting
// false there would be indistinguishable from "needs no delegate" and would hide the real error.
bool pteUsesBackend(const std::string& ptePath, const std::string& backend);

// True if `backend` is registered in this build, i.e. its archive was linked. Answers "was the
// delegate compiled in", NOT "is it configured to run" -- registration is a link-time fact.
//
// Exposed because the two states need different advice and only the caller can tell them apart. A
// build with no delegate cannot run the model at all (re-export is the fix); a build WITH the
// delegate but no OpenVINO runtime for this platform can (supply a runtime). The delegate ships in
// every Linux runtime tarball, including linux-aarch64, while the OpenVINO runtime bundle is
// published for fewer platforms -- so the second case is real, not hypothetical.
bool isBackendRegistered(const std::string& backend);

// The numeric type OpenVINO will use for CPU inference on this host ("f32", "bf16", ...), or
// "unavailable" if it cannot be determined.
//
// Reads through a FRESHLY CREATED ov::Core, not the Core the delegate built inside OpenvinoBackend.
// Those agree today because the choice derives from CPU capability alone; if per-model precision
// control is ever added they could diverge, and this would have to read through the delegate --
// which ExecuTorch exposes no way to do.
//
// Creating a Core loads the CPU plugin and is not cheap. This is an on-demand diagnostic: never
// call it on the hot path or during model load.
std::string openVinoInferencePrecision(const std::string& libPath);

}  // namespace measly::et
#endif  // MEASLY_ET_RUNTIME_H
