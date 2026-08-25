#include "et_runtime.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

#ifndef _WIN32
#include <dlfcn.h>
#else
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#include <sys/stat.h>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/threadpool/threadpool.h>
#include <executorch/runtime/backend/backend_options_map.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/executor/method_meta.h>
#ifdef ET_HAVE_DEVTOOLS
#include <executorch/devtools/etdump/etdump_flatcc.h>
#endif

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
#ifdef ET_HAVE_DEVTOOLS
  // Non-owning: the Module owns the tracer, because its constructor takes the unique_ptr. Null
  // when this runtime is not tracing.
  executorch::etdump::ETDumpGen* tracer = nullptr;
#endif
  // Set by etDump(), cleared by forward(). While set, the cached copy is returned instead of
  // finalizing an already-finalized builder.
  bool dumpFinalized = false;
  // True once a forward has completed. Until then etDump() is empty even when tracing: the only
  // data the tracer holds is the block load_forward() records, which is not a forward's trace.
  bool everForwarded = false;
  std::vector<uint8_t> lastDump;

  RuntimeState(const std::string& path, std::unique_ptr<executorch::runtime::EventTracer> t)
      : module(path, Module::LoadMode::File, std::move(t)) {}
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
  // memory_planned_buffer_size returns Result<int64_t>; a failing entry contributes nothing rather
  // than failing the whole load, because an unreadable arena size must never break model loading.
  size_t arena = 0;
  for (size_t b = 0; b < meta->num_memory_planned_buffers(); ++b) {
    auto planned = meta->memory_planned_buffer_size(b);
    if (planned.ok()) {
      arena += static_cast<size_t>(*planned);
    }
  }
  out.plannedArenaBytes = arena;
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

// Whether a backend is registered in this build. Registration is link-time, so this answers "was
// the delegate compiled in", not "is it configured". Signatures per
// runtime/backend/interface.h:179,184 in the pinned runtime -- note both are size_t-indexed.
bool isBackendAvailable(const char* name) {
  const size_t n = executorch::ET_RUNTIME_NAMESPACE::get_num_registered_backends();
  for (size_t i = 0; i < n; ++i) {
    const auto backendName = executorch::ET_RUNTIME_NAMESPACE::get_backend_name(i);
    if (backendName.ok() && std::strcmp(*backendName, name) == 0) {
      return true;
    }
  }
  return false;
}

// Builds the tracer the Module will own, or nullptr. Kept out of the ctor body so the throw for an
// unsupported runtime happens before any Module exists.
std::unique_ptr<executorch::runtime::EventTracer> makeTracer(bool traceEvents) {
  if (!traceEvents) return nullptr;
#ifdef ET_HAVE_DEVTOOLS
  return std::make_unique<executorch::etdump::ETDumpGen>();
#else
  throw std::runtime_error(
      "EtRuntime: profiling requested but this build links a runtime with no event tracer "
      "(devtools is not provisioned for this platform)");
#endif
}

}  // namespace

EtRuntime::EtRuntime(const std::string& ptePath, int workspaceSharingMode, bool traceEvents)
    : state_(std::make_unique<RuntimeState>(ptePath, makeTracer(traceEvents))) {
#ifdef ET_HAVE_DEVTOOLS
  if (traceEvents) {
    state_->tracer =
        static_cast<executorch::etdump::ETDumpGen*>(state_->module.event_tracer());
  }
#endif
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
  // Refuse an OpenVINO-delegated model that cannot possibly succeed, BEFORE load_forward() -- which
  // is delegate init. This matters more than a typical precondition check: OpenvinoBackend resolves
  // the OpenVINO C API with dlopen under std::call_once and never retries, so a failure that
  // reaches it leaves the whole process broken until restart. Raising here keeps the failure an
  // ordinary exception and the process usable.
  //
  // Duplicated by the Java layer deliberately: EtNative is public and bypasses EtModel, and our own
  // tests call it directly.
  auto etMeta = state_->module.method_meta("forward");
  if (etMeta.ok() && etMeta->uses_backend("OpenvinoBackend")) {
    if (!isBackendAvailable("OpenvinoBackend")) {
      throw std::runtime_error(
          "This .pte uses the OpenvinoBackend delegate, which this build does not provide. "
          "The OpenVINO delegate ships only where the runtime tarball was built with it. "
          "Re-export without the OpenVINO partitioner to run here.");
    }
    const char* lib = std::getenv("OPENVINO_LIB_PATH");
    if (lib == nullptr || *lib == '\0') {
      throw std::runtime_error(
          // Deliberately does NOT name a per-platform artifact. This layer knows neither the
          // platform nor whether an OpenVINO bundle is published for it, and the delegate ships on
          // platforms that have no bundle -- so naming one here told aarch64 users to fetch
          // something that does not exist. EtModel's Java path knows both and says more.
          "This .pte uses the OpenvinoBackend delegate, but OPENVINO_LIB_PATH is not set. "
          "Set it to the FULL PATH OF THE LIBRARY FILE (not a directory) before the first "
          "inference, or load through EtModel, which resolves a bundled runtime when one is "
          "available for this platform.");
    }
    struct stat st {};
    // S_ISREG is a POSIX macro MSVC does not provide; S_IFMT/S_IFREG exist on both, and the
    // expansion is identical, so this form keeps the Windows shim compiling.
    if (stat(lib, &st) != 0 || (st.st_mode & S_IFMT) != S_IFREG) {
      throw std::runtime_error(
          std::string("OPENVINO_LIB_PATH does not name a readable file: '") + lib +
          "'. It must be the full path to the library FILE, not the directory containing it.");
    }
  }
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

size_t EtRuntime::stagingBytes() const {
  size_t total = 0;
  for (const auto& slot : state_->staging) {
    total += slot->capacity();
  }
  return total;
}

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
  // A forward that ran re-opens the dump: upstream resets the generator on the first event block
  // after a finalize, so the cached copy is stale from here on. A forward that threw leaves the
  // cache intact, so etDump() keeps returning the last completed dump.
  state_->dumpFinalized = false;
  state_->everForwarded = true;

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

std::vector<uint8_t> EtRuntime::etDump() {
#ifdef ET_HAVE_DEVTOOLS
  if (state_->tracer == nullptr) return {};
  if (!state_->everForwarded) return {};
  if (state_->dumpFinalized) return state_->lastDump;
  executorch::etdump::ETDumpResult result = state_->tracer->get_etdump_data();
  state_->dumpFinalized = true;
  state_->lastDump.clear();
  if (result.buf != nullptr && result.size > 0) {
    const auto* p = static_cast<const uint8_t*>(result.buf);
    state_->lastDump.assign(p, p + result.size);
    // Caller-owned: get_etdump_data() finalizes into a fresh allocation. free() is the idiom
    // upstream's own consumer uses (examples/devtools/example_runner) and is correct for flatcc's
    // aligned allocator on POSIX. A Windows devtools build must use flatcc_builder_aligned_free
    // instead, since flatcc allocates with _aligned_malloc there.
    std::free(result.buf);
  }
  return state_->lastDump;
#else
  return {};
#endif
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

int64_t xnnpackWorkspaceBytes() {
  // Both strings are hardcoded because XNNPACKBackend.h is not installed in the runtime tarball.
  // The backend id must match the one the sharing-mode path above uses; the key is read-only --
  // set_option on it returns InvalidArgument rather than silently no-op'ing.
  executorch::runtime::BackendOption opt{};
  std::snprintf(opt.key, sizeof(opt.key), "%s", "workspace_size_bytes");
  executorch::runtime::Span<executorch::runtime::BackendOption> span(&opt, 1);
  if (executorch::ET_RUNTIME_NAMESPACE::get_option("XnnpackBackend", span) !=
      executorch::runtime::Error::Ok) {
    return -1;
  }
  // The option is declared int; a different alternative means the runtime's contract changed under
  // us, which is an "unavailable" rather than a value worth guessing at.
  const int* value = std::get_if<int>(&opt.value);
  return (value == nullptr) ? -1 : static_cast<int64_t>(*value);
}

bool isBackendRegistered(const std::string& backend) {
  return isBackendAvailable(backend.c_str());
}

bool devtoolsAvailable() {
#ifdef ET_HAVE_DEVTOOLS
  return true;
#else
  return false;
#endif
}

bool pteUsesBackend(const std::string& ptePath, const std::string& backend) {
  Module probe(ptePath);
  const auto meta = probe.method_meta("forward");
  if (!meta.ok()) {
    throw std::runtime_error(
        "pteUsesBackend: cannot read method metadata from " + ptePath + " (error " +
        std::to_string(static_cast<int>(meta.error())) + ")");
  }
  return meta->uses_backend(backend.c_str());
}

namespace {

// The platform arm of openVinoInferencePrecision(). Isolated so the ov_* call sequence below
// exists exactly once: the two loaders differ only in how a module is opened and a symbol found,
// and a fix applied to one copy but not the other would be invisible on the platform that did not
// get it.
//
// Returns nullptr when the library, or anything it depends on, cannot be resolved.
void* ovLoadLibrary(const std::string& libPath) {
#ifndef _WIN32
  // dlopen'd rather than linked: we have no OpenVINO at link time, and the delegate resolves the
  // same library the same way. Refcounted, so opening it here is safe alongside the delegate's own
  // handle. RTLD_LOCAL so nothing here perturbs the delegate's symbol resolution.
  return dlopen(libPath.c_str(), RTLD_LAZY | RTLD_LOCAL);
#else
  // libPath is a Windows-style absolute path to the vendored openvino_c.dll, UTF-8 as it arrives
  // from JNI.
  //
  // DO NOT SIMPLIFY THIS CALL. Every part of its shape has a measured failure behind it, recorded
  // by executorch-runtime-dist's test/openvino/win_origin_probe.c, which runs the same load against
  // a flat bundle under three modes:
  //
  //   LoadLibraryW(abs)                                  FAILS, error 126. Windows has no $ORIGIN:
  //                                                      the loader searches for a DLL's
  //                                                      dependencies by module name, from the
  //                                                      EXE's directory and PATH, NEVER from the
  //                                                      directory the DLL itself came out of. So
  //                                                      openvino.dll and tbb12.dll are not found
  //                                                      beside openvino_c.dll. If this form ever
  //                                                      appears to work, some OTHER OpenVINO was
  //                                                      found on PATH or in System32 -- that is
  //                                                      the probe's negative control, and a pass
  //                                                      there means a contaminated environment,
  //                                                      not a working load.
  //   LoadLibraryExW(.., SEARCH_DLL_LOAD_DIR)            FAILS, same error 126. Passing ANY
  //                                                      LOAD_LIBRARY_SEARCH_* flag switches the
  //                                                      loader to the alternate search order,
  //                                                      which drops System32 -- where the CRT the
  //                                                      OpenVINO wheel was built against
  //                                                      (MSVCP140, VCRUNTIME140, VCRUNTIME140_1)
  //                                                      lives. Dropping the flag that looks
  //                                                      redundant is what breaks it.
  //   LoadLibraryExW(.., DLL_LOAD_DIR | DEFAULT_DIRS)    Works, and works COLD -- before anything
  //                                                      else has loaded the graph.
  //
  // Both flags are therefore load-bearing, and OpenVinoColdProbeTest is the executable statement of
  // that: it probes in a JVM where no model has been loaded, so any of the failing shapes above
  // turns it red instead of passing on a graph the delegate happened to load first.
  //
  // The A-suffixed entry points are wrong here for a second, independent reason: they convert
  // through the process ANSI codepage, and libPath runs through %LOCALAPPDATA%, which carries the
  // Windows profile name. A non-ASCII profile is unrepresentable in most codepages, so the path
  // would not survive the conversion.
  int wideLen = MultiByteToWideChar(CP_UTF8, 0, libPath.c_str(), -1, nullptr, 0);
  if (wideLen == 0) {
    return nullptr;
  }
  std::wstring widePath(static_cast<size_t>(wideLen), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, libPath.c_str(), -1, widePath.data(), wideLen);
  return LoadLibraryExW(
      widePath.c_str(),
      nullptr,
      LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#endif
}

void* ovSymbol(void* handle, const char* name) {
#ifndef _WIN32
  return dlsym(handle, name);
#else
  // GetProcAddress returns FARPROC; the hop through void* is what keeps MSVC's C4191 quiet on the
  // real function types.
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
#endif
}

// Only ever called on a failure path. On success the handle is deliberately leaked: the delegate
// may hold the same library, and OpenVINO registers plugin state that does not expect to be torn
// down and rebuilt. It is process-lifetime by design, and this is a diagnostic called a handful of
// times at most.
void ovUnloadLibrary(void* handle) {
#ifndef _WIN32
  dlclose(handle);
#else
  FreeLibrary(static_cast<HMODULE>(handle));
#endif
}

}  // namespace

std::string openVinoInferencePrecision(const std::string& libPath) {
  // The probe exists on both shipped platforms, resolving the vendored OpenVINO C API through
  // ovLoadLibrary() above. The accessor's contract (EtEngine) is to degrade to "unavailable"
  // rather than throw, so callers need no platform awareness -- and "unavailable" is the honest
  // answer when no vendored runtime exists to read from.
  void* handle = ovLoadLibrary(libPath);
  if (handle == nullptr) {
    return "unavailable";
  }
  using CoreCreate = int (*)(void**);
  using CoreGetProperty = int (*)(void*, const char*, const char*, char**);
  using CoreFree = void (*)(void*);
  using Free = void (*)(const char*);

  auto create = reinterpret_cast<CoreCreate>(ovSymbol(handle, "ov_core_create"));
  auto getProperty = reinterpret_cast<CoreGetProperty>(ovSymbol(handle, "ov_core_get_property"));
  auto coreFree = reinterpret_cast<CoreFree>(ovSymbol(handle, "ov_core_free"));
  auto ovFree = reinterpret_cast<Free>(ovSymbol(handle, "ov_free"));
  if (create == nullptr || getProperty == nullptr || coreFree == nullptr) {
    ovUnloadLibrary(handle);
    return "unavailable";
  }

  void* core = nullptr;
  if (create(&core) != 0 || core == nullptr) {
    ovUnloadLibrary(handle);
    return "unavailable";
  }
  char* value = nullptr;
  std::string result = "unavailable";
  if (getProperty(core, "CPU", "INFERENCE_PRECISION_HINT", &value) == 0 && value != nullptr) {
    result = value;
    if (ovFree != nullptr) {
      ovFree(value);
    }
  }
  coreFree(core);
  return result;
}

}  // namespace measly::et
