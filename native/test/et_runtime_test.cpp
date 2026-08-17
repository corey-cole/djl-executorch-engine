#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "dtype_size.h"
#include "et_log_level.h"
#include "et_probes.h"
#include "et_runtime.h"
#include "array_size_limits.h"
#include "staging.h"

using namespace measly::et;

// The OpenVINO guard tests must mutate OPENVINO_LIB_PATH in-process. POSIX has setenv/unsetenv;
// MSVC (this suite builds and runs on Windows too) exposes _putenv_s and _putenv instead. The
// helpers keep the test bodies identical across the two.
#ifdef _WIN32
static void setEnvVar(const char* name, const char* value) { _putenv_s(name, value); }
static void unsetEnvVar(const char* name) {
  // "NAME=" removes the variable on MSVC; the entry string is copied by _putenv, so a temporary
  // std::string is safe here.
  const std::string entry = std::string(name) + "=";
  _putenv(entry.c_str());
}
#else
static void setEnvVar(const char* name, const char* value) { setenv(name, value, 1); }
static void unsetEnvVar(const char* name) { unsetenv(name); }
#endif

#ifndef ADD_PTE_PATH
#define ADD_PTE_PATH "add.pte"
#endif

// Intra-op pool tests run FIRST, before ANY EtRuntime construction: Catch2 runs in registration
// order, and the reset guard (issue #26) refuses a reset once a runtime has been constructed --
// including by a throwing ctor, so even "load: missing path throws" below would trip it.
TEST_CASE("intraop: setIntraOpThreads resizes the shared pool and reports the applied count") {
  const uint32_t before = intraOpThreads();
  REQUIRE(setIntraOpThreads(1) == 1);
  REQUIRE(intraOpThreads() == 1);
  // The pool is process-global: restore so sibling tests run on the default pool.
  setIntraOpThreads(before);
  REQUIRE(intraOpThreads() == before);
}

TEST_CASE("intraop: upstream quirks -- 0 is silently ignored, same-count reset is a no-op") {
  const uint32_t cur = intraOpThreads();
  REQUIRE(setIntraOpThreads(0) == cur);   // core guards n < 1 (issue #24): 0 is a no-op before upstream sees it
  REQUIRE(intraOpThreads() == cur);
  REQUIRE(setIntraOpThreads(cur) == cur); // early-returns for the current count: unchanged
  REQUIRE(intraOpThreads() == cur);
}

TEST_CASE("load: missing path throws") {
  REQUIRE_THROWS([] { EtRuntime rt("/nonexistent/definitely-not-here.pte"); }());
}

TEST_CASE("load: valid pte constructs") {
  REQUIRE_NOTHROW([] { EtRuntime rt(ADD_PTE_PATH); }());
}

TEST_CASE("methodMeta: add has two float32 tensor inputs of shape [1]") {
  EtRuntime rt(ADD_PTE_PATH);
  MethodMeta meta = rt.methodMeta();
  REQUIRE(meta.numInputs == 2);
  REQUIRE(meta.inputScalarTypes.size() == 2);
  REQUIRE(meta.inputScalarTypes[0] == 6);  // FLOAT32
  REQUIRE(meta.inputScalarTypes[1] == 6);
  REQUIRE(meta.inputShapes.size() == 2);
  REQUIRE(meta.inputShapes[0] == std::vector<int64_t>{1});
  REQUIRE(meta.inputShapes[1] == std::vector<int64_t>{1});
  REQUIRE(meta.inputMemoryPlanned.size() == 2);
  REQUIRE(meta.inputMemoryPlanned[0] == 1);
  REQUIRE(meta.inputMemoryPlanned[1] == 1);
}

#ifndef ADD_UNPLANNED_PTE_PATH
#define ADD_UNPLANNED_PTE_PATH "add_unplanned.pte"
#endif

TEST_CASE("methodMeta: add_unplanned inputs are borrowed (not memory-planned)") {
  EtRuntime rt(ADD_UNPLANNED_PTE_PATH);
  MethodMeta meta = rt.methodMeta();
  REQUIRE(meta.numInputs == 2);
  REQUIRE(meta.inputScalarTypes[0] == 6);
  REQUIRE(meta.inputScalarTypes[1] == 6);  // same model, different memory plan
  REQUIRE(meta.inputMemoryPlanned.size() == 2);
  REQUIRE(meta.inputMemoryPlanned[0] == 0);
  REQUIRE(meta.inputMemoryPlanned[1] == 0);
}

TEST_CASE("forward: add(2,3) == 5 with correct view metadata") {
  EtRuntime rt(ADD_PTE_PATH);
  float a = 2.0f, b = 3.0f;
  std::vector<InputDesc> inputs = {{&a, {1}, 6}, {&b, {1}, 6}};
  ForwardResult result = rt.forward(inputs);
  auto outs = result.outputs();
  REQUIRE(outs.size() == 1);
  REQUIRE(outs[0].scalarType == 6);
  REQUIRE(outs[0].nbytes == sizeof(float));
  REQUIRE(outs[0].shape == std::vector<int64_t>{1});
  REQUIRE(*static_cast<const float*>(outs[0].data) == 5.0f);
}

TEST_CASE("forward: a second call yields a fresh correct result (view-lifetime happy path)") {
  EtRuntime rt(ADD_PTE_PATH);
  float a1 = 2.0f, b1 = 3.0f;
  std::vector<InputDesc> in1 = {{&a1, {1}, 6}, {&b1, {1}, 6}};
  ForwardResult r1 = rt.forward(in1);
  REQUIRE(*static_cast<const float*>(r1.outputs()[0].data) == 5.0f);

  float a2 = 10.0f, b2 = 7.0f;
  std::vector<InputDesc> in2 = {{&a2, {1}, 6}, {&b2, {1}, 6}};
  ForwardResult r2 = rt.forward(in2);
  REQUIRE(*static_cast<const float*>(r2.outputs()[0].data) == 17.0f);
}

TEST_CASE("level map: ET PAL chars -> slf4j level codes") {
  using namespace measly::et;
  REQUIRE(et_djl_level_to_slf4j('D') == kSlf4jDebug);
  REQUIRE(et_djl_level_to_slf4j('I') == kSlf4jInfo);
  REQUIRE(et_djl_level_to_slf4j('E') == kSlf4jError);
  REQUIRE(et_djl_level_to_slf4j('F') == kSlf4jError);  // slf4j has no FATAL
  REQUIRE(et_djl_level_to_slf4j('?') == kSlf4jWarn);
  REQUIRE(et_djl_level_to_slf4j('X') == kSlf4jInfo);   // unknown -> INFO default
}

TEST_CASE("jni byte[] size limit: outputs above INT32_MAX bytes must be rejected") {
  using measly::et::exceedsJniByteArrayLimit;
  REQUIRE_FALSE(exceedsJniByteArrayLimit(static_cast<size_t>(INT32_MAX)));
  REQUIRE(exceedsJniByteArrayLimit(static_cast<size_t>(INT32_MAX) + 1));
}

// --- W7: grow-only per-slot staging for unplanned inputs ---

namespace {

struct ProbeCounters {
  int grow = 0;         // staging_grow fires
  int stagedInput = 0;  // staging_input fires with staged==1
  int totalInput = 0;   // every staging_input fire
};

ProbeCounters g_probeCounters;

void countProbe(uint32_t id, uint64_t, uint64_t, uint64_t, uint64_t d) {
  switch (id) {
    case kProbeStagingGrow:
      ++g_probeCounters.grow;
      break;
    case kProbeStagingInput:
      ++g_probeCounters.totalInput;
      if (d != 0) {
        ++g_probeCounters.stagedInput;
      }
      break;
  }
}

// RAII registration of the counting probe handler; reset() between scenarios.
struct ProbeGuard {
  ProbeGuard() {
    reset();
    et_probe_set_handler(countProbe);
  }
  ~ProbeGuard() { et_probe_clear_handler(); }
  void reset() { g_probeCounters = ProbeCounters{}; }
  int growCount() const { return g_probeCounters.grow; }
  int stagedInputCount() const { return g_probeCounters.stagedInput; }
  int totalInputCount() const { return g_probeCounters.totalInput; }
};

}  // namespace

TEST_CASE("staging: ensure(kStagingPadding) yields an aligned buffer with padding slack") {
  StagingSlot slot;
  void* p = slot.ensure(kStagingPadding);
  REQUIRE(p != nullptr);
  REQUIRE(reinterpret_cast<uintptr_t>(p) % 64 == 0);
  REQUIRE(slot.capacity() >= kStagingPadding);
}

TEST_CASE("staging: slot carries the padding slack row (100 + kStagingPadding)") {
  StagingSlot slot;
  slot.ensure(100 + kStagingPadding);
  REQUIRE(slot.capacity() >= 100 + kStagingPadding);
}

TEST_CASE("staging: ensure never shrinks (no realloc when capacity suffices)") {
  StagingSlot slot;
  void* first = slot.ensure(64);
  void* again = slot.ensure(32);
  REQUIRE(again == first);
  REQUIRE(slot.capacity() >= 64);
}

TEST_CASE("staging: grow yields a larger aligned buffer (contents not preserved)") {
  // The staging caller overwrites the whole slot right after ensure(), so preserving the old bytes
  // was discarded work. What must hold is that growth produces a bigger, still-64-byte-aligned
  // buffer that is safe to write.
  StagingSlot slot;
  auto* buf = static_cast<uint8_t*>(slot.ensure(64));
  for (size_t k = 0; k < 64; ++k) {
    buf[k] = static_cast<uint8_t>(k);
  }
  auto* grown = static_cast<uint8_t*>(slot.ensure(300));
  REQUIRE(grown != nullptr);
  REQUIRE(reinterpret_cast<uintptr_t>(grown) % 64 == 0);
  REQUIRE(slot.capacity() >= 300);
  for (size_t k = 0; k < 300; ++k) {
    grown[k] = static_cast<uint8_t>(k);  // writable to the full requested extent (ASan checks this)
  }
}

TEST_CASE("forward: unplanned inputs are staged per slot (no grow: slots sized at load)") {
  EtRuntime rt(ADD_UNPLANNED_PTE_PATH);
  float a = 2.0f, b = 3.0f;
  std::vector<InputDesc> inputs = {{&a, {1}, 6}, {&b, {1}, 6}};
  ProbeGuard guard;
  ForwardResult result = rt.forward(inputs);
  REQUIRE(*static_cast<const float*>(result.outputs()[0].data) == 5.0f);
  REQUIRE(guard.growCount() == 0);         // sized from TensorInfo::nbytes() in the ctor
  REQUIRE(guard.totalInputCount() == 2);   // one staging_input per tensor input
  REQUIRE(guard.stagedInputCount() == 2);  // both staged (planned flag = 0)
}

TEST_CASE("methodMeta: declared input byte counts are captured at load") {
  EtRuntime rt(ADD_UNPLANNED_PTE_PATH);
  MethodMeta meta = rt.methodMeta();
  REQUIRE(meta.inputNbytes.size() == 2);
  REQUIRE(meta.inputNbytes[0] == sizeof(float));  // add model: two 1-element f32 inputs
  REQUIRE(meta.inputNbytes[1] == sizeof(float));
}

TEST_CASE("methodMeta: the planned activation arena is captured at load") {
  EtRuntime rt(ADD_PTE_PATH);
  MethodMeta meta = rt.methodMeta();
  // add.pte is memory-planned (the export default), so ExecuTorch allocates a planned arena for
  // its activations. Exact bytes are an ExecuTorch planning detail we deliberately do not pin.
  REQUIRE(meta.plannedArenaBytes > 0);
}

TEST_CASE("methodMeta: the planned arena excludes the XNNPACK delegate workspace") {
  // Documents a known limitation as an executable fact: the number we report is the ExecuTorch
  // planned arena only. xnn_workspace_t is opaque in the shipped xnnpack.h (create/release only),
  // so the delegate workspace cannot be added here. See the runtime-dist issue in Task 9.
  EtRuntime rt(ADD_PTE_PATH);
  MethodMeta meta = rt.methodMeta();
  REQUIRE(meta.plannedArenaBytes < 64u * 1024u * 1024u);  // an arena, not a whole workspace
}

TEST_CASE("staging: slots are sized at load, so repeated forwards never grow") {
  EtRuntime rt(ADD_UNPLANNED_PTE_PATH);
  float a = 2.0f, b = 3.0f;
  std::vector<InputDesc> inputs = {{&a, {1}, 6}, {&b, {1}, 6}};
  ProbeGuard guard;
  for (int k = 0; k < 100; ++k) {
    ForwardResult result = rt.forward(inputs);
    REQUIRE(*static_cast<const float*>(result.outputs()[0].data) == 5.0f);
  }
  REQUIRE(guard.growCount() == 0);            // the whole point of sizing at load
  REQUIRE(guard.stagedInputCount() == 200);   // still staged every call
}

TEST_CASE("stagingBytes: zero for an all-planned model") {
  // add.pte's inputs are memory-planned, so no slot is ever allocated. Zero here is the correct
  // answer, not a missing measurement -- callers distinguish it from -1 ("unavailable").
  EtRuntime rt(ADD_PTE_PATH);
  REQUIRE(rt.stagingBytes() == 0);
}

TEST_CASE("stagingBytes: sums every slot of an unplanned model and is stable across forwards") {
  EtRuntime rt(ADD_UNPLANNED_PTE_PATH);
  // Each slot is ensure(nbytes + kStagingPadding), rounded up to a 64-byte multiple by StagingSlot.
  const size_t perSlot = ((sizeof(float) + kStagingPadding + 63) / 64) * 64;
  REQUIRE(rt.stagingBytes() == 2 * perSlot);

  float a = 2.0f, b = 3.0f;
  std::vector<InputDesc> inputs = {{&a, {1}, 6}, {&b, {1}, 6}};
  ForwardResult result = rt.forward(inputs);
  REQUIRE(*static_cast<const float*>(result.outputs()[0].data) == 5.0f);
  // Slots are sized at load, so a forward must not change the total. If this ever fails, the
  // grow-only invariant that makes steady-state allocation-free has been broken.
  REQUIRE(rt.stagingBytes() == 2 * perSlot);
}

TEST_CASE("forward: an input past its declared bound is rejected before the staging copy") {
  // The staging memcpy's length comes from the caller's shape, so an oversized shape would read
  // past the source buffer -- before module.forward() runs and ExecuTorch gets to reject it. The
  // bound check must fire first: no copy, no slot growth, a diagnostic naming both sizes.
  EtRuntime rt(ADD_UNPLANNED_PTE_PATH);
  std::vector<float> big(64, 1.0f);  // a real 64-float buffer; the shape is what is being tested
  float b = 3.0f;
  std::vector<InputDesc> inputs = {{big.data(), {64}, 6}, {&b, {1}, 6}};
  ProbeGuard guard;
  REQUIRE_THROWS_AS(rt.forward(inputs), std::invalid_argument);
  REQUIRE(guard.growCount() == 0);         // rejected before ensure()
  REQUIRE(guard.stagedInputCount() == 0);  // rejected before the memcpy
}

TEST_CASE("forward: the declared-bound check applies to planned inputs too") {
  // ExecuTorch would catch this one on its own (it copies planned inputs itself, after validating),
  // but the diagnostic should not depend on which memory-plan mode the .pte happens to be in.
  EtRuntime rt(ADD_PTE_PATH);
  std::vector<float> big(64, 1.0f);
  float b = 3.0f;
  std::vector<InputDesc> inputs = {{big.data(), {64}, 6}, {&b, {1}, 6}};
  REQUIRE_THROWS_AS(rt.forward(inputs), std::invalid_argument);
}

TEST_CASE("load: a method with a non-tensor input is rejected") {
  // prim_input.pte's input 1 is a prim double (MethodMeta reports "Tag: 3 input: 1 is not Tensor"),
  // so input_tensor_meta(1) fails and the slot has no scalar type, no shape, and no declared byte
  // bound. InputDesc cannot express a prim value, so the model is undrivable through this engine:
  // reject at load rather than at first inference. This also keeps inputMemoryPlanned == 0 meaning
  // exactly "borrowed tensor" for forward()'s staging branch.
  REQUIRE_THROWS_AS([] { EtRuntime rt(PRIM_INPUT_PTE_PATH); }(), std::invalid_argument);
}

TEST_CASE("forward: an input whose dtype differs from the model's is rejected") {
  // `actual` is dtypeSize(caller's code) x shape product, so a wrong dtype mis-sizes the staging
  // copy. Must be caught before it, not by ExecuTorch inside module.forward().
  EtRuntime rt(ADD_UNPLANNED_PTE_PATH);
  double a = 2.0;  // 8 bytes, declared as FLOAT64 (7) against a FLOAT32 (6) model
  float b = 3.0f;
  std::vector<InputDesc> inputs = {{&a, {1}, 7}, {&b, {1}, 6}};
  REQUIRE_THROWS_AS(rt.forward(inputs), std::invalid_argument);
}

TEST_CASE("dtypeSize: unsupported codes report 0, not a guessed width") {
  REQUIRE(dtypeSize(6) == 4);   // FLOAT32
  REQUIRE(dtypeSize(4) == 8);   // INT64
  REQUIRE(dtypeSize(11) == 1);  // BOOL
  REQUIRE(dtypeSize(5) == 0);   // FLOAT16 — 2 bytes; guessing 4 would double a memcpy length
  REQUIRE(dtypeSize(2) == 0);   // INT16
  REQUIRE(dtypeSize(-1) == 0);  // non-tensor sentinel
}

TEST_CASE("forward: an input at exactly its declared bound is accepted") {
  // Guards the off-by-one: the check is `>`, not `>=`.
  EtRuntime rt(ADD_UNPLANNED_PTE_PATH);
  float a = 2.0f, b = 3.0f;
  std::vector<InputDesc> inputs = {{&a, {1}, 6}, {&b, {1}, 6}};
  ForwardResult result = rt.forward(inputs);
  REQUIRE(*static_cast<const float*>(result.outputs()[0].data) == 5.0f);
}

TEST_CASE("forward: planned inputs pass through (no staging)") {
  EtRuntime rt(ADD_PTE_PATH);
  float a = 2.0f, b = 3.0f;
  std::vector<InputDesc> inputs = {{&a, {1}, 6}, {&b, {1}, 6}};
  ProbeGuard guard;
  ForwardResult result = rt.forward(inputs);
  REQUIRE(*static_cast<const float*>(result.outputs()[0].data) == 5.0f);
  REQUIRE(guard.growCount() == 0);
  REQUIRE(guard.totalInputCount() == 2);
  REQUIRE(guard.stagedInputCount() == 0);  // pass-through: staged flag = 0
}

TEST_CASE("forward: unplanned inputs survive the caller buffer being freed (ASan lifetime)") {
  // Under the QA tree (ASan), any retained dereference of the freed heap buffers after delete[]
  // would trip. Staging copies into engine-owned slots, so this is clean by construction.
  EtRuntime rt(ADD_UNPLANNED_PTE_PATH);
  auto* a = new float[1]{2.0f};
  auto* b = new float[1]{3.0f};
  std::vector<InputDesc> inputs = {{a, {1}, 6}, {b, {1}, 6}};
  ForwardResult r1 = rt.forward(inputs);
  REQUIRE(*static_cast<const float*>(r1.outputs()[0].data) == 5.0f);
  delete[] a;
  delete[] b;
  float c = 10.0f, d = 7.0f;
  std::vector<InputDesc> inputs2 = {{&c, {1}, 6}, {&d, {1}, 6}};
  ForwardResult r2 = rt.forward(inputs2);
  REQUIRE(*static_cast<const float*>(r2.outputs()[0].data) == 17.0f);
}

TEST_CASE("intraop: a reset after a runtime exists is a logged no-op") {
  EtRuntime rt(ADD_PTE_PATH);
  const uint32_t cur = intraOpThreads();
  REQUIRE(setIntraOpThreads(1) == cur);   // refused: returns the current count
  REQUIRE(intraOpThreads() == cur);
}

// Per-model XNNPACK workspace sharing (spec 2026-08-08). add.pte is XNNPACK-delegated (its
// delegate id string is "XnnpackBackend"), so the runtime spec is actually consumed here.
TEST_CASE("workspace: every valid sharing mode loads") {
  REQUIRE_NOTHROW([] { EtRuntime rt(ADD_PTE_PATH, 0); }());  // Disabled
  REQUIRE_NOTHROW([] { EtRuntime rt(ADD_PTE_PATH, 1); }());  // PerModel
  REQUIRE_NOTHROW([] { EtRuntime rt(ADD_PTE_PATH, 2); }());  // Global
}

TEST_CASE("workspace: omitting the mode (-1) loads on the runtime default") {
  REQUIRE_NOTHROW([] { EtRuntime rt(ADD_PTE_PATH, -1); }());
  REQUIRE_NOTHROW([] { EtRuntime rt(ADD_PTE_PATH); }());  // same thing via the default argument
}

// THE WIRING PROOF. XnnpackBackendOptions::resolve_sharing_mode returns InvalidArgument for an
// out-of-range mode and XnnpackBackend::init propagates it, so the load fails -- but ONLY if the
// spec actually reached the XNNPACK backend. If the backend id or the option key were misspelled,
// Method::load would hand the backend an empty span, the mode would silently stay at the default,
// and this load would SUCCEED. There is no read-back API that would otherwise catch that: the
// backend's get_option returns the process-global value, not the per-model resolved one, and
// init does not log the mode it resolved. Do not delete this test as a mere bad-input check.
TEST_CASE("workspace: an out-of-range mode is rejected by the backend (proves the spec lands)") {
  REQUIRE_THROWS([] { EtRuntime rt(ADD_PTE_PATH, 99); }());
}

// conv.pte, not add.pte: delegating and allocating are different properties. add.pte is one node
// with external input and output, and lin129.pte lowers to a GEMM over statically packed weights;
// both delegate to XNNPACK and both grow the arena by exactly 0, so an assertion built on either
// would fail against a CORRECT build. Measured on the pinned runtime: add 0 after load and after
// forward, lin129 0 after forward, conv non-zero. A conv is the cheapest graph that allocates.
//
// Only the transition is asserted, never the value: the figure is a high-water mark including
// allocator alignment padding, so it is not stable across runs or platforms. It is also
// process-wide, which is why this reads ">0 after" rather than a delta -- an earlier case in this
// process may already have grown the arena.
//
// The pre-load 0 is deliberately NOT asserted. Catch2 runs every case in one process in
// registration order, so "reads 0 before the first delegated load" would pass or fail on which
// cases ran first, not on the accessor.
TEST_CASE("workspace: an XNNPACK-delegated conv grows the arena to a readable size") {
  // The forward() is load-bearing, not incidental. EtRuntime's ctor already calls load_forward(),
  // so delegate init has run by the time it returns -- yet the arena is still 0 at that point.
  // Measured on the pinned runtime: it grows on the first EXECUTE, not at delegate init, so the
  // upstream consumer doc's "created lazily during delegate init" does not describe this path.
  // Dropping the forward() turns this test red.
  EtRuntime rt(CONV_PTE_PATH);
  std::vector<float> x(1 * 3 * 16 * 16, 1.0f);
  std::vector<InputDesc> inputs = {{x.data(), {1, 3, 16, 16}, 6}};
  ForwardResult result = rt.forward(inputs);
  const int64_t bytes = xnnpackWorkspaceBytes();
  REQUIRE(bytes != -1);  // -1 is "get_option failed", i.e. the vendored patch is gone
  REQUIRE(bytes > 0);
}


// Detection must work on EVERY platform, including ones where the delegate is not linked -- that
// is precisely the case that needs a good error message. So this asserts metadata reading, not
// delegate availability, and is NOT gated on the backend being present.
TEST_CASE("backend detection: reports which delegate a .pte needs, without loading the method") {
  REQUIRE(pteUsesBackend(OPENVINO_TINY_PTE_PATH, "OpenvinoBackend"));
  REQUIRE_FALSE(pteUsesBackend(OPENVINO_TINY_PTE_PATH, "XnnpackBackend"));
  REQUIRE(pteUsesBackend(CONV_PTE_PATH, "XnnpackBackend"));
  REQUIRE_FALSE(pteUsesBackend(CONV_PTE_PATH, "OpenvinoBackend"));
}

TEST_CASE("backend detection: a missing file throws rather than reporting false") {
  // Reporting false would be indistinguishable from "this model needs no delegate", sending the
  // caller down the non-OpenVINO path and losing the real error until much later.
  REQUIRE_THROWS([] { pteUsesBackend("/nonexistent/definitely-not-here.pte", "OpenvinoBackend"); }());
}

// This is the test that protects the process. Without the guard, constructing an EtRuntime over an
// OpenVINO model with OPENVINO_LIB_PATH unset reaches OpenvinoBackend::init, whose dlopen runs
// under std::call_once and never retries -- so the FIRST bad attempt poisons every later attempt in
// this process, including correct ones. Catch2 runs all cases in one process, which is exactly the
// blast radius this prevents.
// Both cases below match the MESSAGE, not merely "something threw". The guard has three refusal
// branches and they are easy to confuse: until the delegate was linked into this binary, every
// OpenVINO case fell through the first branch ("this build does not provide") and a bare
// REQUIRE_THROWS passed without either OPENVINO_LIB_PATH branch ever running.
//
// Which branch is correct here is a genuine platform property, so it is selected rather than
// skipped: where the runtime tarball ships no delegate (linux-aarch64) the first branch IS the
// right answer, and asserting it there keeps that leg meaningful instead of vacuous.
#ifdef ET_OPENVINO_LINKED
#define ET_EXPECT_UNSET_MSG "OPENVINO_LIB_PATH is not set"
#define ET_EXPECT_DIRECTORY_MSG "does not name a readable file"
#else
#define ET_EXPECT_UNSET_MSG "this build does not provide"
#define ET_EXPECT_DIRECTORY_MSG "this build does not provide"
#endif

TEST_CASE("openvino: an unconfigured OPENVINO_LIB_PATH is refused before delegate init") {
  unsetEnvVar("OPENVINO_LIB_PATH");
  REQUIRE_THROWS_WITH(
      [] { EtRuntime rt(OPENVINO_TINY_PTE_PATH); }(),
      Catch::Matchers::ContainsSubstring(ET_EXPECT_UNSET_MSG));
}

TEST_CASE("openvino: OPENVINO_LIB_PATH pointing at a directory is refused") {
  // Upstream's documented top mistake. The error the delegate would otherwise produce mentions
  // LD_LIBRARY_PATH, which reads like it wants a directory. It does not -- it wants the file.
  setEnvVar("OPENVINO_LIB_PATH", "/tmp");
  REQUIRE_THROWS_WITH(
      [] { EtRuntime rt(OPENVINO_TINY_PTE_PATH); }(),
      Catch::Matchers::ContainsSubstring(ET_EXPECT_DIRECTORY_MSG));
  unsetEnvVar("OPENVINO_LIB_PATH");
}
