#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

#include "dtype_size.h"
#include "et_log_level.h"
#include "et_probes.h"
#include "et_runtime.h"
#include "array_size_limits.h"
#include "staging.h"

using namespace measly::et;

#ifndef ADD_PTE_PATH
#define ADD_PTE_PATH "add.pte"
#endif

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
