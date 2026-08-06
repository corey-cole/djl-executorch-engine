#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

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

TEST_CASE("staging: grow preserves the first min(old, new) bytes") {
  StagingSlot slot;
  auto* buf = static_cast<uint8_t*>(slot.ensure(64));
  for (size_t k = 0; k < 64; ++k) {
    buf[k] = static_cast<uint8_t>(k);
  }
  void* grown = slot.ensure(300);
  REQUIRE(grown != buf);
  REQUIRE(slot.capacity() >= 300);
  auto* g = static_cast<const uint8_t*>(grown);
  for (size_t k = 0; k < 64; ++k) {
    REQUIRE(g[k] == static_cast<uint8_t>(k));
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

TEST_CASE("staging: an input past its declared bound fires staging_grow (anomaly path)") {
  // The slot is sized at nbytes + kStagingPadding, so the probe's threshold is the declared bound
  // *plus* the padding: a 1-float slot holds 4 + 128 bytes rounded to 192, and only an input past
  // that grows it. 64 floats (256 + 128 = 384) clears it. The buffer really is 64 floats, so the
  // copy stays in bounds — ExecuTorch then rejects the shape (static-shape method), which is the
  // pre-existing contract and not what this case is asserting.
  EtRuntime rt(ADD_UNPLANNED_PTE_PATH);
  std::vector<float> big(64, 1.0f);
  float b = 3.0f;
  std::vector<InputDesc> inputs = {{big.data(), {64}, 6}, {&b, {1}, 6}};
  ProbeGuard guard;
  REQUIRE_THROWS(rt.forward(inputs));
  REQUIRE(guard.growCount() == 1);  // slot 0 only; slot 1 was within its bound
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
