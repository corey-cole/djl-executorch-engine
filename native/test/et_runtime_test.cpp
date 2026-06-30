#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

#include "et_log_level.h"
#include "et_runtime.h"

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
