// JNI-free timing harness: load once, then time load/cold/warm forwards over EtRuntime.
// Built Release, no sanitizers (ET_BUILD_BENCH), for the logging/devtools ship-or-not screen
// in docs/benchmarking.md. Model-agnostic: tensor inputs derived from methodMeta(), backed by
// 1-filled host buffers (mirrors et_leak_harness.cpp).
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "dtype_size.h"
#include "et_runtime.h"

using namespace measly::et;
using clock_type = std::chrono::steady_clock;

static double ms_since(clock_type::time_point start) {
  return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

int main(int argc, char** argv) {
  const char* pte = (argc > 1) ? argv[1] : "add.pte";
  const int iters = (argc > 2) ? std::atoi(argv[2]) : 1000;
  const int warmup = (argc > 3) ? std::atoi(argv[3]) : 100;

  try {
    // --- load (single sample; where ET_LOG is concentrated) ---
    auto t_load = clock_type::now();
    EtRuntime rt(pte);
    const double load_ms = ms_since(t_load);

    MethodMeta meta = rt.methodMeta();
    if (meta.numInputs <= 0) {
      std::fprintf(stderr, "et_timing: model %s has no inputs\n", pte);
      return 2;
    }

    // --- 1-filled inputs, kept alive for the whole run (borrowed by InputDesc) ---
    std::vector<std::vector<uint8_t>> buffers(meta.numInputs);
    std::vector<InputDesc> inputs;
    inputs.reserve(meta.numInputs);
    for (int i = 0; i < meta.numInputs; ++i) {
      if (meta.inputScalarTypes[i] < 0) continue;  // non-tensor input
      size_t count = 1;
      for (int64_t d : meta.inputShapes[i]) count *= static_cast<size_t>(d);
      size_t bytes = count * dtypeSize(meta.inputScalarTypes[i]);
      buffers[i].assign(bytes, 0);
      if (meta.inputScalarTypes[i] == 6) {  // float32 -> 1.0f
        float one = 1.0f;
        for (size_t b = 0; b + sizeof(float) <= bytes; b += sizeof(float))
          std::memcpy(buffers[i].data() + b, &one, sizeof(float));
      } else {
        std::memset(buffers[i].data(), 1, bytes);
      }
      inputs.push_back(InputDesc{buffers[i].data(), meta.inputShapes[i],
                                 meta.inputScalarTypes[i]});
    }

    volatile unsigned char sink = 0;  // defeat dead-code elimination of the forward loop

    // --- cold: first forward ---
    auto t_cold = clock_type::now();
    {
      ForwardResult r = rt.forward(inputs);
      auto outs = r.outputs();
      if (outs.empty()) {
        std::fprintf(stderr, "et_timing: forward produced no outputs\n");
        return 3;
      }
      sink ^= *static_cast<const unsigned char*>(outs[0].data);
    }
    const double cold_ms = ms_since(t_cold);

    // --- warmup (discarded) ---
    for (int i = 0; i < warmup; ++i) {
      ForwardResult r = rt.forward(inputs);
      sink ^= *static_cast<const unsigned char*>(r.outputs()[0].data);
    }

    // --- timed warm loop: min / mean / max ---
    double warm_min = 0, warm_max = 0, warm_sum = 0;
    for (int i = 0; i < iters; ++i) {
      auto t = clock_type::now();
      ForwardResult r = rt.forward(inputs);
      sink ^= *static_cast<const unsigned char*>(r.outputs()[0].data);
      double e = ms_since(t);
      warm_sum += e;
      if (i == 0 || e < warm_min) warm_min = e;
      if (i == 0 || e > warm_max) warm_max = e;
    }
    const double warm_mean = iters > 0 ? warm_sum / iters : 0.0;

    std::printf("et_timing: model=%s iters=%d warmup=%d load_ms=%.3f cold_ms=%.3f "
                "warm_min_ms=%.3f warm_mean_ms=%.3f warm_max_ms=%.3f sink=%d\n",
                pte, iters, warmup, load_ms, cold_ms, warm_min, warm_mean, warm_max,
                static_cast<int>(sink));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "et_timing: error: %s\n", e.what());
    return 1;
  }
}
