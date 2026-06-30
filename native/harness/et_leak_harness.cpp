// JNI-free leak harness: load -> forward loop over EtRuntime, built under ASan/LSan.
// LSan reports unfreed allocations at process exit; a leak -> non-zero exit. Model-agnostic:
// tensor inputs are derived from methodMeta() and backed by 1-filled host buffers.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "et_runtime.h"

using namespace measly::et;

static size_t dtypeSize(int8_t st) {
  switch (st) {
    case 6:           // FLOAT32
    case 3: return 4;  // INT32
    case 7:           // FLOAT64
    case 4: return 8;  // INT64
    case 0:           // UINT8
    case 1:           // INT8
    case 11: return 1;  // BOOL
    default: return 4;
  }
}

int main(int argc, char** argv) {
  const char* pte = (argc > 1) ? argv[1] : "add.pte";
  const int outerIters = (argc > 2) ? std::atoi(argv[2]) : 1000;
  const int forwardsPerLoad = 4;

  for (int it = 0; it < outerIters; ++it) {
    EtRuntime rt(pte);  // exercises load/destroy balance across iterations
    MethodMeta meta = rt.methodMeta();

    std::vector<std::vector<uint8_t>> buffers(meta.numInputs);
    std::vector<InputDesc> inputs;
    inputs.reserve(meta.numInputs);
    for (int i = 0; i < meta.numInputs; ++i) {
      if (meta.inputScalarTypes[i] < 0) {
        continue;  // non-tensor input: forward() only consumes tensor inputs
      }
      size_t count = 1;
      for (int64_t d : meta.inputShapes[i]) {
        count *= static_cast<size_t>(d);
      }
      size_t bytes = count * dtypeSize(meta.inputScalarTypes[i]);
      buffers[i].assign(bytes, 0);
      if (meta.inputScalarTypes[i] == 6) {  // fill float32 with 1.0f
        float one = 1.0f;
        for (size_t b = 0; b + sizeof(float) <= bytes; b += sizeof(float)) {
          std::memcpy(buffers[i].data() + b, &one, sizeof(float));
        }
      } else {
        std::memset(buffers[i].data(), 1, bytes);
      }
      inputs.push_back(InputDesc{buffers[i].data(), meta.inputShapes[i],
                                 meta.inputScalarTypes[i]});
    }

    for (int f = 0; f < forwardsPerLoad; ++f) {  // exercises per-forward allocations
      ForwardResult result = rt.forward(inputs);
      auto outs = result.outputs();
      if (!outs.empty()) {
        volatile const unsigned char first =
            *static_cast<const unsigned char*>(outs[0].data);  // touch the view
        (void)first;
      }
    }
  }

  std::printf("OK: %d loads x %d forwards over %s\n", outerIters, forwardsPerLoad, pte);
  return 0;
}
