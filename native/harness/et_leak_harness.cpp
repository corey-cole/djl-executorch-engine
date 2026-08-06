// JNI-free leak harness: load -> forward loop over EtRuntime, built under ASan/LSan.
// LSan reports unfreed allocations at process exit; a leak -> non-zero exit. Model-agnostic:
// tensor inputs are derived from methodMeta() and backed by 1-filled host buffers.
//
// W8: registers a counting USDT-dispatch handler and asserts exact staging counts, so a
// reallocate-every-call staging bug (the > vs >= slip) fails loudly instead of passing the
// aggregate LSan/RSS gates. The planned run (add.pte) additionally guards the runtime pin:
// grow==0 / staged_input==0 means ExecuTorch still copies planned inputs into its arena.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "dtype_size.h"
#include "et_probes.h"
#include "et_runtime.h"

using namespace measly::et;

namespace {

struct ProbeCounters {
  size_t grow = 0;
  size_t stagedInput = 0;
  size_t totalInput = 0;
};

ProbeCounters g_counts;

void countProbe(uint32_t id, uint64_t, uint64_t, uint64_t, uint64_t d) {
  switch (id) {
    case kProbeStagingGrow:
      ++g_counts.grow;
      break;
    case kProbeStagingInput:
      ++g_counts.totalInput;
      if (d != 0) {
        ++g_counts.stagedInput;
      }
      break;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const char* pte = (argc > 1) ? argv[1] : "add.pte";
  const int outerIters = (argc > 2) ? std::atoi(argv[2]) : 1000;
  const int forwardsPerLoad = (argc > 3) ? std::atoi(argv[3]) : 4;

  et_probe_set_handler(countProbe);

  MethodMeta meta;  // filled by the first load; identical across loads
  for (int it = 0; it < outerIters; ++it) {
    EtRuntime rt(pte);  // exercises load/destroy balance across iterations
    meta = rt.methodMeta();

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

  et_probe_clear_handler();

  size_t numTensorInputs = 0;
  size_t numUnplanned = 0;
  for (int i = 0; i < meta.numInputs; ++i) {
    if (meta.inputScalarTypes[i] >= 0) {
      ++numTensorInputs;
      if (meta.inputMemoryPlanned[i] == 0) {
        ++numUnplanned;
      }
    }
  }
  // Slots are sized in EtRuntime's constructor from TensorInfo::nbytes(), so staging_grow is an
  // anomaly signal, not a first-touch event: the correct steady-state count is zero, for any number
  // of loads or forwards. A realloc-per-forward slip, or a slot sized from the caller's shape
  // instead of the model's, both show up here immediately.
  const size_t expectGrow = 0;
  const size_t expectStagedInput =
      numUnplanned * static_cast<size_t>(outerIters) * static_cast<size_t>(forwardsPerLoad);
  const size_t expectTotalInput =
      numTensorInputs * static_cast<size_t>(outerIters) * static_cast<size_t>(forwardsPerLoad);

  if (g_counts.grow != expectGrow || g_counts.stagedInput != expectStagedInput ||
      g_counts.totalInput != expectTotalInput) {
    std::fprintf(stderr,
                 "STAGING ASSERT FAIL: %s grow=%zu (expected %zu) staged_input=%zu (expected %zu) "
                 "total_input=%zu (expected %zu)\n",
                 pte, g_counts.grow, expectGrow, g_counts.stagedInput, expectStagedInput,
                 g_counts.totalInput, expectTotalInput);
    return 1;
  }

  std::printf("OK: %d loads x %d forwards over %s grow=%zu (expected %zu) staged_input=%zu "
              "(expected %zu) total_input=%zu (expected %zu)\n",
              outerIters, forwardsPerLoad, pte, g_counts.grow, expectGrow, g_counts.stagedInput,
              expectStagedInput, g_counts.totalInput, expectTotalInput);
  return 0;
}
