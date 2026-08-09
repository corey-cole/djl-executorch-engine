// Threaded stress harness for ASan/LSan: N threads, each with its OWN EtRuntime over the same .pte,
// forwarding concurrently for a fixed wall-clock duration and asserting bitwise-identical outputs.
//
// What this covers that et_scaling_harness does not: correctness under sanitizers, rather than
// throughput. ASan catches use-after-free on teardown races and overflow from a clobbered workspace;
// LSan catches per-thread leaks; the bitwise check catches silent corruption that leaves the process
// alive and the numbers merely wrong.
//
// Layer 2 of the oracle only -- no golden digests. The JVM arm (StressGateIT) owns the golden
// comparison; pulling a JSON parser into this JNIEnv-free harness would not pay for itself.
//
// Sharing mode arrives through EtRuntime's per-load constructor argument, NOT the process-global
// set_option path that et_scaling_harness uses. That matches how the engine actually does it
// (native/core/et_runtime.cpp) and keeps behaviour independent of load order.
//
//   ET_SHARING_MODE unset -> runtime default (Global for our pin)
//   ET_SHARING_MODE=0|1|2 -> Disabled | PerModel | Global
//
// Usage: et_stress_harness <pte> <threads> <seconds>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "dtype_size.h"
#include "et_runtime.h"

using namespace measly::et;
using clock_type = std::chrono::steady_clock;

namespace {

int env_int(const char* name, int fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return fallback;
  return std::atoi(v);
}

// Per-thread input buffers. Each thread owns its own: InputDesc borrows the pointer, so sharing one
// buffer across threads would be a different (and less interesting) experiment.
struct Workload {
  std::vector<std::vector<uint8_t>> buffers;
  std::vector<InputDesc> inputs;
};

// Fills every f32 input with a deterministic ramp (`(float) e * kRamp + v`). This arm is oracle
// layer 2 ONLY — it compares against its own in-process reference, never the goldens — so kRamp
// only needs to be A deterministic fill; it is deliberately NOT kept in sync with
// stress_golden.json's config.ramp. Non-f32 inputs are byte-filled; the stress model has none,
// but the harness should not silently produce garbage if pointed at another .pte.
Workload buildWorkload(const MethodMeta& meta, float v) {
  constexpr float kRamp = 1e-5f;  // deterministic fill only; see the comment above
  Workload w;
  w.buffers.resize(meta.numInputs);
  w.inputs.reserve(meta.numInputs);
  for (int i = 0; i < meta.numInputs; ++i) {
    if (meta.inputScalarTypes[i] < 0) continue;  // non-tensor input
    size_t count = 1;
    for (int64_t d : meta.inputShapes[i]) count *= static_cast<size_t>(d);
    size_t bytes = count * dtypeSize(meta.inputScalarTypes[i]);
    w.buffers[i].assign(bytes, 0);
    if (meta.inputScalarTypes[i] == 6) {  // float32
      for (size_t e = 0; e < count; ++e) {
        float x = static_cast<float>(e) * kRamp + v;
        std::memcpy(w.buffers[i].data() + e * sizeof(float), &x, sizeof(float));
      }
    } else {
      std::memset(w.buffers[i].data(), 1, bytes);
    }
    w.inputs.push_back(
        InputDesc{w.buffers[i].data(), meta.inputShapes[i], meta.inputScalarTypes[i]});
  }
  return w;
}

// Flattens every output into one byte vector so a whole forward can be compared with one memcmp.
std::vector<uint8_t> capture(const ForwardResult& r) {
  std::vector<uint8_t> out;
  for (const OutputView& v : r.outputs()) {
    const uint8_t* p = static_cast<const uint8_t*>(v.data);
    out.insert(out.end(), p, p + v.nbytes);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: et_stress_harness <pte> <threads> <seconds>\n");
    return 4;
  }
  const char* pte = argv[1];
  const int threads = std::atoi(argv[2]);
  const int seconds = std::atoi(argv[3]);
  if (threads <= 0 || seconds <= 0) {
    std::fprintf(stderr, "threads and seconds must both be positive\n");
    return 4;
  }
  const int sharing_mode = env_int("ET_SHARING_MODE", -1);

  // Two steering values, landing in different buckets, so the data-dependent gather is exercised.
  const float kValues[2] = {0.0f, 0.99f};

  try {
    // Reference outputs, captured single-threaded before any worker starts.
    std::vector<std::vector<uint8_t>> reference;
    {
      EtRuntime rt(pte, sharing_mode);
      MethodMeta meta = rt.methodMeta();
      for (float v : kValues) {
        Workload w = buildWorkload(meta, v);
        reference.push_back(capture(rt.forward(w.inputs)));
      }
    }

    std::atomic<bool> stop{false};
    std::atomic<long long> forwards{0};
    std::atomic<int> diverged{0};
    std::atomic<int> errors{0};
    std::barrier sync(threads + 1);

    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (int t = 0; t < threads; ++t) {
      workers.emplace_back([&, t]() {
        // Load and warm up BEFORE the barrier so construction cost stays out of the timed region.
        // The barrier is arrived at on EVERY path, including the load-failure path: a worker that
        // bails without arriving would strand the main thread's wait forever.
        std::unique_ptr<EtRuntime> rt;
        std::vector<Workload> loads;
        bool ready = false;
        try {
          rt = std::make_unique<EtRuntime>(pte, sharing_mode);
          MethodMeta meta = rt->methodMeta();
          for (float v : kValues) loads.push_back(buildWorkload(meta, v));
          ready = true;
        } catch (const std::exception& e) {
          std::fprintf(stderr, "et_stress: thread %d failed to load: %s\n", t, e.what());
          errors.fetch_add(1);
          stop.store(true);
        }
        sync.arrive_and_wait();
        if (!ready) return;
        try {
          while (!stop.load(std::memory_order_relaxed)) {
            for (size_t c = 0; c < loads.size(); ++c) {
              std::vector<uint8_t> got = capture(rt->forward(loads[c].inputs));
              if (got.size() != reference[c].size() ||
                  std::memcmp(got.data(), reference[c].data(), got.size()) != 0) {
                std::fprintf(stderr,
                             "et_stress: thread %d diverged bitwise on case %zu\n", t, c);
                diverged.fetch_add(1);
                stop.store(true);
                return;
              }
              forwards.fetch_add(1, std::memory_order_relaxed);
            }
          }
        } catch (const std::exception& e) {
          std::fprintf(stderr, "et_stress: thread %d threw: %s\n", t, e.what());
          errors.fetch_add(1);
          stop.store(true);
        }
      });
    }

    sync.arrive_and_wait();
    auto t0 = clock_type::now();
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    stop.store(true);
    for (auto& w : workers) w.join();
    double wall = std::chrono::duration<double>(clock_type::now() - t0).count();

    std::printf("et_stress: %lld forwards, %d threads, %.1fs, mode=%d, diverged=%d, errors=%d\n",
                forwards.load(), threads, wall, sharing_mode, diverged.load(), errors.load());
    if (diverged.load() > 0) return 1;
    if (errors.load() > 0) return 2;
    if (forwards.load() == 0) {
      std::fprintf(stderr, "et_stress: no forwards ran\n");
      return 2;
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "et_stress: %s\n", e.what());
    return 2;
  }
}
