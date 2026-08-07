// JNI-free thread-scaling harness: N threads, each with its OWN EtRuntime over the SAME .pte,
// running forward() concurrently. Measures aggregate throughput vs thread count.
//
// What it exists to answer: ExecuTorch 1.3.1 ships XNNPACK workspace sharing ON by default
// (EXECUTORCH_XNNPACK_SHARED_WORKSPACE, and our runtime tarball is built `--preset linux` which
// does not override it => WorkspaceSharingMode::Global). XNNWorkspace::acquire() takes a mutex
// unless disable_locking() was called, and XNNWorkspaceManager only calls that in the Disabled
// mode. So every XNNPACK delegate call in the process may be serializing on one lock today --
// which would cap this engine's documented "one Model/Predictor per thread" story at 1x.
//
// Both arms come from one binary: the sharing mode is a RUNTIME backend option in 1.3.1, not just
// a build flag, so no runtime rebuild is needed. Set ET_SHARING_MODE before the models load:
//
//   ET_SHARING_MODE unset  -> leave the runtime default alone (Global, today's shipped behavior)
//   ET_SHARING_MODE=2      -> Global   (all delegate instances share one workspace + its lock)
//   ET_SHARING_MODE=1      -> PerModel (one workspace per program)
//   ET_SHARING_MODE=0      -> Disabled (workspace per delegate instance; locking disabled)
//   ET_WEIGHT_CACHE=0|1    -> optional second knob; the cache adds its own process-wide mutex held
//                             for the whole of execute(), so expect it to be strictly worse here.
//
// ET_INTRAOP_THREADS is the knob that makes any of the above legible. XNNPACK here runs on
// ExecuTorch's shared pthreadpool (libxnnpack_backend.a has an undefined reference to
// extension::threadpool::get_pthreadpool), sized to the performance-core count by default. One
// forward therefore already saturates the machine, so aggregate throughput is flat in app-thread
// count for reasons that have nothing to do with any mutex -- the saturation masks the lock
// entirely. Set ET_INTRAOP_THREADS=1 to take intra-op parallelism out of the picture; only then
// does app-thread scaling measure what this harness is named for.
//
// The signature of a workspace-lock ceiling, with ET_INTRAOP_THREADS=1: forwards_per_sec ~flat as
// threads rise, per-thread mean latency growing ~linearly. If instead throughput scales with cores
// under the default sharing mode, the lock is not a ceiling for this workload.
//
// cpu_s/wall gives the average parallelism actually achieved over the timed region -- report it,
// because it is what distinguishes "serialized on a lock" (parallelism ~1 with many threads) from
// "already saturated" (parallelism ~cores at one thread).
//
// Model loads happen sequentially on the main thread and are NOT in the timed region -- this
// measures the execute-path ceiling, not load contention. load_ms and peak RSS are reported
// alongside since Disabled mode trades memory (an activation arena per instance) for that ceiling.
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

#include <sys/resource.h>

#include <executorch/extension/threadpool/threadpool.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/platform/runtime.h>

#include "dtype_size.h"
#include "et_runtime.h"

using namespace measly::et;
using clock_type = std::chrono::steady_clock;

namespace {

double ms_between(clock_type::time_point a, clock_type::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

int env_int(const char* name, int fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return fallback;
  return std::atoi(v);
}

// Process CPU seconds (user+sys, all threads) so far. Differenced across the timed region to get
// the average parallelism actually achieved: cpu_s / (wall_ms/1000).
double cpu_seconds() {
  struct rusage ru{};
  if (getrusage(RUSAGE_SELF, &ru) != 0) return 0.0;
  return (ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1e6) +
         (ru.ru_stime.tv_sec + ru.ru_stime.tv_usec / 1e6);
}

// Peak resident set, in KiB. Linux-only (VmHWM); returns 0 elsewhere or on read failure. Reported
// because the Disabled arm buys throughput by giving every delegate instance its own arena, and
// that cost has to be visible in the same line as the win.
long peak_rss_kb() {
  std::FILE* f = std::fopen("/proc/self/status", "r");
  if (f == nullptr) return 0;
  char line[256];
  long kb = 0;
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    if (std::strncmp(line, "VmHWM:", 6) == 0) {
      kb = std::strtol(line + 6, nullptr, 10);
      break;
    }
  }
  std::fclose(f);
  return kb;
}

// Applies the XNNPACK backend options requested by the environment. Must run before any Module
// loads: the backend reads these at init() and the keys are documented as affecting only
// subsequently loaded models.
//
// The option keys are string literals rather than the constants from
// backends/xnnpack/runtime/XNNPACKBackend.h because the runtime tarball installs runtime/backend/
// but NOT backends/xnnpack/runtime/ -- that header is not available to consumers. If the runtime
// dist starts shipping it, swap these for xnnpack_backend_key / workspace_sharing_mode_option_key /
// weight_cache_option_key.
bool apply_backend_options(int sharing_mode, int weight_cache) {
  if (sharing_mode < 0 && weight_cache < 0) return true;

  executorch::runtime::BackendOptions<2> opts;
  if (sharing_mode >= 0) opts.set_option("workspace_sharing_mode", sharing_mode);
  if (weight_cache >= 0) opts.set_option("weight_cache_enabled", weight_cache != 0);

  auto err = executorch::ET_RUNTIME_NAMESPACE::set_option("XnnpackBackend", opts.view());
  if (err != executorch::runtime::Error::Ok) {
    std::fprintf(stderr, "et_scaling: set_option(XnnpackBackend) failed: 0x%x\n",
                 static_cast<unsigned int>(err));
    return false;
  }
  return true;
}

// 1-filled host buffers for every tensor input, derived from methodMeta(). Per thread, since
// InputDesc borrows and a shared buffer would be a different experiment. Mirrors
// et_timing_harness.cpp.
struct Workload {
  std::vector<std::vector<uint8_t>> buffers;
  std::vector<InputDesc> inputs;
};

Workload buildWorkload(const MethodMeta& meta) {
  Workload w;
  w.buffers.resize(meta.numInputs);
  w.inputs.reserve(meta.numInputs);
  for (int i = 0; i < meta.numInputs; ++i) {
    if (meta.inputScalarTypes[i] < 0) continue;  // non-tensor input
    size_t count = 1;
    for (int64_t d : meta.inputShapes[i]) count *= static_cast<size_t>(d);
    size_t bytes = count * dtypeSize(meta.inputScalarTypes[i]);
    w.buffers[i].assign(bytes, 0);
    if (meta.inputScalarTypes[i] == 6) {  // float32 -> 1.0f
      float one = 1.0f;
      for (size_t b = 0; b + sizeof(float) <= bytes; b += sizeof(float))
        std::memcpy(w.buffers[i].data() + b, &one, sizeof(float));
    } else {
      std::memset(w.buffers[i].data(), 1, bytes);
    }
    w.inputs.push_back(InputDesc{w.buffers[i].data(), meta.inputShapes[i],
                                 meta.inputScalarTypes[i]});
  }
  return w;
}

}  // namespace

int main(int argc, char** argv) {
  const char* pte = (argc > 1) ? argv[1] : "add.pte";
  const int threads = (argc > 2) ? std::atoi(argv[2]) : 1;
  const int iters = (argc > 3) ? std::atoi(argv[3]) : 1000;
  const int warmup = (argc > 4) ? std::atoi(argv[4]) : 100;

  if (threads <= 0 || iters <= 0 || warmup < 0) {
    std::fprintf(stderr, "usage: et_scaling_harness <pte> <threads> <iters> <warmup>\n");
    return 4;
  }

  const int sharing_mode = env_int("ET_SHARING_MODE", -1);
  const int weight_cache = env_int("ET_WEIGHT_CACHE", -1);
  const int intraop = env_int("ET_INTRAOP_THREADS", 0);  // 0 = leave the pool at its default size

  try {
    // Options must be applied before the first Module load, and set_option needs the backend
    // registry populated -- runtime_init() first. (EtRuntime's Module ctor would do this, but by
    // then it is too late to influence that model's delegate init.)
    executorch::runtime::runtime_init();
    if (!apply_backend_options(sharing_mode, weight_cache)) return 5;

    // Resize the shared intra-op pool before anything runs on it. _unsafe_reset_threadpool is
    // "unsafe" only in the sense that it must not race with in-flight work -- nothing has been
    // submitted yet at this point.
    if (intraop > 0 &&
        !executorch::extension::threadpool::get_threadpool()->_unsafe_reset_threadpool(
            static_cast<uint32_t>(intraop))) {
      std::fprintf(stderr, "et_scaling: failed to resize intra-op pool to %d\n", intraop);
      return 6;
    }
    const size_t intraop_actual =
        executorch::extension::threadpool::get_threadpool()->get_thread_count();

    // --- sequential loads, outside the timed region ---
    auto t_load = clock_type::now();
    std::vector<std::unique_ptr<EtRuntime>> runtimes;
    runtimes.reserve(threads);
    for (int t = 0; t < threads; ++t) runtimes.push_back(std::make_unique<EtRuntime>(pte));
    const double load_ms = ms_between(t_load, clock_type::now());

    MethodMeta meta = runtimes[0]->methodMeta();
    if (meta.numInputs <= 0) {
      std::fprintf(stderr, "et_scaling: model %s has no inputs\n", pte);
      return 2;
    }

    std::vector<Workload> workloads;
    workloads.reserve(threads);
    for (int t = 0; t < threads; ++t) workloads.push_back(buildWorkload(meta));

    // Per-thread results. Sized up front so no thread touches the vector's control block.
    std::vector<clock_type::time_point> starts(threads), ends(threads);
    std::vector<double> means(threads, 0.0), maxes(threads, 0.0);
    std::atomic<unsigned> sink{0};  // defeat dead-code elimination of the forward loop
    std::atomic<bool> failed{false};

    // Warmup runs before the barrier, so every thread enters the timed region with its delegate
    // already initialized and its first-call costs paid.
    std::barrier start_gate(threads);

    auto body = [&](int t) {
      try {
        EtRuntime& rt = *runtimes[t];
        const auto& inputs = workloads[t].inputs;
        unsigned local_sink = 0;

        for (int i = 0; i < warmup; ++i) {
          ForwardResult r = rt.forward(inputs);
          local_sink ^= *static_cast<const unsigned char*>(r.outputs()[0].data);
        }

        start_gate.arrive_and_wait();

        starts[t] = clock_type::now();
        double sum = 0, mx = 0;
        for (int i = 0; i < iters; ++i) {
          auto t0 = clock_type::now();
          ForwardResult r = rt.forward(inputs);
          local_sink ^= *static_cast<const unsigned char*>(r.outputs()[0].data);
          double e = ms_between(t0, clock_type::now());
          sum += e;
          if (i == 0 || e > mx) mx = e;
        }
        ends[t] = clock_type::now();
        means[t] = sum / iters;
        maxes[t] = mx;
        sink.fetch_add(local_sink, std::memory_order_relaxed);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "et_scaling: thread %d: %s\n", t, e.what());
        failed.store(true);
        // Deliberately no arrive_and_drop() recovery: a throw before the gate deadlocks the run
        // rather than silently reporting a short-handed measurement. Any failure here is a bug in
        // the harness or an unloadable model, both of which want to be loud.
        throw;
      }
    };

    // Straddles warmup as well as the timed region; warmup is a fixed small fraction of iters, and
    // getrusage has no per-region scope. Keep WARMUP << ITERS for the parallelism figure to mean
    // what it says.
    const double cpu_before = cpu_seconds();

    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (int t = 0; t < threads; ++t) pool.emplace_back(body, t);
    for (auto& th : pool) th.join();
    if (failed.load()) return 1;

    const double cpu_s = cpu_seconds() - cpu_before;

    // Wall clock of the concurrent region: first thread past the gate to last thread done. Using
    // the span rather than any single thread's elapsed keeps the throughput number honest when
    // threads finish unevenly (which is exactly what a contended lock produces).
    auto first_start = starts[0];
    auto last_end = ends[0];
    double mean_of_means = 0, worst_max = 0;
    for (int t = 0; t < threads; ++t) {
      if (starts[t] < first_start) first_start = starts[t];
      if (ends[t] > last_end) last_end = ends[t];
      mean_of_means += means[t];
      if (maxes[t] > worst_max) worst_max = maxes[t];
    }
    mean_of_means /= threads;
    const double wall_ms = ms_between(first_start, last_end);
    const double total_iters = static_cast<double>(threads) * iters;
    const double throughput = wall_ms > 0 ? total_iters / (wall_ms / 1000.0) : 0.0;

    const char* mode_str = sharing_mode < 0 ? "default" :
                           sharing_mode == 0 ? "disabled" :
                           sharing_mode == 1 ? "permodel" :
                           sharing_mode == 2 ? "global" : "invalid";
    const char* cache_str = weight_cache < 0 ? "default" : (weight_cache ? "on" : "off");

    const double parallelism = wall_ms > 0 ? cpu_s / (wall_ms / 1000.0) : 0.0;

    std::printf("et_scaling: model=%s threads=%d intraop=%zu iters=%d warmup=%d sharing=%s "
                "weight_cache=%s load_ms=%.3f wall_ms=%.3f forwards_per_sec=%.1f "
                "per_thread_mean_ms=%.4f worst_max_ms=%.4f cpu_s=%.3f parallelism=%.2f "
                "peak_rss_kb=%ld sink=%u\n",
                pte, threads, intraop_actual, iters, warmup, mode_str, cache_str, load_ms, wall_ms,
                throughput, mean_of_means, worst_max, cpu_s, parallelism, peak_rss_kb(),
                sink.load());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "et_scaling: error: %s\n", e.what());
    return 1;
  }
}
