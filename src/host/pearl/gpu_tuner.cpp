#include "gpu_tuner.h"
#include "hashrate_metrics.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "cuda_compat.h"

#if !defined(PROP_MINER_HOST_ONLY_TESTS)
#include "gpu_worker.h"
#include "job_key.h"
#include "pearl_capi_wrapper.h"
#include "rtx5090_profile.h"
#include "sigma_context.h"
#endif

namespace {
    struct EnvGuard {
        const char* name_ = nullptr;
        const char* old_ = nullptr;
        bool owned_ = false;

        void set(const char* name, const char* value) {
            name_ = name;
            old_ = std::getenv(name);
            if (value) {
                std::string s = std::string(name) + "=" + value;
                char* p = new char[s.size() + 1];
                std::memcpy(p, s.data(), s.size() + 1);
                putenv(p);
                owned_ = true;
            } else {
                unsetenv(name);
            }
        }
        ~EnvGuard() {
            if (!name_) return;
            if (old_) {
                std::string s = std::string(name_) + "=" + old_;
                char* p = new char[s.size() + 1];
                std::memcpy(p, s.data(), s.size() + 1);
                putenv(p);
            } else {
                unsetenv(name_);
            }
            (void)owned_;
        }
    };
}

namespace pearl {

namespace {
    uint64_t now_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
}

GpuTuner::GpuTuner(int device_index) : device_index_(device_index) {}

MiningConfig GpuTuner::shape_for_vram(size_t free_bytes,
                                      int major, int minor) const {
    MiningConfig cfg;
    const bool is_blackwell = (major == 12);
    const bool is_ada = (major == 8 && minor == 9);
    const bool is_ampere = (major == 8 && (minor == 6 || minor == 0));

    // Candidate (M, N) pairs, sorted from largest (most compute intensity) to
    // smallest.  K is always 128 for Pearl V2 with int7 quant.
    std::vector<std::pair<int, int>> candidates;
    if (is_blackwell || is_ada) {
        candidates = {
            {8192, 262144},
            {8192, 131072},
            {8192, 65536},
            {8192, 32768},
            {4096, 32768},
            {4096, 16384},
            {2048, 16384},
        };
    } else if (is_ampere) {
        candidates = {
            {8192, 32768},
            {4096, 32768},
            {4096, 16384},
            {2048, 16384},
        };
    } else {
        candidates = {
            {4096, 16384},
            {2048, 16384},
            {2048, 8192},
        };
    }

    for (const auto& mn : candidates) {
        int M = mn.first, N = mn.second, K = 128;
        int64_t a = int64_t(M) * K;
        int64_t b = int64_t(K) * N;
        int64_t c = int64_t(M) * N;
        // ~25% headroom for leaf CVs, noise matrices, workspace, alignment.
        int64_t bytes = a + b + 4 * c;
        bytes = bytes + bytes / 4;
        if (static_cast<size_t>(bytes) < free_bytes) {
            cfg.m = M;
            cfg.n = N;
            cfg.k = K;
            cfg.r = 128;
            break;
        }
    }

    // Default tile-shape knobs; CMake / Makefile can override for Blackwell.
    cfg.bM = 128; cfg.bN = 256; cfg.bK = 128;
    return cfg;
}

double GpuTuner::benchmark_config(const MiningConfig& cfg,
                                  int batch,
                                  bool use_graph,
                                  int cluster_m,
                                  int carveout_percent,
                                  double seconds,
                                  int min_iters) const {
#if defined(PROP_MINER_HOST_ONLY_TESTS)
    (void)cfg; (void)batch; (void)use_graph; (void)cluster_m;
    (void)carveout_percent; (void)seconds; (void)min_iters;
    return 0.0;
#else
    try {
        EnvGuard cluster_env;
        if (cluster_m > 1) {
            cluster_env.set("PEARL_GEMM_CONSUMER_CLUSTER_M",
                            std::to_string(cluster_m).c_str());
        }
        EnvGuard carveout_env;
        if (carveout_percent >= 0) {
            carveout_env.set("PEARL_GEMM_CONSUMER_CARVEOUT",
                             std::to_string(carveout_percent).c_str());
        }
        EnvGuard graph_env;
        if (!use_graph) {
            graph_env.set("PROPMINER_BENCH_NO_GRAPH", "1");
        }

        GpuWorker worker(device_index_, 0, cfg, /*sink=*/nullptr);
        worker.set_matmuls_per_poll(batch);

        Job job;
        job.sigma.fill(0xab);
        job.b_seed.fill(0xcd);
        job.config = cfg;
        job.job_key = derive_job_key(job.sigma, cfg);
        job.target_nbits = 0;          // impossible target -> no share triggers
        job.audit_k = 0;

        auto ctx = std::make_shared<SigmaContext>(job, cfg);
        worker.set_sigma(ctx);

        uint64_t t0 = now_ms();
        worker.start();
        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<int>(seconds * 1000)));
        worker.stop();
        uint64_t t1 = now_ms();

        const int completed = static_cast<int>(worker.total_iters());
        if (min_iters > 0 && completed < min_iters) {
            std::fprintf(stderr,
                "[batch-tune] GPU %d: batch=%d incomplete (%d < %d iters in %.0fs)\n",
                device_index_, batch, completed, min_iters, seconds);
            return 0.0;
        }

        double elapsed = (t1 - t0) / 1000.0;
        if (elapsed <= 0.0 || completed <= 0) return 0.0;

        double ops_per_iter = static_cast<double>(cfg.m) * cfg.n * cfg.k * 2.0;
        double total_ops = ops_per_iter * completed;
        return total_ops / elapsed;
    } catch (...) {
        return 0.0;
    }
#endif
}

double GpuTuner::benchmark_stable(const MiningConfig& cfg,
                                  int batch,
                                  bool use_graph,
                                  int cluster_m,
                                  int carveout_percent,
                                  double seconds,
                                  int repeats,
                                  int min_iters) const {
    std::vector<double> rates;
    rates.reserve(repeats);
    for (int i = 0; i < repeats; ++i) {
        double r = benchmark_config(cfg, batch, use_graph, cluster_m,
                                    carveout_percent, seconds, min_iters);
        if (r > 0.0) rates.push_back(r);
    }
    if (rates.empty()) return 0.0;
    if (rates.size() > 2) {
        std::sort(rates.begin(), rates.end());
        rates.erase(rates.begin());
    }
    double sum = 0.0;
    for (double r : rates) sum += r;
    return sum / rates.size();
}

TuningResult GpuTuner::tune_shapes(const std::vector<MiningConfig>& candidates,
                                   double seconds_per_candidate,
                                   int repeats) {
    TuningResult best;
    best.hashrate = 0.0;
    best.repeats = repeats;

    // Sweep batch sizes: smaller values reduce share latency, larger values
    // amortize launch overhead and CPU wake-ups.  We use a denser grid than
    // before because the default tuning window is now longer.
    const std::vector<int> batches = {
        1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48
    };

    // CUDA graphs almost always win on Pearl because the kernel body is tiny
    // and launch overhead dominates; still keep the option for odd drivers.
    const std::vector<bool> graph_opts = {true, false};

    for (const MiningConfig& base : candidates) {
        std::vector<int> cluster_ms = {1};
        std::vector<int> carveouts = {-1};
        const bool is_blackwell = (base.bM == 128 && base.bN == 256);
        if (is_blackwell) {
            cluster_ms = {1, 2, 4};
            carveouts = {-1, 50, 80};
        }

        for (int batch : batches) {
            for (bool use_graph : graph_opts) {
                for (int cluster_m : cluster_ms) {
                    for (int carveout : carveouts) {
                        double rate = benchmark_stable(
                            base, batch, use_graph, cluster_m, carveout,
                            seconds_per_candidate, repeats);
                        if (rate > best.hashrate) {
                            best.config = base;
                            best.batch_size = batch;
                            best.use_graph = use_graph;
                            best.cluster_m = cluster_m;
                            best.carveout_percent = carveout;
                            best.hashrate = rate;
                        }
                    }
                }
            }
        }
    }

    return best;
}

TuningResult GpuTuner::tune_mine_batch(const MiningConfig& cfg,
                                       double seconds_per_candidate,
                                       int repeats) {
    TuningResult best;
    best.config = cfg;
    best.batch_size = Rtx5090Profile::kDefaultMineBatch;
    best.use_graph = true;
    best.hashrate = 0.0;
    best.repeats = repeats;

#if !defined(PROP_MINER_HOST_ONLY_TESTS)
    int cluster_m = 1;
    if (const char* env = std::getenv("PEARL_GEMM_CONSUMER_CLUSTER_M")) {
        cluster_m = std::max(1, std::atoi(env));
    }
    int carveout = -1;
    if (const char* env = std::getenv("PEARL_GEMM_CONSUMER_CARVEOUT")) {
        carveout = std::atoi(env);
    }

    for (int batch : Rtx5090Profile::kMineBatchCandidates) {
        if (batch > Rtx5090Profile::kMaxMineBatch) continue;
        const double timeout =
            std::max(seconds_per_candidate, 10.0 * static_cast<double>(batch));
        const double rate = benchmark_stable(
            cfg, batch, true, cluster_m, carveout, timeout, repeats, batch);
        const double tmad = tmad_from_tuner_ops_per_sec(rate);
        std::fprintf(stderr,
            "[batch-tune] GPU %d: M=%d N=%d batch=%d -> %.2f TMAD/s (pool TH/s) "
            "(%.0fs window)\n",
            device_index_, cfg.m, cfg.n, batch, tmad, timeout);
        if (rate > best.hashrate) {
            best.batch_size = batch;
            best.hashrate = rate;
        }
    }

    std::fprintf(stderr,
        "[batch-tune] GPU %d: selected batch=%d graph=yes cluster_m=%d -> %.2f TMAD/s\n",
        device_index_, best.batch_size, cluster_m,
        tmad_from_tuner_ops_per_sec(best.hashrate));
#else
    (void)seconds_per_candidate;
#endif

    return best;
}

TuningResult GpuTuner::tune_cluster_sweep(const MiningConfig& cfg,
                                          int batch,
                                          double seconds_per_candidate,
                                          int repeats) {
    TuningResult best;
    best.config = cfg;
    best.batch_size = batch;
    best.use_graph = true;
    best.cluster_m = 1;
    best.hashrate = 0.0;
    best.repeats = repeats;

#if !defined(PROP_MINER_HOST_ONLY_TESTS)
    int carveout = -1;
    if (const char* env = std::getenv("PEARL_GEMM_CONSUMER_CARVEOUT")) {
        carveout = std::atoi(env);
    }

    // Kernel accepts cluster_m in {1, 2, 4} only; 3 falls back to default 1.
    const int clusters[] = {1, 2, 4};
    for (int cluster_m : clusters) {
        const double timeout =
            std::max(seconds_per_candidate, 10.0 * static_cast<double>(batch));
        const double rate = benchmark_stable(
            cfg, batch, true, cluster_m, carveout, timeout, repeats, batch);
        std::fprintf(stderr,
            "[cluster-tune] GPU %d: M=%d N=%d batch=%d cluster_m=%d "
            "-> %.2f TMAD/s (%.0fs x %d repeats)\n",
            device_index_, cfg.m, cfg.n, batch, cluster_m,
            tmad_from_tuner_ops_per_sec(rate), timeout, repeats);
        if (rate > best.hashrate) {
            best.cluster_m = cluster_m;
            best.hashrate = rate;
        }
    }

    std::fprintf(stderr,
        "[cluster-tune] GPU %d: winner cluster_m=%d -> %.2f TMAD/s\n",
        device_index_, best.cluster_m,
        tmad_from_tuner_ops_per_sec(best.hashrate));
#else
    (void)cfg; (void)batch; (void)seconds_per_candidate;
#endif

    return best;
}

TuningResult GpuTuner::tune(double seconds_per_candidate, int repeats) {
    TuningResult best;
    best.hashrate = 0.0;
    best.repeats = repeats;

#if !defined(PROP_MINER_HOST_ONLY_TESTS)
    cudaSetDevice(device_index_);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device_index_);

    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);
    free = free > (1024ULL << 20) ? free - (1024ULL << 20) : free;

    MiningConfig base = shape_for_vram(free, prop.major, prop.minor);

    // Build a small list of candidate shapes around the VRAM-derived default.
    std::vector<MiningConfig> candidates = {base};
    if (base.n >= 65536) {
        MiningConfig smaller = base;
        smaller.n /= 2;
        candidates.push_back(smaller);
    }
    if (base.n < 262144) {
        MiningConfig bigger = base;
        bigger.n *= 2;
        int64_t c = int64_t(bigger.m) * bigger.n * 4;
        int64_t bytes = (int64_t(bigger.m) * bigger.k) +
                        (int64_t(bigger.k) * bigger.n) + c;
        bytes = bytes + bytes / 4;
        if (static_cast<size_t>(bytes) < free) {
            candidates.push_back(bigger);
        }
    }

    best = tune_shapes(candidates, seconds_per_candidate, repeats);

    std::fprintf(stderr,
        "[autotune] GPU %d: selected M=%d N=%d K=%d batch=%d graph=%s "
        "cluster_m=%d carveout=%d repeats=%d -> %.2f TMAD/s\n",
        device_index_, best.config.m, best.config.n, best.config.k,
        best.batch_size, best.use_graph ? "yes" : "no",
        best.cluster_m, best.carveout_percent, best.repeats,
        tmad_from_tuner_ops_per_sec(best.hashrate));
#else
    (void)seconds_per_candidate;
#endif

    return best;
}

} // namespace pearl
