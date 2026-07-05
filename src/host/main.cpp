#include <cstdio>
#include <cstring>
#include <csignal>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <atomic>
#include <unistd.h>
#include <chrono>

#include "worker_orchestrator.h"
#include "pearl_capi_wrapper.h"
#include "pearl_types.h"
#include "gpu_tuner.h"
#include "gpu_worker.h"
#include "job_key.h"
#include "share_builder.h"
#include "sigma_context.h"
#include "rtx5090_profile.h"

#include "cuda_compat.h"
#include <condition_variable>

static pearl::WorkerOrchestrator* g_orch = nullptr;
static std::atomic<bool> g_shutdown{false};

namespace {

// Lightweight share sink that captures the first share for self-test.
struct CaptureSink : pearl::IShareSink {
    std::mutex mtx;
    std::condition_variable cv;
    pearl::ShareFound share;
    bool have = false;

    void submit(const pearl::ShareFound& s) override {
        std::lock_guard<std::mutex> lk(mtx);
        if (!have) {
            share = s;
            have = true;
            cv.notify_one();
        }
    }

    bool wait(pearl::ShareFound& out, int seconds) {
        std::unique_lock<std::mutex> lk(mtx);
        if (!cv.wait_for(lk, std::chrono::seconds(seconds),
                         [this] { return have; })) {
            return false;
        }
        out = share;
        return true;
    }
};

// Run a tiny problem with an easy target on GPU 0 and verify the first share.
static int run_self_test() {
    using namespace pearl;
    int n = GemmCapi::device_count();
    if (n <= 0) {
        fprintf(stderr, "[self-test] No CUDA devices\n");
        return 1;
    }

    const char* prod_env = std::getenv("PROP_MINER_SELF_TEST_PROD");
    const bool prod_mode = prod_env && prod_env[0] == '1';

    MiningConfig cfg = prod_mode ? rtx5090_mining_config()
                                 : MiningConfig{};
    if (!prod_mode) {
        cfg.m = 2048;
        cfg.n = 4096;
        cfg.k = 128;
        cfg.r = 128;
    }

    Job job;
    job.sigma.fill(0xab);
    job.b_seed.fill(0xcd);
    job.config = cfg;
    job.job_key = derive_job_key(job.sigma, cfg);
    job.audit_k = 4;
    job.target_nbits = 0x01111111;

    auto ctx = std::make_shared<SigmaContext>(job, cfg);
    CaptureSink sink;
    GpuWorker worker(0, 0, cfg, &sink);
    worker.set_sigma(ctx);
    worker.set_target_nbits(job.target_nbits);
    worker.set_matmuls_per_poll(prod_mode ? 8 : 8);
    worker.start();

    ShareFound share;
    bool ok = sink.wait(share, prod_mode ? 180 : 120);
    worker.stop();

    if (!ok) {
        fprintf(stderr, "[self-test] Timed out waiting for a share\n");
        return 1;
    }

    if (!share.sigma_ctx) share.sigma_ctx = ctx;

    bool verified = ShareBuilder::VerifyShare(share, *ctx);
    if (!verified) {
        fprintf(stderr, "[self-test] Share verification FAILED\n");
        return 1;
    }

    ShareBuilder builder(cfg);
    auto proof = builder.build(share, *ctx);
    if (proof.empty()) {
        fprintf(stderr, "[self-test] Share build FAILED (target guard or proof)\n");
        return 1;
    }

    fprintf(stderr,
        "[self-test] Share verified OK (nonce=%llu, rows=%zu, cols=%zu, proof=%zu B)\n",
        (unsigned long long)share.nonce,
        share.a_row_indices.size(), share.b_col_indices.size(), proof.size());
    return 0;
}

} // namespace

void signal_handler(int sig) {
    if (sig == SIGINT || sig == SIGTERM) {
        fprintf(stderr, "\n[main] Shutdown signal received...\n");
        g_shutdown.store(true);
        if (g_orch) g_orch->stop();
    }
}

static void print_usage(const char* prog) {
    fprintf(stderr,
        "\nPropMiner v2.0 — Pearl NoisyGEMM GPU Miner (0%% dev fee)\n"
        "\nUsage: %s [OPTIONS]\n"
        "\nOptions:\n"
        "  --pool HOST:PORT     Pool address (default: prl.kryptex.network:443)\n"
        "  --wallet ADDRESS     Your Pearl wallet address (required)\n"
        "  --worker NAME        Worker name (default: propminer)\n"
        "  --gpus INDEXES       GPU indices, comma-separated (default: all)\n"
        "  --bench SECONDS      Run benchmark only, don't connect to pool\n"
        "  --self-test          Mine tiny problem + verify first share, then exit\n"
        "  --config M,N,K,R     Override default mining dimensions\n"
        "  --rtx5090            Use hard-coded RTX 5090 profile (M=8192,N=32768)\n"
        "  --no-watchdog        Disable GPU watchdog/auto-recovery thread\n"
        "  --disable-cpu        Explicitly disable CPU mining (default; no-op)\n"
        "  --tls 0/1            Use TLS (default: 1)\n"
        "  --help               Show this help\n"
        "\nExamples:\n"
        "  %s --wallet krxX2P3Z84.mine-crypto-script\n"
        "  %s --pool prl-eu.kryptex.network:443 --wallet krxX2P3Z84.worker1 --gpus 0,1\n"
        "\n", prog, prog, prog);
}

static std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    size_t start = 0;
    while (start < s.size()) {
        size_t comma = s.find(',', start);
        if (comma == std::string::npos) comma = s.size();
        try {
            out.push_back(std::stoi(s.substr(start, comma - start)));
        } catch (...) {
            fprintf(stderr, "[main] Invalid integer: %s\n", s.substr(start, comma - start).c_str());
            std::exit(1);
        }
        start = comma + 1;
    }
    return out;
}

int main(int argc, char* argv[]) {
    fprintf(stderr,
        "\n  ____ _   _ ____  ____  _       _    _ _ \n"
        " / ___| | | | __ )| __ )| | __ _| | _|| |\n"
        " \\___ \\| | | |  _ \\|  _ \\| |/ _` | |/ / |\n"
        "  ___) | |_| | |_| | (_) | | (_| |   <| |\n"
        " |____/ \\___/|_____|\\____/|_|\\__,_|_|\\_\\_|\n"
        "          v2.0  Pearl NoisyGEMM GPU Miner  0%% dev fee\n\n");

    pearl::WorkerOrchestrator::Config cfg;
    cfg.pool_host = "prl.kryptex.network";
    cfg.pool_port = 443;
    cfg.use_tls = true;
    cfg.worker_name = "propminer";
    cfg.miner_version = "propminer/2.0";
    int bench_seconds = 0;
    bool self_test = false;
    bool use_rtx5090_profile = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "--pool") == 0 && i + 1 < argc) {
            std::string pool_str = argv[++i];
            size_t colon = pool_str.rfind(':');
            if (colon != std::string::npos) {
                cfg.pool_host = pool_str.substr(0, colon);
                try { cfg.pool_port = std::stoi(pool_str.substr(colon + 1)); }
                catch (...) {
                    fprintf(stderr, "[main] Invalid port\n");
                    return 1;
                }
            } else {
                cfg.pool_host = pool_str;
            }
        }
        else if (strcmp(argv[i], "--wallet") == 0 && i + 1 < argc) {
            cfg.wallet_address = argv[++i];
        }
        else if (strcmp(argv[i], "--worker") == 0 && i + 1 < argc) {
            cfg.worker_name = argv[++i];
        }
        else if (strcmp(argv[i], "--gpus") == 0 && i + 1 < argc) {
            cfg.gpu_indices = parse_int_list(argv[++i]);
        }
        else if (strcmp(argv[i], "--bench") == 0 && i + 1 < argc) {
            bench_seconds = std::stoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--self-test") == 0) {
            self_test = true;
        }
        else if (strcmp(argv[i], "--tls") == 0 && i + 1 < argc) {
            cfg.use_tls = std::stoi(argv[++i]) != 0;
        }
        else if (strcmp(argv[i], "--rtx5090") == 0) {
            use_rtx5090_profile = true;
        }
        else if (strcmp(argv[i], "--no-watchdog") == 0) {
            cfg.enable_watchdog = false;
        }
        else if (strcmp(argv[i], "--disable-cpu") == 0) {
            // PropMiner has no CPU mining path; this flag exists only to make
            // the no-CPU-mining guarantee explicit for users coming from other
            // miners.  CPU remains used for hosting/seeds/streaming only.
            cfg.disable_cpu_mining = true;
        }
        else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            // Format: M,N,K,R
            auto dims = parse_int_list(argv[++i]);
            if (dims.size() == 4) {
                cfg.mining_config.m = dims[0];
                cfg.mining_config.n = dims[1];
                cfg.mining_config.k = dims[2];
                cfg.mining_config.r = dims[3];
            } else {
                fprintf(stderr, "[main] --config requires M,N,K,R\n");
                return 1;
            }
        }
        else {
            fprintf(stderr, "[main] Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    cfg.speed_test_seconds = bench_seconds;
    if (bench_seconds > 0) {
        cfg.enable_watchdog = false;
    }

    if (self_test) {
        return run_self_test();
    }

    if (bench_seconds == 0 && cfg.wallet_address.empty()) {
        fprintf(stderr, "[main] Error: --wallet is required for mining\n\n");
        print_usage(argv[0]);
        return 1;
    }

    int n = pearl::GemmCapi::device_count();
    if (n <= 0) {
        fprintf(stderr, "[main] No CUDA devices found\n");
        return 1;
    }
    if (cfg.gpu_indices.empty()) {
        for (int i = 0; i < n; ++i) cfg.gpu_indices.push_back(i);
    }

    // Auto-shape mining config to each selected GPU unless the user explicitly
    // overrode the dimensions via --config.
    bool user_overrode_config = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--config") == 0) { user_overrode_config = true; break; }
    }
    if (use_rtx5090_profile) {
        size_t vram_free = 0;
        if (!cfg.gpu_indices.empty()) {
            cudaSetDevice(cfg.gpu_indices[0]);
            size_t total = 0;
            cudaMemGetInfo(&vram_free, &total);
            if (vram_free > (512ULL << 20)) vram_free -= (512ULL << 20);
        }
        const int n_cap = (bench_seconds > 0) ? pearl::Rtx5090Profile::kBenchMaxN : 0;
        cfg.mining_config = pearl::rtx5090_mining_config(vram_free, n_cap);
        cfg.batch_size = pearl::Rtx5090Profile::kDefaultBatch;
        const int ctas = pearl::Rtx5090Profile::tiles(cfg.mining_config.m,
                                                      cfg.mining_config.n);
        const int tail = ctas % pearl::Rtx5090Profile::kSMCount;
        fprintf(stderr,
                "[main] RTX 5090 profile: M=%d N=%d batch=%d CTAs=%d "
                "waves~%d tail=%d swizzle=<3,4,3>%s\n",
                cfg.mining_config.m, cfg.mining_config.n, cfg.batch_size,
                ctas, (ctas + pearl::Rtx5090Profile::kSMCount - 1)
                          / pearl::Rtx5090Profile::kSMCount,
                tail,
                n_cap > 0 ? " (bench N cap)" : "");
        // Rtx5090 profile is pre-tuned; skip multi-minute autotune unless requested.
        const char* autotune_env = std::getenv("PROPMINER_AUTOTUNE");
        if (!autotune_env || autotune_env[0] == '\0' ||
            std::strcmp(autotune_env, "0") == 0) {
            cfg.autotune = false;
            fprintf(stderr, "[main] RTX 5090: autotune off (set PROPMINER_AUTOTUNE=1 to enable)\n");
        }
    } else if (!user_overrode_config) {
        for (int idx : cfg.gpu_indices) {
            if (idx < 0 || idx >= n) continue;
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, idx) == cudaSuccess) {
                cfg.mining_config = pearl::MiningConfig::auto_shape_for_gpu(prop, 0);
                // Use the first valid GPU's shape globally. Per-GPU shapes would
                // require extending JobBus/WorkerOrchestrator; homogeneous rigs are
                // the common case.
                break;
            }
        }
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    pearl::WorkerOrchestrator orch(cfg);
    g_orch = &orch;

    fprintf(stderr, "[main] Starting PropMiner v2 on %zu GPU(s) -> %s:%d\n",
            cfg.gpu_indices.size(), cfg.pool_host.c_str(), cfg.pool_port);
    if (bench_seconds > 0) {
        fprintf(stderr, "[main] Benchmark mode: %d seconds\n", bench_seconds);
    } else {
        fprintf(stderr, "[main] Wallet: %s Worker: %s\n\n",
                cfg.wallet_address.c_str(), cfg.worker_name.c_str());
    }

    return orch.run();
}
