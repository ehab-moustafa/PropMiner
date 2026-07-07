#include <cstdio>
#include <cstring>
#include <cctype>
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
#include "hashrate_metrics.h"
#include "gpu_worker.h"
#include "job_key.h"
#include "share_builder.h"
#include "sigma_context.h"
#include "rtx5090_profile.h"
#include "kernel_knob_cache.h"
#include "mine_batch_cache.h"
#include "env_tuning.h"
#include "tune_cache.h"

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

    MiningConfig cfg = prod_mode
                           ? rtx5090_mining_config(0, resolve_n_cap())
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

static int run_mine_batch_tune(const std::vector<int>& gpu_indices,
                               const pearl::MiningConfig& cfg,
                               double seconds_per_candidate,
                               int repeats) {
    using namespace pearl;
    if (gpu_indices.empty()) {
        fprintf(stderr, "[batch-tune] No GPU indices\n");
        return 1;
    }
    const int idx = gpu_indices[0];
    fprintf(stderr,
            "[batch-tune] Sweeping mine batch at M=%d N=%d (%ds x %d repeats/candidate)\n",
            cfg.m, cfg.n, static_cast<int>(seconds_per_candidate), repeats);

    GpuTuner tuner(idx);
    auto result = tuner.tune_mine_batch(cfg, seconds_per_candidate, repeats);
    if (result.hashrate <= 0.0) {
        fprintf(stderr, "[batch-tune] No candidate completed a full batch\n");
        return 1;
    }

    MineBatchResult cached;
    cached.m = cfg.m;
    cached.n = cfg.n;
    cached.batch = result.batch_size;
    cached.hashrate = result.hashrate;
    cached.use_graph = result.use_graph;
    MineBatchCache cache;
    cache.save(idx, cached);

    fprintf(stderr,
            "[batch-tune] Winner: batch=%d (%.2f TMAD/s) -> %s\n",
            cached.batch, tmad_from_tuner_ops_per_sec(cached.hashrate),
            MineBatchCache::cache_path().c_str());
    return 0;
}

static int run_cluster_sweep(const std::vector<int>& gpu_indices,
                             const pearl::MiningConfig& cfg,
                             int batch,
                             double seconds_per_candidate,
                             int repeats) {
    using namespace pearl;
    if (gpu_indices.empty()) {
        fprintf(stderr, "[cluster-tune] No GPU indices\n");
        return 1;
    }
    const int idx = gpu_indices[0];
    fprintf(stderr,
            "[cluster-tune] Sweeping cluster_m in {1,2,4} at M=%d N=%d batch=%d "
            "(%ds x %d repeats; cluster_m=3 unsupported by kernel)\n",
            cfg.m, cfg.n, batch,
            static_cast<int>(seconds_per_candidate), repeats);

    GpuTuner tuner(idx);
    auto result = tuner.tune_cluster_sweep(cfg, batch, seconds_per_candidate, repeats);
    if (result.hashrate <= 0.0) {
        fprintf(stderr, "[cluster-tune] No candidate completed a full batch\n");
        return 1;
    }

    fprintf(stderr,
            "[cluster-tune] Winner: cluster_m=%d (%.2f TMAD/s)\n"
            "[cluster-tune] Apply for prod: export PEARL_GEMM_CONSUMER_CLUSTER_M=%d\n",
            result.cluster_m, tmad_from_tuner_ops_per_sec(result.hashrate),
            result.cluster_m);
    return 0;
}

static int run_runtime_autotune(const std::vector<int>& gpu_indices,
                                double seconds_per_candidate,
                                int repeats,
                                bool production_shape) {
    using namespace pearl;
    if (gpu_indices.empty()) {
        fprintf(stderr, "[autotune] No GPU indices\n");
        return 1;
    }
    const int idx = gpu_indices[0];
    cudaSetDevice(idx);
    size_t vram_free = 0;
    size_t total = 0;
    cudaMemGetInfo(&vram_free, &total);
    if (vram_free > (512ULL << 20)) {
        vram_free -= (512ULL << 20);
    }

    GpuTuner tuner(idx);
    TuningResult result;
    if (production_shape) {
        const bool stratum =
            (std::getenv("PROPMINER_USE_STRATUM") &&
             std::getenv("PROPMINER_USE_STRATUM")[0] == '1');
        const int n_cap = resolve_n_cap();
        const MiningConfig cfg = stratum
            ? stratum_pool_mining_config(vram_free, n_cap)
            : rtx5090_mining_config(vram_free, n_cap);
        fprintf(stderr,
                "[autotune] Production-shape sweep at M=%d N=%d K=%d "
                "(batch x graph_batch x cluster x carveout; %gs x %d repeats)\n",
                cfg.m, cfg.n, cfg.k, seconds_per_candidate, repeats);
        result = tuner.tune_production(cfg, seconds_per_candidate, repeats);
    } else {
        fprintf(stderr,
                "[autotune] Full runtime sweep (M/N, batch, graph_batch, cluster, "
                "carveout) (%gs x %d repeats/candidate)\n",
                seconds_per_candidate, repeats);
        result = tuner.tune(seconds_per_candidate, repeats);
    }
    if (result.hashrate <= 0.0) {
        fprintf(stderr, "[autotune] No candidate produced measurable hashrate\n");
        return 1;
    }

    TuneCache cache;
    cache.save(idx, result);

    fprintf(stderr,
            "[autotune] Winner: M=%d N=%d batch=%d graph_batch=%d graph=%s "
            "cluster_m=%d carveout=%d -> %.2f TMAD/s\n"
            "[autotune] Cached at %s\n"
            "[autotune] Prod: PROPMINER_MODE=mine PROPMINER_USE_TUNE_CACHE=1 "
            "PROPMINER_AUTOTUNE=0\n",
            result.config.m, result.config.n, result.batch_size,
            result.graph_batch_size,
            result.use_graph ? "yes" : "no",
            result.cluster_m, result.carveout_percent,
            tmad_from_tuner_ops_per_sec(result.hashrate),
            TuneCache::cache_path().c_str());
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
        "  --pool HOST:PORT[,HOST2:PORT2]  gRPC pool (optional; Kryptex uses Stratum by default)\n"
        "  --wallet ADDRESS     Your Pearl wallet address (required)\n"
        "  --worker NAME        Worker name (default: propminer)\n"
        "  --gpus INDEXES       GPU indices, comma-separated (default: all)\n"
        "  --bench SECONDS      Run benchmark only, don't connect to pool\n"
        "  --self-test          Mine tiny problem + verify first share, then exit\n"
        "  --tune-autotune S    Production autotune (batch/graph_batch/cluster), exit\n"
        "  --config M,N,K,R     Override default mining dimensions\n"
        "  --rtx5090            RTX 5090 profile (M=8192, N default cap 65536)\n"
        "  --no-watchdog        Disable GPU watchdog/auto-recovery thread\n"
        "  --disable-cpu        Explicitly disable CPU mining (default; no-op)\n"
        "  --tls 0/1            Use TLS (default: 1)\n"
        "  Env: PROPMINER_AUTOTUNE=0|1|2|force  Runtime autotune + knob cache (5090)\n"
        "  Env: PROPMINER_N_CAP=N          Cap N (default 65536; 0 = uncapped VRAM pick)\n"
        "  Env: PROPMINER_BATCH=N            Matmuls per poll (default 32; env or cache)\n"
        "  Env: PROPMINER_GRAPH_BATCH=N      CUDA graph depth (default 8; must divide batch)\n"
        "  Env: PEARL_GEMM_CONSUMER_CLUSTER_M  Cluster M (default 4 on RTX 5090 prod)\n"
        "  Env: PROPMINER_STRATUM_DIFF=N     Stratum share difficulty (default 32768)\n"
        "  Env: PROPMINER_BENCH_BATCH=N      Override bench graph batch\n"
        "  Env: PROPMINER_USE_TUNE_CACHE=1   Load autotune.json (default on for mine)\n"
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

static void split_wallet_worker(std::string& wallet, std::string& worker,
                                bool worker_set) {
    if (worker_set || wallet.empty()) return;
    const size_t dot = wallet.find('.');
    if (dot == std::string::npos || dot == 0 || dot + 1 >= wallet.size()) return;
    worker = wallet.substr(dot + 1);
    wallet = wallet.substr(0, dot);
    fprintf(stderr, "[main] Parsed wallet.worker -> wallet=%s worker=%s\n",
            wallet.c_str(), worker.c_str());
}

static void warn_kryptex_worker_name(const std::string& worker) {
    if (worker.empty() || worker.size() > 32) {
        fprintf(stderr,
                "[main] WARNING: Kryptex worker name should be 1-32 chars "
                "(current len=%zu). See https://www.kryptex.com/en/articles/kryptex-pools-direct-mining-en\n",
                worker.size());
    }
    for (unsigned char c : worker) {
        if (std::isalnum(c)) continue;
        fprintf(stderr,
                "[main] WARNING: Kryptex worker name should use only letters "
                "and digits (no '%c' in \"%s\"). Dashboard may show the worker "
                "online but not credit shares. Use e.g. propminerhigh.\n",
                c, worker.c_str());
        break;
    }
}

static bool parse_pool_endpoint(const std::string& pool_str,
                                pearl::WorkerOrchestrator::PoolEndpoint& out) {
    size_t colon = pool_str.rfind(':');
    if (colon == std::string::npos) {
        out.host = pool_str;
        out.port = 443;
        out.use_tls = true;
        return !out.host.empty();
    }
    out.host = pool_str.substr(0, colon);
    try {
        out.port = std::stoi(pool_str.substr(colon + 1));
    } catch (...) {
        return false;
    }
    out.use_tls = true;
    return !out.host.empty();
}

static void append_pool_endpoint(pearl::WorkerOrchestrator::Config& cfg,
                                 const std::string& pool_str) {
    pearl::WorkerOrchestrator::PoolEndpoint ep;
    if (!parse_pool_endpoint(pool_str, ep)) {
        fprintf(stderr, "[main] Invalid pool endpoint: %s\n", pool_str.c_str());
        std::exit(1);
    }
    if (ep.port == 7048) {
        cfg.stratum_endpoints.push_back(ep);
    } else {
        cfg.pool_endpoints.push_back(ep);
    }
}

static void parse_pool_list(const std::string& pool_str,
                            pearl::WorkerOrchestrator::Config& cfg) {
    size_t start = 0;
    while (start < pool_str.size()) {
        size_t comma = pool_str.find(',', start);
        if (comma == std::string::npos) comma = pool_str.size();
        const std::string part = pool_str.substr(start, comma - start);
        if (!part.empty()) append_pool_endpoint(cfg, part);
        start = comma + 1;
    }
    if (!cfg.pool_endpoints.empty()) {
        cfg.pool_host = cfg.pool_endpoints[0].host;
        cfg.pool_port = cfg.pool_endpoints[0].port;
        cfg.use_tls = cfg.pool_endpoints[0].use_tls;
    }
}

static void parse_stratum_list(const std::string& pool_str,
                               pearl::WorkerOrchestrator::Config& cfg) {
    size_t start = 0;
    while (start < pool_str.size()) {
        size_t comma = pool_str.find(',', start);
        if (comma == std::string::npos) comma = pool_str.size();
        const std::string part = pool_str.substr(start, comma - start);
        if (!part.empty()) {
            pearl::WorkerOrchestrator::PoolEndpoint ep;
            if (!parse_pool_endpoint(part, ep)) {
                fprintf(stderr, "[main] Invalid stratum endpoint: %s\n", part.c_str());
                std::exit(1);
            }
            ep.use_tls = false;
            cfg.stratum_endpoints.push_back(ep);
        }
        start = comma + 1;
    }
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
    cfg.miner_version = "propminer/2.1";
    int bench_seconds = 0;
    bool self_test = false;
    bool use_rtx5090_profile = false;
    int tune_mine_batch_seconds = 0;
    int tune_mine_batch_repeats = 2;
    int tune_cluster_seconds = 0;
    int tune_cluster_repeats = 3;
    int tune_autotune_seconds = 0;
    int tune_autotune_repeats = 3;

    bool worker_set = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--worker") == 0) { worker_set = true; break; }
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        else if (strcmp(argv[i], "--pool") == 0 && i + 1 < argc) {
            parse_pool_list(argv[++i], cfg);
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
        else if (strcmp(argv[i], "--tune-mine-batch") == 0 && i + 1 < argc) {
            tune_mine_batch_seconds = std::stoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--tune-cluster") == 0 && i + 1 < argc) {
            tune_cluster_seconds = std::stoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--tune-autotune") == 0 && i + 1 < argc) {
            tune_autotune_seconds = std::stoi(argv[++i]);
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

    split_wallet_worker(cfg.wallet_address, cfg.worker_name, worker_set);
    warn_kryptex_worker_name(cfg.worker_name);

    // gRPC endpoints only when explicitly configured (--pool or PROPMINER_POOL).
    // Kryptex Pearl is Stratum-only; do not invent a gRPC :443 default.
    if (cfg.pool_endpoints.empty()) {
        if (const char* env = std::getenv("PROPMINER_POOL"); env && env[0]) {
            parse_pool_list(env, cfg);
        }
        if (const char* fb = std::getenv("PROPMINER_POOL_FALLBACK"); fb && fb[0]) {
            append_pool_endpoint(cfg, fb);
        }
    }
    if (const char* st = std::getenv("PROPMINER_STRATUM_POOL"); st && st[0]) {
        parse_stratum_list(st, cfg);
    }
    if (cfg.stratum_endpoints.empty()) {
        parse_stratum_list("prl.kryptex.network:7048,prl-eu.kryptex.network:7048", cfg);
    }

    // Default Stratum-first for Kryptex Pearl unless operator overrides.
    if (!std::getenv("PROPMINER_USE_STRATUM") && !std::getenv("PROPMINER_POOL_MODE")) {
        setenv("PROPMINER_USE_STRATUM", "1", 0);
    }
    if (cfg.pool_endpoints.empty() && !cfg.stratum_endpoints.empty()) {
        fprintf(stderr, "[main] Stratum-only mode (no gRPC endpoints configured)\n");
    }

    cfg.speed_test_seconds = bench_seconds;
    if (bench_seconds > 0) {
        cfg.enable_watchdog = false;
    }

    if (self_test) {
        return run_self_test();
    }

    int n = pearl::GemmCapi::device_count();
    if (n <= 0) {
        fprintf(stderr, "[main] No CUDA devices found\n");
        return 1;
    }
    if (cfg.gpu_indices.empty()) {
        for (int i = 0; i < n; ++i) cfg.gpu_indices.push_back(i);
    }
    if (cfg.gpu_indices.size() > 1) {
        fprintf(stderr, "[main] Using %zu GPU(s)\n", cfg.gpu_indices.size());
    }

    bool user_overrode_config = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--config") == 0) { user_overrode_config = true; break; }
    }
    cfg.autotune = pearl::autotune_env_enabled();
    cfg.use_tune_cache = pearl::tune_cache_env_enabled();
    if (use_rtx5090_profile) {
        size_t vram_free = 0;
        if (!cfg.gpu_indices.empty()) {
            cudaSetDevice(cfg.gpu_indices[0]);
            size_t total = 0;
            cudaMemGetInfo(&vram_free, &total);
            if (vram_free > (512ULL << 20)) vram_free -= (512ULL << 20);
        }
        const int n_cap = pearl::resolve_n_cap();
        cfg.mining_config = pearl::rtx5090_mining_config(vram_free, n_cap);
        if (bench_seconds == 0 && !user_overrode_config) {
            cfg.batch_size = pearl::MineBatchCache::resolve(
                cfg.gpu_indices[0], cfg.mining_config.m, cfg.mining_config.n,
                pearl::Rtx5090Profile::kDefaultMineBatch);
            cfg.graph_batch_size = pearl::resolve_graph_batch();
        } else if (bench_seconds > 0) {
            const char* bb = std::getenv("PROPMINER_BENCH_BATCH");
            if (bb && bb[0] != '\0') {
                cfg.batch_size = std::atoi(bb);
            } else {
                cfg.batch_size = pearl::MineBatchCache::resolve(
                    cfg.gpu_indices[0], cfg.mining_config.m, cfg.mining_config.n,
                    pearl::Rtx5090Profile::kDefaultMineBatch);
            }
        }
        pearl::log_resolved_tuning("[main]");
        const int ctas = pearl::Rtx5090Profile::tiles(cfg.mining_config.m,
                                                      cfg.mining_config.n);
        const int tail = ctas % pearl::Rtx5090Profile::kSMCount;
        fprintf(stderr,
                "[main] RTX 5090 profile: M=%d N=%d batch=%d graph_batch=%d CTAs=%d "
                "waves~%d tail=%d swizzle=<3,4,3>%s\n",
                cfg.mining_config.m, cfg.mining_config.n, cfg.batch_size,
                cfg.graph_batch_size,
                ctas, (ctas + pearl::Rtx5090Profile::kSMCount - 1)
                          / pearl::Rtx5090Profile::kSMCount,
                tail,
                pearl::n_cap_env_set() ? " (PROPMINER_N_CAP)"
                                       : (n_cap > 0 ? " (default N cap)" : ""));

        pearl::GemmCapi capi;
        if (const int kv = capi.validate_kernel_selection(); kv != 0) {
            return 1;
        }
        const char* built_knobs = capi.build_knobs();
        fprintf(stderr, "[main] Kernel knobs (built): %s\n",
                built_knobs ? built_knobs : "unknown");
        fprintf(stderr, "[main] Active transcript kernel: %s\n",
                capi.active_kernel_name() ? capi.active_kernel_name() : "unknown");

        const char* autotune_env = std::getenv("PROPMINER_AUTOTUNE");
        int autotune_level = cfg.autotune ? 1 : 0;
        if (autotune_env && autotune_env[0] != '\0') {
            if (std::strcmp(autotune_env, "force") == 0) {
                autotune_level = 2;
            } else if (std::strcmp(autotune_env, "2") == 0) {
                autotune_level = 2;
            } else if (std::strcmp(autotune_env, "1") == 0) {
                autotune_level = 1;
            } else if (std::strcmp(autotune_env, "0") == 0) {
                autotune_level = 0;
            } else {
                fprintf(stderr,
                        "[main] WARN: unknown PROPMINER_AUTOTUNE='%s' (use 0|1|2|force)\n",
                        autotune_env);
            }
        }

        if (cfg.autotune) {
            fprintf(stderr,
                    "[main] RTX 5090: runtime autotune on (PROPMINER_AUTOTUNE=%s)\n",
                    autotune_env);
            pearl::KernelKnobCache knob_cache;
            for (int idx : cfg.gpu_indices) {
                if (auto cached = knob_cache.load(idx)) {
                    fprintf(stderr,
                            "[main] Kernel knob cache GPU %d: %s (%.0f H/s, self-test=%s)\n",
                            idx, cached->manifest.c_str(), cached->hashrate,
                            cached->self_test_ok ? "ok" : "no");
                    if (built_knobs &&
                        !pearl::KernelKnobCache::manifest_matches(
                            built_knobs, cached->manifest.c_str())) {
                        fprintf(stderr,
                                "[main] WARN: built .so knobs != cached winner; "
                                "run PROPMINER_MODE=tune or ./scripts/tune_blackwell_knobs.sh\n");
                    }
                } else if (autotune_level >= 2) {
                    fprintf(stderr,
                            "[main] No kernel knob cache for GPU %d; "
                            "run PROPMINER_MODE=tune or ./scripts/tune_blackwell_knobs.sh\n",
                            idx);
                }
            }
        } else {
            fprintf(stderr,
                    "[main] RTX 5090 prod: aggressive defaults "
                    "(cluster_m=%d, tune cache %s; "
                    "PROPMINER_USE_TUNE_CACHE=0 to skip cache, "
                    "PEARL_GEMM_CONSUMER_CLUSTER_M=1 to disable clusters)\n",
                    pearl::Rtx5090Profile::kProdDefaultClusterM,
                    cfg.use_tune_cache ? "on" : "off");
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

    if (tune_mine_batch_seconds > 0) {
        if (!use_rtx5090_profile) {
            fprintf(stderr, "[batch-tune] --tune-mine-batch requires --rtx5090\n");
            return 1;
        }
        if (const char* rep = std::getenv("PROPMINER_BATCH_TUNE_REPEATS")) {
            tune_mine_batch_repeats = std::max(1, std::atoi(rep));
        }
        return run_mine_batch_tune(cfg.gpu_indices, cfg.mining_config,
                                   static_cast<double>(tune_mine_batch_seconds),
                                   tune_mine_batch_repeats);
    }

    if (tune_cluster_seconds > 0) {
        if (!use_rtx5090_profile) {
            fprintf(stderr, "[cluster-tune] --tune-cluster requires --rtx5090\n");
            return 1;
        }
        if (const char* rep = std::getenv("PROPMINER_CLUSTER_TUNE_REPEATS")) {
            tune_cluster_repeats = std::max(1, std::atoi(rep));
        }
        int batch = cfg.batch_size;
        if (const char* b = std::getenv("PROPMINER_BENCH_BATCH")) {
            batch = std::max(1, std::atoi(b));
        } else {
            batch = pearl::MineBatchCache::resolve(
                cfg.gpu_indices[0], cfg.mining_config.m, cfg.mining_config.n,
                cfg.batch_size);
        }
        return run_cluster_sweep(cfg.gpu_indices, cfg.mining_config, batch,
                                 static_cast<double>(tune_cluster_seconds),
                                 tune_cluster_repeats);
    }

    if (tune_autotune_seconds > 0) {
        if (!use_rtx5090_profile) {
            fprintf(stderr, "[autotune] --tune-autotune requires --rtx5090\n");
            return 1;
        }
        if (const char* rep = std::getenv("PROPMINER_AUTOTUNE_REPEATS")) {
            tune_autotune_repeats = std::max(1, std::atoi(rep));
        }
        return run_runtime_autotune(cfg.gpu_indices,
                                    static_cast<double>(tune_autotune_seconds),
                                    tune_autotune_repeats,
                                    use_rtx5090_profile);
    }

    if (bench_seconds == 0 && cfg.wallet_address.empty()) {
        fprintf(stderr, "[main] Error: --wallet is required for mining\n\n");
        print_usage(argv[0]);
        return 1;
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
        fprintf(stderr, "[main] Production mine mode: pool %s:%d TLS=%s\n",
                cfg.pool_host.c_str(), cfg.pool_port, cfg.use_tls ? "on" : "off");
        fprintf(stderr, "[main] Wallet: %s Worker: %s\n",
                cfg.wallet_address.c_str(), cfg.worker_name.c_str());
        if (use_rtx5090_profile) {
            fprintf(stderr,
                    "[main] Expect first pool job + sigma install before hashrate > 0 "
                    "(full N may take 60-120s for first batch).\n\n");
        } else {
            fprintf(stderr, "\n");
        }
    }

    return orch.run();
}
