#include "worker_orchestrator.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <cstdio>

#include "gpu_tuner.h"
#include "job_key.h"
#include "pearl_capi_wrapper.h"
#include "share_builder.h"
#include "tune_cache.h"

namespace pearl {

namespace {
    int64_t now_unix_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
}

WorkerOrchestrator::WorkerOrchestrator(const Config& cfg) : cfg_(cfg) {}

WorkerOrchestrator::~WorkerOrchestrator() {
    stop();
}

std::vector<proto::GpuCard> WorkerOrchestrator::enumerate_gpu_cards() {
    std::vector<proto::GpuCard> cards;
    int n = GemmCapi::device_count();
    if (n <= 0) return cards;

    std::vector<int> indices = cfg_.gpu_indices;
    if (indices.empty()) {
        for (int i = 0; i < n; ++i) indices.push_back(i);
    }

    for (int idx : indices) {
        if (idx < 0 || idx >= n) continue;
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, idx) != cudaSuccess) continue;

        char hex[33] = {0};
        for (int b = 0; b < 16; ++b) {
            std::snprintf(hex + b * 2, 3, "%02x",
                          static_cast<unsigned char>(prop.uuid.bytes[b]));
        }

        proto::GpuCard card;
        card.uuid = hex;
        card.model = prop.name;
        card.index = static_cast<uint32_t>(idx);
        card.hashrate = 0.0;
        cards.push_back(card);
    }
    return cards;
}

bool WorkerOrchestrator::ensure_connected_and_registered() {
    if (client_ && client_->connected()) return true;

    PearlGrpcClient::Options opts;
    opts.host = cfg_.pool_host;
    opts.port = cfg_.pool_port;
    opts.use_tls = cfg_.use_tls;
    client_ = std::make_unique<PearlGrpcClient>(opts);

    if (!client_->connect()) {
        std::cerr << "grpc connect failed: " << client_->last_error() << std::endl;
        return false;
    }

    proto::RegisterRequest req;
    req.wallet_address = cfg_.wallet_address;
    req.worker_name = cfg_.worker_name;
    req.miner_version = cfg_.miner_version;
    req.protocol_version = 2;
    req.k = static_cast<uint32_t>(cfg_.mining_config.k);
    req.gpu_cards = enumerate_gpu_cards();

    proto::RegisterResponse resp;
    if (!client_->register_miner(req, resp)) {
        std::cerr << "register failed: " << client_->last_error() << std::endl;
        return false;
    }
    if (!resp.success) {
        std::cerr << "register rejected: " << resp.error_message << std::endl;
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(session_mtx_);
        miner_id_ = resp.miner_id;
        session_token_ = resp.session_token;
    }
    if (resp.initial_difficulty_nbits != 0) {
        pending_target_nbits_.store(resp.initial_difficulty_nbits);
        for (auto& w : workers_) {
            w->set_target_nbits(resp.initial_difficulty_nbits);
        }
    }
    registered_ = true;
    return true;
}

void WorkerOrchestrator::publish_job_from_assignment(const proto::JobAssignment& ja) {
    Job job;
    job.sigma = ja.sigma;
    job.b_seed = ja.b_seed;
    job.target_nbits = ja.target_nbits;
    job.audit_k = ja.audit_k;
    job.block_height = static_cast<uint64_t>(ja.block_height);
    job.job_id = ja.job_id;
    job.config = cfg_.mining_config;
    job.job_key = derive_job_key(job.sigma, job.config);

    auto ctx = std::make_shared<SigmaContext>(job, cfg_.mining_config);
    bus_.publish(ctx);

    for (auto& w : workers_) {
        w->set_sigma(ctx);
        w->set_target_nbits(ja.target_nbits);
    }
}

void WorkerOrchestrator::handle_pool_event(const proto::PoolEvent& evt) {
    switch (evt.type) {
        case proto::PoolEventType::Job:
            publish_job_from_assignment(evt.job);
            break;
        case proto::PoolEventType::Vardiff: {
            const uint32_t nbits = evt.vardiff.new_target_nbits;
            for (auto& w : workers_) w->set_target_nbits(nbits);
            auto entry = bus_.drain_latest();
            if (entry.ctx) {
                entry.ctx->set_target_nbits(nbits);
            }
            break;
        }
        case proto::PoolEventType::ShareResult:
            // Log accepted/rejected. The pool does not need a client-side ack.
            std::cout << "share " << evt.share_result.outcome
                      << (evt.share_result.is_block_find ? " BLOCK" : "")
                      << std::endl;
            break;
        case proto::PoolEventType::Error:
            std::cerr << "pool error: " << evt.error.message << std::endl;
            if (evt.error.fatal) stop();
            break;
        case proto::PoolEventType::Reconnect:
            std::cout << "pool reconnect in " << evt.reconnect.wait_seconds << "s" << std::endl;
            break;
        default:
            break;
    }
}

void WorkerOrchestrator::network_thread_func() {
    while (!stop_flag_) {
        if (!ensure_connected_and_registered()) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }

        proto::AuthEvent auth;
        {
            std::lock_guard<std::mutex> lk(session_mtx_);
            auth.miner_id = miner_id_;
            auth.session_token = session_token_;
        }
        if (!client_->start_mining_stream(auth)) {
            std::cerr << "mining stream failed: " << client_->last_error() << std::endl;
            client_->disconnect();
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }

        while (!stop_flag_ && client_->connected()) {
            proto::PoolEvent evt;
            if (client_->receive_event(evt)) {
                handle_pool_event(evt);
            } else if (!client_->connected()) {
                break;
            }
        }
        client_->disconnect();
    }
}

void WorkerOrchestrator::share_sender_thread_func() {
    while (!stop_flag_) {
        std::vector<ShareFound> batch;
        {
            std::unique_lock<std::mutex> lk(share_mtx_);
            share_cv_.wait(lk, [this] { return stop_flag_ || !pending_share_found_.empty(); });
            batch.swap(pending_share_found_);
        }
        if (stop_flag_) break;
        for (const ShareFound& raw : batch) {
            if (!raw.sigma_ctx) {
                std::cerr << "share dropped: missing sigma context\n";
                continue;
            }
            ShareBuilder builder(raw.job.config);
            auto proof = builder.build(raw, *raw.sigma_ctx);
            if (proof.empty()) {
                std::cerr << "share dropped: stale target or build failed\n";
                continue;
            }
            proto::MinerEvent evt;
            evt.type = proto::MinerEventType::Share;
            evt.payload = std::move(proof);
            {
                std::lock_guard<std::mutex> lk(session_mtx_);
                evt.seq = ++miner_event_seq_;
            }
            if (client_ && client_->connected()) {
                if (!client_->send_event(evt)) {
                    std::cerr << "share send failed: " << client_->last_error() << std::endl;
                }
            }
        }
    }
}

void WorkerOrchestrator::heartbeat_thread_func() {
    while (!stop_flag_) {
        std::this_thread::sleep_for(std::chrono::seconds(30));
        if (stop_flag_) break;
        if (!client_ || !client_->connected()) continue;

        proto::Heartbeat hb;
        hb.timestamp = now_unix_ms();
        double wire_total = 0.0;
        std::vector<proto::PerGpuHashrate> gpu_rates;
        for (size_t i = 0; i < workers_.size(); ++i) {
            proto::PerGpuHashrate r;
            r.gpu_uuid = i < gpu_uuids_.size() ? gpu_uuids_[i]
                                               : ("gpu" + std::to_string(i));
            const double wire_hr = workers_[i]->tmads_per_sec() * kTmadsToHashesPerSec;
            r.hashrate_5m = wire_hr;
            r.shares_5m = 0;
            wire_total += wire_hr;
            gpu_rates.push_back(r);
        }
        hb.current_hashrate = wire_total;
        hb.sequence_number = static_cast<uint64_t>(now_unix_ms());
        hb.gpu_hashrates = std::move(gpu_rates);
        client_->send_heartbeat(hb);
    }
}

void WorkerOrchestrator::submit(const ShareFound& share) {
    std::lock_guard<std::mutex> lk(share_mtx_);
    pending_share_found_.push_back(share);
    share_cv_.notify_one();
}

void WorkerOrchestrator::submit_share(const ShareFound& share) {
    submit(share);
}

int WorkerOrchestrator::run() {
    std::vector<int> indices = cfg_.gpu_indices;
    int n = GemmCapi::device_count();
    if (indices.empty()) {
        for (int i = 0; i < n; ++i) indices.push_back(i);
    }
    if (indices.empty()) {
        std::cerr << "no CUDA devices" << std::endl;
        return 1;
    }

    if (cfg_.disable_cpu_mining) {
        std::cerr << "[orchestrator] CPU mining disabled; CPU is used only for "
                     "host management, seed generation, and PCIe feeding.\n";
    }

    MiningConfig tuned_config = cfg_.mining_config;
    int tuned_batch = cfg_.batch_size;
    int tuned_cluster_m = 1;
    int tuned_carveout = -1;

    const char* cluster_env = std::getenv("PEARL_GEMM_CONSUMER_CLUSTER_M");
    if (cluster_env && cluster_env[0]) {
        tuned_cluster_m = std::max(1, std::atoi(cluster_env));
    }
    MiningConfig::warn_if_cluster_m_mismatch(tuned_cluster_m);

    TuneCache tune_cache;
    if (cfg_.autotune && cfg_.speed_test_seconds == 0) {
        // Try the persistent cache first so reconnects pick up the last good
        // config without re-benchmarking.
        bool cache_hit = false;
        for (int idx : indices) {
            if (auto cached = tune_cache.load(idx)) {
                tuned_config = cached->config;
                tuned_batch = cached->batch_size;
                tuned_cluster_m = cached->cluster_m;
                tuned_carveout = cached->carveout_percent;
                cache_hit = true;
                std::cerr << "[orchestrator] Using cached autotune for GPU "
                          << idx << " (M=" << tuned_config.m
                          << " N=" << tuned_config.n
                          << " batch=" << tuned_batch
                          << " graph=" << (cached->use_graph ? "yes" : "no")
                          << ")" << std::endl;
            }
        }

        if (!cache_hit) {
            std::cerr << "[orchestrator] Running per-GPU autotune..." << std::endl;
            for (int idx : indices) {
                GpuTuner tuner(idx);
                auto result = tuner.tune(5.0, 3);
                if (result.hashrate > 0.0) {
                    tuned_config = result.config;
                    tuned_batch = result.batch_size;
                    tuned_cluster_m = result.cluster_m;
                    tuned_carveout = result.carveout_percent;
                    tune_cache.save(idx, result);
                }
            }
            if (tuned_cluster_m > 1) {
                setenv("PEARL_GEMM_CONSUMER_CLUSTER_M",
                       std::to_string(tuned_cluster_m).c_str(), 1);
            }
            if (tuned_carveout >= 0) {
                setenv("PEARL_GEMM_CONSUMER_CARVEOUT",
                       std::to_string(tuned_carveout).c_str(), 1);
            }
        }
    }

    const auto gpu_cards = enumerate_gpu_cards();
    gpu_uuids_.clear();
    for (const auto& c : gpu_cards) gpu_uuids_.push_back(c.uuid);

    const bool bench_mode = cfg_.speed_test_seconds > 0;

    if (cfg_.enable_watchdog && !bench_mode) {
        watchdog_ = std::make_unique<Watchdog>();
        watchdog_->start([this]() {
            std::cerr << "[watchdog] stall detected — republishing current job\n";
            auto entry = bus_.drain_latest();
            if (entry.ctx) {
                for (auto& w : workers_) w->set_sigma(entry.ctx);
            }
        });
    }

    uint32_t bench_target_nbits = 0;
    std::shared_ptr<SigmaContext> bench_ctx;
    if (bench_mode) {
        Job job;
        job.sigma.fill(0xab);
        job.b_seed.fill(0xcd);
        job.config = tuned_config;
        job.job_key = derive_job_key(job.sigma, tuned_config);
        job.audit_k = 4;
        job.target_nbits = 0x01111111;
        bench_target_nbits = job.target_nbits;
        bench_ctx = std::make_shared<SigmaContext>(job, tuned_config);
        // Full production batch (20) can take >2 min per graph launch on WSL2.
        // Use a smaller batch so at least one completes inside the bench window.
        int bench_batch = 4;
        if (const char* bb = std::getenv("PROPMINER_BENCH_BATCH")) {
            bench_batch = std::max(1, std::atoi(bb));
        }
        tuned_batch = bench_batch;
        const bool bench_graph = !std::getenv("PROPMINER_BENCH_NO_GRAPH");
        std::cerr << "[orchestrator] Benchmark mode: local job, no pool connection"
                  << " (batch=" << tuned_batch
                  << " graph=" << (bench_graph ? "on" : "off") << ")\n";
    }

    for (int idx : indices) {
        auto w = std::make_unique<GpuWorker>(idx, static_cast<int>(workers_.size()),
                                             tuned_config, this);
        w->set_matmuls_per_poll(tuned_batch);
        if (watchdog_) w->set_watchdog(watchdog_.get());
        if (bench_ctx) {
            w->set_sigma(bench_ctx);
            w->set_target_nbits(bench_target_nbits);
        } else if (const uint32_t nbits = pending_target_nbits_.load()) {
            w->set_target_nbits(nbits);
        }
        workers_.push_back(std::move(w));
    }

    for (auto& w : workers_) w->start();

    if (!bench_mode) {
        threads_.emplace_back(&WorkerOrchestrator::network_thread_func, this);
        threads_.emplace_back(&WorkerOrchestrator::share_sender_thread_func, this);
        threads_.emplace_back(&WorkerOrchestrator::heartbeat_thread_func, this);
    }

    // Stats loop.
    while (!stop_flag_) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        double h = 0.0;
        for (auto& w : workers_) h += w->hashrate();
        total_hashrate_ = h;
        std::cout << "hashrate: " << h << " H/s" << std::endl;
        if (cfg_.speed_test_seconds > 0) {
            static int elapsed = 0;
            elapsed += 5;
            if (h == 0.0 && elapsed >= 30 && (elapsed % 30 == 0)) {
                std::cerr << "[bench] " << elapsed << "s elapsed, still awaiting first "
                          << "batch completion (check nvidia-smi for GPU util)\n";
            }
            if (elapsed >= cfg_.speed_test_seconds) {
                std::cout << "benchmark complete: " << h << " H/s" << std::endl;
                std::cerr << "[bench] finished after " << elapsed << "s, final "
                          << h << " H/s\n";
                stop();
                break;
            }
        }
    }

    stop();
    if (watchdog_) watchdog_->stop();
    for (auto& t : threads_) if (t.joinable()) t.join();
    for (auto& w : workers_) w->stop();
    return 0;
}

void WorkerOrchestrator::stop() {
    stop_flag_ = true;
    share_cv_.notify_all();
}

double WorkerOrchestrator::total_hashrate() const {
    return total_hashrate_.load();
}

} // namespace pearl
