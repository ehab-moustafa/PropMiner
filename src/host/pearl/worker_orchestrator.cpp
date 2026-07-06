#include "worker_orchestrator.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <cstdio>
#include <string>

#include "gpu_tuner.h"
#include "hashrate_metrics.h"
#include "job_key.h"
#include "kernel_knob_cache.h"
#include "mine_batch_cache.h"
#include "pearl_capi_wrapper.h"
#include "pow_target_utils.h"
#include "rtx5090_profile.h"
#include "share_builder.h"
#include "env_flags.h"
#include "share_trace.h"
#include "tune_cache.h"

namespace pearl {

namespace {
    int64_t now_unix_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }

    HashrateMetrics aggregate_worker_metrics(const std::vector<std::unique_ptr<GpuWorker>>& workers,
                                             const MiningConfig& cfg) {
        HashrateMetrics total{};
        total.m = cfg.m;
        total.n = cfg.n;
        total.k = cfg.k;
        double batch_ms_sum = 0.0;
        int batch_ms_count = 0;
        for (const auto& w : workers) {
            total.tmad_per_sec += w->tmads_per_sec();
            total.protocol_hps += w->hashrate();
            const double batch_ms = w->last_iter_ms();
            if (batch_ms > 0.0) {
                batch_ms_sum += batch_ms;
                ++batch_ms_count;
            }
            if (total.batch == 0) {
                total.batch = w->matmuls_per_poll();
            }
        }
        if (batch_ms_count > 0) {
            total.batch_ms = batch_ms_sum / static_cast<double>(batch_ms_count);
        }
        if (total.tmad_per_sec > 0.0) {
            total.tops_pct =
                (total.tmad_per_sec * 1e12 / kRtx5090RatedInt8Tops) * 100.0;
        }
        const double daf = static_cast<double>(cfg.difficulty_adjustment_factor());
        if (daf > 0.0 && total.protocol_hps > 0.0) {
            total.tiles_per_sec = total.protocol_hps / daf;
        }
        if (total.batch_ms > 0.0 && total.batch > 0) {
            total.iters_per_sec =
                (static_cast<double>(total.batch) * 1000.0) / total.batch_ms;
        }
        return total;
    }

    int normalize_cluster_m(int cluster_m) {
        if (cluster_m == 2 || cluster_m == 4) return cluster_m;
        if (cluster_m == 3) {
            std::fprintf(stderr,
                "[orchestrator] WARN: cluster_m=3 unsupported; using 1\n");
            return 1;
        }
        return 1;
    }

    bool bench_json_enabled() {
        const char* env = std::getenv("PROPMINER_BENCH_JSON");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }

    int bench_grace_seconds() {
        const char* env = std::getenv("PROPMINER_BENCH_GRACE_SECONDS");
        if (env && env[0] != '\0') {
            return std::max(0, std::atoi(env));
        }
        return Rtx5090Profile::kBenchGraceSeconds;
    }

    std::string resolve_git_sha() {
        if (const char* env = std::getenv("PROP_MINER_GIT_SHA");
            env && env[0] != '\0') {
            return env;
        }
        auto read_version_file = [](const char* path) -> std::string {
            FILE* vf = std::fopen(path, "r");
            if (!vf) return {};
            char buf[64] = {};
            if (!std::fgets(buf, sizeof(buf), vf)) {
                std::fclose(vf);
                return {};
            }
            std::fclose(vf);
            size_t len = std::strlen(buf);
            while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r')) {
                buf[--len] = '\0';
            }
            return len > 0 ? std::string(buf, len) : std::string{};
        };
        for (const char* path : {"VERSION", "/opt/propminer/VERSION"}) {
            if (const std::string v = read_version_file(path); !v.empty()) {
                return v;
            }
        }
        std::string sha;
        FILE* pipe = popen("git rev-parse --short HEAD 2>/dev/null", "r");
        if (pipe) {
            char buf[64] = {};
            if (std::fgets(buf, sizeof(buf), pipe)) {
                size_t len = std::strlen(buf);
                while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r')) {
                    buf[--len] = '\0';
                }
                if (len > 0) sha.assign(buf, len);
            }
            pclose(pipe);
        }
        if (!sha.empty()) return sha;
#ifdef PROP_MINER_GIT_SHA
        return PROP_MINER_GIT_SHA;
#else
        return "unknown";
#endif
    }

    double resolve_claimed_tmad_per_gpu() {
        if (const char* env = std::getenv("PROPMINER_CLAIMED_TMAD"); env && env[0]) {
            return std::max(0.0, std::atof(env));
        }
        // Conservative RTX 5090 estimate until live bench populates heartbeats.
        return 15.0;
    }

    std::string build_stratum_password() {
        std::string pw = "x";
        if (const char* env = std::getenv("PROPMINER_STRATUM_PASSWORD"); env && env[0]) {
            pw = env;
        }
        if (const char* d = std::getenv("PROPMINER_STRATUM_DIFF"); d && d[0]) {
            const long long diff = std::atoll(d);
            if (diff > 0 && pw.find("d=") == std::string::npos) {
                pw += ";d=" + std::to_string(diff);
            }
        }
        return pw;
    }

    void pool_log(const std::string& msg) {
        std::cout << "pool: " << msg << std::endl;
    }

    std::string miner_id_hex(const std::array<uint8_t, 16>& id) {
        char hex[33] = {0};
        for (int b = 0; b < 16; ++b) {
            std::snprintf(hex + b * 2, 3, "%02x",
                          static_cast<unsigned char>(id[b]));
        }
        return hex;
    }
}

WorkerOrchestrator::WorkerOrchestrator(const Config& cfg) : cfg_(cfg) {
    if (cfg_.pool_endpoints.empty()) {
        cfg_.pool_endpoints.push_back({cfg_.pool_host, cfg_.pool_port, cfg_.use_tls});
    }
    if (cfg_.stratum_endpoints.empty()) {
        cfg_.stratum_endpoints.push_back({"prl.kryptex.network", 7048, false});
    }
    if (const char* env = std::getenv("PROPMINER_USE_STRATUM"); env && env[0] == '1') {
        use_stratum_.store(true);
    }
    if (const char* env = std::getenv("PROPMINER_POOL_MODE"); env) {
        if (std::strcmp(env, "stratum") == 0) use_stratum_.store(true);
        if (std::strcmp(env, "grpc") == 0) use_stratum_.store(false);
    }
}

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

const WorkerOrchestrator::PoolEndpoint& WorkerOrchestrator::active_pool() const {
    const size_t idx = active_pool_index_.load() % cfg_.pool_endpoints.size();
    return cfg_.pool_endpoints[idx];
}

void WorkerOrchestrator::set_pool_state(PoolState state) {
    pool_state_.store(state);
}

std::string WorkerOrchestrator::pool_status_line() const {
    switch (pool_state_.load()) {
        case PoolState::Disconnected:
            if (!last_pool_error_.empty()) {
                return "hashrate: pool disconnected (" + last_pool_error_ + ")";
            }
            return "hashrate: pool disconnected (retrying)";
        case PoolState::Connecting:
            if (use_stratum_.load() && !cfg_.stratum_endpoints.empty()) {
                return "hashrate: connecting Stratum " + cfg_.stratum_endpoints[0].host + ":" +
                       std::to_string(cfg_.stratum_endpoints[0].port) + "...";
            }
            return "hashrate: connecting gRPC pool " + active_pool().host + "...";
        case PoolState::Registering:
            if (use_stratum_.load()) {
                return "hashrate: authorizing with Stratum pool...";
            }
            return "hashrate: registering with gRPC pool " + active_pool().host + "...";
        case PoolState::Streaming:
        case PoolState::AwaitingJob:
            return "hashrate: connected, awaiting first job from pool...";
        case PoolState::Mining:
            return "";
    }
    return "hashrate: pool status unknown";
}

int WorkerOrchestrator::backoff_ms(int attempt) const {
    const int base_ms = 5000;
    int max_ms = 120000;
    if (const char* env = std::getenv("PROPMINER_POOL_BACKOFF_MAX_MS"); env && env[0]) {
        max_ms = std::max(5000, std::atoi(env));
    }
    const int exp = std::min(attempt, 5);
    int ms = std::min(base_ms * (1 << exp), max_ms);
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<int> jitter(0, std::max(1, ms / 4));
    return ms + jitter(rng);
}

void WorkerOrchestrator::reset_pool_session() {
    registered_ = false;
    if (client_) {
        client_->disconnect();
        client_.reset();
    }
}

void WorkerOrchestrator::start_watchdog_if_needed() {
    if (!cfg_.enable_watchdog || watchdog_) return;
    watchdog_ = std::make_unique<Watchdog>();
    watchdog_->start([this]() {
        std::cerr << "[watchdog] stall detected — republishing current job\n";
        auto entry = bus_.drain_latest();
        if (entry.ctx) {
            for (auto& w : workers_) w->set_sigma(entry.ctx);
        }
    });
    for (auto& w : workers_) {
        w->set_watchdog(watchdog_.get());
    }
}

bool WorkerOrchestrator::ensure_connected_and_registered() {
    if (client_ && client_->connected() && registered_.load()) return true;

    reset_pool_session();
    set_pool_state(PoolState::Connecting);

    const auto& ep = active_pool();
    PearlGrpcClient::Options opts;
    opts.host = ep.host;
    opts.port = ep.port;
    opts.use_tls = ep.use_tls;
    opts.user_agent = cfg_.miner_version;
    if (const char* t = std::getenv("PROPMINER_GRPC_TIMEOUT_MS"); t && t[0]) {
        opts.connect_timeout_ms = std::max(5000, std::atoi(t));
    }
    client_ = std::make_unique<PearlGrpcClient>(opts);

    std::cerr << "[orchestrator] connecting to pool " << ep.host << ":" << ep.port
              << " (endpoint " << (active_pool_index_.load() + 1) << "/"
              << cfg_.pool_endpoints.size() << ")\n";

    if (!client_->connect()) {
        last_pool_error_ = client_->last_error();
        std::cerr << "grpc connect failed: " << last_pool_error_ << std::endl;
        reset_pool_session();
        active_pool_index_ = (active_pool_index_.load() + 1) % cfg_.pool_endpoints.size();
        return false;
    }

    set_pool_state(PoolState::Registering);

    proto::RegisterRequest req;
    req.wallet_address = cfg_.wallet_address;
    req.worker_name = cfg_.worker_name;
    req.miner_version = cfg_.miner_version;
    req.git_sha = resolve_git_sha();
    req.protocol_version = 2;
    req.k = static_cast<uint32_t>(cfg_.mining_config.k);
    req.gpu_cards = enumerate_gpu_cards();

    const double claimed_tmad = resolve_claimed_tmad_per_gpu();
    const size_t n_gpus = std::max<size_t>(1, req.gpu_cards.size());
    const double per_gpu_hr = claimed_tmad * kTmadsToHashesPerSec;
    req.claimed_total_hashrate = per_gpu_hr * static_cast<double>(n_gpus);
    for (auto& card : req.gpu_cards) {
        card.hashrate = per_gpu_hr;
    }

    proto::RegisterResponse resp;
    if (!client_->register_miner(req, resp)) {
        last_pool_error_ = client_->last_error();
        std::cerr << "register failed: " << last_pool_error_ << std::endl;
        reset_pool_session();
        const size_t prev = active_pool_index_.load();
        active_pool_index_ = (prev + 1) % cfg_.pool_endpoints.size();
        if (cfg_.pool_endpoints.size() > 1) {
            const auto& next = cfg_.pool_endpoints[active_pool_index_.load()];
            std::cerr << "[orchestrator] failing over to " << next.host << ":"
                      << next.port << "\n";
        }
        return false;
    }
    if (!resp.success) {
        last_pool_error_ = resp.error_message.empty() ? "register rejected" : resp.error_message;
        std::cerr << "register rejected: " << last_pool_error_ << std::endl;
        reset_pool_session();
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
    reconnect_attempt_ = 0;
    last_pool_error_.clear();
    std::cerr << "[orchestrator] pool registered on " << ep.host << " (miner_id="
              << miner_id_hex(miner_id_) << ")\n";
    return true;
}

void WorkerOrchestrator::publish_job_from_assignment(const proto::JobAssignment& ja) {
    const bool first_job = !jobs_received_.exchange(true);
    if (first_job) {
        std::cerr << "[orchestrator] first pool job received (job_id="
                  << miner_id_hex(ja.job_id)
                  << ", block_height=" << ja.block_height << ")\n";
        set_pool_state(PoolState::Mining);
        start_watchdog_if_needed();
    }

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
        w->set_target_nbits(ja.target_nbits);
        w->set_sigma(ctx);
    }
    live_share_target_nbits_.store(ja.target_nbits);
    share_trace("job-installed",
                "job_id=" + miner_id_hex(ja.job_id) +
                " height=" + std::to_string(ja.block_height) +
                " share_target=" + nbits_hex(ja.target_nbits) +
                " sigma=" + hex_prefix(ja.sigma.data(), ja.sigma.size(), 4));
}

void WorkerOrchestrator::handle_pool_event(const proto::PoolEvent& evt) {
    switch (evt.type) {
        case proto::PoolEventType::Job:
            publish_job_from_assignment(evt.job);
            break;
        case proto::PoolEventType::Vardiff: {
            const uint32_t nbits = evt.vardiff.new_target_nbits;
            live_share_target_nbits_.store(nbits);
            for (auto& w : workers_) w->set_target_nbits(nbits);
            auto entry = bus_.drain_latest();
            if (entry.ctx) {
                entry.ctx->set_target_nbits(nbits);
            }
            break;
        }
        case proto::PoolEventType::ShareResult:
            if (evt.share_result.accepted) shares_accepted_.fetch_add(1);
            else shares_rejected_.fetch_add(1);
            share_log(evt.share_result.accepted ? "accepted" : "rejected",
                      evt.share_result.outcome +
                      (evt.share_result.is_block_find ? " BLOCK" : "") +
                      " (totals " + std::to_string(shares_accepted_.load()) + "a/" +
                      std::to_string(shares_rejected_.load()) + "r)");
            break;
        case proto::PoolEventType::Error:
            std::cerr << "pool error: " << evt.error.message << std::endl;
            if (evt.error.fatal) stop();
            break;
        case proto::PoolEventType::Reconnect: {
            const uint32_t wait_s = evt.reconnect.wait_seconds;
            std::cerr << "[orchestrator] pool requested reconnect in " << wait_s
                      << "s\n";
            if (wait_s > 0) {
                std::this_thread::sleep_for(std::chrono::seconds(wait_s));
            }
            reset_pool_session();
            set_pool_state(PoolState::Disconnected);
            pool_reconnect_requested_.store(true);
            break;
        }
        default:
            break;
    }
}

bool WorkerOrchestrator::run_grpc_session() {
    if (!ensure_connected_and_registered()) {
        return false;
    }

    proto::AuthEvent auth;
    {
        std::lock_guard<std::mutex> lk(session_mtx_);
        auth.miner_id = miner_id_;
        auth.session_token = session_token_;
    }
    set_pool_state(PoolState::Streaming);
    if (!client_->start_mining_stream(auth)) {
        last_pool_error_ = client_->last_error();
        std::cerr << "mining stream failed: " << last_pool_error_ << std::endl;
        reset_pool_session();
        return false;
    }
    std::cerr << "[orchestrator] gRPC mining stream open, awaiting JobAssignment\n";
    set_pool_state(PoolState::AwaitingJob);
    int stream_idle_polls = 0;

    while (!stop_flag_ && client_->connected()) {
        proto::PoolEvent evt;
        if (client_->receive_event(evt)) {
            stream_idle_polls = 0;
            handle_pool_event(evt);
            if (pool_reconnect_requested_.exchange(false)) break;
        } else if (!client_->connected()) {
            break;
        } else {
            ++stream_idle_polls;
            if (stream_idle_polls % 12 == 0) {
                std::cerr << "[orchestrator] gRPC streaming, no pool event yet ("
                          << (stream_idle_polls * 5) << "s)\n";
            }
        }
    }
    last_pool_error_ = client_ ? client_->last_error() : "gRPC stream closed";
    reset_pool_session();
    set_pool_state(PoolState::Disconnected);
    return true;
}

void WorkerOrchestrator::run_stratum_session() {
    set_pool_state(PoolState::Connecting);
    for (const auto& ep : cfg_.stratum_endpoints) {
        if (stop_flag_) break;

        PearlStratumClient::Options opts;
        opts.host = ep.host;
        opts.port = ep.port;
        opts.wallet = cfg_.wallet_address;
        opts.worker = cfg_.worker_name;
        opts.password = build_stratum_password();
        opts.user_agent = cfg_.miner_version;
        if (const char* t = std::getenv("PROPMINER_GRPC_TIMEOUT_MS"); t && t[0]) {
            opts.connect_timeout_ms = std::max(5000, std::atoi(t));
        }

        stratum_client_ = std::make_unique<PearlStratumClient>(opts);
        stratum_client_->set_callbacks(
            [this](const proto::JobAssignment& ja, const std::string& job_id) {
                {
                    std::lock_guard<std::mutex> lk(stratum_job_mtx_);
                    current_stratum_job_id_ = job_id;
                }
                publish_job_from_assignment(ja);
            },
            [this](uint32_t nbits) {
                live_share_target_nbits_.store(nbits);
                for (auto& w : workers_) w->set_target_nbits(nbits);
                auto entry = bus_.drain_latest();
                if (entry.ctx) entry.ctx->set_target_nbits(nbits);
            },
            [this](bool accepted, const std::string& msg) {
                if (accepted) shares_accepted_.fetch_add(1);
                else shares_rejected_.fetch_add(1);
                share_log(accepted ? "accepted" : "rejected",
                          msg + " (totals " + std::to_string(shares_accepted_.load()) +
                          "a/" + std::to_string(shares_rejected_.load()) + "r/" +
                          std::to_string(shares_dropped_.load()) + "d)");
            });

        std::cerr << "[orchestrator] connecting Stratum " << ep.host << ":" << ep.port
                  << " (Kryptex direct mining :7048)\n";
        set_pool_state(PoolState::Registering);
        if (!stratum_client_->connect()) {
            last_pool_error_ = stratum_client_->last_error();
            std::cerr << "[orchestrator] Stratum connect failed: " << last_pool_error_ << "\n";
            stratum_client_.reset();
            continue;
        }

        reconnect_attempt_ = 0;
        last_pool_error_.clear();
        set_pool_state(PoolState::AwaitingJob);
        while (!stop_flag_ && stratum_client_->connected()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        last_pool_error_ = stratum_client_ ? stratum_client_->last_error() : "stratum disconnected";
        stratum_client_->disconnect();
        stratum_client_.reset();
    }
    set_pool_state(PoolState::Disconnected);
    const int wait_ms = backoff_ms(reconnect_attempt_++);
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
}

void WorkerOrchestrator::network_thread_func() {
    while (!stop_flag_) {
        if (use_stratum_.load()) {
            run_stratum_session();
            continue;
        }

        if (run_grpc_session()) {
            grpc_fail_cycles_ = 0;
            continue;
        }

        const int wait_ms = backoff_ms(reconnect_attempt_++);
        std::cerr << "[orchestrator] gRPC pool connect/register failed; retry in "
                  << (wait_ms / 1000) << "s"
                  << (last_pool_error_.empty() ? "" : " (" + last_pool_error_ + ")")
                  << std::endl;
        set_pool_state(PoolState::Disconnected);
        grpc_fail_cycles_++;
        if (grpc_fail_cycles_ >= 1 && !cfg_.stratum_endpoints.empty()) {
            use_stratum_.store(true);
            std::cerr << "[orchestrator] gRPC unavailable on all endpoints; "
                         "switching to Kryptex Stratum :7048 fallback\n";
            grpc_fail_cycles_ = 0;
            continue;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
    }
}

void WorkerOrchestrator::ping_thread_func() {
    while (!stop_flag_) {
        std::this_thread::sleep_for(std::chrono::seconds(15));
        if (stop_flag_) break;
        if (!client_ || !client_->connected() || !registered_.load()) continue;

        proto::MinerEvent evt;
        evt.type = proto::MinerEventType::Ping;
        proto::PingEvent ping;
        ping.timestamp = now_unix_ms();
        evt.payload = ping.encode();
        {
            std::lock_guard<std::mutex> lk(session_mtx_);
            evt.seq = ++miner_event_seq_;
        }
        if (!client_->send_event(evt)) {
            std::cerr << "[orchestrator] pool ping failed: " << client_->last_error()
                      << std::endl;
        }
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
                shares_dropped_.fetch_add(1);
                share_log("dropped", "reason=missing_sigma_context nonce=" +
                                     std::to_string(raw.nonce));
                continue;
            }
            ShareBuilder builder(raw.job.config);
            if (!ShareBuilder::VerifyShare(raw, *raw.sigma_ctx)) {
                shares_dropped_.fetch_add(1);
                share_log("dropped", "reason=verify_failed nonce=" +
                                     std::to_string(raw.nonce));
                continue;
            }
            // Drop if vardiff tightened while share was queued (ARC ShareTargetGuard).
            {
                const uint32_t mined = raw.installed_target_nbits
                    ? raw.installed_target_nbits
                    : raw.job.target_nbits;
                const uint32_t live = live_share_target_nbits_.load();
                const uint32_t guard = tighter_target_nbits(live, mined);
                if (guard != 0 && guard != mined && !raw.claimed_hash.empty()) {
                    const uint64_t daf = raw.job.config.difficulty_adjustment_factor();
                    const auto diag = diagnose_share_target(
                        raw.claimed_hash.data(), guard, daf);
                    if (!diag.clears_with_daf) {
                        shares_dropped_.fetch_add(1);
                        share_log("dropped",
                                  "reason=vardiff_tightened nonce=" +
                                  std::to_string(raw.nonce) +
                                  " mined=" + nbits_hex(mined) +
                                  " live=" + nbits_hex(guard));
                        continue;
                    }
                }
            }
            std::vector<uint8_t> proof;
            if (use_stratum_.load()) {
                proof = builder.build_stratum_plain_proof(raw, *raw.sigma_ctx);
            } else {
                proof = builder.build(raw, *raw.sigma_ctx);
            }
            if (proof.empty()) {
                shares_dropped_.fetch_add(1);
                share_log("dropped", "reason=build_failed_below_target nonce=" +
                                     std::to_string(raw.nonce));
                continue;
            }
            share_log("build-ok",
                      "nonce=" + std::to_string(raw.nonce) +
                      " proof_bytes=" + std::to_string(proof.size()) +
                      " format=" + (use_stratum_.load() ? "stratum" : "grpc"));
            if (use_stratum_.load() && stratum_client_ && stratum_client_->connected()) {
                const std::string job_id =
                    stratum_client_->job_id_for_sigma(raw.job.sigma);
                if (job_id.empty()) {
                    shares_dropped_.fetch_add(1);
                    share_log("dropped", "reason=no_job_id_for_sigma nonce=" +
                                         std::to_string(raw.nonce));
                    continue;
                }
                shares_submitted_.fetch_add(1);
                share_log("submitting",
                          "nonce=" + std::to_string(raw.nonce) +
                          " job_id=" + job_id.substr(0, std::min<size_t>(16, job_id.size())) +
                          " proof_bytes=" + std::to_string(proof.size()) +
                          " target=" + nbits_hex(raw.installed_target_nbits
                                                    ? raw.installed_target_nbits
                                                    : raw.job.target_nbits));
                if (!stratum_client_->submit_plain_proof(job_id, proof, raw.nonce)) {
                    shares_dropped_.fetch_add(1);
                    share_log("dropped", "reason=send_failed nonce=" +
                                         std::to_string(raw.nonce) +
                                         " err=" + stratum_client_->last_error());
                }
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
    shares_found_.fetch_add(1);
    share_log("found",
              "nonce=" + std::to_string(share.nonce) +
              " rows=" + std::to_string(share.a_row_indices.size()) +
              " cols=" + std::to_string(share.b_col_indices.size()) +
              " target=" + nbits_hex(share.installed_target_nbits
                                        ? share.installed_target_nbits
                                        : share.job.target_nbits) +
              " sigma=" + hex_prefix(share.job.sigma.data(), share.job.sigma.size(), 4));
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
                     "host management, seed upload, and PCIe feeding.\n";
    }
    if (share_trace_enabled()) {
        std::cerr << "[orchestrator] PROPMINER_VERBOSE_SHARES=1 — intensive share "
                     "pipeline logging enabled\n";
    }

    MiningConfig tuned_config = cfg_.mining_config;
    const int profile_n = cfg_.mining_config.n;
    int tuned_batch = cfg_.batch_size;
    int tuned_cluster_m = Rtx5090Profile::kProdDefaultClusterM;
    int tuned_carveout = -1;

    const char* cluster_env = std::getenv("PEARL_GEMM_CONSUMER_CLUSTER_M");
    if (cluster_env && cluster_env[0]) {
        tuned_cluster_m = normalize_cluster_m(std::max(1, std::atoi(cluster_env)));
    }

    GemmCapi capi;
    if (const int kv = capi.validate_kernel_selection(); kv != 0) {
        return 1;
    }
    const char* built_knobs = capi.build_knobs();
    std::cerr << "[orchestrator] Active transcript kernel: "
              << (capi.active_kernel_name() ? capi.active_kernel_name()
                                            : "unknown")
              << "\n";
    KernelKnobCache knob_cache;
    for (int idx : indices) {
        if (auto cached = knob_cache.load(idx)) {
            if (auto err = KernelKnobCache::strict_validate(built_knobs, *cached)) {
                std::cerr << "[orchestrator] ERROR: PROPMINER_STRICT_KNOB_CACHE=1: "
                          << *err << " (GPU " << idx << ")\n";
                return 1;
            }
            if (built_knobs &&
                !KernelKnobCache::manifest_matches(built_knobs,
                                                   cached->manifest.c_str())) {
                std::cerr << "[orchestrator] WARN: built kernel knobs ("
                          << built_knobs << ") != cached sweep winner ("
                          << cached->manifest << ")\n";
            } else if (cached->self_test_ok) {
                std::cerr << "[orchestrator] Kernel knob cache GPU " << idx
                          << ": " << cached->manifest << " ("
                          << cached->hashrate << " H/s)\n";
            }
        }
    }

    const char* autotune_env = std::getenv("PROPMINER_AUTOTUNE");
    const bool force_autotune =
        autotune_env && std::strcmp(autotune_env, "force") == 0;

    // Offline tune cache: cluster/carveout only (N and batch stay on production path).
    if (cfg_.use_tune_cache && !cfg_.autotune && cfg_.speed_test_seconds == 0) {
        TuneCache tune_cache;
        for (int idx : indices) {
            if (auto cached = tune_cache.load(idx)) {
                tuned_cluster_m = normalize_cluster_m(cached->cluster_m);
                tuned_carveout = cached->carveout_percent;
                std::cerr << "[orchestrator] Tune cache GPU " << idx
                          << ": cluster_m=" << tuned_cluster_m
                          << " carveout=" << tuned_carveout
                          << " (N=" << tuned_config.n
                          << ", batch via mine_batch cache)\n";
            }
        }
    }

    MiningConfig::warn_if_cluster_m_mismatch(tuned_cluster_m);

    TuneCache tune_cache;
    if (cfg_.autotune && cfg_.speed_test_seconds == 0) {
        // Try the persistent cache first so reconnects pick up the last good
        // config without re-benchmarking.
        bool cache_hit = false;
        if (!force_autotune) {
            for (int idx : indices) {
                if (auto cached = tune_cache.load(idx)) {
                    tuned_cluster_m = normalize_cluster_m(cached->cluster_m);
                    tuned_carveout = cached->carveout_percent;
                    cache_hit = true;
                    std::cerr << "[orchestrator] Using cached autotune dispatch for GPU "
                              << idx << " (cluster_m=" << tuned_cluster_m
                              << " carveout=" << tuned_carveout
                              << "; N/batch unchanged from production path)"
                              << std::endl;
                }
            }
        } else {
            std::cerr << "[orchestrator] PROPMINER_AUTOTUNE=force: "
                         "ignoring runtime autotune cache\n";
        }

        if (!cache_hit) {
            std::cerr << "[orchestrator] Running per-GPU autotune..." << std::endl;
            for (int idx : indices) {
                GpuTuner tuner(idx);
                auto result = tuner.tune(5.0, 3);
                if (result.hashrate > 0.0) {
                    tuned_config = result.config;
                    tuned_batch = result.batch_size;
                    tuned_cluster_m = normalize_cluster_m(result.cluster_m);
                    tuned_carveout = result.carveout_percent;
                    tune_cache.save(idx, result);
                }
            }
        }

        if (tuned_config.m == Rtx5090Profile::kDefaultM &&
            tuned_config.n < profile_n) {
            std::cerr << "[orchestrator] RTX 5090: clamping autotune N "
                      << tuned_config.n << " -> " << profile_n << "\n";
            tuned_config.n = profile_n;
        }
        if (tuned_config.m == Rtx5090Profile::kDefaultM &&
            tuned_batch > Rtx5090Profile::kMaxMineBatch) {
            std::cerr << "[orchestrator] RTX 5090: clamping autotune batch "
                      << tuned_batch << " -> " << Rtx5090Profile::kMaxMineBatch
                      << "\n";
            tuned_batch = Rtx5090Profile::kMaxMineBatch;
        }
    } else if (cfg_.speed_test_seconds == 0 &&
               tuned_config.m == Rtx5090Profile::kDefaultM) {
        tuned_batch = MineBatchCache::resolve(
            indices.empty() ? 0 : indices[0],
            tuned_config.m, tuned_config.n,
            cfg_.batch_size > 0 ? cfg_.batch_size : Rtx5090Profile::kDefaultMineBatch);

        const char* batch_tune_env = std::getenv("PROPMINER_BATCH_TUNE");
        if (batch_tune_env && batch_tune_env[0] == '1' && !std::getenv("PROPMINER_BATCH")) {
            std::cerr << "[orchestrator] Running mine batch autotune..." << std::endl;
            for (int idx : indices) {
                GpuTuner tuner(idx);
                auto result = tuner.tune_mine_batch(tuned_config, 12.0, 2);
                if (result.hashrate > 0.0) {
                    tuned_batch = result.batch_size;
                    MineBatchResult cached;
                    cached.m = tuned_config.m;
                    cached.n = tuned_config.n;
                    cached.batch = tuned_batch;
                    cached.hashrate = result.hashrate;
                    cached.use_graph = result.use_graph;
                    MineBatchCache cache;
                    cache.save(idx, cached);
                }
            }
        }
    }

    const bool bench_mode = cfg_.speed_test_seconds > 0;
    if (!bench_mode && tuned_config.m == Rtx5090Profile::kDefaultM) {
        if (tuned_cluster_m > 1) {
            const char* allow = std::getenv("PROPMINER_ALLOW_CLUSTER_M");
            if (!allow || allow[0] != '1') {
                std::cerr << "[orchestrator] WARN: cluster_m=" << tuned_cluster_m
                          << " disabled for share safety (gpu-hit + verify-fail). "
                          << "Set PROPMINER_ALLOW_CLUSTER_M=1 to override.\n";
                tuned_cluster_m = 1;
            }
        }
        setenv("PEARL_GEMM_CONSUMER_CLUSTER_M",
               std::to_string(tuned_cluster_m).c_str(), 1);
        if (tuned_carveout >= 0) {
            setenv("PEARL_GEMM_CONSUMER_CARVEOUT",
                   std::to_string(tuned_carveout).c_str(), 1);
        }
        std::cerr << "[orchestrator] RTX 5090 prod: cluster_m=" << tuned_cluster_m;
        if (tuned_carveout >= 0) {
            std::cerr << " carveout=" << tuned_carveout << "%";
        }
        std::cerr << " (tune cache "
                  << (cfg_.use_tune_cache ? "on" : "off") << ")\n";
    }

    const bool rtx5090_shape =
        (tuned_config.m == Rtx5090Profile::kDefaultM &&
         tuned_config.n >= Rtx5090Profile::kDefaultN);
    if (rtx5090_shape) {
        std::cerr << "[orchestrator] RTX 5090: GPU-isolated path (VRAM-resident B, "
                     "CUDA graphs, pinned PCIe, no CPU mining)\n";
    }
    {
        const int ctas = Rtx5090Profile::tiles(tuned_config.m, tuned_config.n);
        const int tail = ctas % Rtx5090Profile::kSMCount;
        std::cerr << "[orchestrator] GEMM grid: M=" << tuned_config.m
                  << " N=" << tuned_config.n << " K=" << tuned_config.k
                  << " CTAs=" << ctas
                  << " waves~" << ((ctas + Rtx5090Profile::kSMCount - 1)
                                    / Rtx5090Profile::kSMCount)
                  << " tail_slots=" << tail
                  << " batch=" << tuned_batch << "\n";
    }

    if (!bench_mode) {
        std::cerr << "[orchestrator] Production mine mode: "
                  << cfg_.pool_endpoints.size() << " gRPC endpoint(s), "
                  << cfg_.stratum_endpoints.size() << " Stratum fallback(s)"
                  << (use_stratum_.load() ? " [Stratum-first]" : " [gRPC-first]")
                  << "\n";
    }

    const auto gpu_cards = enumerate_gpu_cards();
    gpu_uuids_.clear();
    for (const auto& c : gpu_cards) gpu_uuids_.push_back(c.uuid);

    uint32_t bench_target_nbits = 0;
    std::shared_ptr<SigmaContext> bench_ctx;
    std::string bench_git_sha;
    if (bench_mode) {
        bench_git_sha = resolve_git_sha();
        Job job;
        job.sigma.fill(0xab);
        job.b_seed.fill(0xcd);
        job.config = tuned_config;
        job.job_key = derive_job_key(job.sigma, tuned_config);
        job.audit_k = 4;
        job.target_nbits = 0x01111111;
        bench_target_nbits = job.target_nbits;
        bench_ctx = std::make_shared<SigmaContext>(job, tuned_config);
        // WSL2/Salad: batch=1 finishes inside the 180s bench window; batch=4 may not.
        int bench_batch = 1;
        if (const char* bb = std::getenv("PROPMINER_BENCH_BATCH")) {
            bench_batch = std::max(1, std::atoi(bb));
        }
        tuned_batch = bench_batch;
        const bool bench_graph = !bench_no_graph_enabled();
        std::cerr << "[orchestrator] Benchmark mode: local job, no pool connection"
                  << " (batch=" << tuned_batch
                  << " graph=" << (bench_graph ? "on" : "off") << ")\n";
        std::cerr << "[orchestrator] Bench reports TMAD/s (= pool/community TH/s) "
                     "and protocol H/s (DAF-normalized; ~1000x smaller label)\n";
        std::cerr << "[orchestrator] Set PROPMINER_BENCH_JSON=1 for one-line JSON "
                     "at bench end (track optimizations over time)\n";
    }

    for (int idx : indices) {
        auto w = std::make_unique<GpuWorker>(idx, static_cast<int>(workers_.size()),
                                             tuned_config, this);
        w->set_matmuls_per_poll(tuned_batch);
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
        threads_.emplace_back(&WorkerOrchestrator::ping_thread_func, this);
    }

    // Stats loop.
    const int bench_grace_sec =
        cfg_.speed_test_seconds > 0 ? bench_grace_seconds() : 0;
    const int bench_max_seconds =
        cfg_.speed_test_seconds > 0
            ? cfg_.speed_test_seconds + bench_grace_sec
            : 0;

    while (!stop_flag_) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        const HashrateMetrics metrics =
            aggregate_worker_metrics(workers_, tuned_config);
        total_hashrate_ = metrics.protocol_hps;
        uint64_t bench_iters = 0;
        for (auto& w : workers_) {
            bench_iters += w->total_iters();
        }

        if (metrics.tmad_per_sec > 0.0 || metrics.protocol_hps > 0.0) {
            print_hashrate_metrics_line(stdout, "hashrate: ", metrics);
            const size_t pending_acks =
                (use_stratum_.load() && stratum_client_ && stratum_client_->connected())
                    ? stratum_client_->pending_submit_count()
                    : 0;
            std::printf("shares: build=%s found=%llu submitted=%llu accepted=%llu "
                        "rejected=%llu dropped=%llu pending_acks=%zu\n",
                        resolve_git_sha().c_str(),
                        static_cast<unsigned long long>(shares_found_.load()),
                        static_cast<unsigned long long>(shares_submitted_.load()),
                        static_cast<unsigned long long>(shares_accepted_.load()),
                        static_cast<unsigned long long>(shares_rejected_.load()),
                        static_cast<unsigned long long>(shares_dropped_.load()),
                        pending_acks);
        } else if (bench_mode) {
            std::cout << "hashrate: waiting for first batch..." << std::endl;
        } else {
            const std::string status = pool_status_line();
            if (!status.empty()) {
                std::cout << status << std::endl;
            }
        }
        if (cfg_.speed_test_seconds > 0) {
            static int elapsed = 0;
            elapsed += 5;
            if (metrics.protocol_hps == 0.0 && metrics.tmad_per_sec == 0.0 &&
                elapsed >= 30 && (elapsed % 30 == 0)) {
                std::cerr << "[bench] " << elapsed << "s elapsed, still awaiting first "
                          << "batch completion (iters=" << bench_iters
                          << "; check nvidia-smi for GPU util)\n";
            }
            if (elapsed >= cfg_.speed_test_seconds) {
                bool finish = false;
                HashrateMetrics final_metrics = metrics;
                if (bench_iters > 0) {
                    final_metrics = hashrate_metrics_from_iters(
                        tuned_config, bench_iters, elapsed, tuned_batch);
                    total_hashrate_ = final_metrics.protocol_hps;
                }
                if (final_metrics.protocol_hps > 0.0 ||
                    final_metrics.tmad_per_sec > 0.0) {
                    finish = true;
                } else if (elapsed >= bench_max_seconds) {
                    finish = true;
                    std::cerr << "[bench] no completed batch after " << elapsed
                              << "s (iters=" << bench_iters << ")\n";
                } else if (elapsed == cfg_.speed_test_seconds) {
                    std::cerr << "[bench] extending up to " << bench_max_seconds
                              << "s grace for first batch (iters="
                              << bench_iters << ")\n";
                }

                if (finish) {
                    if (final_metrics.protocol_hps > 0.0 ||
                        final_metrics.tmad_per_sec > 0.0) {
                        std::cout << "benchmark complete" << std::endl;
                        print_hashrate_metrics_line(stdout, "benchmark: ", final_metrics);
                        std::cerr << "[bench] finished after " << elapsed << "s\n";
                        print_hashrate_metrics_line(stderr, "[bench] ", final_metrics);
                        if (bench_json_enabled()) {
                            print_hashrate_metrics_json(
                                stdout, final_metrics, tuned_config,
                                elapsed, bench_iters, cfg_.speed_test_seconds,
                                bench_git_sha.c_str(),
                                cfg_.miner_version.c_str());
                        }
                    } else {
                        std::cout << "benchmark incomplete: no completed batch"
                                  << std::endl;
                        std::cerr << "[bench] finished after " << elapsed
                                  << "s with no completed batch (iters="
                                  << bench_iters << ")\n";
                    }
                    stop();
                    break;
                }
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
