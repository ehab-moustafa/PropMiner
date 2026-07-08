#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gpu_worker.h"
#include "grpc_client.h"
#include "pearl_stratum_client.h"
#include "job_bus.h"
#include "pearl_types.h"
#include "rtx5090_profile.h"
#include "watchdog.h"

namespace pearl {

// High-level orchestrator: owns the gRPC session, JobBus, and GPU workers.
class WorkerOrchestrator : public IShareSink {
public:
    struct PoolEndpoint {
        std::string host;
        int port = 443;
        bool use_tls = true;
    };

    enum class PoolState {
        Disconnected,
        Connecting,
        Registering,
        Streaming,
        AwaitingJob,
        Mining,
    };

    struct Config {
        std::string pool_host;
        int pool_port = 443;
        bool use_tls = true;
        std::vector<PoolEndpoint> pool_endpoints;
        std::vector<PoolEndpoint> stratum_endpoints;
        std::string wallet_address;
        std::string worker_name = "propminer";
        std::string miner_version = "propminer/1.0";
        std::vector<int> gpu_indices; // empty = all
        MiningConfig mining_config;
        int batch_size = Rtx5090Profile::kDefaultMineBatch;       // PROPMINER_BATCH
        int graph_batch_size = Rtx5090Profile::kDefaultGraphBatch; // PROPMINER_GRAPH_BATCH
        int speed_test_seconds = 0;   // >0 => benchmark and exit
        bool autotune = false;        // PROPMINER_AUTOTUNE=1 to sweep at startup
        bool use_tune_cache = true;   // PROPMINER_USE_TUNE_CACHE=0 to disable
        bool enable_watchdog = true;  // reset GPU context on TDR/timeout
        bool disable_cpu_mining = true; // PropMiner has no CPU mining path
    };

    explicit WorkerOrchestrator(const Config& cfg);
    ~WorkerOrchestrator();

    // Connect/register/start workers. Blocks until stop().
    int run();

    // Async stop.
    void stop();

    // IShareSink implementation — queues raw share for async build/send.
    void submit(const ShareFound& share) override;

    // Current aggregate hashrate (H/s, DAF-normalized tiles/s).
    double total_hashrate() const;

private:
    void network_thread_func();
    void share_sender_thread_func();
    void heartbeat_thread_func();
    void publish_job_from_assignment(const proto::JobAssignment& ja);
    void handle_pool_event(const proto::PoolEvent& evt);
    void submit_share(const ShareFound& share);
    bool ensure_connected_and_registered();
    std::vector<proto::GpuCard> enumerate_gpu_cards();
    void reset_pool_session();
    void start_watchdog_if_needed();
    int backoff_ms(int attempt) const;
    const PoolEndpoint& active_pool() const;
    void set_pool_state(PoolState state);
    std::string pool_status_line() const;
    void ping_thread_func();

    void run_stratum_session();
    bool run_grpc_session();

    static constexpr double kTmadsToHashesPerSec = 1e12;

    Config cfg_;
    std::unique_ptr<PearlGrpcClient> client_;
    std::unique_ptr<Watchdog> watchdog_;
    JobBus bus_;
    std::vector<std::unique_ptr<GpuWorker>> workers_;
    std::vector<std::string> gpu_uuids_;
    std::vector<std::thread> threads_;

    std::atomic<bool> stop_flag_{false};
    std::atomic<bool> registered_{false};
    std::atomic<bool> jobs_received_{false};
    std::atomic<PoolState> pool_state_{PoolState::Disconnected};
    std::atomic<size_t> active_pool_index_{0};
    std::atomic<bool> pool_reconnect_requested_{false};
    std::atomic<bool> use_stratum_{false};
    int grpc_fail_cycles_ = 0;
    std::unique_ptr<PearlStratumClient> stratum_client_;
    std::mutex stratum_job_mtx_;
    std::string current_stratum_job_id_;
    std::atomic<uint32_t> pending_target_nbits_{0};
    std::atomic<uint32_t> live_share_target_nbits_{0};
    std::atomic<double> total_hashrate_{0.0};
    std::atomic<uint64_t> shares_found_{0};
    std::atomic<uint64_t> shares_submitted_{0};
    std::atomic<uint64_t> shares_accepted_{0};
    std::atomic<uint64_t> shares_rejected_{0};
    bool thermal_paused_ = false;
    int reconnect_attempt_ = 0;
    std::string last_pool_error_;

    // Session state.
    std::mutex session_mtx_;
    std::array<uint8_t, 16> miner_id_{};
    std::string session_token_;
    uint64_t miner_event_seq_ = 1;

    // Share queue: raw ShareFound built/sent on share_sender thread.
    std::mutex share_mtx_;
    std::condition_variable share_cv_;
    std::vector<ShareFound> pending_share_found_;
};

} // namespace pearl
