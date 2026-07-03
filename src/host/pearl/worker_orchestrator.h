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
#include "job_bus.h"
#include "pearl_types.h"

namespace pearl {

// High-level orchestrator: owns the gRPC session, JobBus, and GPU workers.
class WorkerOrchestrator : public IShareSink {
public:
    struct Config {
        std::string pool_host;
        int pool_port = 443;
        bool use_tls = true;
        std::string wallet_address;
        std::string worker_name = "propminer";
        std::string miner_version = "propminer/1.0";
        std::vector<int> gpu_indices; // empty = all
        MiningConfig mining_config;
        int batch_size = 8;           // matmuls per poll
        int speed_test_seconds = 0;   // >0 => benchmark and exit
        bool autotune = true;         // run per-GPU autotune at startup
        bool enable_watchdog = true;  // reset GPU context on TDR/timeout
        bool disable_cpu_mining = true; // PropMiner has no CPU mining path
    };

    explicit WorkerOrchestrator(const Config& cfg);
    ~WorkerOrchestrator();

    // Connect/register/start workers. Blocks until stop().
    int run();

    // Async stop.
    void stop();

    // IShareSink implementation.
    void submit(const ShareFound& share) override;

    // Current aggregate hashrate (H/s).
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

    Config cfg_;
    std::unique_ptr<PearlGrpcClient> client_;
    JobBus bus_;
    std::vector<std::unique_ptr<GpuWorker>> workers_;
    std::vector<std::thread> threads_;

    std::atomic<bool> stop_flag_{false};
    std::atomic<bool> registered_{false};
    std::atomic<double> total_hashrate_{0.0};

    // Session state.
    std::mutex session_mtx_;
    std::array<uint8_t, 16> miner_id_{};
    std::string session_token_;
    uint64_t miner_event_seq_ = 1;

    // Share queue.
    std::mutex share_mtx_;
    std::condition_variable share_cv_;
    std::vector<std::vector<uint8_t>> pending_shares_;
};

} // namespace pearl
