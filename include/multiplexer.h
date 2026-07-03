#pragma once
#include <vector>
#include <atomic>
#include <thread>
#include <memory>
#include <mutex>
#include <string>
#include <functional>
#include <condition_variable>

#include "stratum.h"

namespace propminer {

struct MinerConfig {
    std::string pool_host = "prl.kryptex.network";
    int pool_port = 7048;
    std::string wallet;
    std::string worker = "propminer";
    std::string password = "x";
    std::vector<int> gpu_indices;  // empty = all GPUs
    int intensity = 100;           // 0-100
    bool show_hashrate = true;
};

struct GpuStats {
    int index = -1;
    std::string name;
    double hashrate = 0.0;       // H/s
    uint64_t accepted = 0;
    uint64_t rejected = 0;
    uint64_t total_computed = 0;
    float temperature = 0.0f;
    float power = 0.0f;          // watts
};

class Multiplexer {
public:
    explicit Multiplexer(const MinerConfig& config);
    ~Multiplexer();

    int init();
    void start();
    void stop();
    void wait();
    bool is_connected() const;
    bool reconnect();

    std::vector<GpuStats> get_stats() const;
    double get_total_hashrate() const;

private:
    void gpu_worker(int gpu_index, std::shared_ptr<GpuStats> stats);
    void monitor_thread_func();

    MinerConfig config_;
    std::vector<std::thread> gpu_threads_;
    std::unique_ptr<std::thread> monitor_thread_;

    std::atomic<bool> running_{false};

    std::unique_ptr<StratumClient> stratum_;

    std::mutex stats_mutex_;
    std::vector<std::shared_ptr<GpuStats>> stats_;

    // Job notification
    std::mutex job_notify_mutex_;
    std::condition_variable job_notify_cv_;
    bool new_job_{false};

    int gpu_count_ = 0;
};

} // namespace propminer