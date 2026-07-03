#pragma once
#include <string>
#include <functional>
#include <cstdint>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <queue>
#include <unordered_map>

namespace propminer {

struct JobAssignment {
    std::string job_id;
    std::string prevhash;
    std::string coinb1;
    std::string coinb2;
    std::string merkle_branch;  // JSON array string
    std::string version;
    uint32_t target_nbits = 0;
    std::string ntime;
    bool clean_jobs = false;
    std::string extranonce2;
    size_t extranonce_size = 0;
};

struct ShareResult {
    bool accepted = false;
    std::string outcome;
    bool is_block_find = false;
    int gpu_index = -1;
};

struct DifficultyAdjust {
    double new_difficulty = 0;
};

using JobCallback = std::function<void(const JobAssignment&)>;
using ShareResultCallback = std::function<void(const ShareResult&)>;
using VardiffCallback = std::function<void(const DifficultyAdjust&)>;

// Ring buffer for lock-free share submissions from GPU threads to the network thread.
struct ShareSubmission {
    std::string job_id;
    uint64_t extranonce2_counter;  // raw counter, formatted as hex in submit_loop
    uint32_t nonce;
    std::string ntime;
    int gpu_index;                 // which GPU submitted this share
};

static constexpr int SHARE_RING_SIZE = 1024;

class StratumClient {
public:
    StratumClient();
    ~StratumClient();

    bool connect(const std::string& host, int port,
                 const std::string& wallet, const std::string& worker,
                 const std::string& password = "x");

    void start_mining();
    void stop_mining();
    bool reconnect();

    // Call from GPU threads to enqueue a share for submission
    bool submit_share(const std::string& job_id, uint32_t nonce,
                      const std::string& ntime,
                      int gpu_index);

    void set_callbacks(JobCallback job_cb,
                       ShareResultCallback result_cb,
                       VardiffCallback vardiff_cb);

    bool is_connected() const { return connected_.load(); }
    double get_hashrate() const { return hashrate_.load(); }

    // Public accessors for GPU worker threads
    std::mutex& get_job_mutex() { return job_mutex_; }
    JobAssignment& get_current_job() { return current_job_; }

    // Thread-safe ring buffer push (called from GPU worker threads)
    bool push_share(const ShareSubmission& share);
    // Pop a share (called from network thread)
    bool pop_share(ShareSubmission& share);

private:
    bool subscribe();
    bool authorize();
    void receive_loop();
    void submit_loop();
    void send_raw(const std::string& data);
    void handle_message(const std::string& msg);

    void handle_notify(const std::string& params);
    void handle_set_difficulty(const std::string& params);
    void handle_response(const std::string& id, const std::string& result,
                         const std::string& error);

    int sock_ = -1;
    std::string host_;
    int port_ = 0;
    std::string wallet_;
    std::string worker_;
    std::string password_;

    std::atomic<bool> connected_{false};
    std::atomic<bool> running_{false};
    std::atomic<double> hashrate_{0.0};

    // JSON-RPC request counter
    std::atomic<int> request_id_{0};

    // Current job state
    std::mutex job_mutex_;
    JobAssignment current_job_;
    std::string notify_method_;
    std::string submit_method_;
    std::string difficulty_method_;
    std::string ping_id_;             // mining.ping id from subscribe response
    double current_difficulty_ = 1.0;

    // Extranonce state (extracted from subscribe response)
    std::string extranonce1_;
    size_t extranonce2_size_ = 0;
    std::atomic<uint64_t> share_counter_{0};

    // Callbacks
    JobCallback job_cb_;
    ShareResultCallback result_cb_;
    VardiffCallback vardiff_cb_;

    // Receive buffer
    std::string recv_buf_;

    // Share ring buffer (lock-free single-producer / single-consumer style)
    ShareSubmission ring_[SHARE_RING_SIZE];
    std::atomic<size_t> ring_head_{0};  // writer (GPU thread)
    std::atomic<size_t> ring_tail_{0};  // reader (network thread)

    // Network thread
    std::unique_ptr<std::thread> recv_thread_;
    std::unique_ptr<std::thread> submit_thread_;

    // Send mutex for thread-safe socket writes
    std::mutex send_mutex_;

    // Map request_id -> gpu_index for share result tracking
    std::mutex request_map_mutex_;
    std::unordered_map<int, int> request_gpu_map_;

    // Reconnect state
    int reconnect_delay_ = 1000;
};

} // namespace propminer