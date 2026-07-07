#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <unordered_map>
#include <unordered_set>

#include "simple_json.h"
#include "pow_target_utils.h"
#include "proto/mining_v2.h"
#include "rtx5090_profile.h"

namespace pearl {

// Default share difficulty when pool has not sent vardiff / notify target yet.
inline constexpr double kStratumDefaultShareDiff =
    static_cast<double>(Rtx5090Profile::kDefaultStratumShareDiff);

// Pearl Stratum client (Kryptex :7048 / ARC-compatible object notify+submit).
class PearlStratumClient {
public:
    struct Options {
        std::string host;
        int port = 7048;
        std::string wallet;
        std::string worker;
        std::string password = "x";
        std::string user_agent = "propminer/2.1";
        int connect_timeout_ms = 15000;
    };

    explicit PearlStratumClient(const Options& opts);
    ~PearlStratumClient();

    bool connect();
    void disconnect();
    bool connected() const;

    using JobCallback = std::function<void(const proto::JobAssignment&, const std::string& job_id_str)>;
    using VardiffCallback = std::function<void(uint32_t nbits)>;
    using ShareResultCallback = std::function<void(bool accepted, const std::string& msg)>;

    void set_callbacks(JobCallback job_cb, VardiffCallback vardiff_cb,
                       ShareResultCallback share_cb);

    bool submit_plain_proof(const std::string& job_id,
                            const std::vector<uint8_t>& proof_bytes,
                            uint64_t nonce = 0);

    std::string job_id_for_sigma(const std::array<uint8_t, kSigmaHeaderBytes>& sigma) const;

    // Pool mining.notify cert_version (1=dense V1 proof, 2=V2 + moe suffix).
    int cert_version_for_job(const std::string& job_id) const;

    // Outstanding mining.submit RPCs awaiting a pool JSON-RPC ack.
    size_t pending_submit_count() const;

    std::string last_error() const;

    // Active share difficulty (pool vardiff or PROPMINER_STRATUM_DIFF default).
    double effective_share_difficulty() const;

private:
    enum class ReadStatus { Line, Timeout, Closed };

    bool send_line(const std::string& line);
    ReadStatus read_line_status(std::string& out, int timeout_ms);
    bool read_line(std::string& out, int timeout_ms);
    bool subscribe();
    bool authorize();
    // pearl/v1 challenge-first handshake (Kryptex :7048 / AlphaPool / HeroMiners):
    // detect a pool-first `pearl.challenge`, solve it, then configure/subscribe/
    // authorize. Returns true if the challenge path handled the handshake.
    enum class HandshakeResult { NotChallenge, Ok, Failed };
    HandshakeResult try_challenge_handshake();
    bool solve_and_respond_challenge(const propminer::JsonValue& params, int resp_id);
    bool pearl_v1_finish_handshake(int authorize_id);
    void handle_challenge_async(const propminer::JsonValue& params);
    void receive_loop();
    void handle_message(const std::string& line);
    bool parse_notify_object(const std::string& params_json);
    bool parse_notify_array(const propminer::JsonValue& params);
    proto::JobAssignment make_job(const std::string& job_id,
                                    const std::string& header_hex,
                                    const std::string& target_hex,
                                    int64_t height,
                                    int cert_version = 1);
    static std::array<uint8_t, 16> job_id_bytes(const std::string& job_id);
    static std::vector<uint8_t> hex_to_bytes(const std::string& hex);
    static double read_difficulty_param(const propminer::JsonValue& params);
    void flush_stale_pending_submits();
    void flush_all_pending_submits(const char* reason);
    uint32_t share_target_nbits() const;

    Options opts_;
    bool use_object_submit_ = false;
    int sock_ = -1;
    std::atomic<bool> connected_{false};
    std::atomic<bool> running_{false};
    std::atomic<bool> pearl_v1_{false};
    std::atomic<bool> stop_solving_{false};
    // Request ids of in-flight pearl.challenge_response messages so the receive
    // loop does not mistake the pool's ack for a share result.
    std::mutex challenge_mtx_;
    std::unordered_set<int> challenge_resp_ids_;
    std::string last_error_;
    std::string recv_buf_;
    int request_id_ = 1;
    double last_difficulty_ = 0.0;
    std::mutex send_mtx_;
    mutable std::mutex job_map_mtx_;
    mutable std::mutex pending_submit_mtx_;
    struct PendingSubmit {
        uint64_t nonce = 0;
        std::chrono::steady_clock::time_point sent_at{};
    };
    std::unordered_map<int, PendingSubmit> pending_submit_nonces_;
    std::unordered_map<std::string, std::string> sigma_hex_to_job_id_;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> job_received_at_;
    std::unordered_map<std::string, int> job_cert_version_;
    std::string current_job_id_;
    std::chrono::steady_clock::time_point current_job_at_{};

    JobCallback job_cb_;
    VardiffCallback vardiff_cb_;
    ShareResultCallback share_cb_;
    std::unique_ptr<std::thread> recv_thread_;
};

}  // namespace pearl
