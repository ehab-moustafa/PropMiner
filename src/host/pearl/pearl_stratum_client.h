#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <unordered_map>

#include "simple_json.h"
#include "pow_target_utils.h"
#include "proto/mining_v2.h"

namespace pearl {

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
                            const std::vector<uint8_t>& proof_bytes);

    std::string job_id_for_sigma(const std::array<uint8_t, kSigmaHeaderBytes>& sigma) const;

    std::string last_error() const;

private:
    bool send_line(const std::string& line);
    bool read_line(std::string& out, int timeout_ms);
    bool subscribe();
    bool authorize();
    void receive_loop();
    void handle_message(const std::string& line);
    bool parse_notify_object(const std::string& params_json);
    bool parse_notify_array(const propminer::JsonValue& params);
    proto::JobAssignment make_job(const std::string& job_id,
                                    const std::string& header_hex,
                                    const std::string& target_hex,
                                    int64_t height);
    static std::array<uint8_t, 16> job_id_bytes(const std::string& job_id);
    static std::vector<uint8_t> hex_to_bytes(const std::string& hex);
    static double read_difficulty_param(const propminer::JsonValue& params);
    double effective_share_difficulty() const;
    uint32_t share_target_nbits() const;

    Options opts_;
    int sock_ = -1;
    std::atomic<bool> connected_{false};
    std::atomic<bool> running_{false};
    std::string last_error_;
    std::string recv_buf_;
    int request_id_ = 1;
    double last_difficulty_ = 0.0;
    std::mutex send_mtx_;
    mutable std::mutex job_map_mtx_;
    std::unordered_map<std::string, std::string> sigma_hex_to_job_id_;

    JobCallback job_cb_;
    VardiffCallback vardiff_cb_;
    ShareResultCallback share_cb_;
    std::unique_ptr<std::thread> recv_thread_;
};

}  // namespace pearl
