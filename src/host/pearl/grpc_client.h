#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "proto/mining_v2.h"

namespace pearl {

// Minimal gRPC-over-HTTP/2 client for Pearl V2 (no protoc dependency).
// Supports TLS (OpenSSL) or plaintext. Only the MinerService bidi stream
// is fully implemented; Register/Resume are available as unary calls.
class PearlGrpcClient {
public:
    struct Options {
        std::string host;
        int port = 443;
        bool use_tls = true;
        std::string user_agent = "propminer/1.0";
        int connect_timeout_ms = 10000;
    };

    explicit PearlGrpcClient(const Options& opts);
    ~PearlGrpcClient();

    // Connect TCP (+ TLS) and perform HTTP/2 handshake.
    bool connect();
    void disconnect();
    bool connected() const;

    // Unary Register RPC.
    bool register_miner(const proto::RegisterRequest& req, proto::RegisterResponse& out);

    // Start the bidi MiningStream. First outbound message must be AuthEvent.
    bool start_mining_stream(const proto::AuthEvent& auth);

    // Send a MinerEvent on the open stream.
    bool send_event(const proto::MinerEvent& evt);

    // Blocking receive of the next PoolEvent. Returns false on error/EOF.
    bool receive_event(proto::PoolEvent& out);

    // Non-blocking peek.
    bool try_receive_event(proto::PoolEvent& out);

    // Send a heartbeat. Convenience wrapper around send_event.
    bool send_heartbeat(const proto::Heartbeat& hb);

    std::string last_error() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pearl
