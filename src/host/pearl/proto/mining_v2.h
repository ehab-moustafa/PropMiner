#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace pearl {
namespace proto {

// Minimal protobuf wire helpers.
struct WireField {
    uint32_t field;
    uint32_t wire;
    uint64_t value;      // for varint/fixed32/fixed64
    const uint8_t* data; // for length-delimited
    size_t len;
};

class ProtoReader {
public:
    ProtoReader(const uint8_t* data, size_t len);
    bool next(WireField& out);

    static uint64_t decode_varint(const uint8_t*& p, const uint8_t* end);

private:
    const uint8_t* p_;
    const uint8_t* end_;
};

class ProtoWriter {
public:
    void write_varint(uint64_t v);
    void write_tag(uint32_t field, uint32_t wire);
    void write_uint64(uint32_t field, uint64_t v);
    void write_uint32(uint32_t field, uint32_t v);
    void write_int64(uint32_t field, int64_t v);
    void write_int32(uint32_t field, int32_t v);
    void write_bool(uint32_t field, bool v);
    void write_bytes(uint32_t field, const uint8_t* data, size_t len);
    void write_string(uint32_t field, const std::string& s);
    void write_double(uint32_t field, double v);
    void write_embedded(uint32_t field, const std::vector<uint8_t>& msg);

    const std::vector<uint8_t>& buf() const { return buf_; }
    std::vector<uint8_t> take() { return std::move(buf_); }

private:
    std::vector<uint8_t> buf_;
};

struct GpuCard {
    std::string uuid;
    std::string model;
    uint32_t index = 0;
    double hashrate = 0.0;

    std::vector<uint8_t> encode() const;
    bool decode(const uint8_t* data, size_t len);
};

struct RegisterRequest {
    std::array<uint8_t, 16> miner_id{};
    std::string identity_key;
    std::string wallet_address;
    std::string worker_name;
    std::vector<GpuCard> gpu_cards;
    double claimed_total_hashrate = 0.0;
    std::string miner_version;
    std::string git_sha;
    uint32_t protocol_version = 2;
    uint32_t k = 128;

    std::vector<uint8_t> encode() const;
    bool decode(const std::vector<uint8_t>& data);
    bool decode(const uint8_t* data, size_t len);
};

struct RegisterResponse {
    bool success = false;
    std::array<uint8_t, 16> miner_id{};
    std::string session_token;
    std::string identity_key;
    uint32_t initial_difficulty_nbits = 0;
    std::array<uint8_t, 32> pool_id{};
    std::string error_message;

    bool decode(const uint8_t* data, size_t len);
};

struct AuthEvent {
    std::array<uint8_t, 16> miner_id{};
    std::string session_token;
    std::unique_ptr<RegisterRequest> register_req;

    std::vector<uint8_t> encode() const;
};

struct PerGpuHashrate {
    std::string gpu_uuid;
    double hashrate_5m = 0.0;
    uint32_t shares_5m = 0;
};

struct Heartbeat {
    int64_t timestamp = 0;
    std::vector<PerGpuHashrate> gpu_hashrates;
    uint64_t sequence_number = 0;
    double current_hashrate = 0.0;
    double latency = 0.0;

    std::vector<uint8_t> encode() const;
};

struct PingEvent {
    int64_t timestamp = 0;
    std::vector<uint8_t> encode() const;
};

enum class MinerEventType : uint32_t {
    None = 0,
    Auth = 2,
    Share = 3,
    Heartbeat = 4,
    Ping = 5,
};

struct MinerEvent {
    uint64_t seq = 0;
    MinerEventType type = MinerEventType::None;
    std::vector<uint8_t> payload; // encoded AuthEvent/ShareSubmission/Heartbeat/PingEvent

    std::vector<uint8_t> encode() const;
};

struct JobAssignment {
    std::array<uint8_t, 16> job_id{};
    std::array<uint8_t, 32> sigma{};
    uint32_t target_nbits = 0;
    uint32_t network_target_nbits = 0;
    int64_t block_height = 0;
    uint32_t protocol_version = 2;
    std::array<uint8_t, 32> b_seed{};
    uint32_t audit_k = 0;

    bool decode(const uint8_t* data, size_t len);
};

struct ShareResult {
    std::array<uint8_t, 32> computed_hash{};
    bool accepted = false;
    std::string outcome;
    std::string message;
    bool is_block_find = false;

    bool decode(const uint8_t* data, size_t len);
};

struct DifficultyAdjust {
    uint32_t new_target_nbits = 0;
    std::string reason;
    double measured_hashrate = 0.0;

    bool decode(const uint8_t* data, size_t len);
};

struct PongEvent {
    int64_t timestamp = 0;
    bool decode(const uint8_t* data, size_t len);
};

struct PoolError {
    std::string code;
    std::string message;
    bool fatal = false;
    bool decode(const uint8_t* data, size_t len);
};

struct ReconnectHint {
    int32_t wait_seconds = 0;
    std::string reason;
    bool decode(const uint8_t* data, size_t len);
};

enum class PoolEventType : uint32_t {
    None = 0,
    Job = 2,
    ShareResult = 3,
    Vardiff = 4,
    Pong = 5,
    Error = 6,
    Reconnect = 7,
};

struct PoolEvent {
    uint64_t seq = 0;
    PoolEventType type = PoolEventType::None;
    // Decoded payload depending on type.
    JobAssignment job;
    ShareResult share_result;
    DifficultyAdjust vardiff;
    PongEvent pong;
    PoolError error;
    ReconnectHint reconnect;

    bool decode(const uint8_t* data, size_t len);
};

} // namespace proto
} // namespace pearl
