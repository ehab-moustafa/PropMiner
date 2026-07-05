#include "mining_v2.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "pow_target_utils.h"

namespace pearl {
namespace proto {

static void append(std::vector<uint8_t>& out, const uint8_t* p, size_t n) {
    out.insert(out.end(), p, p + n);
}

static double read_double_le(const uint8_t* p) {
    double d;
    static_assert(sizeof(d) == 8, "double size");
    std::memcpy(&d, p, 8);
    return d;
}

static void write_double_le(std::vector<uint8_t>& out, double d) {
    uint8_t tmp[8];
    std::memcpy(tmp, &d, 8);
    append(out, tmp, 8);
}

// ProtoReader
ProtoReader::ProtoReader(const uint8_t* data, size_t len)
    : p_(data), end_(data + len) {}

uint64_t ProtoReader::decode_varint(const uint8_t*& p, const uint8_t* end) {
    uint64_t v = 0;
    int shift = 0;
    while (p < end) {
        uint8_t b = *p++;
        v |= static_cast<uint64_t>(b & 0x7F) << shift;
        if ((b & 0x80) == 0) return v;
        shift += 7;
        if (shift >= 64) throw std::runtime_error("varint overflow");
    }
    throw std::runtime_error("truncated varint");
}

bool ProtoReader::next(WireField& out) {
    if (p_ >= end_) return false;
    uint64_t tag = decode_varint(p_, end_);
    out.field = static_cast<uint32_t>(tag >> 3);
    out.wire = static_cast<uint32_t>(tag & 7);
    out.value = 0;
    out.data = nullptr;
    out.len = 0;
    switch (out.wire) {
        case 0:
            out.value = decode_varint(p_, end_);
            break;
        case 1:
            if (p_ + 8 > end_) throw std::runtime_error("truncated fixed64");
            std::memcpy(&out.value, p_, 8);
            p_ += 8;
            break;
        case 2: {
            uint64_t len = decode_varint(p_, end_);
            if (p_ + len > end_) throw std::runtime_error("truncated bytes");
            out.len = static_cast<size_t>(len);
            out.data = p_;
            p_ += out.len;
            break;
        }
        case 5:
            if (p_ + 4 > end_) throw std::runtime_error("truncated fixed32");
            std::memcpy(&out.value, p_, 4);
            p_ += 4;
            break;
        default:
            throw std::runtime_error("unknown wire type");
    }
    return true;
}

// ProtoWriter
void ProtoWriter::write_varint(uint64_t v) {
    while (v >= 0x80) {
        buf_.push_back(static_cast<uint8_t>(v | 0x80));
        v >>= 7;
    }
    buf_.push_back(static_cast<uint8_t>(v));
}

void ProtoWriter::write_tag(uint32_t field, uint32_t wire) {
    write_varint((static_cast<uint64_t>(field) << 3) | wire);
}

void ProtoWriter::write_uint64(uint32_t field, uint64_t v) {
    write_tag(field, 0);
    write_varint(v);
}

void ProtoWriter::write_uint32(uint32_t field, uint32_t v) {
    write_uint64(field, v);
}

void ProtoWriter::write_int64(uint32_t field, int64_t v) {
    write_uint64(field, static_cast<uint64_t>(v));
}

void ProtoWriter::write_int32(uint32_t field, int32_t v) {
    write_uint64(field, static_cast<uint32_t>(v));
}

void ProtoWriter::write_bool(uint32_t field, bool v) {
    write_uint64(field, v ? 1 : 0);
}

void ProtoWriter::write_bytes(uint32_t field, const uint8_t* data, size_t len) {
    write_tag(field, 2);
    write_varint(len);
    append(buf_, data, len);
}

void ProtoWriter::write_string(uint32_t field, const std::string& s) {
    write_bytes(field, reinterpret_cast<const uint8_t*>(s.data()), s.size());
}

void ProtoWriter::write_double(uint32_t field, double v) {
    write_tag(field, 1);
    write_double_le(buf_, v);
}

void ProtoWriter::write_embedded(uint32_t field, const std::vector<uint8_t>& msg) {
    write_tag(field, 2);
    write_varint(msg.size());
    append(buf_, msg.data(), msg.size());
}

// GpuCard
std::vector<uint8_t> GpuCard::encode() const {
    ProtoWriter w;
    w.write_string(1, uuid);
    w.write_string(2, model);
    w.write_uint32(3, index);
    w.write_double(4, hashrate);
    return w.take();
}

bool GpuCard::decode(const uint8_t* data, size_t len) {
    ProtoReader r(data, len);
    WireField f;
    while (r.next(f)) {
        switch (f.field) {
            case 1: uuid.assign(reinterpret_cast<const char*>(f.data), f.len); break;
            case 2: model.assign(reinterpret_cast<const char*>(f.data), f.len); break;
            case 3: index = static_cast<uint32_t>(f.value); break;
            case 4:
                if (f.len == sizeof(double)) {
                    std::memcpy(&hashrate, f.data, sizeof(double));
                }
                break;
        }
    }
    return true;
}

// RegisterRequest
std::vector<uint8_t> RegisterRequest::encode() const {
    ProtoWriter w;
    w.write_bytes(1, miner_id.data(), miner_id.size());
    w.write_string(2, identity_key);
    w.write_string(3, wallet_address);
    w.write_string(4, worker_name);
    for (const auto& g : gpu_cards) {
        auto enc = g.encode();
        w.write_embedded(5, enc);
    }
    w.write_double(6, claimed_total_hashrate);
    w.write_string(7, miner_version);
    w.write_string(8, git_sha);
    w.write_uint32(9, protocol_version);
    w.write_uint32(10, k);
    return w.take();
}

bool RegisterRequest::decode(const std::vector<uint8_t>& data) {
    return decode(data.data(), data.size());
}

bool RegisterRequest::decode(const uint8_t* data, size_t len) {
    ProtoReader r(data, len);
    WireField f;
    while (r.next(f)) {
        switch (f.field) {
            case 1:
                if (f.len == 16) std::memcpy(miner_id.data(), f.data, 16);
                break;
            case 2: identity_key.assign(reinterpret_cast<const char*>(f.data), f.len); break;
            case 3: wallet_address.assign(reinterpret_cast<const char*>(f.data), f.len); break;
            case 4: worker_name.assign(reinterpret_cast<const char*>(f.data), f.len); break;
            case 5: {
                GpuCard g;
                if (g.decode(f.data, f.len)) gpu_cards.push_back(g);
                break;
            }
            case 6: {
                if (f.len == sizeof(double)) {
                    std::memcpy(&claimed_total_hashrate, f.data, sizeof(double));
                }
                break;
            }
            case 7: miner_version.assign(reinterpret_cast<const char*>(f.data), f.len); break;
            case 8: git_sha.assign(reinterpret_cast<const char*>(f.data), f.len); break;
            case 9: protocol_version = static_cast<uint32_t>(f.value); break;
            case 10: k = static_cast<uint32_t>(f.value); break;
        }
    }
    return true;
}

// RegisterResponse
bool RegisterResponse::decode(const uint8_t* data, size_t len) {
    ProtoReader r(data, len);
    WireField f;
    while (r.next(f)) {
        switch (f.field) {
            case 1: success = f.value != 0; break;
            case 2:
                if (f.len == 16) std::memcpy(miner_id.data(), f.data, 16);
                break;
            case 3: session_token.assign(reinterpret_cast<const char*>(f.data), f.len); break;
            case 4: identity_key.assign(reinterpret_cast<const char*>(f.data), f.len); break;
            case 5: initial_difficulty_nbits = static_cast<uint32_t>(f.value); break;
            case 6:
                if (f.len == 32) std::memcpy(pool_id.data(), f.data, 32);
                break;
            case 7: error_message.assign(reinterpret_cast<const char*>(f.data), f.len); break;
        }
    }
    return true;
}

// AuthEvent
std::vector<uint8_t> AuthEvent::encode() const {
    ProtoWriter w;
    w.write_bytes(1, miner_id.data(), miner_id.size());
    w.write_string(2, session_token);
    if (register_req) {
        auto enc = register_req->encode();
        w.write_embedded(3, enc);
    }
    return w.take();
}

// Heartbeat
std::vector<uint8_t> Heartbeat::encode() const {
    ProtoWriter w;
    w.write_int64(1, timestamp);
    for (const auto& g : gpu_hashrates) {
        ProtoWriter gw;
        gw.write_string(1, g.gpu_uuid);
        gw.write_double(2, g.hashrate_5m);
        gw.write_uint32(3, g.shares_5m);
        w.write_embedded(2, gw.buf());
    }
    w.write_uint64(3, sequence_number);
    w.write_double(4, current_hashrate);
    w.write_double(5, latency);
    return w.take();
}

// PingEvent
std::vector<uint8_t> PingEvent::encode() const {
    ProtoWriter w;
    w.write_int64(1, timestamp);
    return w.take();
}

// MinerEvent
std::vector<uint8_t> MinerEvent::encode() const {
    ProtoWriter w;
    w.write_uint64(1, seq);
    if (!payload.empty()) {
        w.write_embedded(static_cast<uint32_t>(type), payload);
    }
    return w.take();
}

// JobAssignment
bool JobAssignment::decode(const uint8_t* data, size_t len) {
    ProtoReader r(data, len);
    WireField f;
    while (r.next(f)) {
        switch (f.field) {
            case 1:
                if (f.len == 16) std::memcpy(job_id.data(), f.data, 16);
                break;
            case 2:
                if (f.len == pearl::kSigmaHeaderBytes) {
                    std::memcpy(sigma.data(), f.data, pearl::kSigmaHeaderBytes);
                } else if (f.len == 32) {
                    // Self-test / legacy: zero-pad to full header size.
                    sigma.fill(0);
                    std::memcpy(sigma.data(), f.data, 32);
                }
                break;
            case 3: target_nbits = static_cast<uint32_t>(f.value); break;
            case 4: network_target_nbits = static_cast<uint32_t>(f.value); break;
            case 5: block_height = static_cast<int64_t>(f.value); break;
            case 6: protocol_version = static_cast<uint32_t>(f.value); break;
            case 7:
                if (f.len == 32) std::memcpy(b_seed.data(), f.data, 32);
                break;
            case 8: audit_k = static_cast<uint32_t>(f.value); break;
        }
    }
    return true;
}

// ShareResult
bool ShareResult::decode(const uint8_t* data, size_t len) {
    ProtoReader r(data, len);
    WireField f;
    while (r.next(f)) {
        switch (f.field) {
            case 1:
                if (f.len == 32) std::memcpy(computed_hash.data(), f.data, 32);
                break;
            case 2: accepted = f.value != 0; break;
            case 3: outcome.assign(reinterpret_cast<const char*>(f.data), f.len); break;
            case 4: message.assign(reinterpret_cast<const char*>(f.data), f.len); break;
            case 5: is_block_find = f.value != 0; break;
        }
    }
    return true;
}

// DifficultyAdjust
bool DifficultyAdjust::decode(const uint8_t* data, size_t len) {
    ProtoReader r(data, len);
    WireField f;
    while (r.next(f)) {
        switch (f.field) {
            case 1: new_target_nbits = static_cast<uint32_t>(f.value); break;
            case 2: reason.assign(reinterpret_cast<const char*>(f.data), f.len); break;
            case 3: measured_hashrate = read_double_le(reinterpret_cast<const uint8_t*>(&f.value)); break;
        }
    }
    return true;
}

// PongEvent
bool PongEvent::decode(const uint8_t* data, size_t len) {
    ProtoReader r(data, len);
    WireField f;
    while (r.next(f)) {
        if (f.field == 1) timestamp = static_cast<int64_t>(f.value);
    }
    return true;
}

// PoolError
bool PoolError::decode(const uint8_t* data, size_t len) {
    ProtoReader r(data, len);
    WireField f;
    while (r.next(f)) {
        switch (f.field) {
            case 1: code.assign(reinterpret_cast<const char*>(f.data), f.len); break;
            case 2: message.assign(reinterpret_cast<const char*>(f.data), f.len); break;
            case 3: fatal = f.value != 0; break;
        }
    }
    return true;
}

// ReconnectHint
bool ReconnectHint::decode(const uint8_t* data, size_t len) {
    ProtoReader r(data, len);
    WireField f;
    while (r.next(f)) {
        switch (f.field) {
            case 1: wait_seconds = static_cast<int32_t>(f.value); break;
            case 2: reason.assign(reinterpret_cast<const char*>(f.data), f.len); break;
        }
    }
    return true;
}

// PoolEvent
bool PoolEvent::decode(const uint8_t* data, size_t len) {
    ProtoReader r(data, len);
    WireField f;
    while (r.next(f)) {
        switch (f.field) {
            case 1: seq = f.value; break;
            case 2:
                type = PoolEventType::Job;
                job.decode(f.data, f.len);
                break;
            case 3:
                type = PoolEventType::ShareResult;
                share_result.decode(f.data, f.len);
                break;
            case 4:
                type = PoolEventType::Vardiff;
                vardiff.decode(f.data, f.len);
                break;
            case 5:
                type = PoolEventType::Pong;
                pong.decode(f.data, f.len);
                break;
            case 6:
                type = PoolEventType::Error;
                error.decode(f.data, f.len);
                break;
            case 7:
                type = PoolEventType::Reconnect;
                reconnect.decode(f.data, f.len);
                break;
        }
    }
    return true;
}

} // namespace proto
} // namespace pearl
