#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <string>

namespace pearl {

// Local drops — share never reached the pool (or was discarded before ack).
enum class ShareDropKind : uint8_t {
    MissingSigmaContext = 0,
    VerifyFailed,
    VardiffTightened,
    NoJobId,
    BuildFailed,
    SendFailed,
    SupersededJob,
    PoolAckTimeout,
    ConnectionLost,
    Count
};

// Pool-side rejections after submit.
enum class ShareRejectKind : uint8_t {
    InvalidShare = 0,
    DuplicateShare,
    JobNotFound,
    LowDifficulty,
    Unauthorized,
    Other,
    Count
};

struct PoolRejectInfo {
    ShareRejectKind kind = ShareRejectKind::Other;
    std::string pool_message;   // e.g. "Invalid share"
    std::string raw_error;      // full JSON from pool
    std::string human_summary;  // one-line explanation for logs
};

const char* share_drop_kind_label(ShareDropKind kind);
const char* share_reject_kind_label(ShareRejectKind kind);

void share_drop(ShareDropKind kind, const std::string& detail);
void share_reject(ShareRejectKind kind, const std::string& detail);

PoolRejectInfo parse_pool_reject(const std::string& err_json);

uint64_t share_drop_count(ShareDropKind kind);
uint64_t share_reject_count(ShareRejectKind kind);
uint64_t share_drop_total();
uint64_t share_reject_total();

// Compact breakdown for periodic telemetry, e.g. "stale_job=2 verify_failed=1".
std::string format_share_drop_breakdown();
std::string format_share_reject_breakdown();

}  // namespace pearl
