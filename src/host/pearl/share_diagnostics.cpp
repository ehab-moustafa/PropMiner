#include "share_diagnostics.h"

#include "share_trace.h"
#include "simple_json.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace pearl {

namespace {

std::array<std::atomic<uint64_t>, static_cast<size_t>(ShareDropKind::Count)> g_drop_counts{};
std::array<std::atomic<uint64_t>, static_cast<size_t>(ShareRejectKind::Count)> g_reject_counts{};

std::string lower_ascii(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

ShareRejectKind classify_pool_message(const std::string& msg) {
    const std::string m = lower_ascii(msg);
    if (m.find("duplicate") != std::string::npos) return ShareRejectKind::DuplicateShare;
    if (m.find("job not found") != std::string::npos ||
        m.find("unknown job") != std::string::npos ||
        m.find("job expired") != std::string::npos ||
        m.find("stale") != std::string::npos) {
        return ShareRejectKind::JobNotFound;
    }
    if (m.find("low difficulty") != std::string::npos ||
        m.find("not good enough") != std::string::npos ||
        m.find("below target") != std::string::npos) {
        return ShareRejectKind::LowDifficulty;
    }
    if (m.find("unauthorized") != std::string::npos ||
        m.find("not authorized") != std::string::npos) {
        return ShareRejectKind::Unauthorized;
    }
    if (m.find("invalid share") != std::string::npos ||
        m.find("invalid proof") != std::string::npos ||
        m.find("invalid") != std::string::npos) {
        return ShareRejectKind::InvalidShare;
    }
    return ShareRejectKind::Other;
}

std::string human_reject_summary(ShareRejectKind kind, const std::string& pool_message) {
    switch (kind) {
        case ShareRejectKind::InvalidShare:
            return "REJECTED: invalid share — proof failed pool verification "
                   "(check GPU stability / overclock, or stale work)";
        case ShareRejectKind::DuplicateShare:
            return "REJECTED: duplicate share — pool already credited this work";
        case ShareRejectKind::JobNotFound:
            return "REJECTED: job not found — share arrived too late (stale job)";
        case ShareRejectKind::LowDifficulty:
            return "REJECTED: below pool difficulty — share did not meet target";
        case ShareRejectKind::Unauthorized:
            return "REJECTED: unauthorized — check wallet.worker and pool password";
        case ShareRejectKind::Other:
        case ShareRejectKind::Count:
            break;
    }
    if (!pool_message.empty()) {
        return "REJECTED: " + pool_message;
    }
    return "REJECTED: pool declined share";
}

std::string human_drop_summary(ShareDropKind kind) {
    switch (kind) {
        case ShareDropKind::MissingSigmaContext:
            return "DROP: internal error (job context missing)";
        case ShareDropKind::VerifyFailed:
            return "DROP: local verify failed — bad proof math "
                   "(GPU overclock unstable or corrupted work)";
        case ShareDropKind::VardiffTightened:
            return "DROP: stale share — pool raised difficulty while share was queued";
        case ShareDropKind::NoJobId:
            return "DROP: job not found — no pool job id for this work unit";
        case ShareDropKind::BuildFailed:
            return "DROP: below share target — proof build failed local difficulty check";
        case ShareDropKind::SendFailed:
            return "DROP: network send failed — socket error submitting to pool";
        case ShareDropKind::SupersededJob:
            return "DROP: stale share — job expired before submit (high latency or slow GPU)";
        case ShareDropKind::PoolAckTimeout:
            return "DROP: no pool ack — share may be stale (Kryptex often silent on :7048)";
        case ShareDropKind::ConnectionLost:
            return "DROP: connection lost — pool socket closed before share was acked";
        case ShareDropKind::Count:
            break;
    }
    return "DROP: unknown reason";
}

void append_breakdown(std::ostringstream& out, const char* key, uint64_t n) {
    if (n == 0) return;
    if (!out.str().empty()) out << ' ';
    out << key << '=' << n;
}

}  // namespace

const char* share_drop_kind_label(ShareDropKind kind) {
    switch (kind) {
        case ShareDropKind::MissingSigmaContext: return "internal_error";
        case ShareDropKind::VerifyFailed: return "local_verify_failed";
        case ShareDropKind::VardiffTightened: return "vardiff_stale";
        case ShareDropKind::NoJobId: return "job_not_found";
        case ShareDropKind::BuildFailed: return "below_target";
        case ShareDropKind::SendFailed: return "network_send_failed";
        case ShareDropKind::SupersededJob: return "stale_job";
        case ShareDropKind::PoolAckTimeout: return "pool_ack_timeout";
        case ShareDropKind::ConnectionLost: return "connection_lost";
        case ShareDropKind::Count: return "unknown";
    }
    return "unknown";
}

const char* share_reject_kind_label(ShareRejectKind kind) {
    switch (kind) {
        case ShareRejectKind::InvalidShare: return "invalid_share";
        case ShareRejectKind::DuplicateShare: return "duplicate_share";
        case ShareRejectKind::JobNotFound: return "job_not_found";
        case ShareRejectKind::LowDifficulty: return "low_difficulty";
        case ShareRejectKind::Unauthorized: return "unauthorized";
        case ShareRejectKind::Other: return "other";
        case ShareRejectKind::Count: return "unknown";
    }
    return "unknown";
}

void share_drop(ShareDropKind kind, const std::string& detail) {
    if (static_cast<size_t>(kind) < g_drop_counts.size()) {
        g_drop_counts[static_cast<size_t>(kind)].fetch_add(1);
    }
    share_log("dropped",
              human_drop_summary(kind) + " [" + share_drop_kind_label(kind) + "] " + detail);
}

void share_reject(ShareRejectKind kind, const std::string& detail) {
    if (static_cast<size_t>(kind) < g_reject_counts.size()) {
        g_reject_counts[static_cast<size_t>(kind)].fetch_add(1);
    }
    share_log("rejected",
              human_reject_summary(kind, detail) + " [" + share_reject_kind_label(kind) + "] " +
              detail);
}

PoolRejectInfo parse_pool_reject(const std::string& err_json) {
    PoolRejectInfo info;
    info.raw_error = err_json;
    if (err_json.empty() || err_json == "rejected") {
        info.kind = ShareRejectKind::Other;
        info.pool_message = err_json.empty() ? "rejected" : err_json;
        info.human_summary = human_reject_summary(info.kind, info.pool_message);
        return info;
    }

    const auto parsed = propminer::JsonValue::parse(err_json);
    if (parsed.is_array()) {
        for (size_t i = 0; i < parsed.size(); ++i) {
            if (parsed[i].is_string()) {
                const std::string s = parsed[i].to_string();
                if (!s.empty() && !std::isdigit(static_cast<unsigned char>(s[0]))) {
                    info.pool_message = s;
                    break;
                }
            }
        }
    } else if (parsed.is_string()) {
        info.pool_message = parsed.to_string();
    } else if (parsed.is_object() && parsed["message"].is_string()) {
        info.pool_message = parsed["message"].to_string();
    }

    if (info.pool_message.empty()) info.pool_message = err_json;
    info.kind = classify_pool_message(info.pool_message);
    info.human_summary = human_reject_summary(info.kind, info.pool_message);
    return info;
}

uint64_t share_drop_count(ShareDropKind kind) {
    if (static_cast<size_t>(kind) >= g_drop_counts.size()) return 0;
    return g_drop_counts[static_cast<size_t>(kind)].load();
}

uint64_t share_reject_count(ShareRejectKind kind) {
    if (static_cast<size_t>(kind) >= g_reject_counts.size()) {
        return 0;
    }
    return g_reject_counts[static_cast<size_t>(kind)].load();
}

uint64_t share_drop_total() {
    uint64_t total = 0;
    for (const auto& c : g_drop_counts) total += c.load();
    return total;
}

uint64_t share_reject_total() {
    uint64_t total = 0;
    for (const auto& c : g_reject_counts) total += c.load();
    return total;
}

std::string format_share_drop_breakdown() {
    std::ostringstream out;
    append_breakdown(out, "stale_job", share_drop_count(ShareDropKind::SupersededJob));
    append_breakdown(out, "vardiff_stale", share_drop_count(ShareDropKind::VardiffTightened));
    append_breakdown(out, "local_verify_failed", share_drop_count(ShareDropKind::VerifyFailed));
    append_breakdown(out, "below_target", share_drop_count(ShareDropKind::BuildFailed));
    append_breakdown(out, "job_not_found", share_drop_count(ShareDropKind::NoJobId));
    append_breakdown(out, "network_send_failed", share_drop_count(ShareDropKind::SendFailed));
    append_breakdown(out, "pool_ack_timeout", share_drop_count(ShareDropKind::PoolAckTimeout));
    append_breakdown(out, "connection_lost", share_drop_count(ShareDropKind::ConnectionLost));
    append_breakdown(out, "internal_error", share_drop_count(ShareDropKind::MissingSigmaContext));
    return out.str();
}

std::string format_share_reject_breakdown() {
    std::ostringstream out;
    append_breakdown(out, "invalid_share", share_reject_count(ShareRejectKind::InvalidShare));
    append_breakdown(out, "duplicate_share", share_reject_count(ShareRejectKind::DuplicateShare));
    append_breakdown(out, "job_not_found", share_reject_count(ShareRejectKind::JobNotFound));
    append_breakdown(out, "low_difficulty", share_reject_count(ShareRejectKind::LowDifficulty));
    append_breakdown(out, "unauthorized", share_reject_count(ShareRejectKind::Unauthorized));
    append_breakdown(out, "other", share_reject_count(ShareRejectKind::Other));
    return out.str();
}

}  // namespace pearl
