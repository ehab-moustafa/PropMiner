#include "pearl_stratum_client.h"

#include "pow_target_utils.h"
#include "share_trace.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

namespace pearl {
namespace {

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\\' || c == '"') out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

std::string base64_encode(const std::vector<uint8_t>& in) {
    static const char* tbl =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((in.size() + 2) / 3) * 4);
    size_t i = 0;
    while (i + 3 <= in.size()) {
        const uint32_t n = (static_cast<uint32_t>(in[i]) << 16) |
                           (static_cast<uint32_t>(in[i + 1]) << 8) |
                           static_cast<uint32_t>(in[i + 2]);
        i += 3;
        out.push_back(tbl[(n >> 18) & 0x3F]);
        out.push_back(tbl[(n >> 12) & 0x3F]);
        out.push_back(tbl[(n >> 6) & 0x3F]);
        out.push_back(tbl[n & 0x3F]);
    }
    if (i < in.size()) {
        uint32_t n = static_cast<uint32_t>(in[i]) << 16;
        if (i + 1 < in.size()) {
            n |= static_cast<uint32_t>(in[i + 1]) << 8;
        }
        out.push_back(tbl[(n >> 18) & 0x3F]);
        out.push_back(tbl[(n >> 12) & 0x3F]);
        if (i + 1 < in.size()) {
            out.push_back(tbl[(n >> 6) & 0x3F]);
            out.push_back('=');
        } else {
            out.push_back('=');
            out.push_back('=');
        }
    }
    return out;
}

void stratum_log(const std::string& msg) {
    std::cout << "pool: " << msg << std::endl;
}

void pool_diag_log(const char* stage, const std::string& detail) {
    stratum_log(std::string(stage) + ": " + detail);
    share_log(stage, detail);
}

bool stratum_verbose_recv() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("PROPMINER_VERBOSE_STRATUM");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached != 0;
}

std::string truncate_line(const std::string& line, size_t max_len) {
    if (line.size() <= max_len) return line;
    return line.substr(0, max_len) + "…";
}

// JSON-RPC id may be number or string depending on pool.
bool parse_request_id(const propminer::JsonValue& id_field, int* out) {
    if (!out) return false;
    if (id_field.is_number()) {
        *out = static_cast<int>(id_field.to_int());
        return true;
    }
    if (id_field.is_string()) {
        const std::string& s = id_field.to_string();
        if (s.empty()) return false;
        char* end = nullptr;
        const long v = std::strtol(s.c_str(), &end, 10);
        if (end == s.c_str() || *end != '\0') return false;
        *out = static_cast<int>(v);
        return true;
    }
    return false;
}

bool share_result_accepted(const propminer::JsonValue& parsed, std::string* err_out) {
    if (!parsed["error"].is_null()) {
        if (err_out) *err_out = parsed["error"].serialize();
        return false;
    }
    const auto& result = parsed["result"];
    if (result.is_bool()) return result.to_bool();
    if (result.is_null()) {
        if (err_out) *err_out = "null result";
        return false;
    }
    if (result.is_string()) {
        const std::string r = result.to_string();
        if (r == "false" || r == "0") {
            if (err_out) *err_out = "rejected";
            return false;
        }
        return true;
    }
    // Kryptex / pearl pools usually return true or an error object.
    return true;
}

constexpr double kStratumDefaultShareDiff = 32768.0;

double parse_password_difficulty(const std::string& password) {
    const std::string key = ";d=";
    size_t pos = password.find(key);
    if (pos == std::string::npos) {
        if (password.rfind("d=", 0) == 0) {
            pos = 0;
        } else {
            return 0.0;
        }
    } else {
        pos += 1;  // skip ';' so we land on "d="
    }
    const double diff = std::atof(password.c_str() + pos + 2);
    return diff > 0.0 ? diff : 0.0;
}

}  // namespace

PearlStratumClient::PearlStratumClient(const Options& opts) : opts_(opts) {
    // pearl/v1 (Kryptex :7048) requires positional submit; object form is rejected.
    use_object_submit_ = false;
    if (const char* env = std::getenv("PROPMINER_STRATUM_OBJECT_SUBMIT"); env && env[0]) {
        use_object_submit_ = (env[0] == '1' || env[0] == 'y' || env[0] == 'Y');
    }
    if (const char* d = std::getenv("PROPMINER_STRATUM_DIFF"); d && d[0]) {
        const double req = std::atof(d);
        if (req > 0.0) last_difficulty_ = req;
    }
    if (const double pw_diff = parse_password_difficulty(opts_.password); pw_diff > 0.0) {
        last_difficulty_ = pw_diff;
    }
    std::string build = "unknown";
    if (const char* v = std::getenv("PROP_MINER_GIT_SHA"); v && v[0]) {
        build = v;
    } else {
        FILE* vf = std::fopen("VERSION", "r");
        if (vf) {
            char buf[64] = {};
            if (std::fgets(buf, sizeof(buf), vf)) {
                size_t len = std::strlen(buf);
                while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r')) {
                    buf[--len] = '\0';
                }
                if (len > 0) build.assign(buf, len);
            }
            std::fclose(vf);
        }
    }
    stratum_log("propminer build=" + build +
                " stratum_submit=" + (use_object_submit_ ? "object" : "positional") +
                " (stratum ack diagnostics enabled)");
}

PearlStratumClient::~PearlStratumClient() { disconnect(); }

double PearlStratumClient::effective_share_difficulty() const {
    if (last_difficulty_ > 0.0) return last_difficulty_;
    if (const char* d = std::getenv("PROPMINER_STRATUM_DIFF"); d && d[0]) {
        const double req = std::atof(d);
        if (req > 0.0) return req;
    }
    return kStratumDefaultShareDiff;
}

uint32_t PearlStratumClient::share_target_nbits() const {
    return difficulty_to_nbits_pdif(effective_share_difficulty());
}

bool PearlStratumClient::connect() {
    disconnect();
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    char port_str[16];
    std::snprintf(port_str, sizeof(port_str), "%d", opts_.port);
    if (getaddrinfo(opts_.host.c_str(), port_str, &hints, &res) != 0) {
        last_error_ = "getaddrinfo failed for " + opts_.host;
        return false;
    }
    sock_ = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sock_ < 0) {
        freeaddrinfo(res);
        last_error_ = "socket() failed";
        return false;
    }
    int yes = 1;
    setsockopt(sock_, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
    setsockopt(sock_, SOL_SOCKET, SO_KEEPALIVE, &yes, sizeof(yes));
#ifdef TCP_KEEPIDLE
    {
        int idle = 30;
        int intvl = 10;
        int cnt = 4;
        setsockopt(sock_, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));
        setsockopt(sock_, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof(intvl));
        setsockopt(sock_, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof(cnt));
    }
#endif
    if (::connect(sock_, res->ai_addr, res->ai_addrlen) < 0) {
        last_error_ = std::string("connect failed: ") + strerror(errno);
        close(sock_);
        sock_ = -1;
        freeaddrinfo(res);
        return false;
    }
    freeaddrinfo(res);

    stratum_log("stratum connected " + opts_.host + ":" + std::to_string(opts_.port));

    // Pearl/Kryptex :7048 does not require mining.subscribe (ARC skips it too).
    if (!authorize()) {
        disconnect();
        return false;
    }

    connected_ = true;
    running_ = true;
    recv_thread_ = std::make_unique<std::thread>(&PearlStratumClient::receive_loop, this);
    return true;
}

void PearlStratumClient::disconnect() {
    flush_all_pending_submits("disconnect");
    running_ = false;
    connected_ = false;
    if (recv_thread_ && recv_thread_->joinable()) recv_thread_->join();
    recv_thread_.reset();
    if (sock_ >= 0) {
        close(sock_);
        sock_ = -1;
    }
}

bool PearlStratumClient::connected() const { return connected_.load(); }
std::string PearlStratumClient::last_error() const { return last_error_; }

void PearlStratumClient::set_callbacks(JobCallback job_cb, VardiffCallback vardiff_cb,
                                       ShareResultCallback share_cb) {
    job_cb_ = std::move(job_cb);
    vardiff_cb_ = std::move(vardiff_cb);
    share_cb_ = std::move(share_cb);
}

bool PearlStratumClient::send_line(const std::string& line) {
    std::lock_guard<std::mutex> lk(send_mtx_);
    if (sock_ < 0) return false;
    std::string msg = line;
    if (msg.empty() || msg.back() != '\n') msg.push_back('\n');
    size_t sent = 0;
    while (sent < msg.size()) {
        ssize_t n = ::send(sock_, msg.data() + sent, msg.size() - sent, MSG_NOSIGNAL);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

PearlStratumClient::ReadStatus PearlStratumClient::read_line_status(std::string& out,
                                                                  int timeout_ms) {
    out.clear();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true) {
        auto nl = recv_buf_.find('\n');
        if (nl != std::string::npos) {
            out = recv_buf_.substr(0, nl);
            recv_buf_.erase(0, nl + 1);
            if (!out.empty() && out.back() == '\r') out.pop_back();
            return ReadStatus::Line;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return ReadStatus::Timeout;
        }
        pollfd pfd{sock_, POLLIN, 0};
        int remain = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count());
        if (remain <= 0) return ReadStatus::Timeout;
        int pr = poll(&pfd, 1, remain);
        if (pr == 0) return ReadStatus::Timeout;
        if (pr < 0) return ReadStatus::Closed;
        char buf[8192];
        ssize_t n = recv(sock_, buf, sizeof(buf), 0);
        if (n == 0) return ReadStatus::Closed;
        if (n < 0) return ReadStatus::Closed;
        recv_buf_.append(buf, static_cast<size_t>(n));
    }
}

bool PearlStratumClient::read_line(std::string& out, int timeout_ms) {
    return read_line_status(out, timeout_ms) == ReadStatus::Line;
}

bool PearlStratumClient::subscribe() {
    request_id_ = 1;
    std::string line = "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":[\"" +
                       json_escape(opts_.user_agent) + "\"]}";
    if (!send_line(line)) {
        last_error_ = "subscribe send failed";
        return false;
    }
    std::string resp;
    if (!read_line(resp, opts_.connect_timeout_ms)) {
        last_error_ = "subscribe response timeout";
        return false;
    }
    auto parsed = propminer::JsonValue::parse(resp);
    if (!parsed["error"].is_null()) {
        last_error_ = "subscribe error: " + parsed["error"].serialize();
        return false;
    }
    stratum_log("stratum subscribed");
    return true;
}

bool PearlStratumClient::authorize() {
    const std::string user = opts_.wallet + "." + opts_.worker;
    request_id_ = 2;
    std::string line = "{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\"" +
                       json_escape(user) + "\",\"" + json_escape(opts_.password) + "\"]}";
    if (!send_line(line)) {
        last_error_ = "authorize send failed";
        return false;
    }
    for (int attempt = 0; attempt < 30; ++attempt) {
        std::string resp;
        if (!read_line(resp, opts_.connect_timeout_ms)) {
            last_error_ = "authorize response timeout";
            return false;
        }
        auto parsed = propminer::JsonValue::parse(resp);
        if (parsed["method"].is_string()) {
            handle_message(resp);
            continue;
        }
        if (parsed["id"].is_number() && parsed["id"].to_int() == 2) {
            if (!parsed["error"].is_null()) {
                last_error_ = "authorize error: " + parsed["error"].serialize();
                return false;
            }
            if (parsed["result"].is_bool() && !parsed["result"].to_bool()) {
                last_error_ = "authorize rejected (check wallet.worker format)";
                return false;
            }
            stratum_log("stratum authorized as " + user + " password=" + opts_.password);
            return true;
        }
    }
    last_error_ = "authorize: no matching response";
    return false;
}

double PearlStratumClient::read_difficulty_param(const propminer::JsonValue& params) {
    if (params.is_array()) {
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i].is_number()) return params[i].to_number();
            if (params[i].is_string()) return std::atof(params[i].to_string().c_str());
        }
        return 0.0;
    }
    if (params.is_number()) return params.to_number();
    if (params.is_string()) return std::atof(params.to_string().c_str());
    return 0.0;
}

void PearlStratumClient::flush_all_pending_submits(const char* reason) {
    std::vector<std::pair<int, PendingSubmit>> pending;
    {
        std::lock_guard<std::mutex> lk(pending_submit_mtx_);
        pending.reserve(pending_submit_nonces_.size());
        for (auto& entry : pending_submit_nonces_) {
            pending.emplace_back(entry.first, entry.second);
        }
        pending_submit_nonces_.clear();
    }
    const auto now = std::chrono::steady_clock::now();
    for (const auto& [id, ps] : pending) {
        const auto waited = std::chrono::duration_cast<std::chrono::seconds>(
            now - ps.sent_at).count();
        pool_diag_log("pool-no-response",
                      "nonce=" + std::to_string(ps.nonce) +
                      " stratum_id=" + std::to_string(id) +
                      " waited_sec=" + std::to_string(waited) +
                      " reason=" + (reason ? reason : "unknown") +
                      " (Kryptex :7048 often sends no mining.submit ack)");
    }
}

void PearlStratumClient::flush_stale_pending_submits() {
    constexpr auto kAckTimeout = std::chrono::seconds(45);
    const auto now = std::chrono::steady_clock::now();
    std::vector<std::pair<int, PendingSubmit>> stale;
    {
        std::lock_guard<std::mutex> lk(pending_submit_mtx_);
        for (auto it = pending_submit_nonces_.begin(); it != pending_submit_nonces_.end();) {
            if (now - it->second.sent_at >= kAckTimeout) {
                stale.emplace_back(it->first, it->second);
                it = pending_submit_nonces_.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (const auto& [id, pending] : stale) {
        const auto waited = std::chrono::duration_cast<std::chrono::seconds>(
            now - pending.sent_at).count();
        pool_diag_log("pool-no-response",
                      "nonce=" + std::to_string(pending.nonce) +
                      " stratum_id=" + std::to_string(id) +
                      " waited_sec=" + std::to_string(waited) +
                      " (Kryptex :7048 often sends no mining.submit ack)");
    }
}

void PearlStratumClient::receive_loop() {
    while (running_.load()) {
        std::string line;
        const ReadStatus st = read_line_status(line, 120000);
        if (st == ReadStatus::Timeout) {
            flush_stale_pending_submits();
            continue;
        }
        if (st == ReadStatus::Closed) {
            if (running_.load()) {
                flush_all_pending_submits("connection_lost");
                stratum_log("stratum connection lost");
                connected_ = false;
            }
            break;
        }
        if (!line.empty()) handle_message(line);
        flush_stale_pending_submits();
    }
}

void PearlStratumClient::handle_message(const std::string& line) {
    if (stratum_verbose_recv()) {
        stratum_log("recv: " + truncate_line(line, 400));
    }

    auto parsed = propminer::JsonValue::parse(line);
    if (parsed.is_null()) {
        if (stratum_verbose_recv()) {
            stratum_log("recv: unparseable JSON");
        }
        return;
    }

    if (parsed["method"].is_string()) {
        const std::string method = parsed["method"].to_string();
        const auto& params = parsed["params"];
        if (method == "mining.notify") {
            if (params.is_array()) {
                parse_notify_array(params);
            } else {
                parse_notify_object(params.serialize());
            }
        } else if (method == "mining.set_difficulty") {
            const double diff = read_difficulty_param(params);
            if (diff > 0.0) {
                last_difficulty_ = diff;
                const uint32_t nbits = difficulty_to_nbits_pdif(last_difficulty_);
                char buf[96];
                std::snprintf(buf, sizeof(buf),
                              "stratum set_difficulty=%.3f nbits=0x%08x",
                              last_difficulty_, nbits);
                stratum_log(buf);
                if (vardiff_cb_) vardiff_cb_(nbits);
            }
        } else if (stratum_verbose_recv()) {
            stratum_log("notify: method=" + method);
        }
        return;
    }

    int msg_id = 0;
    if (!parse_request_id(parsed["id"], &msg_id)) return;

    // Share submit ack: JSON-RPC response with id, no method (ARC StratumSession).
    bool is_share = msg_id > 2;
    if (!is_share) {
        std::lock_guard<std::mutex> lk(pending_submit_mtx_);
        is_share = pending_submit_nonces_.count(msg_id) > 0;
    }
    if (!is_share) {
        stratum_log("rpc id=" + std::to_string(msg_id) + " (not a share ack): " +
                    truncate_line(line, 200));
        return;
    }

    std::string err;
    const bool accepted = share_result_accepted(parsed, &err);
    uint64_t nonce = 0;
    {
        std::lock_guard<std::mutex> lk(pending_submit_mtx_);
        auto it = pending_submit_nonces_.find(msg_id);
        if (it != pending_submit_nonces_.end()) {
            nonce = it->second.nonce;
            pending_submit_nonces_.erase(it);
        }
    }
    if (share_cb_) share_cb_(accepted, accepted ? "Accepted" : err);
    pool_diag_log("pool-response",
                  std::string(accepted ? "accepted" : "rejected") +
                  " nonce=" + std::to_string(nonce) +
                  " stratum_id=" + std::to_string(msg_id) +
                  (accepted ? "" : (" err=" + err)));
    share_trace("pool-response",
                std::string(accepted ? "accepted" : ("rejected err=" + err)) +
                " id=" + std::to_string(msg_id) +
                (nonce ? (" nonce=" + std::to_string(nonce)) : ""));
}

bool PearlStratumClient::parse_notify_object(const std::string& params_json) {
    auto obj = propminer::JsonValue::parse(params_json);
    if (!obj["job_id"].is_string() || !obj["header"].is_string()) return false;
    const int64_t height = obj["height"].is_number() ? obj["height"].to_int() : 0;
    const std::string job_id = obj["job_id"].to_string();
    // Kryptex object notify carries the *network* target in `target` — never use
    // it for share PoW (nbits ~0x1a07ffff). Share difficulty comes from d= / vardiff.
    auto ja = make_job(job_id, obj["header"].to_string(), "", height);
    if (job_cb_) job_cb_(ja, job_id);
    return true;
}

bool PearlStratumClient::parse_notify_array(const propminer::JsonValue& params) {
    if (params.size() < 4) return false;
    const std::string job_id = params[0].to_string();
    const std::string header = params[2].to_string();
    const int64_t height = params[3].is_number() ? params[3].to_int() : 0;
    auto ja = make_job(job_id, header, "", height);
    if (job_cb_) job_cb_(ja, job_id);
    return true;
}

proto::JobAssignment PearlStratumClient::make_job(const std::string& job_id,
                                                   const std::string& header_hex,
                                                   const std::string& target_hex,
                                                   int64_t height) {
    proto::JobAssignment ja{};
    auto id_bytes = job_id_bytes(job_id);
    std::memcpy(ja.job_id.data(), id_bytes.data(), 16);
    auto sigma = hex_to_bytes(header_hex);
    if (sigma.size() >= kSigmaHeaderBytes) {
        std::memcpy(ja.sigma.data(), sigma.data(), kSigmaHeaderBytes);
    }
    const uint32_t share_nbits = share_target_nbits();
    ja.target_nbits = share_nbits;
    ja.network_target_nbits = network_nbits_from_sigma(ja.sigma);
    if (ja.network_target_nbits == 0) {
        ja.network_target_nbits = share_nbits;
    }
    if (!target_hex.empty()) {
        const uint32_t wire_nbits = hex_target_to_nbits(target_hex);
        if (wire_nbits != share_nbits && wire_nbits == ja.network_target_nbits) {
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true)) {
                stratum_log("stratum: ignoring wire network target for share PoW "
                            "(use PROPMINER_STRATUM_DIFF or mining.set_difficulty)");
            }
        }
    }
    if (last_difficulty_ <= 0.0) {
        static std::atomic<bool> warned_default{false};
        if (!warned_default.exchange(true)) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                          "stratum: no vardiff yet; using share diff=%.0f "
                          "(set PROPMINER_STRATUM_DIFF to override)",
                          effective_share_difficulty());
            stratum_log(buf);
        }
    }
    ja.block_height = height;
    ja.protocol_version = 2;
    ja.audit_k = 0;
    ja.b_seed.fill(0);

    std::string sigma_key;
    sigma_key.reserve(header_hex.size());
    for (char c : header_hex) sigma_key.push_back(static_cast<char>(std::tolower(c)));
    {
        std::lock_guard<std::mutex> lk(job_map_mtx_);
        sigma_hex_to_job_id_[sigma_key] = job_id;
        job_received_at_[job_id] = std::chrono::steady_clock::now();
        current_job_id_ = job_id;
        current_job_at_ = job_received_at_[job_id];
    }
    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  "stratum new job %s height=%lld share_nbits=0x%08x "
                  "network_nbits=0x%08x diff=%.3f",
                  job_id.substr(0, std::min<size_t>(12, job_id.size())).c_str(),
                  static_cast<long long>(height), ja.target_nbits,
                  ja.network_target_nbits, effective_share_difficulty());
    stratum_log(buf);
    return ja;
}

std::string PearlStratumClient::job_id_for_sigma(
    const std::array<uint8_t, kSigmaHeaderBytes>& sigma) const {
    std::string hex;
    hex.reserve(kSigmaHeaderBytes * 2);
    for (size_t i = 0; i < kSigmaHeaderBytes; ++i) {
        char buf[3];
        std::snprintf(buf, sizeof(buf), "%02x",
                      static_cast<unsigned char>(sigma[i]));
        hex += buf;
    }
    std::lock_guard<std::mutex> lk(job_map_mtx_);
    auto it = sigma_hex_to_job_id_.find(hex);
    if (it != sigma_hex_to_job_id_.end()) return it->second;
    return hex.substr(0, std::min<size_t>(32, hex.size()));
}

bool PearlStratumClient::submit_plain_proof(const std::string& job_id,
                                             const std::vector<uint8_t>& proof_bytes,
                                             uint64_t nonce) {
    if (!connected_.load()) return false;

    int64_t job_age_ms = -1;
    bool superseded = false;
    {
        std::lock_guard<std::mutex> lk(job_map_mtx_);
        auto it = job_received_at_.find(job_id);
        if (it != job_received_at_.end()) {
            job_age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - it->second).count();
        }
        superseded = !current_job_id_.empty() && job_id != current_job_id_;
    }
    if (superseded) {
        stratum_log("submit dropped superseded job_id=" +
                    job_id.substr(0, std::min<size_t>(12, job_id.size())) +
                    " current=" +
                    current_job_id_.substr(0, std::min<size_t>(12, current_job_id_.size())));
        share_log("submit-drop", "nonce=" + std::to_string(nonce) + " reason=superseded_job");
        return false;
    }
    if (job_age_ms > 15000) {
        stratum_log("submit dropped stale job age_ms=" + std::to_string(job_age_ms));
        share_log("submit-drop", "nonce=" + std::to_string(nonce) + " reason=stale_job");
        return false;
    }
    if (job_age_ms >= 0) {
        stratum_log("submit job_age_ms=" + std::to_string(job_age_ms) +
                    " job=" + job_id.substr(0, std::min<size_t>(12, job_id.size())));
    }

    const std::string b64 = base64_encode(proof_bytes);
    const int id = ++request_id_;
    if (nonce != 0) {
        std::lock_guard<std::mutex> lk(pending_submit_mtx_);
        pending_submit_nonces_[id] = PendingSubmit{
            nonce, std::chrono::steady_clock::now()};
    }
    const std::string user = opts_.wallet + "." + opts_.worker;
    std::string line;
    if (use_object_submit_) {
        // SRBMiner / suprnova object form (Kryptex officially supports SRBMiner).
        line = "{\"id\":" + std::to_string(id) +
               ",\"method\":\"mining.submit\",\"params\":{" +
               "\"job_id\":\"" + json_escape(job_id) + "\"," +
               "\"plain_proof\":\"" + b64 + "\"}}";
    } else {
        // pearl/v1 positional: [worker, job_id, plain_proof_b64]
        line = "{\"id\":" + std::to_string(id) +
               ",\"method\":\"mining.submit\",\"params\":[\"" +
               json_escape(user) + "\",\"" + json_escape(job_id) + "\",\"" +
               b64 + "\"]}";
    }
    stratum_log(std::string("submit-wire format=") +
                (use_object_submit_ ? "object" : "positional") +
                " user=" + user + " proof_bytes=" + std::to_string(proof_bytes.size()));
    share_trace("submit-wire",
                "job=" + job_id.substr(0, std::min<size_t>(16, job_id.size())) +
                " proof=" + std::to_string(proof_bytes.size()) + "B id=" +
                std::to_string(id) +
                (nonce ? (" nonce=" + std::to_string(nonce)) : ""));
    share_log("submit-wire",
              "nonce=" + std::to_string(nonce) +
              " job_id=" + job_id.substr(0, std::min<size_t>(16, job_id.size())) +
              " proof_bytes=" + std::to_string(proof_bytes.size()) +
              " stratum_id=" + std::to_string(id));
    stratum_log("share submitting job=" + job_id.substr(0, std::min<size_t>(12, job_id.size())) +
                " proof=" + std::to_string(proof_bytes.size()) + "B id=" + std::to_string(id) +
                " nonce=" + std::to_string(nonce) + " awaiting_ack=45s");
    if (!send_line(line)) {
        share_log("submit-fail",
                  "nonce=" + std::to_string(nonce) +
                  " stratum_id=" + std::to_string(id) +
                  " reason=send_line_failed");
        return false;
    }
    return true;
}

std::array<uint8_t, 16> PearlStratumClient::job_id_bytes(const std::string& job_id) {
    std::array<uint8_t, 16> out{};
    if (job_id.size() == 32) {
        auto raw = hex_to_bytes(job_id);
        if (raw.size() == 16) std::memcpy(out.data(), raw.data(), 16);
        return out;
    }
    uint64_t h = 0;
    for (unsigned char c : job_id) h = h * 131 + c;
    std::memcpy(out.data(), &h, 8);
    std::memcpy(out.data() + 8, &h, 8);
    return out;
}

std::vector<uint8_t> PearlStratumClient::hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> out;
    std::string h = hex;
    if (h.size() >= 2 && h.rfind("0x", 0) == 0) h = h.substr(2);
    if (h.size() % 2) h = "0" + h;
    out.reserve(h.size() / 2);
    for (size_t i = 0; i + 1 < h.size(); i += 2) {
        out.push_back(static_cast<uint8_t>(std::strtoul(h.substr(i, 2).c_str(), nullptr, 16)));
    }
    return out;
}

size_t PearlStratumClient::pending_submit_count() const {
    std::lock_guard<std::mutex> lk(pending_submit_mtx_);
    return pending_submit_nonces_.size();
}

}  // namespace pearl
