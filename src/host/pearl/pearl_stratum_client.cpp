#include "pearl_stratum_client.h"

#include "pow_target_utils.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstring>
#include <iostream>
#include <thread>

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

void stratum_log(const std::string& msg) {
    std::cout << "pool: " << msg << std::endl;
}

}  // namespace

PearlStratumClient::PearlStratumClient(const Options& opts) : opts_(opts) {
    if (const char* d = std::getenv("PROPMINER_STRATUM_DIFF"); d && d[0]) {
        const double req = std::atof(d);
        if (req > 0.0) last_difficulty_ = req;
    }
}

PearlStratumClient::~PearlStratumClient() { disconnect(); }

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
    if (::connect(sock_, res->ai_addr, res->ai_addrlen) < 0) {
        last_error_ = std::string("connect failed: ") + strerror(errno);
        close(sock_);
        sock_ = -1;
        freeaddrinfo(res);
        return false;
    }
    freeaddrinfo(res);

    stratum_log("stratum connected " + opts_.host + ":" + std::to_string(opts_.port));

    if (!subscribe()) {
        stratum_log("stratum subscribe skipped/failed; continuing to authorize");
    }
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

bool PearlStratumClient::read_line(std::string& out, int timeout_ms) {
    out.clear();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true) {
        auto nl = recv_buf_.find('\n');
        if (nl != std::string::npos) {
            out = recv_buf_.substr(0, nl);
            recv_buf_.erase(0, nl + 1);
            if (!out.empty() && out.back() == '\r') out.pop_back();
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) return false;
        pollfd pfd{sock_, POLLIN, 0};
        int remain = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count());
        if (remain <= 0) return false;
        int pr = poll(&pfd, 1, remain);
        if (pr <= 0) return false;
        char buf[8192];
        ssize_t n = recv(sock_, buf, sizeof(buf), 0);
        if (n <= 0) return false;
        recv_buf_.append(buf, static_cast<size_t>(n));
    }
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

void PearlStratumClient::receive_loop() {
    while (running_.load()) {
        std::string line;
        if (!read_line(line, 30000)) {
            if (running_.load()) {
                stratum_log("stratum connection lost");
                connected_ = false;
            }
            break;
        }
        if (!line.empty()) handle_message(line);
    }
}

void PearlStratumClient::handle_message(const std::string& line) {
    auto parsed = propminer::JsonValue::parse(line);
    if (parsed.is_null()) return;

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
        }
        return;
    }

    if (parsed["id"].is_number() && parsed["id"].to_int() > 2) {
        bool accepted = true;
        std::string err;
        if (!parsed["error"].is_null()) {
            accepted = false;
            err = parsed["error"].serialize();
        } else if (parsed["result"].is_bool() && !parsed["result"].to_bool()) {
            accepted = false;
            err = "rejected";
        } else if (parsed["result"].is_null()) {
            accepted = false;
            err = "null result";
        }
        if (share_cb_) share_cb_(accepted, accepted ? "Accepted" : err);
    }
}

bool PearlStratumClient::parse_notify_object(const std::string& params_json) {
    auto obj = propminer::JsonValue::parse(params_json);
    if (!obj["job_id"].is_string() || !obj["header"].is_string()) return false;
    const std::string target = obj["target"].is_string() ? obj["target"].to_string() : "";
    const int64_t height = obj["height"].is_number() ? obj["height"].to_int() : 0;
    const std::string job_id = obj["job_id"].to_string();
    auto ja = make_job(job_id, obj["header"].to_string(), target, height);
    if (job_cb_) job_cb_(ja, job_id);
    return true;
}

bool PearlStratumClient::parse_notify_array(const propminer::JsonValue& params) {
    if (params.size() < 4) return false;
    const std::string job_id = params[0].to_string();
    const std::string header = params[2].to_string();
    const int64_t height = params[3].is_number() ? params[3].to_int() : 0;
    double diff = last_difficulty_ > 0 ? last_difficulty_ : 32.0;
    const uint32_t nbits = difficulty_to_nbits_pdif(diff);
    std::array<uint8_t, 32> target_le = nbits_to_target_le(nbits);
    std::string target_hex;
    target_hex.reserve(64);
    for (int i = 31; i >= 0; --i) {
        char buf[3];
        std::snprintf(buf, sizeof(buf), "%02x", target_le[static_cast<size_t>(i)]);
        target_hex += buf;
    }
    auto ja = make_job(job_id, header, target_hex, height);
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
    if (!target_hex.empty()) {
        ja.target_nbits = hex_target_to_nbits(target_hex);
    } else if (last_difficulty_ > 0.0) {
        ja.target_nbits = difficulty_to_nbits_pdif(last_difficulty_);
    } else {
        ja.target_nbits = 0x207fffff;
    }
    ja.network_target_nbits = ja.target_nbits;
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
    }
    char buf[128];
    std::snprintf(buf, sizeof(buf),
                  "stratum new job %s height=%lld nbits=0x%08x diff=%.3f",
                  job_id.substr(0, std::min<size_t>(12, job_id.size())).c_str(),
                  static_cast<long long>(height), ja.target_nbits, last_difficulty_);
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
                                             const std::vector<uint8_t>& proof_bytes) {
    if (!connected_.load()) return false;
    std::string b64;
    static const char* tbl =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t i = 0;
    while (i < proof_bytes.size()) {
        uint32_t octet_a = i < proof_bytes.size() ? proof_bytes[i++] : 0;
        uint32_t octet_b = i < proof_bytes.size() ? proof_bytes[i++] : 0;
        uint32_t octet_c = i < proof_bytes.size() ? proof_bytes[i++] : 0;
        uint32_t triple = (octet_a << 16) + (octet_b << 8) + octet_c;
        b64.push_back(tbl[(triple >> 18) & 0x3F]);
        b64.push_back(tbl[(triple >> 12) & 0x3F]);
        b64.push_back(i > proof_bytes.size() + 1 ? '=' : tbl[(triple >> 6) & 0x3F]);
        b64.push_back(i > proof_bytes.size() ? '=' : tbl[triple & 0x3F]);
    }
    const int id = ++request_id_;
    std::string line = "{\"id\":" + std::to_string(id) +
                       ",\"method\":\"mining.submit\",\"params\":{\"job_id\":\"" +
                       json_escape(job_id) + "\",\"plain_proof\":\"" + b64 + "\"}}";
    stratum_log("share submitting job=" + job_id.substr(0, std::min<size_t>(12, job_id.size())) +
                " proof=" + std::to_string(proof_bytes.size()) + "B id=" + std::to_string(id));
    return send_line(line);
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

}  // namespace pearl
