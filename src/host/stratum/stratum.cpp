#include "stratum.h"
#include "simple_json.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <poll.h>
#include <chrono>
#include <algorithm>
#include <cerrno>
#include <sstream>

namespace propminer {

StratumClient::StratumClient() = default;
StratumClient::~StratumClient() { stop_mining(); }

bool StratumClient::connect(const std::string& host, int port,
                            const std::string& wallet, const std::string& worker,
                            const std::string& password) {
    host_ = host;
    port_ = port;
    wallet_ = wallet;
    worker_ = worker;
    password_ = password;

    // Resolve hostname
    struct addrinfo hints, *result = nullptr;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);

    int ret = getaddrinfo(host.c_str(), port_str, &hints, &result);
    if (ret != 0) {
        fprintf(stderr, "[stratum] Failed to resolve %s: %s\n", host.c_str(), gai_strerror(ret));
        return false;
    }

    // Create socket
    sock_ = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (sock_ < 0) {
        fprintf(stderr, "[stratum] socket() failed: %s\n", strerror(errno));
        freeaddrinfo(result);
        return false;
    }

    // Disable Nagle's algorithm for lower latency
    int opt = 1;
    setsockopt(sock_, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

    // Connect
    ret = connect(sock_, result->ai_addr, result->ai_addrlen);
    freeaddrinfo(result);

    if (ret < 0) {
        fprintf(stderr, "[stratum] connect() failed: %s\n", strerror(errno));
        close(sock_);
        sock_ = -1;
        return false;
    }

    fprintf(stderr, "[stratum] Connected to %s:%d\n", host.c_str(), port);

    // Subscribe
    if (!subscribe()) {
        fprintf(stderr, "[stratum] Subscribe failed\n");
        close(sock_);
        sock_ = -1;
        return false;
    }

    // Authorize
    if (!authorize()) {
        fprintf(stderr, "[stratum] Authorization failed\n");
        close(sock_);
        sock_ = -1;
        return false;
    }

    connected_.store(true);
    fprintf(stderr, "[stratum] Authorized successfully\n");
    return true;
}

void StratumClient::start_mining() {
    running_.store(true);
    recv_thread_ = std::make_unique<std::thread>(&StratumClient::receive_loop, this);
    submit_thread_ = std::make_unique<std::thread>(&StratumClient::submit_loop, this);
}

void StratumClient::stop_mining() {
    running_.store(false);

    if (recv_thread_ && recv_thread_->joinable()) recv_thread_->join();
    if (submit_thread_ && submit_thread_->joinable()) submit_thread_->join();
    recv_thread_.reset();
    submit_thread_.reset();

    if (sock_ >= 0) {
        close(sock_);
        sock_ = -1;
    }
    connected_.store(false);
}

bool StratumClient::reconnect() {
    stop_mining();

    // Reset state
    share_counter_.store(0);
    {
        std::lock_guard<std::mutex> lock(request_map_mutex_);
        request_gpu_map_.clear();
    }

    // Reconnect with stored credentials
    if (!host_.empty() && !wallet_.empty()) {
        return connect(host_, port_, wallet_, worker_, password_);
    }
    return false;
}

bool StratumClient::subscribe() {
    request_id_.store(1);
    std::string msg = "{\"id\": 1, \"method\": \"mining.subscribe\", \"params\": []}\n";
    send_raw(msg);

    // Read responses until we get one with "id": 1
    // Use member recv_buf_ so leftover data survives to receive_loop
    std::string& recv_buffer = recv_buf_;
    for (int attempt = 0; attempt < 20; attempt++) {
        fd_set fds;
        struct timeval tv;
        FD_ZERO(&fds);
        FD_SET(sock_, &fds);
        tv.tv_sec = 5;
        tv.tv_usec = 0;

        int ret = select(sock_ + 1, &fds, nullptr, nullptr, &tv);
        if (ret <= 0) return false;

        char buf[4096];
        int bytes = recv(sock_, buf, sizeof(buf) - 1, 0);
        if (bytes <= 0) return false;
        buf[bytes] = '\0';
        recv_buffer.append(buf, bytes);

        // Process complete messages
        while (true) {
            auto nl = recv_buffer.find('\n');
            if (nl == std::string::npos) break;
            std::string line = recv_buffer.substr(0, nl);
            recv_buffer.erase(0, nl + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;

            auto parsed = JsonValue::parse(line);
            if (parsed["id"].is_number() && parsed["id"].to_int() == 1) {
                // This is our subscribe response
                if (!parsed["error"].is_null()) {
                    fprintf(stderr, "[stratum] Subscribe error\n");
                    return false;
                }

                auto& result = parsed["result"];
                if (result.is_array()) {
                    // Parse methods array: result[0]
                    if (result[0].is_array()) {
                        auto& methods = result[0];
                        for (size_t i = 0; i < methods.size(); i++) {
                            auto& pair = methods[i];
                            if (!pair.is_array() || pair.size() < 2) continue;
                            std::string method_name = pair[0].to_string();
                            std::string method_id = pair[1].to_string();
                            if (i == 0) notify_method_ = method_name;
                            else if (i == 1) submit_method_ = method_name;
                            else if (method_name == "mining.set_difficulty") difficulty_method_ = method_name;
                            else if (method_name == "mining.ping") ping_id_ = method_id;
                        }
                    }
                    // Parse extranonce1: result[1]
                    if (result.size() > 1 && result[1].is_string()) {
                        extranonce1_ = result[1].to_string();
                    }
                    // Parse extranonce2 size: result[2]
                    if (result.size() > 2 && result[2].is_number()) {
                        extranonce2_size_ = static_cast<size_t>(result[2].to_int());
                    }
                }

                fprintf(stderr, "[stratum] Subscribed (notify=%s, submit=%s, extranonce1=%s, en2_size=%zu)\n",
                        notify_method_.c_str(), submit_method_.c_str(),
                        extranonce1_.c_str(), extranonce2_size_);
                return true;
            }
            // Non-id messages (notifications) — ignore or process
        }
    }

    return false;
}

bool StratumClient::authorize() {
    request_id_.store(2);
    std::string worker_full = worker_ + "." + wallet_;

    char msg_buf[2048];
    snprintf(msg_buf, sizeof(msg_buf),
             "{\"id\": 2, \"method\": \"mining.authorize\", \"params\": [\"%s\", \"%s\"]}\n",
             worker_full.c_str(), password_.c_str());
    send_raw(msg_buf);

    // Read responses until we get one with "id": 2
    // Prepend any leftover data from subscribe(), use member recv_buf_ for leftovers
    std::string recv_buffer = std::move(recv_buf_);
    for (int attempt = 0; attempt < 20; attempt++) {
        fd_set fds;
        struct timeval tv;
        FD_ZERO(&fds);
        FD_SET(sock_, &fds);
        tv.tv_sec = 5;
        tv.tv_usec = 0;

        int ret = select(sock_ + 1, &fds, nullptr, nullptr, &tv);
        if (ret <= 0) return false;

        char buf[4096];
        int bytes = recv(sock_, buf, sizeof(buf) - 1, 0);
        if (bytes <= 0) return false;
        buf[bytes] = '\0';
        recv_buffer.append(buf, bytes);

        while (true) {
            auto nl = recv_buffer.find('\n');
            if (nl == std::string::npos) break;
            std::string line = recv_buffer.substr(0, nl);
            recv_buffer.erase(0, nl + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;

            auto parsed = JsonValue::parse(line);
            if (parsed["id"].is_number() && parsed["id"].to_int() == 2) {
                auto& result = parsed["result"];
                if (!result.is_bool() || !result.to_bool()) {
                    fprintf(stderr, "[stratum] Authorization rejected\n");
                    recv_buf_ = std::move(recv_buffer);
                    return false;
                }

                fprintf(stderr, "[stratum] Authorized successfully\n");
                recv_buf_ = std::move(recv_buffer);
                return true;
            }
        }
    }

    recv_buf_ = std::move(recv_buffer);
    return false;
}

bool StratumClient::submit_share(const std::string& job_id, uint32_t nonce,
                                  const std::string& ntime,
                                  int gpu_index) {
    uint64_t counter = share_counter_.fetch_add(1, std::memory_order_relaxed);
    char en2_hex[32];
    size_t hex_len = extranonce2_size_ * 2;
    snprintf(en2_hex, sizeof(en2_hex), "%0*llx", (int)hex_len, (unsigned long long)counter);
    return push_share({job_id, counter, nonce, ntime, gpu_index});
}

void StratumClient::set_callbacks(JobCallback job_cb,
                                   ShareResultCallback result_cb,
                                   VardiffCallback vardiff_cb) {
    job_cb_ = std::move(job_cb);
    result_cb_ = std::move(result_cb);
    vardiff_cb_ = std::move(vardiff_cb);
}

bool StratumClient::push_share(const ShareSubmission& share) {
    size_t head = ring_head_.load(std::memory_order_relaxed);
    while (true) {
        size_t next = (head + 1) % SHARE_RING_SIZE;
        if (next == ring_tail_.load(std::memory_order_acquire)) {
            return false; // Ring full
        }
        // Try to claim this slot atomically
        if (ring_head_.compare_exchange_weak(head, next, std::memory_order_release)) {
            ring_[head] = share;
            return true;
        }
        // head is updated by compare_exchange_weak on failure
    }
}

bool StratumClient::pop_share(ShareSubmission& share) {
    size_t tail = ring_tail_.load(std::memory_order_relaxed);
    if (tail == ring_head_.load(std::memory_order_acquire)) {
        return false; // empty
    }
    share = ring_[tail];
    ring_tail_.store((tail + 1) % SHARE_RING_SIZE, std::memory_order_release);
    return true;
}

void StratumClient::send_raw(const std::string& data) {
    std::lock_guard<std::mutex> lock(send_mutex_);
    if (sock_ < 0) return;
    size_t sent = 0;
    while (sent < data.size()) {
        ssize_t n = send(sock_, data.c_str() + sent, data.size() - sent, 0);
        if (n <= 0) break;
        sent += n;
    }
}

void StratumClient::receive_loop() {
    char buf[65536];
    std::string buffer = std::move(recv_buf_);  // prepend leftover data from subscribe/authorize
    auto last_heartbeat = std::chrono::steady_clock::now();

    while (running_.load()) {
        struct pollfd pfd;
        pfd.fd = sock_;
        pfd.events = POLLIN;
        int ret = poll(&pfd, 1, 1000); // 1s timeout for heartbeat

        if (ret > 0 && (pfd.revents & POLLIN)) {
            int bytes = recv(sock_, buf, sizeof(buf), 0);
            if (bytes <= 0) {
                fprintf(stderr, "[stratum] Connection lost, reconnecting...\n");
                connected_.store(false);
                break;
            }
            buffer.append(buf, bytes);
        }

        // Process complete messages (newline-delimited)
        while (true) {
            auto nl = buffer.find('\n');
            if (nl == std::string::npos) break;
            std::string msg = buffer.substr(0, nl);
            buffer.erase(0, nl + 1);

            // Remove trailing \r
            if (!msg.empty() && msg.back() == '\r') msg.pop_back();

            if (!msg.empty()) {
                handle_message(msg);
            }
        }

        // Send keepalive ping every 15 seconds if pool supports it
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_heartbeat).count() >= 15 && !ping_id_.empty()) {
            char ping_buf[256];
            snprintf(ping_buf, sizeof(ping_buf), "{\"id\": %d, \"method\": \"mining.ping\", \"params\": [%s]}\n",
                     request_id_.load(), ping_id_.c_str());
            send_raw(ping_buf);
            last_heartbeat = now;
        }
    }
}

void StratumClient::submit_loop() {
    while (running_.load()) {
        ShareSubmission share;
        if (pop_share(share)) {
            int id = request_id_.fetch_add(1) + 1;
            // Generate extranonce2 hex string from counter
            char en2_hex[32];
            size_t hex_len = extranonce2_size_ * 2;
            snprintf(en2_hex, sizeof(en2_hex), "%0*llx", (int)hex_len, (unsigned long long)share.extranonce2_counter);

            char msg_buf[1024];
            // Standard Stratum submit: ["worker", "job_id", "extranonce2", "ntime", "nonce"]
            int len = snprintf(msg_buf, sizeof(msg_buf),
                "{\"id\": %d, \"method\": \"mining.submit\", \"params\": [\"%s\", \"%s\", \"%s\", \"%s\", \"%08x\"]}\n",
                id,
                (worker_ + "." + wallet_).c_str(),
                share.job_id.c_str(),
                en2_hex,
                share.ntime.c_str(),
                share.nonce);

            if (len > 0 && len < static_cast<int>(sizeof(msg_buf))) {
                send_raw(std::string(msg_buf));
                // Track which GPU submitted this share for result attribution
                {
                    std::lock_guard<std::mutex> lock(request_map_mutex_);
                    request_gpu_map_[id] = share.gpu_index;
                }
            }
        }
        usleep(1000); // 1ms sleep to avoid busy loop
    }
}

void StratumClient::handle_message(const std::string& msg) {
    auto parsed = JsonValue::parse(msg);
    if (parsed.is_null()) return;

    // Check if this is a notification (no "id")
    if (!parsed["id"].is_number()) {
        std::string method = parsed["method"].to_string();
        std::string params_str = parsed["params"].serialize();

        if (method == "mining.notify" || method == notify_method_) {
            handle_notify(params_str);
        } else if (method == "mining.set_difficulty" || method == difficulty_method_) {
            handle_set_difficulty(params_str);
        }
        return;
    }

    // It's a response to our request
    std::string id_str;
    if (parsed["id"].is_string())
        id_str = parsed["id"].to_string();
    else if (parsed["id"].is_number())
        id_str = std::to_string(parsed["id"].to_int());

    handle_response(id_str, parsed["result"].serialize(),
                    parsed["error"].serialize());
}

void StratumClient::handle_notify(const std::string& params_str) {
    auto params = JsonValue::parse(params_str);
    if (!params.is_array() || params.size() < 9) return;

    JobAssignment job;
    job.job_id = params[0].to_string();
    job.prevhash = params[1].to_string();
    job.coinb1 = params[2].to_string();
    job.coinb2 = params[3].to_string();

    // Merkle branch may be an array or string
    if (params[4].is_array()) {
        job.merkle_branch = params[4].serialize();
    } else {
        job.merkle_branch = params[4].to_string();
    }

    // version can be a number or a hex string
    if (params[5].is_string())
        job.version = params[5].to_string();
    else if (params[5].is_number())
        job.version = std::to_string(params[5].to_int());

    // nbits is a hex string, not a JSON number
    job.target_nbits = static_cast<uint32_t>(std::stoul(params[6].to_string(), nullptr, 16));
    job.ntime = params[7].to_string();

    if (params.size() > 8)
        job.clean_jobs = params[8].to_bool();

    {
        std::lock_guard<std::mutex> lock(job_mutex_);
        current_job_ = job;
    }

    if (job_cb_) {
        job_cb_(job);
    }

    fprintf(stderr, "[stratum] New job: %s (nbits=0x%08x)\n",
            job.job_id.substr(0, 8).c_str(), job.target_nbits);
}

void StratumClient::handle_set_difficulty(const std::string& params_str) {
    auto params = JsonValue::parse(params_str);
    if (!params.is_array() || params.size() < 1) return;

    current_difficulty_ = params[0].to_number();
    fprintf(stderr, "[stratum] New difficulty: %f\n", current_difficulty_);

    if (vardiff_cb_) {
        DifficultyAdjust adj;
        adj.new_difficulty = current_difficulty_;
        vardiff_cb_(adj);
    }
}

void StratumClient::handle_response(const std::string& id,
                                     const std::string& result,
                                     const std::string& error) {
    if (id == "2") {
        // Authorization response
        auto parsed = JsonValue::parse(result);
        if (!parsed.is_bool() || !parsed.to_bool()) {
            fprintf(stderr, "[stratum] Auth rejected\n");
            connected_.store(false);
        }
    } else if (!error.empty() && error != "null") {
        fprintf(stderr, "[stratum] Error response (id=%s): %s\n", id.c_str(), error.c_str());
    }

    // Handle share submission results
    if (id != "1" && id != "2") {
        auto parsed = JsonValue::parse(result);
        if (parsed.is_bool()) {
            if (result_cb_) {
                ShareResult sr;
                sr.accepted = parsed.to_bool();
                sr.outcome = sr.accepted ? "Accepted" : "Rejected";
                sr.is_block_find = false;
                // Look up which GPU submitted this share
                int id_num = std::stoi(id);
                {
                    std::lock_guard<std::mutex> lock(request_map_mutex_);
                    auto it = request_gpu_map_.find(id_num);
                    if (it != request_gpu_map_.end()) {
                        sr.gpu_index = it->second;
                        request_gpu_map_.erase(it);
                    }
                }
                result_cb_(sr);
            }
        }
    }
}

} // namespace propminer