#include "grpc_client.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace pearl {

namespace {
    constexpr uint32_t FRAME_HEADER_SIZE = 9;
    enum FrameType : uint8_t {
        FRAME_DATA = 0x0,
        FRAME_HEADERS = 0x1,
        FRAME_PRIORITY = 0x2,
        FRAME_RST_STREAM = 0x3,
        FRAME_SETTINGS = 0x4,
        FRAME_PING = 0x6,
        FRAME_GOAWAY = 0x7,
        FRAME_WINDOW_UPDATE = 0x8,
    };
    enum FrameFlag : uint8_t {
        FLAG_END_STREAM = 0x1,
        FLAG_ACK = 0x1,
        FLAG_END_HEADERS = 0x4,
    };

    void write_u24_be(uint8_t* p, uint32_t v) {
        p[0] = static_cast<uint8_t>(v >> 16);
        p[1] = static_cast<uint8_t>(v >> 8);
        p[2] = static_cast<uint8_t>(v);
    }

    uint32_t read_u24_be(const uint8_t* p) {
        return (static_cast<uint32_t>(p[0]) << 16) |
               (static_cast<uint32_t>(p[1]) << 8) |
               static_cast<uint32_t>(p[2]);
    }

    uint32_t read_u32_be(const uint8_t* p) {
        return (static_cast<uint32_t>(p[0]) << 24) |
               (static_cast<uint32_t>(p[1]) << 16) |
               (static_cast<uint32_t>(p[2]) << 8) |
               static_cast<uint32_t>(p[3]);
    }

    void write_u32_be(uint8_t* p, uint32_t v) {
        p[0] = static_cast<uint8_t>(v >> 24);
        p[1] = static_cast<uint8_t>(v >> 16);
        p[2] = static_cast<uint8_t>(v >> 8);
        p[3] = static_cast<uint8_t>(v);
    }

    // Minimal HPACK encoder for the request headers we need.
    class HpackEncoder {
    public:
        std::vector<uint8_t> encode_request_headers(const std::string& method,
                                                     const std::string& scheme,
                                                     const std::string& authority,
                                                     const std::string& path,
                                                     const std::string& user_agent,
                                                     const std::string& content_type) {
            std::vector<uint8_t> out;
            encode_literal(out, ":method", method);
            encode_literal(out, ":scheme", scheme);
            encode_literal(out, ":authority", authority);
            encode_literal(out, ":path", path);
            encode_literal(out, "content-type", content_type);
            encode_literal(out, "user-agent", user_agent);
            encode_literal(out, "te", "trailers");
            return out;
        }

    private:
        void encode_literal(std::vector<uint8_t>& out,
                            const std::string& name,
                            const std::string& value) {
            out.push_back(0x40);
            encode_string(out, name);
            encode_string(out, value);
        }

        void encode_string(std::vector<uint8_t>& out, const std::string& s) {
            size_t len = s.size();
            if (len < 128) {
                out.push_back(static_cast<uint8_t>(len));
            } else {
                uint64_t v = len;
                out.push_back(static_cast<uint8_t>((v & 0x7F) | 0x80));
                v >>= 7;
                while (v) {
                    out.push_back(static_cast<uint8_t>((v & 0x7F) | 0x80));
                    v >>= 7;
                }
                out.back() &= 0x7F;
            }
            out.insert(out.end(), s.begin(), s.end());
        }
    };
} // namespace

struct PearlGrpcClient::Impl {
    Options opts;
    int sock_ = -1;
    SSL_CTX* ctx_ = nullptr;
    SSL* ssl_ = nullptr;
    bool connected_ = false;
    uint32_t next_stream_id_ = 1;
    uint32_t stream_id_ = 0;
    std::string last_error_;

    std::mutex recv_mtx_;
    std::vector<uint8_t> read_buf_;
    std::queue<proto::PoolEvent> event_queue_;
    bool headers_received_ = false;
    bool stream_closed_ = false;

    ~Impl() { disconnect(); }

    bool set_error(const std::string& msg) {
        last_error_ = msg;
        return false;
    }

    bool read_exact(uint8_t* out, size_t n, int timeout_ms) {
        size_t got = 0;
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (got < n) {
            int r = 0;
            if (ssl_) {
                r = SSL_read(ssl_, out + got, static_cast<int>(n - got));
                if (r <= 0) {
                    int err = SSL_get_error(ssl_, r);
                    if (err == SSL_ERROR_WANT_READ || err == SSL_ERROR_WANT_WRITE) {
                        if (!poll_socket(err == SSL_ERROR_WANT_READ ? POLLIN : POLLOUT,
                                         static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                             deadline - std::chrono::steady_clock::now()).count())))
                            return false;
                        continue;
                    }
                    return set_error("SSL_read failed");
                }
            } else {
                r = static_cast<int>(recv(sock_, out + got, n - got, 0));
                if (r <= 0) {
                    if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                        if (!poll_socket(POLLIN, static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                            deadline - std::chrono::steady_clock::now()).count())))
                            return false;
                        continue;
                    }
                    return set_error("recv failed");
                }
            }
            got += r;
        }
        return true;
    }

    bool write_exact(const uint8_t* data, size_t n) {
        size_t sent = 0;
        while (sent < n) {
            int r = 0;
            if (ssl_) {
                r = SSL_write(ssl_, data + sent, static_cast<int>(n - sent));
                if (r <= 0) {
                    int err = SSL_get_error(ssl_, r);
                    if (err == SSL_ERROR_WANT_READ || err == SSL_ERROR_WANT_WRITE) {
                        if (!poll_socket(err == SSL_ERROR_WANT_READ ? POLLIN : POLLOUT, 5000))
                            return false;
                        continue;
                    }
                    return set_error("SSL_write failed");
                }
            } else {
                r = static_cast<int>(send(sock_, data + sent, n - sent, MSG_NOSIGNAL));
                if (r <= 0) {
                    if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                        if (!poll_socket(POLLOUT, 5000)) return false;
                        continue;
                    }
                    return set_error("send failed");
                }
            }
            sent += r;
        }
        return true;
    }

    bool poll_socket(short events, int timeout_ms) {
        if (timeout_ms <= 0) return set_error("poll timeout");
        pollfd pfd{sock_, events, 0};
        int r = poll(&pfd, 1, timeout_ms);
        if (r < 0) return set_error("poll failed");
        if (r == 0) return set_error("poll timeout");
        return true;
    }

    bool send_frame(uint8_t type, uint8_t flags, uint32_t stream_id,
                    const uint8_t* payload, uint32_t len) {
        uint8_t hdr[FRAME_HEADER_SIZE];
        write_u24_be(hdr, len);
        hdr[3] = type;
        hdr[4] = flags;
        write_u32_be(hdr + 5, stream_id);
        if (!write_exact(hdr, sizeof(hdr))) return false;
        if (len > 0 && !write_exact(payload, len)) return false;
        return true;
    }

    bool send_settings_ack() {
        return send_frame(FRAME_SETTINGS, FLAG_ACK, 0, nullptr, 0);
    }

    bool send_window_update(uint32_t stream_id, uint32_t increment) {
        uint8_t payload[4];
        write_u32_be(payload, increment);
        return send_frame(FRAME_WINDOW_UPDATE, 0, stream_id, payload, 4);
    }

    bool send_preface_and_settings() {
        const char* preface = "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";
        if (!write_exact(reinterpret_cast<const uint8_t*>(preface), std::strlen(preface)))
            return false;
        if (!send_frame(FRAME_SETTINGS, 0, 0, nullptr, 0)) return false;
        if (!send_window_update(0, 0x7FFFFFFFu)) return false;
        return true;
    }

    bool connect_tcp() {
        struct hostent* he = gethostbyname(opts.host.c_str());
        if (!he) return set_error("gethostbyname failed");
        sock_ = socket(AF_INET, SOCK_STREAM, 0);
        if (sock_ < 0) return set_error("socket failed");
        int flags = fcntl(sock_, F_GETFL, 0);
        fcntl(sock_, F_SETFL, flags | O_NONBLOCK);
        int yes = 1;
        setsockopt(sock_, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(opts.port));
        std::memcpy(&addr.sin_addr, he->h_addr_list[0], he->h_length);
        int rc = ::connect(sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        if (rc < 0 && errno != EINPROGRESS) return set_error("connect failed");
        if (rc != 0) {
            if (!poll_socket(POLLOUT, opts.connect_timeout_ms)) return false;
            int soerr = 0;
            socklen_t soerr_len = sizeof(soerr);
            getsockopt(sock_, SOL_SOCKET, SO_ERROR, &soerr, &soerr_len);
            if (soerr != 0) return set_error("connect error");
        }
        return true;
    }

    bool connect_tls() {
        SSL_library_init();
        SSL_load_error_strings();
        OpenSSL_add_all_algorithms();
        ctx_ = SSL_CTX_new(TLS_client_method());
        if (!ctx_) return set_error("SSL_CTX_new failed");
        SSL_CTX_set_default_verify_paths(ctx_);
        ssl_ = SSL_new(ctx_);
        if (!ssl_) return set_error("SSL_new failed");
        SSL_set_fd(ssl_, sock_);
        SSL_set_tlsext_host_name(ssl_, opts.host.c_str());
        SSL_set_connect_state(ssl_);
        while (true) {
            int r = SSL_do_handshake(ssl_);
            if (r == 1) break;
            int err = SSL_get_error(ssl_, r);
            if (err == SSL_ERROR_WANT_READ) {
                if (!poll_socket(POLLIN, opts.connect_timeout_ms)) return false;
            } else if (err == SSL_ERROR_WANT_WRITE) {
                if (!poll_socket(POLLOUT, opts.connect_timeout_ms)) return false;
            } else {
                return set_error("TLS handshake failed");
            }
        }
        return true;
    }

    bool perform_http2_handshake() {
        if (!send_preface_and_settings()) return false;
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(opts.connect_timeout_ms);
        bool got_settings = false, got_ack = false;
        while (!got_settings || !got_ack) {
            if (std::chrono::steady_clock::now() > deadline)
                return set_error("HTTP/2 handshake timeout");
            uint8_t hdr[FRAME_HEADER_SIZE];
            if (!read_exact(hdr, sizeof(hdr), 1000)) return false;
            uint32_t len = read_u24_be(hdr);
            uint8_t type = hdr[3];
            uint8_t flags = hdr[4];
            uint32_t sid = read_u32_be(hdr + 5) & 0x7FFFFFFFu;
            std::vector<uint8_t> payload(len);
            if (len > 0 && !read_exact(payload.data(), len, 1000)) return false;
            if (type == FRAME_SETTINGS) {
                if (flags & FLAG_ACK) got_ack = true;
                else { got_settings = true; send_settings_ack(); }
            } else if (type == FRAME_WINDOW_UPDATE && sid == 0) {
                // ignore
            } else if (type == FRAME_GOAWAY) {
                return set_error("GOAWAY during handshake");
            }
        }
        return true;
    }

    bool send_headers(uint32_t stream_id, const std::string& path,
                      bool end_stream) {
        HpackEncoder enc;
        auto headers = enc.encode_request_headers(
            "POST",
            impl_->opts.use_tls ? "https" : "http",
            impl_->opts.host + ":" + std::to_string(impl_->opts.port),
            path,
            impl_->opts.user_agent,
            "application/grpc");
        return send_frame(FRAME_HEADERS,
                          static_cast<uint8_t>(FLAG_END_HEADERS | (end_stream ? FLAG_END_STREAM : 0)),
                          stream_id, headers.data(), static_cast<uint32_t>(headers.size()));
    }

    bool send_data_message(uint32_t stream_id, const std::vector<uint8_t>& grpc_msg,
                           bool end_stream) {
        if (grpc_msg.size() > 0xFFFFFFu) return set_error("grpc message too large");
        std::vector<uint8_t> frame(5 + grpc_msg.size());
        frame[0] = 0; // compression flag
        write_u32_be(frame.data() + 1, static_cast<uint32_t>(grpc_msg.size()));
        std::memcpy(frame.data() + 5, grpc_msg.data(), grpc_msg.size());
        return send_frame(FRAME_DATA,
                          static_cast<uint8_t>(end_stream ? FLAG_END_STREAM : 0),
                          stream_id, frame.data(), static_cast<uint32_t>(frame.size()));
    }

    void drain_frames(int timeout_ms) {
        uint8_t hdr[FRAME_HEADER_SIZE];
        if (!read_exact(hdr, sizeof(hdr), timeout_ms)) return;
        uint32_t len = read_u24_be(hdr);
        uint8_t type = hdr[3];
        uint8_t flags = hdr[4];
        uint32_t sid = read_u32_be(hdr + 5) & 0x7FFFFFFFu;
        if (len > 0xFFFFFFu) { set_error("oversized frame"); return; }
        std::vector<uint8_t> payload(len);
        if (len > 0 && !read_exact(payload.data(), len, 1000)) return;

        if (type == FRAME_HEADERS && sid == stream_id_) {
            headers_received_ = true;
            if (flags & FLAG_END_STREAM) stream_closed_ = true;
        } else if (type == FRAME_DATA && sid == stream_id_) {
            read_buf_.insert(read_buf_.end(), payload.begin(), payload.end());
            if (flags & FLAG_END_STREAM) stream_closed_ = true;
            parse_grpc_messages();
        } else if (type == FRAME_SETTINGS) {
            if (!(flags & FLAG_ACK)) send_settings_ack();
        } else if (type == FRAME_GOAWAY) {
            stream_closed_ = true;
        } else if (type == FRAME_WINDOW_UPDATE) {
            // ignore
        } else if (type == FRAME_PING) {
            if (payload.size() == 8)
                send_frame(FRAME_PING, FLAG_ACK, 0, payload.data(), 8);
        }
    }

    void parse_grpc_messages() {
        while (read_buf_.size() >= 5) {
            bool compressed = read_buf_[0] != 0;
            uint32_t msg_len = read_u32_be(read_buf_.data() + 1);
            if (read_buf_.size() < 5 + msg_len) break;
            if (!compressed) {
                proto::PoolEvent evt;
                if (evt.decode(read_buf_.data() + 5, msg_len)) {
                    std::lock_guard<std::mutex> lk(recv_mtx_);
                    event_queue_.push(evt);
                }
            }
            read_buf_.erase(read_buf_.begin(), read_buf_.begin() + 5 + msg_len);
        }
    }

    void disconnect() {
        if (ssl_) { SSL_shutdown(ssl_); SSL_free(ssl_); ssl_ = nullptr; }
        if (ctx_) { SSL_CTX_free(ctx_); ctx_ = nullptr; }
        if (sock_ >= 0) { close(sock_); sock_ = -1; }
        connected_ = false;
        stream_id_ = 0;
    }
};

PearlGrpcClient::PearlGrpcClient(const Options& opts)
    : impl_(std::make_unique<Impl>()) {
    impl_->opts = opts;
}

PearlGrpcClient::~PearlGrpcClient() = default;

bool PearlGrpcClient::connect() {
    if (!impl_->connect_tcp()) return false;
    if (impl_->opts.use_tls && !impl_->connect_tls()) return false;
    if (!impl_->perform_http2_handshake()) return false;
    impl_->connected_ = true;
    return true;
}

void PearlGrpcClient::disconnect() {
    impl_->disconnect();
}

bool PearlGrpcClient::connected() const {
    return impl_->connected_;
}

std::string PearlGrpcClient::last_error() const {
    return impl_->last_error_;
}

bool PearlGrpcClient::register_miner(const proto::RegisterRequest& req, proto::RegisterResponse& out) {
    if (!impl_->connected_) return impl_->set_error("not connected");
    uint32_t sid = impl_->next_stream_id_;
    impl_->next_stream_id_ += 2;
    if (!impl_->send_headers(sid, "/pearlpool.mining.v2.MinerService/Register", true))
        return false;
    auto body = req.encode();
    if (!impl_->send_data_message(sid, body, false)) return false;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(impl_->opts.connect_timeout_ms);
    std::vector<uint8_t> response_body;
    while (std::chrono::steady_clock::now() < deadline) {
        uint8_t hdr[FRAME_HEADER_SIZE];
        if (!impl_->read_exact(hdr, sizeof(hdr), 1000)) return false;
        uint32_t len = read_u24_be(hdr);
        uint8_t type = hdr[3];
        uint8_t flags = hdr[4];
        uint32_t rsid = read_u32_be(hdr + 5) & 0x7FFFFFFFu;
        if (rsid != sid) {
            std::vector<uint8_t> discard(len);
            if (len > 0 && !impl_->read_exact(discard.data(), len, 1000)) return false;
            continue;
        }
        std::vector<uint8_t> payload(len);
        if (len > 0 && !impl_->read_exact(payload.data(), len, 1000)) return false;
        if (type == FRAME_DATA) {
            response_body.insert(response_body.end(), payload.begin(), payload.end());
            if (flags & FLAG_END_STREAM) break;
        } else if (type == FRAME_GOAWAY) {
            return impl_->set_error("GOAWAY");
        }
    }
    if (response_body.size() < 5) return impl_->set_error("empty register response");
    return out.decode(response_body.data() + 5, response_body.size() - 5);
}

bool PearlGrpcClient::start_mining_stream(const proto::AuthEvent& auth) {
    if (!impl_->connected_) return impl_->set_error("not connected");
    uint32_t sid = impl_->next_stream_id_;
    impl_->next_stream_id_ += 2;
    impl_->stream_id_ = sid;
    impl_->headers_received_ = false;
    impl_->stream_closed_ = false;
    if (!impl_->send_headers(sid, "/pearlpool.mining.v2.MinerService/MiningStream", false))
        return false;
    proto::MinerEvent evt;
    evt.seq = 1;
    evt.type = proto::MinerEventType::Auth;
    evt.payload = auth.encode();
    return send_event(evt);
}

bool PearlGrpcClient::send_event(const proto::MinerEvent& evt) {
    if (!impl_->connected_ || impl_->stream_id_ == 0)
        return impl_->set_error("mining stream not started");
    auto enc = evt.encode();
    return impl_->send_data_message(impl_->stream_id_, enc, false);
}

bool PearlGrpcClient::send_heartbeat(const proto::Heartbeat& hb) {
    proto::MinerEvent evt;
    evt.type = proto::MinerEventType::Heartbeat;
    evt.payload = hb.encode();
    return send_event(evt);
}

bool PearlGrpcClient::receive_event(proto::PoolEvent& out) {
    {
        std::lock_guard<std::mutex> lk(impl_->recv_mtx_);
        if (!impl_->event_queue_.empty()) {
            out = impl_->event_queue_.front();
            impl_->event_queue_.pop();
            return true;
        }
    }
    impl_->drain_frames(5000);
    std::lock_guard<std::mutex> lk(impl_->recv_mtx_);
    if (!impl_->event_queue_.empty()) {
        out = impl_->event_queue_.front();
        impl_->event_queue_.pop();
        return true;
    }
    return false;
}

bool PearlGrpcClient::try_receive_event(proto::PoolEvent& out) {
    std::lock_guard<std::mutex> lk(impl_->recv_mtx_);
    if (!impl_->event_queue_.empty()) {
        out = impl_->event_queue_.front();
        impl_->event_queue_.pop();
        return true;
    }
    return false;
}

} // namespace pearl
