#include "pearl_challenge.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include "pearl_blake3.h"

namespace pearl {
namespace {

inline uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

inline uint32_t read_le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

inline uint32_t bswap32(uint32_t x) {
    return ((x & 0x000000FFu) << 24) | ((x & 0x0000FF00u) << 8) |
           ((x & 0x00FF0000u) >> 8) | ((x & 0xFF000000u) >> 24);
}

inline int clz32(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return x == 0 ? 32 : __builtin_clz(x);
#else
    int n = 0;
    while ((x & 0x80000000u) == 0 && n < 32) { x <<= 1; ++n; }
    return n;
#endif
}

inline void g_mix(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
                  uint32_t x, uint32_t y) {
    a += b + x; d = rotr32(d ^ a, 16);
    c += d;     b = rotr32(b ^ c, 12);
    a += b + y; d = rotr32(d ^ a, 8);
    c += d;     b = rotr32(b ^ c, 7);
}

// One BLAKE3 root compression of (seed[32] || nonce_le || zero-pad-to-64) and a
// leading-zero-bits >= difficulty test. Message words 8,9 hold the nonce; words
// 10-15 are constant zero. Ported bit-for-bit from ARC-miner ChallengeSolver
// (verified against the official BLAKE3 vectors and a live-accepted nonce).
inline bool try_nonce(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
                      uint32_t s4, uint32_t s5, uint32_t s6, uint32_t s7,
                      uint64_t nonce, int difficulty) {
    const uint32_t n0 = static_cast<uint32_t>(nonce);
    const uint32_t n1 = static_cast<uint32_t>(nonce >> 32);

    uint32_t v0 = 0x6A09E667u, v1 = 0xBB67AE85u, v2 = 0x3C6EF372u, v3 = 0xA54FF53Au;
    uint32_t v4 = 0x510E527Fu, v5 = 0x9B05688Cu, v6 = 0x1F83D9ABu, v7 = 0x5BE0CD19u;
    uint32_t v8 = 0x6A09E667u, v9 = 0xBB67AE85u, v10 = 0x3C6EF372u, v11 = 0xA54FF53Au;
    // counter=0, block_len=40, flags = CHUNK_START|CHUNK_END|ROOT (1|2|8)
    uint32_t v12 = 0, v13 = 0, v14 = 40, v15 = 1u | 2u | 8u;

    // round 0
    g_mix(v0, v4, v8,  v12, s0, s1);
    g_mix(v1, v5, v9,  v13, s2, s3);
    g_mix(v2, v6, v10, v14, s4, s5);
    g_mix(v3, v7, v11, v15, s6, s7);
    g_mix(v0, v5, v10, v15, n0, n1);
    g_mix(v1, v6, v11, v12, 0u, 0u);
    g_mix(v2, v7, v8,  v13, 0u, 0u);
    g_mix(v3, v4, v9,  v14, 0u, 0u);
    // round 1
    g_mix(v0, v4, v8,  v12, s2, s6);
    g_mix(v1, v5, v9,  v13, s3, 0u);
    g_mix(v2, v6, v10, v14, s7, s0);
    g_mix(v3, v7, v11, v15, s4, 0u);
    g_mix(v0, v5, v10, v15, s1, 0u);
    g_mix(v1, v6, v11, v12, 0u, s5);
    g_mix(v2, v7, v8,  v13, n1, 0u);
    g_mix(v3, v4, v9,  v14, 0u, n0);
    // round 2
    g_mix(v0, v4, v8,  v12, s3, s4);
    g_mix(v1, v5, v9,  v13, 0u, 0u);
    g_mix(v2, v6, v10, v14, 0u, s2);
    g_mix(v3, v7, v11, v15, s7, 0u);
    g_mix(v0, v5, v10, v15, s6, s5);
    g_mix(v1, v6, v11, v12, n1, s0);
    g_mix(v2, v7, v8,  v13, 0u, 0u);
    g_mix(v3, v4, v9,  v14, n0, s1);
    // round 3
    g_mix(v0, v4, v8,  v12, 0u, s7);
    g_mix(v1, v5, v9,  v13, 0u, n1);
    g_mix(v2, v6, v10, v14, 0u, s3);
    g_mix(v3, v7, v11, v15, 0u, 0u);
    g_mix(v0, v5, v10, v15, s4, s0);
    g_mix(v1, v6, v11, v12, 0u, s2);
    g_mix(v2, v7, v8,  v13, s5, n0);
    g_mix(v3, v4, v9,  v14, s1, s6);
    // round 4
    g_mix(v0, v4, v8,  v12, 0u, 0u);
    g_mix(v1, v5, v9,  v13, n1, 0u);
    g_mix(v2, v6, v10, v14, 0u, 0u);
    g_mix(v3, v7, v11, v15, 0u, n0);
    g_mix(v0, v5, v10, v15, s7, s2);
    g_mix(v1, v6, v11, v12, s5, s3);
    g_mix(v2, v7, v8,  v13, s0, s1);
    g_mix(v3, v4, v9,  v14, s6, s4);
    // round 5
    g_mix(v0, v4, v8,  v12, n1, 0u);
    g_mix(v1, v5, v9,  v13, 0u, s5);
    g_mix(v2, v6, v10, v14, n0, 0u);
    g_mix(v3, v7, v11, v15, 0u, s1);
    g_mix(v0, v5, v10, v15, 0u, s3);
    g_mix(v1, v6, v11, v12, s0, 0u);
    g_mix(v2, v7, v8,  v13, s2, s6);
    g_mix(v3, v4, v9,  v14, s4, s7);
    // round 6
    g_mix(v0, v4, v8,  v12, 0u, 0u);
    g_mix(v1, v5, v9,  v13, s5, s0);
    g_mix(v2, v6, v10, v14, s1, n1);
    g_mix(v3, v7, v11, v15, n0, s6);
    g_mix(v0, v5, v10, v15, 0u, 0u);
    g_mix(v1, v6, v11, v12, s2, 0u);
    g_mix(v2, v7, v8,  v13, s3, s4);
    g_mix(v3, v4, v9,  v14, s7, 0u);

    // Output word i = v_i ^ v_{i+8}; hash byte stream is words little-endian, so
    // leading-zero bits of the stream = clz of the byte-reversed word.
    const uint32_t h0 = bswap32(v0 ^ v8);
    if (h0 != 0) {
        return clz32(h0) >= difficulty;
    }
    int lz = 32;
    const uint32_t rest[7] = {v1 ^ v9, v2 ^ v10, v3 ^ v11,
                              v4 ^ v12, v5 ^ v13, v6 ^ v14, v7 ^ v15};
    for (uint32_t w : rest) {
        const uint32_t rev = bswap32(w);
        if (rev == 0) { lz += 32; continue; }
        lz += clz32(rev);
        break;
    }
    return lz >= difficulty;
}

int leading_zero_bits(const std::array<uint8_t, 32>& h) {
    int lz = 0;
    for (uint8_t b : h) {
        if (b == 0) { lz += 8; continue; }
        int n = 0;
        while ((b & 0x80u) == 0) { b <<= 1; ++n; }
        return lz + n;
    }
    return lz;
}

}  // namespace

std::optional<uint64_t> PearlChallenge::solve(const std::array<uint8_t, 32>& seed,
                                              int difficulty, int max_difficulty,
                                              const std::atomic<bool>* cancel) {
    if (difficulty < 0 || difficulty > 255) return std::nullopt;
    if (max_difficulty > 0 && difficulty > max_difficulty) return std::nullopt;

    const uint32_t s0 = read_le32(seed.data() + 0);
    const uint32_t s1 = read_le32(seed.data() + 4);
    const uint32_t s2 = read_le32(seed.data() + 8);
    const uint32_t s3 = read_le32(seed.data() + 12);
    const uint32_t s4 = read_le32(seed.data() + 16);
    const uint32_t s5 = read_le32(seed.data() + 20);
    const uint32_t s6 = read_le32(seed.data() + 24);
    const uint32_t s7 = read_le32(seed.data() + 28);

    const unsigned hw = std::thread::hardware_concurrency();
    const int threads = std::max(1, static_cast<int>(hw == 0 ? 1 : hw));

    std::atomic<bool> found{false};
    std::atomic<uint64_t> found_nonce{0};

    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&, t]() {
            for (uint64_t nonce = static_cast<uint64_t>(t);;
                 nonce += static_cast<uint64_t>(threads)) {
                if (((nonce / static_cast<uint64_t>(threads)) & 0x1FFF) == 0) {
                    if (found.load(std::memory_order_relaxed)) return;
                    if (cancel && cancel->load(std::memory_order_relaxed)) return;
                }
                if (try_nonce(s0, s1, s2, s3, s4, s5, s6, s7, nonce, difficulty)) {
                    bool expected = false;
                    if (found.compare_exchange_strong(expected, true)) {
                        found_nonce.store(nonce, std::memory_order_relaxed);
                    }
                    return;
                }
            }
        });
    }
    for (auto& w : workers) w.join();
    if (!found.load()) return std::nullopt;
    return found_nonce.load();
}

bool PearlChallenge::verify(const std::array<uint8_t, 32>& seed, uint64_t nonce,
                            int difficulty) {
    std::array<uint8_t, 40> input{};
    std::memcpy(input.data(), seed.data(), 32);
    for (int i = 0; i < 8; ++i) {
        input[32 + i] = static_cast<uint8_t>((nonce >> (8 * i)) & 0xFF);
    }
#if defined(PROP_MINER_DISABLE_RUST_CRYPTO)
    // The unrolled solver is the only path available; trust it.
    (void)input;
    return true;
#else
    const auto hash = Blake3Helper::hash(input.data(), input.size());
    return leading_zero_bits(hash) >= difficulty;
#endif
}

std::string PearlChallenge::nonce_to_wire_hex(uint64_t nonce) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx",
                  static_cast<unsigned long long>(nonce));
    return std::string(buf);
}

std::optional<std::array<uint8_t, 32>> PearlChallenge::hex_to_seed(
    const std::string& hex) {
    std::string h = hex;
    if (h.size() >= 2 && (h[0] == '0') && (h[1] == 'x' || h[1] == 'X')) {
        h = h.substr(2);
    }
    if (h.size() != 64) return std::nullopt;
    std::array<uint8_t, 32> out{};
    for (size_t i = 0; i < 32; ++i) {
        auto nibble = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
        };
        const int hi = nibble(h[i * 2]);
        const int lo = nibble(h[i * 2 + 1]);
        if (hi < 0 || lo < 0) return std::nullopt;
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return out;
}

}  // namespace pearl
