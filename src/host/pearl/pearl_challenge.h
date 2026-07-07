#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <optional>
#include <string>

namespace pearl {

// pearl/v1 connection challenge solver (Kryptex :7048 / AlphaPool / HeroMiners).
//
// Challenge-first Pearl stratum pools send a `pearl.challenge` line immediately
// after the TCP/TLS connect and refuse to hand out jobs until the miner returns
// a winning nonce. A u64 nonce wins when
//
//     BLAKE3( seed[32] || nonce_as_8_bytes_LITTLE_ENDIAN )
//
// has at least `difficulty` leading-zero BITS (counted over the 32-byte output
// as a big-endian byte stream). See ARC-miner docs/PEARL-V1-CHALLENGE.md.
class PearlChallenge {
public:
    // Multithreaded solver. Returns the winning u64 nonce, or nullopt if
    // cancelled / difficulty exceeds max_difficulty. `difficulty` is the number
    // of required leading-zero bits; expected work is ~2^difficulty hashes.
    static std::optional<uint64_t> solve(const std::array<uint8_t, 32>& seed,
                                         int difficulty,
                                         int max_difficulty,
                                         const std::atomic<bool>* cancel);

    // Independent verification path (uses a different BLAKE3 implementation than
    // the unrolled solver so a bug in one is caught by the other).
    static bool verify(const std::array<uint8_t, 32>& seed, uint64_t nonce,
                       int difficulty);

    // The u64 nonce is transmitted as zero-padded 16-hex, most-significant first
    // (e.g. 0x2cc663e5 -> "000000002cc663e5"). Independent of the little-endian
    // byte order fed to the hash.
    static std::string nonce_to_wire_hex(uint64_t nonce);

    static std::optional<std::array<uint8_t, 32>> hex_to_seed(const std::string& hex);
};

}  // namespace pearl
