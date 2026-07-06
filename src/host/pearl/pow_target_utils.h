#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace pearl {

// Incomplete block header bytes on Pearl V2 wire (akoya-miner SigmaContext.HeaderSize).
constexpr size_t kSigmaHeaderBytes = 76;

std::array<uint8_t, 32> nbits_to_target_le(uint32_t nbits);

// Convert 32-byte big-endian target (Bitcoin-style) to compact nbits (ARC-compatible).
uint32_t target_be_to_nbits(const std::array<uint8_t, 32>& target_be);

// Stratum pdiff: share_target = diff1_target / difficulty.
uint32_t difficulty_to_nbits_pdif(double difficulty);

// Parse hex target string (big-endian) to nbits.
uint32_t hex_target_to_nbits(const std::string& target_hex);

// Expand compact nbits to 64-char big-endian target hex (ARC NbitsToTargetHex).
std::string nbits_to_target_hex_be(uint32_t nbits);

// Read compact network difficulty from Pearl sigma header bytes [72:76] (LE).
uint32_t network_nbits_from_sigma(const std::array<uint8_t, kSigmaHeaderBytes>& sigma);

// Multiply a 256-bit little-endian target by factor; clamp to 2^256-1 on overflow.
std::array<uint8_t, 32> multiply_target_le_by_u64(const std::array<uint8_t, 32>& target_le,
                                                  uint64_t factor);

std::array<uint32_t, 8> target_le_to_pow_u32(const std::array<uint8_t, 32>& target_le);

inline std::array<uint8_t, 32> adjusted_pow_target_le(uint32_t nbits, uint64_t daf) {
    return multiply_target_le_by_u64(nbits_to_target_le(nbits), daf);
}

// Returns true when claimed_hash (LE) <= nbits_target * DAF (ShareTargetGuard).
bool claimed_hash_clears_target(const uint8_t claimed_hash[32],
                                uint32_t nbits,
                                uint64_t daf);

}  // namespace pearl
