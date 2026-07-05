#pragma once

#include <array>
#include <cstdint>

namespace pearl {

// Incomplete block header bytes on Pearl V2 wire (akoya-miner SigmaContext.HeaderSize).
constexpr size_t kSigmaHeaderBytes = 76;

std::array<uint8_t, 32> nbits_to_target_le(uint32_t nbits);

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
