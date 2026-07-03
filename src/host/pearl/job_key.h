#pragma once

#include <array>
#include <cstdint>
#include <cstddef>

#include "pearl_types.h"

namespace pearl {

// Derive the canonical V2 jobKey: BLAKE3(sigma || config_bytes).
std::array<uint8_t, 32> derive_job_key(const uint8_t sigma[32], const MiningConfig& cfg);

inline std::array<uint8_t, 32> derive_job_key(const std::array<uint8_t, 32>& sigma,
                                               const MiningConfig& cfg) {
    return derive_job_key(sigma.data(), cfg);
}

} // namespace pearl
