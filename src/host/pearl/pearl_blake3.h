#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace pearl {

// Minimal BLAKE3 helper that delegates to libpearl_mining_capi.
class Blake3Helper {
public:
    static std::array<uint8_t, 32> keyed(const uint8_t* data, size_t len,
                                         const uint8_t key[32]);
    // Standard (unkeyed) BLAKE3 hash. Matches C# Blake3.Hash(input).
    static std::array<uint8_t, 32> hash(const uint8_t* data, size_t len);
    // BLAKE3 XOF output. Matches C# Blake3.Xof(input, out_len).
    static std::vector<uint8_t> xof(const uint8_t* data, size_t len, size_t out_len);
};

} // namespace pearl
