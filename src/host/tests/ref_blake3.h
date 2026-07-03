#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace pearl {
namespace ref {

// C++ wrapper around the official BLAKE3 reference implementation.
// Used for host-only tests when libpearl_mining_capi.so is not available.
class Blake3Ref {
public:
    static constexpr size_t DIGEST_SIZE = 32;

    static std::array<uint8_t, DIGEST_SIZE> hash(const uint8_t* data, size_t len);
    static std::array<uint8_t, DIGEST_SIZE> keyed_hash(
        const uint8_t* data, size_t len, const uint8_t key[DIGEST_SIZE]);

    static std::vector<uint8_t> xof(const uint8_t* data, size_t len, size_t out_len);
    static std::vector<uint8_t> keyed_xof(
        const uint8_t* data, size_t len, const uint8_t key[DIGEST_SIZE], size_t out_len);

    // In-place XOF variants avoiding vector allocations on hot paths.
    static void xof_into(const uint8_t* data, size_t len, uint8_t* out, size_t out_len);
    static void keyed_xof_into(
        const uint8_t* data, size_t len, const uint8_t key[DIGEST_SIZE],
        uint8_t* out, size_t out_len);

    // Pearl-Merkle helpers: keyed BLAKE3 over 1024-byte leaves / 64-byte parents.
    static std::array<uint8_t, DIGEST_SIZE> chunk_cv(
        const uint8_t* chunk, size_t len, uint64_t chunk_index,
        const uint8_t key[DIGEST_SIZE]);

    static std::array<uint8_t, DIGEST_SIZE> parent_cv(
        const uint8_t left[DIGEST_SIZE], const uint8_t right[DIGEST_SIZE],
        const uint8_t key[DIGEST_SIZE]);

    static std::array<uint8_t, DIGEST_SIZE> root_cv(
        const uint8_t left[DIGEST_SIZE], const uint8_t right[DIGEST_SIZE],
        const uint8_t key[DIGEST_SIZE]);
};

} // namespace ref
} // namespace pearl
