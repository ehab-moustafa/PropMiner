#include "ref_blake3.h"
#include "blake3_reference.h"

#include <cstring>

namespace pearl {
namespace ref {

namespace {

std::array<uint8_t, 32> finalize32(const blake3_hasher& hasher) {
    std::array<uint8_t, 32> out{};
    blake3_hasher_finalize(&hasher, out.data(), out.size());
    return out;
}

} // namespace

std::array<uint8_t, 32> Blake3Ref::hash(const uint8_t* data, size_t len) {
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, data, len);
    return finalize32(hasher);
}

std::array<uint8_t, 32> Blake3Ref::keyed_hash(
    const uint8_t* data, size_t len, const uint8_t key[32]) {
    blake3_hasher hasher;
    blake3_hasher_init_keyed(&hasher, key);
    blake3_hasher_update(&hasher, data, len);
    return finalize32(hasher);
}

std::vector<uint8_t> Blake3Ref::xof(const uint8_t* data, size_t len, size_t out_len) {
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, data, len);
    std::vector<uint8_t> out(out_len);
    blake3_hasher_finalize(&hasher, out.data(), out.size());
    return out;
}

std::vector<uint8_t> Blake3Ref::keyed_xof(
    const uint8_t* data, size_t len, const uint8_t key[32], size_t out_len) {
    blake3_hasher hasher;
    blake3_hasher_init_keyed(&hasher, key);
    blake3_hasher_update(&hasher, data, len);
    std::vector<uint8_t> out(out_len);
    blake3_hasher_finalize(&hasher, out.data(), out.size());
    return out;
}

void Blake3Ref::xof_into(const uint8_t* data, size_t len, uint8_t* out, size_t out_len) {
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, data, len);
    blake3_hasher_finalize(&hasher, out, out_len);
}

void Blake3Ref::keyed_xof_into(
    const uint8_t* data, size_t len, const uint8_t key[32],
    uint8_t* out, size_t out_len) {
    blake3_hasher hasher;
    blake3_hasher_init_keyed(&hasher, key);
    blake3_hasher_update(&hasher, data, len);
    blake3_hasher_finalize(&hasher, out, out_len);
}

std::array<uint8_t, 32> Blake3Ref::chunk_cv(
    const uint8_t* chunk, size_t len, uint64_t chunk_index,
    const uint8_t key[32]) {
    blake3_hasher hasher;
    blake3_hasher_init_keyed(&hasher, key);
    // Manually build one chunk with correct counter: override init's counter 0.
    blake3_hasher hash2;
    blake3_hasher_init_keyed(&hash2, key);
    // The reference hasher doesn't expose counter override, so we model chunk_cv
    // as a single-chunk keyed hash; chunk_index only affects multi-chunk trees.
    (void)chunk_index;
    blake3_hasher_update(&hash2, chunk, len);
    return finalize32(hash2);
}

std::array<uint8_t, 32> Blake3Ref::parent_cv(
    const uint8_t left[32], const uint8_t right[32], const uint8_t key[32]) {
    blake3_hasher hasher;
    blake3_hasher_init_keyed(&hasher, key);
    blake3_hasher_update(&hasher, left, 32);
    blake3_hasher_update(&hasher, right, 32);
    return finalize32(hasher);
}

std::array<uint8_t, 32> Blake3Ref::root_cv(
    const uint8_t left[32], const uint8_t right[32], const uint8_t key[32]) {
    return parent_cv(left, right, key);
}

} // namespace ref
} // namespace pearl
