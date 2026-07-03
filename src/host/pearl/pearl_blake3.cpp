#include "pearl_blake3.h"

#include <stdexcept>

#include "pearl_mining_wrapper.h"

namespace pearl {

#if !defined(PROP_MINER_DISABLE_RUST_CRYPTO)
std::array<uint8_t, 32> Blake3Helper::keyed(const uint8_t* data, size_t len,
                                            const uint8_t key[32]) {
    MiningCapi capi;
    std::array<uint8_t, 32> out{};
    capi.blake3_keyed(data, len, key, out.data());
    return out;
}

std::array<uint8_t, 32> Blake3Helper::hash(const uint8_t* data, size_t len) {
    MiningCapi capi;
    std::array<uint8_t, 32> out{};
    capi.blake3_hash(data, len, out.data());
    return out;
}

std::vector<uint8_t> Blake3Helper::xof(const uint8_t* data, size_t len, size_t out_len) {
    MiningCapi capi;
    std::vector<uint8_t> out(out_len);
    capi.blake3_xof(data, len, out.data(), out_len);
    return out;
}
#else
std::array<uint8_t, 32> Blake3Helper::keyed(const uint8_t*, size_t,
                                            const uint8_t[32]) {
    throw std::runtime_error("Blake3Helper::keyed unavailable without Rust crypto");
}

std::array<uint8_t, 32> Blake3Helper::hash(const uint8_t*, size_t) {
    throw std::runtime_error("Blake3Helper::hash unavailable without Rust crypto");
}

std::vector<uint8_t> Blake3Helper::xof(const uint8_t*, size_t, size_t) {
    throw std::runtime_error("Blake3Helper::xof unavailable without Rust crypto");
}
#endif

} // namespace pearl
