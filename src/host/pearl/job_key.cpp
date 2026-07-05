#include "job_key.h"

#include <cstring>
#include <vector>

#include "pearl_blake3.h"

#if defined(PROP_MINER_DISABLE_RUST_CRYPTO)
#include "../tests/ref_blake3.h"
#endif

namespace pearl {

std::array<uint8_t, 32> derive_job_key(const uint8_t* sigma, size_t sigma_len,
                                        const MiningConfig& cfg) {
    auto cfg_bytes = cfg.to_bytes();
    std::vector<uint8_t> input(sigma_len + cfg_bytes.size());
    std::memcpy(input.data(), sigma, sigma_len);
    std::memcpy(input.data() + sigma_len, cfg_bytes.data(), cfg_bytes.size());
#if defined(PROP_MINER_DISABLE_RUST_CRYPTO)
    return ref::Blake3Ref::hash(input.data(), input.size());
#else
    return Blake3Helper::hash(input.data(), input.size());
#endif
}

} // namespace pearl
