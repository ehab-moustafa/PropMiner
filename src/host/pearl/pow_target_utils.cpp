#include "pow_target_utils.h"

#include <cstring>

namespace pearl {

std::array<uint8_t, 32> nbits_to_target_le(uint32_t nbits) {
    int exp = static_cast<int>(nbits >> 24);
    uint32_t mant = nbits & 0xFFFFFFu;
    std::array<uint8_t, 32> t{};
    for (int i = 0; i < 3; ++i) {
        int pos = 32 - exp + i;
        if (pos >= 0 && pos < 32) {
            t[static_cast<size_t>(pos)] =
                static_cast<uint8_t>(mant >> (8 * (2 - i)));
        }
    }
    return t;
}

std::array<uint8_t, 32> multiply_target_le_by_u64(const std::array<uint8_t, 32>& target_le,
                                                  uint64_t factor) {
    if (factor <= 1) {
        return target_le;
    }
    std::array<uint8_t, 32> out{};
    unsigned __int128 carry = 0;
    for (int i = 0; i < 32; ++i) {
        unsigned __int128 prod =
            static_cast<unsigned __int128>(target_le[static_cast<size_t>(i)]) * factor + carry;
        out[static_cast<size_t>(i)] = static_cast<uint8_t>(prod & 0xFF);
        carry = prod >> 8;
    }
    if (carry != 0) {
        out.fill(0xFF);
    }
    return out;
}

std::array<uint32_t, 8> target_le_to_pow_u32(const std::array<uint8_t, 32>& target_le) {
    std::array<uint32_t, 8> words{};
    for (int i = 0; i < 8; ++i) {
        std::memcpy(&words[static_cast<size_t>(i)],
                    target_le.data() + i * 4, 4);
    }
    return words;
}

bool claimed_hash_clears_target(const uint8_t claimed_hash[32],
                                uint32_t nbits,
                                uint64_t daf) {
    auto target = adjusted_pow_target_le(nbits, daf);
    for (int i = 31; i >= 0; --i) {
        const uint8_t h = claimed_hash[i];
        const uint8_t t = target[static_cast<size_t>(i)];
        if (h < t) return true;
        if (h > t) return false;
    }
    return true;
}

}  // namespace pearl
