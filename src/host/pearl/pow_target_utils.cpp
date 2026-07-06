#include "pow_target_utils.h"

#include <cmath>
#include <cctype>
#include <cstdlib>
#include <string>
#include <cstring>

namespace pearl {

namespace {

std::array<uint8_t, 32> diff1_target_be() {
    std::array<uint8_t, 32> t{};
    t[4] = 0xFF;
    t[5] = 0xFF;
    return t;
}

void be_mul_u32_inplace(std::array<uint8_t, 32>& be, uint32_t m) {
    uint64_t carry = 0;
    for (int i = 31; i >= 0; --i) {
        const uint64_t v = static_cast<uint64_t>(be[static_cast<size_t>(i)]) * m + carry;
        be[static_cast<size_t>(i)] = static_cast<uint8_t>(v & 0xFF);
        carry = v >> 8;
    }
    if (carry != 0) {
        be.fill(0xFF);
    }
}

std::array<uint8_t, 32> be_div_u64(std::array<uint8_t, 32> be, uint64_t d) {
    if (d == 0) return be;
    uint64_t rem = 0;
    for (int i = 0; i < 32; ++i) {
        rem = (rem << 8) | be[static_cast<size_t>(i)];
        be[static_cast<size_t>(i)] = static_cast<uint8_t>(rem / d);
        rem %= d;
    }
    return be;
}

}  // namespace

uint32_t target_be_to_nbits(const std::array<uint8_t, 32>& target_bytes) {
    int first_non_zero = -1;
    for (int i = 0; i < 32; ++i) {
        if (target_bytes[static_cast<size_t>(i)] != 0) {
            first_non_zero = i;
            break;
        }
    }
    if (first_non_zero < 0) return 0;

    const int len = 32 - first_non_zero;
    uint32_t mantissa = 0;
    if (len >= 3) {
        mantissa = (static_cast<uint32_t>(target_bytes[static_cast<size_t>(first_non_zero)]) << 16) |
                   (static_cast<uint32_t>(target_bytes[static_cast<size_t>(first_non_zero + 1)]) << 8) |
                   static_cast<uint32_t>(target_bytes[static_cast<size_t>(first_non_zero + 2)]);
    } else if (len == 2) {
        mantissa = (static_cast<uint32_t>(target_bytes[static_cast<size_t>(first_non_zero)]) << 16) |
                   (static_cast<uint32_t>(target_bytes[static_cast<size_t>(first_non_zero + 1)]) << 8);
    } else {
        mantissa = static_cast<uint32_t>(target_bytes[static_cast<size_t>(first_non_zero)]) << 16;
    }

    int out_len = len;
    if ((mantissa & 0x00800000u) != 0) {
        mantissa >>= 8;
        ++out_len;
    }
    return (static_cast<uint32_t>(out_len) << 24) | (mantissa & 0xFFFFFFu);
}

uint32_t difficulty_to_nbits_pdif(double difficulty) {
    if (!(difficulty > 0.0) || !std::isfinite(difficulty)) {
        difficulty = 1.0;
    }
    const uint64_t scaled_diff =
        std::max<uint64_t>(1ULL, static_cast<uint64_t>(difficulty * 1000000.0));
    auto be = diff1_target_be();
    be_mul_u32_inplace(be, 1000000U);
    be = be_div_u64(be, scaled_diff);
    return target_be_to_nbits(be);
}

uint32_t hex_target_to_nbits(const std::string& target_hex) {
    std::string h = target_hex;
    if (h.size() >= 2 && (h.rfind("0x", 0) == 0 || h.rfind("0X", 0) == 0)) {
        h = h.substr(2);
    }
    if (h.size() % 2) h = "0" + h;
    std::array<uint8_t, 32> bytes{};
    const size_t copy_len = std::min(h.size() / 2, size_t{32});
    const size_t offset = 32 - copy_len;
    for (size_t i = 0; i < copy_len; ++i) {
        const std::string byte_hex = h.substr(i * 2, 2);
        bytes[offset + i] =
            static_cast<uint8_t>(std::strtoul(byte_hex.c_str(), nullptr, 16));
    }
    return target_be_to_nbits(bytes);
}

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
