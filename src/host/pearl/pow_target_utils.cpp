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

std::string nbits_to_target_hex_be(uint32_t nbits) {
    const auto target = nbits_to_target_le(nbits);
    std::string hex;
    hex.reserve(64);
    // Big-endian wire/display order (MSB byte first).
    for (int i = 31; i >= 0; --i) {
        char buf[3];
        std::snprintf(buf, sizeof(buf), "%02x",
                      static_cast<unsigned char>(target[static_cast<size_t>(i)]));
        hex += buf;
    }
    return hex;
}

uint32_t network_nbits_from_sigma(const std::array<uint8_t, kSigmaHeaderBytes>& sigma) {
    uint32_t nbits = 0;
    std::memcpy(&nbits, sigma.data() + 72, sizeof(nbits));
    return nbits;
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
    const int exp = static_cast<int>(nbits >> 24);
    const uint32_t mant = nbits & 0xFFFFFFu;
    std::array<uint8_t, 32> t{};
    if (exp <= 3) {
        const uint64_t v = static_cast<uint64_t>(mant >> (8 * (3 - exp)));
        std::memcpy(t.data(), &v, sizeof(v));
        return t;
    }
    // ARC/Bitcoin compact target: mantissa * 256^(exp-3), stored as uint256 LE.
    // The legacy placement at bytes (32-exp..) broke GPU uint32 MSW-first
    // comparisons and made share targets impossibly hard (gpu_hits_30s=0).
#if defined(__SIZEOF_INT128__)
    unsigned __int128 val = static_cast<unsigned __int128>(mant);
    val <<= static_cast<unsigned>(8 * (exp - 3));
    for (int i = 0; i < 32; ++i) {
        t[static_cast<size_t>(i)] =
            static_cast<uint8_t>(static_cast<uint64_t>(val >> (8 * i)) & 0xFF);
    }
#else
    uint64_t lo = mant;
    uint64_t hi = 0;
    const int shift_bytes = exp - 3;
    if (shift_bytes >= 8) {
        hi = lo << (8 * (shift_bytes - 8));
        lo = 0;
    } else if (shift_bytes > 0) {
        lo <<= (8 * shift_bytes);
    }
    std::memcpy(t.data(), &lo, sizeof(lo));
    std::memcpy(t.data() + 8, &hi, sizeof(hi));
#endif
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
