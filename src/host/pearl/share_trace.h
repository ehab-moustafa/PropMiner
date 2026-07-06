#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

namespace pearl {

// Enable with PROPMINER_VERBOSE_SHARES=1 (alias: PROPMINER_SHARE_TRACE=1).
// Logs every stage: GPU hit, proof rebuild, verify, build, submit, pool response.
inline bool share_trace_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("PROPMINER_VERBOSE_SHARES");
        if (!e) e = std::getenv("PROPMINER_SHARE_TRACE");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached != 0;
}

inline std::string hex_prefix(const uint8_t* data, size_t len, size_t nbytes = 8) {
    std::string hex;
    const size_t n = len < nbytes ? len : nbytes;
    hex.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        char buf[3];
        std::snprintf(buf, sizeof(buf), "%02x", data[i]);
        hex += buf;
    }
    return hex;
}

inline std::string nbits_hex(uint32_t nbits) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "0x%08x", nbits);
    return buf;
}

inline void share_trace(const char* stage, const std::string& detail) {
    if (!share_trace_enabled()) return;
    std::cout << "[share-trace] " << stage << ": " << detail << std::endl;
}

// Key pipeline events — always logged (found, verify fail, drop, submit, pool result).
inline void share_log(const char* stage, const std::string& detail) {
    std::cout << "[share] " << stage << ": " << detail << std::endl;
}

}  // namespace pearl
