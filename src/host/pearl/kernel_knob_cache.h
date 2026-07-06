#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>

namespace pearl {

struct KernelKnobResult {
    int kblock = 128;
    int stages = 2;
    int swizzle_bits = 3;
    int min_blocks = 1;
    std::string load_policy = "cp_async";
    std::string manifest;
    double hashrate = 0.0;
    bool self_test_ok = false;
};

// Persistent per-GPU kernel knob recommendations from offline sweeps.
// Written by scripts/tune_blackwell_knobs.sh; read-only at miner startup.
class KernelKnobCache {
public:
    KernelKnobCache();

    std::optional<KernelKnobResult> load(int device_index) const;
    void clear();

    static std::string cache_path();
    static bool strict_cache_enabled() {
        const char* v = std::getenv("PROPMINER_STRICT_KNOB_CACHE");
        return v && v[0] == '1';
    }
    static std::optional<std::string> strict_validate(const char* built_knobs,
                                                      const KernelKnobResult& cached) {
        if (!strict_cache_enabled()) return std::nullopt;
        if (!built_knobs || !*built_knobs || std::strcmp(built_knobs, "unknown") == 0) {
            return std::string("built kernel knobs unknown; rebuild libpearl_gemm_capi.so");
        }
        if (cached.manifest.empty()) {
            return std::string("cached manifest empty in kernel_knobs.json");
        }
        if (!manifest_matches(built_knobs, cached.manifest.c_str())) {
            std::ostringstream oss;
            oss << "built kernel knobs (" << built_knobs
                << ") != cached manifest (" << cached.manifest << ")";
            return oss.str();
        }
        if (!cached.self_test_ok) {
            return std::string("cached knob entry has self_test_ok=0; "
                               "re-run ./scripts/tune_blackwell_knobs.sh");
        }
        return std::nullopt;
    }
    static bool manifest_matches(const char* built, const char* cached) {
        if (!built || !*built) return true;
        if (!cached || !*cached) return true;
        return std::strcmp(built, cached) == 0;
    }

private:
    struct Key {
        std::array<char, 16> uuid{};
        int major = 0;
        int minor = 0;
        int driver_version = 0;
        std::string arch_profile;

        std::string to_string() const;
    };

    Key make_key(int device_index) const;
    mutable std::mutex mtx_;
};

} // namespace pearl
