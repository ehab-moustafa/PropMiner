#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

#include "cuda_compat.h"
#include "gpu_tuner.h"

namespace pearl {

// Persistent per-SKU autotune cache.
// Key = GPU UUID + compute capability + driver version + CUDA arch profile.
// Value = the TuningResult produced by GpuTuner::tune().
// The cache is written to a small JSON file in the user's cache directory so
// benchmark results survive miner restarts and pool reconnects.
class TuneCache {
public:
    TuneCache();

    // Look up a cached result for the currently selected GPU.  Returns nullopt
    // if no entry exists or if the cached entry is incompatible (e.g. built for
    // a different PEARL_GEMM_ARCH).
    std::optional<TuningResult> load(int device_index) const;

    // Store a result for the currently selected GPU.
    void save(int device_index, const TuningResult& result);

    // Invalidate all cached entries (e.g. after a major miner update).
    void clear();

    // Path to the cache file.
    static std::string cache_path();

private:
    struct Key {
        std::array<char, 16> uuid{};
        int major = 0;
        int minor = 0;
        int driver_version = 0;
        std::string arch_profile;

        std::string to_string() const;
        static std::optional<Key> from_string(const std::string& s);
    };

    Key make_key(int device_index) const;
    mutable std::mutex mtx_;
};

} // namespace pearl
