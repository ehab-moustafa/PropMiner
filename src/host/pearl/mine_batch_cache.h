#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

namespace pearl {

struct MineBatchResult {
    int m = 0;
    int n = 0;
    int batch = 4;
    double hashrate = 0.0;
    bool use_graph = true;
};

// Per-GPU mine batch recommendations from offline sweeps at production M/N.
class MineBatchCache {
public:
    MineBatchCache();

    std::optional<MineBatchResult> load(int device_index, int m, int n) const;
    void save(int device_index, const MineBatchResult& result);
    void clear();

    static std::string cache_path();

    // PROPMINER_BATCH env > cache (matching M,N) > Rtx5090Profile::kDefaultMineBatch.
    static int resolve(int device_index, int m, int n, int fallback);

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
