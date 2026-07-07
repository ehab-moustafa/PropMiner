#include "mine_batch_cache.h"

#include "env_tuning.h"
#include "rtx5090_profile.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <cuda_runtime.h>

namespace pearl {

namespace {
    std::string cache_dir() {
        const char* xdg = std::getenv("XDG_CACHE_HOME");
        if (xdg && *xdg) return std::string(xdg) + "/propminer";
        const char* home = std::getenv("HOME");
        if (home && *home) return std::string(home) + "/.cache/propminer";
        return ".propminer_cache";
    }

    int driver_version() {
        int v = 0;
        cudaRuntimeGetVersion(&v);
        return v;
    }

    std::string get_env(const char* name) {
        const char* v = std::getenv(name);
        return v ? v : "";
    }
}

MineBatchCache::MineBatchCache() = default;

std::string MineBatchCache::cache_path() {
    std::filesystem::create_directories(cache_dir());
    return cache_dir() + "/mine_batch.json";
}

MineBatchCache::Key MineBatchCache::make_key(int device_index) const {
    Key k;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device_index);
    k.major = prop.major;
    k.minor = prop.minor;
    std::memcpy(k.uuid.data(), prop.uuid.bytes, 16);
    k.driver_version = driver_version();
    k.arch_profile = get_env("PEARL_GEMM_ARCH");
    if (k.arch_profile.empty()) k.arch_profile = "auto";
    return k;
}

std::string MineBatchCache::Key::to_string() const {
    std::ostringstream oss;
    for (auto c : uuid) oss << std::hex << (static_cast<int>(c) & 0xff);
    oss << "-" << major << "." << minor;
    oss << "-drv" << driver_version;
    oss << "-" << arch_profile;
    return oss.str();
}

std::optional<MineBatchResult> MineBatchCache::load(int device_index,
                                                    int m, int n) const {
    std::lock_guard<std::mutex> lk(mtx_);
    std::ifstream f(cache_path());
    if (!f) return std::nullopt;

    const Key want = make_key(device_index);
    const std::string want_str = want.to_string();
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        const size_t sep = line.find(' ');
        if (sep == std::string::npos) continue;
        if (line.substr(0, sep) != want_str) continue;

        MineBatchResult r;
        int graph_int = 1;
        if (std::sscanf(line.c_str() + sep + 1,
                        "%d,%d,%d,%lf,%d",
                        &r.m, &r.n, &r.batch, &r.hashrate, &graph_int) == 5) {
            if (r.m != m || r.n != n) return std::nullopt;
            r.use_graph = (graph_int != 0);
            if (r.batch < 1) return std::nullopt;
            return r;
        }
        return std::nullopt;
    }
    return std::nullopt;
}

void MineBatchCache::save(int device_index, const MineBatchResult& result) {
    std::lock_guard<std::mutex> lk(mtx_);
    const Key key = make_key(device_index);
    const std::string key_str = key.to_string();

    std::vector<std::string> lines;
    std::ifstream in(cache_path());
    if (in) {
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') {
                lines.push_back(line);
                continue;
            }
            const size_t sep = line.find(' ');
            if (sep != std::string::npos && line.substr(0, sep) == key_str) {
                continue;
            }
            lines.push_back(line);
        }
    }

    std::ofstream out(cache_path());
    for (const auto& line : lines) {
        if (!line.empty()) out << line << '\n';
    }
    out << key_str << ' '
        << result.m << ',' << result.n << ',' << result.batch << ','
        << result.hashrate << ',' << (result.use_graph ? 1 : 0) << '\n';
}

void MineBatchCache::clear() {
    std::lock_guard<std::mutex> lk(mtx_);
    std::ofstream f(cache_path(), std::ios::trunc);
}

int MineBatchCache::resolve(int device_index, int m, int n, int fallback) {
    if (mine_batch_env_set()) {
        return resolve_mine_batch();
    }
    MineBatchCache cache;
    if (auto cached = cache.load(device_index, m, n)) {
        return cached->batch;
    }
    return std::max(1, fallback);
}

} // namespace pearl
