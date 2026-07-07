#include "tune_cache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <cuda_runtime.h>

namespace pearl {

namespace {
    std::string get_env(const char* name) {
        const char* v = std::getenv(name);
        return v ? v : "";
    }

    std::string cache_dir() {
        std::string dir;
        const char* xdg = std::getenv("XDG_CACHE_HOME");
        if (xdg && *xdg) {
            dir = std::string(xdg) + "/propminer";
        } else {
            const char* home = std::getenv("HOME");
            if (home && *home) {
                dir = std::string(home) + "/.cache/propminer";
            } else {
                dir = ".propminer_cache";
            }
        }
        std::filesystem::create_directories(dir);
        return dir;
    }

int driver_version() {
    int v = 0;
#if defined(__has_include)
#  if __has_include(<cuda.h>)
    cuDriverGetVersion(&v);
#  elif __has_include(<cuda_runtime.h>)
    cudaRuntimeGetVersion(&v);
#  endif
#endif
    return v;
}
}

std::string TuneCache::cache_path() {
    return cache_dir() + "/autotune.json";
}

TuneCache::TuneCache() = default;

TuneCache::Key TuneCache::make_key(int device_index) const {
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

std::string TuneCache::Key::to_string() const {
    std::ostringstream oss;
    for (auto c : uuid) oss << std::hex << (static_cast<int>(c) & 0xff);
    oss << "-" << major << "." << minor;
    oss << "-drv" << driver_version;
    oss << "-" << arch_profile;
    return oss.str();
}

std::optional<TuneCache::Key> TuneCache::Key::from_string(const std::string& s) {
    // Expected: <32 hex chars>-<major>.<minor>-drv<N>-<arch>
    Key k;
    if (s.size() < 38) return std::nullopt;
    if (s[32] != '-') return std::nullopt;
    for (int i = 0; i < 16; ++i) {
        int b = 0;
        if (std::sscanf(s.c_str() + i * 2, "%2x", &b) != 1) return std::nullopt;
        k.uuid[i] = static_cast<char>(b);
    }
    if (std::sscanf(s.c_str() + 33, "%d.%d", &k.major, &k.minor) != 2)
        return std::nullopt;
    size_t drv = s.find("-drv");
    if (drv == std::string::npos) return std::nullopt;
    if (std::sscanf(s.c_str() + drv + 4, "%d", &k.driver_version) != 1)
        return std::nullopt;
    size_t arch = s.find('-', drv + 4);
    if (arch == std::string::npos || arch + 1 >= s.size()) return std::nullopt;
    k.arch_profile = s.substr(arch + 1);
    return k;
}

std::optional<TuningResult> TuneCache::load(int device_index) const {
    std::lock_guard<std::mutex> lk(mtx_);
    std::ifstream f(cache_path());
    if (!f) return std::nullopt;

    std::string line;
    Key want = make_key(device_index);
    std::string want_str = want.to_string();
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t sep = line.find(' ');
        if (sep == std::string::npos) continue;
        std::string key_str = line.substr(0, sep);
        if (key_str != want_str) continue;
        std::string val = line.substr(sep + 1);
        TuningResult r;
        int use_graph_int = 1;
        int graph_batch = 0;
        const int n = std::sscanf(val.c_str(),
                                  "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%lf",
                                  &r.config.m, &r.config.n, &r.config.k, &r.config.r,
                                  &r.config.bM, &r.config.bN, &r.config.bK,
                                  &r.batch_size,
                                  &graph_batch,
                                  &use_graph_int,
                                  &r.cluster_m,
                                  &r.carveout_percent,
                                  &r.hashrate);
        if (n == 14) {
            r.use_graph = (use_graph_int != 0);
            r.graph_batch_size =
                graph_batch > 0 ? graph_batch : r.batch_size;
            return r;
        }
        if (n == 12) {
            r.use_graph = (use_graph_int != 0);
            r.graph_batch_size = r.use_graph ? r.batch_size : 1;
            return r;
        }
        return std::nullopt;
    }
    return std::nullopt;
}

void TuneCache::save(int device_index, const TuningResult& result) {
    std::lock_guard<std::mutex> lk(mtx_);
    std::vector<std::string> lines;
    {
        std::ifstream f(cache_path());
        std::string line;
        Key want = make_key(device_index);
        std::string want_str = want.to_string();
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') {
                lines.push_back(line);
                continue;
            }
            size_t sep = line.find(' ');
            if (sep == std::string::npos || line.substr(0, sep) != want_str) {
                lines.push_back(line);
            }
        }
    }

    std::ofstream f(cache_path(), std::ios::trunc);
    if (!f) return;
    for (const auto& l : lines) f << l << '\n';
    Key k = make_key(device_index);
    f << k.to_string() << " "
      << result.config.m << "," << result.config.n << "," << result.config.k << ","
      << result.config.r << ","
      << result.config.bM << "," << result.config.bN << "," << result.config.bK << ","
      << result.batch_size << ","
      << (result.graph_batch_size > 0 ? result.graph_batch_size : result.batch_size)
      << ","
      << (result.use_graph ? 1 : 0) << ","
      << result.cluster_m << ","
      << result.carveout_percent << ","
      << result.hashrate << '\n';
}

void TuneCache::clear() {
    std::lock_guard<std::mutex> lk(mtx_);
    std::ofstream f(cache_path(), std::ios::trunc);
}

} // namespace pearl
