#include "kernel_knob_cache.h"

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

KernelKnobCache::KernelKnobCache() = default;

std::string KernelKnobCache::cache_path() {
    std::filesystem::create_directories(cache_dir());
    return cache_dir() + "/kernel_knobs.json";
}

KernelKnobCache::Key KernelKnobCache::make_key(int device_index) const {
    Key k;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device_index);
    k.major = prop.major;
    k.minor = prop.minor;
    std::memcpy(k.uuid.data(), prop.uuid.bytes, 16);
    k.driver_version = driver_version();
    k.arch_profile = get_env("PEARL_GEMM_ARCH");
    if (k.arch_profile.empty()) k.arch_profile = "blackwell";
    return k;
}

std::string KernelKnobCache::Key::to_string() const {
    std::ostringstream oss;
    for (auto c : uuid) oss << std::hex << (static_cast<int>(c) & 0xff);
    oss << "-" << major << "." << minor;
    oss << "-drv" << driver_version;
    oss << "-" << arch_profile;
    return oss.str();
}

std::optional<KernelKnobResult> KernelKnobCache::load(int device_index) const {
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

        KernelKnobResult r;
        int self_test_int = 0;
        char manifest_buf[128] = {};
        char policy_buf[32] = {};
        if (std::sscanf(line.c_str() + sep + 1,
                        "%d,%d,%d,%d,%31[^,],%127[^,],%lf,%d",
                        &r.kblock, &r.stages, &r.swizzle_bits, &r.min_blocks,
                        policy_buf, manifest_buf, &r.hashrate, &self_test_int) == 8) {
            r.load_policy = policy_buf;
            r.manifest = manifest_buf;
            r.self_test_ok = (self_test_int != 0);
            return r;
        }
        return std::nullopt;
    }
    return std::nullopt;
}

void KernelKnobCache::clear() {
    std::lock_guard<std::mutex> lk(mtx_);
    std::ofstream f(cache_path(), std::ios::trunc);
}

} // namespace pearl
