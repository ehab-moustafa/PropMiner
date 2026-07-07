#include "system_telemetry.h"

#include "cuda_compat.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

namespace pearl {

namespace {

int64_t steady_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

unsigned long long parse_ull(const char* s) {
    if (!s || !s[0]) return 0;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(s, &end, 10);
    return (end && end != s) ? v : 0;
}

void format_gibibytes(unsigned long long used_mb, unsigned long long total_mb,
                      char* buf, size_t buflen) {
    if (total_mb == 0) {
        std::snprintf(buf, buflen, "n/a");
        return;
    }
    const double used_gib = static_cast<double>(used_mb) / 1024.0;
    const double total_gib = static_cast<double>(total_mb) / 1024.0;
    const int vram_pct =
        static_cast<int>((used_mb * 100ULL) / std::max(1ULL, total_mb));
    std::snprintf(buf, buflen, "%.1f/%.1fGiB(%d%%)", used_gib, total_gib,
                  vram_pct);
}

const char* resolve_nvidia_smi_binary() {
    if (const char* env = std::getenv("PROPMINER_NVIDIA_SMI"); env && env[0]) {
        return env;
    }
    static const char* kCandidates[] = {
        "/usr/bin/nvidia-smi",
        "/usr/local/nvidia/bin/nvidia-smi",
        "/usr/lib/wsl/lib/nvidia-smi",
        "nvidia-smi",
    };
    for (const char* cand : kCandidates) {
        if (std::strchr(cand, '/') != nullptr) {
            if (access(cand, X_OK) == 0) return cand;
        } else {
            return cand;
        }
    }
    return nullptr;
}

void warn_nvidia_smi_once() {
    static bool warned = false;
    if (warned) return;
    warned = true;
    std::fprintf(stderr,
                 "[telemetry] WARN: nvidia-smi unavailable — gpu%%, mem%%, "
                 "temp, power, clocks omitted from hashrate lines "
                 "(set PROPMINER_NVIDIA_SMI or install host nvidia-utils)\n");
}

}  // namespace

bool SystemTelemetry::read_cpu_util(int& out_pct) {
    std::ifstream stat("/proc/stat");
    if (!stat) return false;
    std::string line;
    if (!std::getline(stat, line) || line.rfind("cpu ", 0) != 0) return false;

    unsigned long long user = 0, nice = 0, system = 0, idle = 0;
    unsigned long long iowait = 0, irq = 0, softirq = 0, steal = 0;
    std::istringstream iss(line.substr(4));
    iss >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
    const unsigned long long idle_all = idle + iowait;
    const unsigned long long total =
        user + nice + system + idle_all + irq + softirq + steal;
    if (total <= idle_all) return false;

    if (!cpu_initialized_) {
        prev_cpu_idle_ = idle_all;
        prev_cpu_total_ = total;
        cpu_initialized_ = true;
        out_pct = 0;
        return true;
    }

    const unsigned long long total_delta = total - prev_cpu_total_;
    const unsigned long long idle_delta = idle_all - prev_cpu_idle_;
    prev_cpu_idle_ = idle_all;
    prev_cpu_total_ = total;
    if (total_delta == 0) {
        out_pct = 0;
        return true;
    }
    const unsigned long long used_delta = total_delta - idle_delta;
    out_pct = static_cast<int>((used_delta * 100ULL) / total_delta);
    if (out_pct < 0) out_pct = 0;
    if (out_pct > 100) out_pct = 100;
    return true;
}

bool SystemTelemetry::read_ram(unsigned long long& used_mb, unsigned long long& total_mb) {
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo) return false;
    unsigned long long mem_total_kb = 0;
    unsigned long long mem_avail_kb = 0;
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.rfind("MemTotal:", 0) == 0) {
            mem_total_kb = parse_ull(line.c_str() + 9);
        } else if (line.rfind("MemAvailable:", 0) == 0) {
            mem_avail_kb = parse_ull(line.c_str() + 13);
        }
    }
    if (mem_total_kb == 0) return false;
    total_mb = mem_total_kb / 1024;
    const unsigned long long used_kb =
        mem_total_kb > mem_avail_kb ? mem_total_kb - mem_avail_kb : 0;
    used_mb = used_kb / 1024;
    return true;
}

bool SystemTelemetry::read_vram_cuda(int gpu_index, unsigned long long& used_mb,
                                     unsigned long long& total_mb) {
    if (cudaSetDevice(gpu_index) != cudaSuccess) return false;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return false;
    total_mb = total_bytes / (1024ULL * 1024ULL);
    used_mb = (total_bytes > free_bytes)
                  ? (total_bytes - free_bytes) / (1024ULL * 1024ULL)
                  : 0;
    return total_mb > 0;
}

bool SystemTelemetry::read_gpu_nvidia_smi(int gpu_index, SystemSnapshot& out) {
    const char* smi = resolve_nvidia_smi_binary();
    if (!smi) {
        warn_nvidia_smi_once();
        return false;
    }

    char cmd[512];
    std::snprintf(cmd, sizeof(cmd),
                  "%s -i %d "
                  "--query-gpu=utilization.gpu,utilization.memory,memory.used,"
                  "memory.total,temperature.gpu,power.draw,power.limit,"
                  "clocks.current.sm "
                  "--format=csv,noheader,nounits 2>/dev/null",
                  smi, gpu_index);
    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        warn_nvidia_smi_once();
        return false;
    }
    char buf[384] = {};
    if (!std::fgets(buf, sizeof(buf), pipe)) {
        pclose(pipe);
        warn_nvidia_smi_once();
        return false;
    }
    pclose(pipe);

    int util = -1;
    int mem_util = -1;
    unsigned long long mem_used = 0;
    unsigned long long mem_total = 0;
    int temp = -1;
    double power = -1.0;
    double power_limit = -1.0;
    int sm_clock = -1;
    const int n = std::sscanf(buf, "%d,%d,%llu,%llu,%d,%lf,%lf,%d", &util,
                              &mem_util, &mem_used, &mem_total, &temp, &power,
                              &power_limit, &sm_clock);
    if (n < 2) {
        warn_nvidia_smi_once();
        return false;
    }
    out.nvidia_smi = true;
    if (util >= 0) out.gpu_util_pct = util;
    if (mem_util >= 0) out.gpu_mem_util_pct = mem_util;
    if (mem_used > 0) out.vram_used_mb = mem_used;
    if (mem_total > 0) out.vram_total_mb = mem_total;
    if (n >= 5 && temp >= 0) out.gpu_temp_c = temp;
    if (n >= 6 && power >= 0.0) out.gpu_power_w = static_cast<int>(power + 0.5);
    if (n >= 7 && power_limit >= 0.0) {
        out.gpu_power_limit_w = static_cast<int>(power_limit + 0.5);
    }
    if (n >= 8 && sm_clock >= 0) out.sm_clock_mhz = sm_clock;
    return true;
}

SystemSnapshot SystemTelemetry::sample(int gpu_index, int interval_ms) {
    SystemSnapshot snap{};
    int cpu_pct = -1;
    if (read_cpu_util(cpu_pct)) {
        snap.cpu_util_pct = cpu_pct;
    }
    read_ram(snap.ram_used_mb, snap.ram_total_mb);

    const int64_t now = steady_ms();
    if (cached_gpu_index_ == gpu_index && (now - last_gpu_sample_ms_) < interval_ms &&
        (cached_gpu_.nvidia_smi || cached_gpu_.vram_total_mb > 0)) {
        snap.nvidia_smi = cached_gpu_.nvidia_smi;
        snap.gpu_util_pct = cached_gpu_.gpu_util_pct;
        snap.gpu_mem_util_pct = cached_gpu_.gpu_mem_util_pct;
        snap.vram_used_mb = cached_gpu_.vram_used_mb;
        snap.vram_total_mb = cached_gpu_.vram_total_mb;
        snap.gpu_temp_c = cached_gpu_.gpu_temp_c;
        snap.gpu_power_w = cached_gpu_.gpu_power_w;
        snap.gpu_power_limit_w = cached_gpu_.gpu_power_limit_w;
        snap.sm_clock_mhz = cached_gpu_.sm_clock_mhz;
    } else {
        SystemSnapshot gpu{};
        if (!read_gpu_nvidia_smi(gpu_index, gpu)) {
            read_vram_cuda(gpu_index, gpu.vram_used_mb, gpu.vram_total_mb);
        } else if (gpu.vram_total_mb == 0) {
            read_vram_cuda(gpu_index, gpu.vram_used_mb, gpu.vram_total_mb);
        }
        cached_gpu_ = gpu;
        cached_gpu_index_ = gpu_index;
        last_gpu_sample_ms_ = now;
        snap.nvidia_smi = gpu.nvidia_smi;
        snap.gpu_util_pct = gpu.gpu_util_pct;
        snap.gpu_mem_util_pct = gpu.gpu_mem_util_pct;
        snap.vram_used_mb = gpu.vram_used_mb;
        snap.vram_total_mb = gpu.vram_total_mb;
        snap.gpu_temp_c = gpu.gpu_temp_c;
        snap.gpu_power_w = gpu.gpu_power_w;
        snap.gpu_power_limit_w = gpu.gpu_power_limit_w;
        snap.sm_clock_mhz = gpu.sm_clock_mhz;
    }

    snap.valid = snap.ram_total_mb > 0 || snap.vram_total_mb > 0 ||
                 snap.gpu_util_pct >= 0 || snap.nvidia_smi;
    return snap;
}

void append_system_snapshot_to_line(FILE* out, const SystemSnapshot& sys) {
    if (!sys.valid) return;

    if (sys.gpu_util_pct >= 0) {
        if (sys.gpu_mem_util_pct >= 0) {
            std::fprintf(out, " | gpu=%d%%/mem=%d%%", sys.gpu_util_pct,
                         sys.gpu_mem_util_pct);
        } else {
            std::fprintf(out, " | gpu=%d%%", sys.gpu_util_pct);
        }
    }
    if (sys.vram_total_mb > 0) {
        char vram_buf[56];
        format_gibibytes(sys.vram_used_mb, sys.vram_total_mb, vram_buf,
                         sizeof(vram_buf));
        std::fprintf(out, " | vram=%s", vram_buf);
    }
    if (sys.cpu_util_pct >= 0) {
        std::fprintf(out, " | cpu=%d%%", sys.cpu_util_pct);
    }
    if (sys.ram_total_mb > 0) {
        char ram_buf[48];
        format_gibibytes(sys.ram_used_mb, sys.ram_total_mb, ram_buf,
                         sizeof(ram_buf));
        std::fprintf(out, " | ram=%s", ram_buf);
    }
    if (sys.gpu_temp_c >= 0) {
        std::fprintf(out, " | gpu_temp=%dC", sys.gpu_temp_c);
    }
    if (sys.gpu_power_w >= 0) {
        if (sys.gpu_power_limit_w > 0) {
            std::fprintf(out, " | power=%d/%dW", sys.gpu_power_w,
                         sys.gpu_power_limit_w);
        } else {
            std::fprintf(out, " | power=%dW", sys.gpu_power_w);
        }
    }
    if (sys.sm_clock_mhz >= 0) {
        std::fprintf(out, " | sm_clock=%dMHz", sys.sm_clock_mhz);
    }
    if (sys.pool_share_diff > 0.0) {
        std::fprintf(out, " | pool_diff=%.0f", sys.pool_share_diff);
    }
}

}  // namespace pearl
