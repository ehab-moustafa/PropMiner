#pragma once

#include <cstdint>
#include <cstdio>

namespace pearl {

// Host + GPU resource snapshot for hashrate/status lines.
struct SystemSnapshot {
    bool valid = false;
    bool nvidia_smi = false;
    int gpu_util_pct = -1;
    int gpu_mem_util_pct = -1;
    unsigned long long vram_used_mb = 0;
    unsigned long long vram_total_mb = 0;
    int gpu_temp_c = -1;
    int gpu_power_w = -1;
    int gpu_power_limit_w = -1;
    int sm_clock_mhz = -1;
    int cpu_util_pct = -1;
    unsigned long long ram_used_mb = 0;
    unsigned long long ram_total_mb = 0;
    double pool_share_diff = 0.0;  // 0 = unknown / not connected yet
};

class SystemTelemetry {
public:
    // Sample CPU/RAM every call; GPU via nvidia-smi at most once per interval_ms.
    // Default 30s cache: GPU stats (temp, power, clocks) change slowly;
    // 4s was overkill and caused unnecessary popen() overhead.
    SystemSnapshot sample(int gpu_index = 0, int interval_ms = 30000);

private:
    bool read_cpu_util(int& out_pct);
    bool read_ram(unsigned long long& used_mb, unsigned long long& total_mb);
    bool read_gpu_nvidia_smi(int gpu_index, SystemSnapshot& out);
    bool read_vram_cuda(int gpu_index, unsigned long long& used_mb,
                        unsigned long long& total_mb);

    unsigned long long prev_cpu_idle_ = 0;
    unsigned long long prev_cpu_total_ = 0;
    bool cpu_initialized_ = false;

    int64_t last_gpu_sample_ms_ = 0;
    int cached_gpu_index_ = -1;
    SystemSnapshot cached_gpu_{};
};

void append_system_snapshot_to_line(FILE* out, const SystemSnapshot& sys);

}  // namespace pearl
