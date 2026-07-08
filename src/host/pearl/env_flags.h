#pragma once

#include <cstdlib>
#include <cstring>

namespace pearl {

// True when env is set to a non-empty value other than "0".
inline bool env_truthy(const char* name) {
    const char* env = std::getenv(name);
    return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

inline bool bench_no_graph_enabled() {
    return env_truthy("PROPMINER_BENCH_NO_GRAPH");
}

// Max wall time a single GPU batch may take before we treat the worker as
// wedged (GPU pinned at 100% but not making progress — a hung CUDA stream that
// in-process recovery cannot clear). On stall we exit so the supervisor
// (PROPMINER_RESTART_ON_EXIT) relaunches with a fresh context. 0 disables.
inline long long stall_restart_ms() {
    const char* env = std::getenv("PROPMINER_STALL_RESTART_MS");
    if (env && env[0] != '\0') {
        char* end = nullptr;
        long long v = std::strtoll(env, &end, 10);
        if (end != env && v >= 0) return v;
    }
    return 30000;  // 30s: a healthy batch is well under 1s.
}

// PeakMiner/BzMiner-style thermal pause (0 = disabled).
// PROPMINER_GPU_TEMP_STOP: pause mining when GPU temp (C) reaches this value.
// PROPMINER_GPU_TEMP_START: resume when temp drops to or below this (default:
//   stop-10 when only stop is set).
inline int gpu_temp_stop_c() {
    const char* env = std::getenv("PROPMINER_GPU_TEMP_STOP");
    if (!env || !env[0]) return 0;
    const int v = std::atoi(env);
    return v > 0 ? v : 0;
}

inline int gpu_temp_start_c() {
    const char* env = std::getenv("PROPMINER_GPU_TEMP_START");
    const int stop = gpu_temp_stop_c();
    if (!env || !env[0]) {
        return stop > 10 ? stop - 10 : 0;
    }
    const int v = std::atoi(env);
    return v > 0 ? v : 0;
}

// Max failure retries per tune combo (process-isolated sweep scripts).
inline int tune_max_retries() {
    const char* env = std::getenv("PROPMINER_TUNE_MAX_RETRIES");
    if (env && env[0]) {
        const int v = std::atoi(env);
        if (v >= 1 && v <= 10) return v;
    }
    return 3;
}

// Fast supervisor restart delay after wedged GPU exit (exit code 42).
inline int stall_restart_delay_sec() {
    const char* env = std::getenv("PROPMINER_STALL_RESTART_DELAY_SEC");
    if (env && env[0]) {
        const int v = std::atoi(env);
        if (v >= 0 && v <= 60) return v;
    }
    return 3;
}

}  // namespace pearl
