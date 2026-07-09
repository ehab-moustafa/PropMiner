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

// Targeted B-column expansion in compute_claimed_hash: expand only the opened
// columns (w*k bytes via BLAKE3 XOF seek) instead of the full n*k stream.
// Byte-identical output (see pearl_capi_bseed_expand_range_raw docs); the
// same per-column pattern is already used by protobuf_encoder.cpp.
// PROPMINER_BCOL_CACHE=0 restores the legacy full expansion.
inline bool bcol_targeted_expand_enabled() {
    const char* env = std::getenv("PROPMINER_BCOL_CACHE");
    return !(env && env[0] == '0');
}

// Async job installation (PROPMINER_ASYNC_JOB_INSTALL, default ON): when a new
// pool job arrives, install its resident B (GPU expand + tensor-hash + noise +
// Merkle) on a background thread while the current job keeps mining, then do a
// fast swap on the worker thread. A cudaMemGetInfo headroom check runs before
// each background install: if there is not enough free VRAM for a second
// resident-B set (+ one-time staging workspace + margin) it self-disables and
// does a synchronous install instead, so it can never OOM — at large N
// (262144) with high VRAM use it simply falls back to the old behavior.
// PROPMINER_ASYNC_JOB_INSTALL=0 forces the synchronous job switch.
inline bool async_job_install_enabled() {
    const char* env = std::getenv("PROPMINER_ASYNC_JOB_INSTALL");
    return !(env && env[0] == '0');
}

// Triple half-buffer (PROPMINER_TRIPLE_BUFFER, default ON): add a third
// compute workspace so share reconstruction on one half never blocks GEMM on
// the other two. Requires PROPMINER_DEFER_SHARE_GPU (default ON). A VRAM
// headroom check at GpuWorker construction auto-disables when tight.
// PROPMINER_TRIPLE_BUFFER=0 to disable.
inline bool triple_buffer_enabled() {
    const char* env = std::getenv("PROPMINER_TRIPLE_BUFFER");
    return !(env && env[0] == '0');
}

// Async seed conveyor (default ON): upload the next graph sub-batch's seed on
// the dedicated copy stream while the CPU copies out the previous sub-batch's
// headers, instead of a synchronous H2D on the compute stream. The upload is
// issued only after the current sub-batch has fully completed (seed_dev must
// never change while a graph is reading it), and the next launch orders after
// the copy via cudaStreamWaitEvent. PROPMINER_ASYNC_SEED=0 restores the
// synchronous seed upload.
inline bool async_seed_enabled() {
    const char* env = std::getenv("PROPMINER_ASYNC_SEED");
    return !(env && env[0] == '0');
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

// Health monitor: no iter progress + wedge signature (high util, low power).
inline int64_t progress_stall_warn_ms() {
    const char* env = std::getenv("PROPMINER_PROGRESS_STALL_WARN_MS");
    if (env && env[0]) {
        const long long v = std::atoll(env);
        if (v >= 5000 && v <= 120000) return v;
    }
    return 12000;  // warn + soft recovery at 12s (healthy batch << 1s)
}

inline int64_t progress_stall_abort_ms() {
    const char* env = std::getenv("PROPMINER_PROGRESS_STALL_ABORT_MS");
    if (env && env[0]) {
        const long long v = std::atoll(env);
        if (v >= 10000 && v <= 120000) return v;
    }
    return 18000;  // abort batch wait at 18s (before default 30s stall exit)
}

inline int wedge_power_threshold_w() {
    const char* env = std::getenv("PROPMINER_WEDGE_POWER_THRESHOLD_W");
    if (env && env[0]) {
        const int v = std::atoi(env);
        if (v > 0 && v <= 500) return v;
    }
    return 150;  // 100% util + <150W = spinning wedge (healthy ~450W)
}

}  // namespace pearl
