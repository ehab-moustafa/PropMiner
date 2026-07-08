#pragma once

#include "pearl_types.h"
#include "system_telemetry.h"

#include <cstdio>
#include <cmath>
#include <ctime>

namespace pearl {

// RTX 5090 rated dense INT8 tensor throughput (TOPS).
inline constexpr double kRtx5090RatedInt8Tops = 838.0e12;

// Snapshot of mining throughput in all common unit systems.
struct HashrateMetrics {
    double tmad_per_sec = 0.0;     // trillion int8 MAC/s (community "TH/s")
    double protocol_hps = 0.0;     // DAF-normalized tiles/s (Pearl wire H/s)
    double tops_pct = 0.0;         // tmad_per_sec * 1e12 / rated INT8 TOPS
    double tiles_per_sec = 0.0;
    double iters_per_sec = 0.0;
    double batch_ms = 0.0;
    int batch = 0;
    int graph_batch = 0;
    int m = 0;
    int n = 0;
    int k = 0;
};

// M*N*K exceeds INT32_MAX at production shapes (8192*32768*128 = 2^35).
inline double mining_mac_volume(const MiningConfig& cfg) {
    return static_cast<double>(static_cast<int64_t>(cfg.m) * cfg.n * cfg.k);
}

inline HashrateMetrics hashrate_metrics_from_rates(const MiningConfig& cfg,
                                                 double tmad_per_sec,
                                                 double protocol_hps,
                                                 double batch_ms,
                                                 int batch,
                                                 int graph_batch = 0) {
    HashrateMetrics m{};
    m.tmad_per_sec = tmad_per_sec;
    m.protocol_hps = protocol_hps;
    m.batch_ms = batch_ms;
    m.batch = batch;
    m.graph_batch = graph_batch > 0 ? graph_batch : batch;
    m.m = cfg.m;
    m.n = cfg.n;
    m.k = cfg.k;
    if (tmad_per_sec > 0.0) {
        m.tops_pct = (tmad_per_sec * 1e12 / kRtx5090RatedInt8Tops) * 100.0;
    }
    if (protocol_hps > 0.0) {
        const double daf = static_cast<double>(cfg.difficulty_adjustment_factor());
        if (daf > 0.0) {
            m.tiles_per_sec = protocol_hps / daf;
        }
    }
    if (batch_ms > 0.0 && batch > 0) {
        m.iters_per_sec =
            (static_cast<double>(batch) * 1000.0) / batch_ms;
    }
    return m;
}

inline HashrateMetrics hashrate_metrics_from_iters(const MiningConfig& cfg,
                                                   uint64_t iters,
                                                   int elapsed_sec,
                                                   int batch) {
    if (iters == 0 || elapsed_sec <= 0) {
        return hashrate_metrics_from_rates(cfg, 0.0, 0.0, 0.0, batch);
    }
    const double sec = static_cast<double>(elapsed_sec);
    const double ips = static_cast<double>(iters) / sec;
    const double tmad = mining_mac_volume(cfg) * ips / 1e12;
    const double tiles_per_iter =
        static_cast<double>(cfg.m / cfg.bM) * (cfg.n / cfg.bN);
    const double protocol =
        ips * tiles_per_iter *
        static_cast<double>(cfg.difficulty_adjustment_factor());
    return hashrate_metrics_from_rates(cfg, tmad, protocol, 0.0, batch);
}

// GpuTuner counts 2*MAC per matmul; convert to TMAD/s for pool-scale reporting.
inline double tmad_from_tuner_ops_per_sec(double ops_per_sec) {
    return ops_per_sec / 2.0e12;
}

inline void format_scaled_rate(double rate, char* buf, size_t buflen) {
    if (rate >= 1e15) {
        std::snprintf(buf, buflen, "%.2f PH/s", rate / 1e15);
    } else if (rate >= 1e12) {
        std::snprintf(buf, buflen, "%.2f TH/s", rate / 1e12);
    } else if (rate >= 1e9) {
        std::snprintf(buf, buflen, "%.2f GH/s", rate / 1e9);
    } else if (rate >= 1e6) {
        std::snprintf(buf, buflen, "%.2f MH/s", rate / 1e6);
    } else if (rate >= 1e3) {
        std::snprintf(buf, buflen, "%.2f KH/s", rate / 1e3);
    } else {
        std::snprintf(buf, buflen, "%.0f H/s", rate);
    }
}

inline void print_hashrate_metrics_line(FILE* out,
                                        const char* prefix,
                                        const HashrateMetrics& m,
                                        const SystemSnapshot* sys = nullptr) {
    char protocol_buf[32];
    format_scaled_rate(m.protocol_hps, protocol_buf, sizeof(protocol_buf));

    std::fprintf(out,
        "%sTMAD/s=%.3f (pool TH/s) | protocol=%s (%.3e H/s) | "
        "TOPS=%.1f%% | tiles/s=%.2e",
        prefix, m.tmad_per_sec, protocol_buf, m.protocol_hps, m.tops_pct,
        m.tiles_per_sec);

    if (m.batch_ms > 0.0) {
        std::fprintf(out, " | batch=%.0fms", m.batch_ms);
    }
    if (m.iters_per_sec > 0.0) {
        std::fprintf(out, " | matmuls/s=%.3f", m.iters_per_sec);
    }
    if (m.batch > 0) {
        std::fprintf(out, " | batch=%d", m.batch);
    }
    if (m.graph_batch > 0) {
        std::fprintf(out, " | graph_batch=%d", m.graph_batch);
    }
    if (m.m > 0 && m.n > 0 && m.k > 0) {
        std::fprintf(out, " | M=%d N=%d K=%d", m.m, m.n, m.k);
    }
    if (sys) {
        append_system_snapshot_to_line(out, *sys);
    }
    std::fputc('\n', out);
}

// Emit a one-time health verdict comparing measured throughput against the
// GPU's power budget so a throttled / power-capped / shared instance is obvious
// at a glance. `sys` may be null (no telemetry) — then only the compute verdict
// is printed. Heuristics are tuned for a single RTX 5090 (~300-360 TMAD/s,
// ~35-40% of rated INT8 at ~450-500 W).
inline void print_hashrate_health(FILE* out, const char* prefix,
                                  const HashrateMetrics& m,
                                  const SystemSnapshot* sys) {
    // Compute-side verdict from TOPS%.
    const char* compute = "unknown";
    if (m.tops_pct >= 33.0)      compute = "GOOD (near single-5090 ceiling)";
    else if (m.tops_pct >= 22.0) compute = "OK (some headroom)";
    else if (m.tops_pct > 0.0)   compute = "LOW (large headroom)";

    std::fprintf(out, "%shealth: TOPS=%.1f%% -> compute=%s",
                 prefix, m.tops_pct, compute);

    if (sys && sys->gpu_power_w >= 0 && sys->gpu_power_limit_w > 0) {
        const double pct = 100.0 * sys->gpu_power_w / sys->gpu_power_limit_w;
        const char* pwr = "ok";
        if (pct < 45.0)      pwr = "SEVERELY CAPPED";
        else if (pct < 70.0) pwr = "capped/low";
        std::fprintf(out, " | power=%d/%dW (%.0f%% -> %s)",
                     sys->gpu_power_w, sys->gpu_power_limit_w, pct, pwr);
    }
    if (sys && sys->sm_clock_mhz >= 0) {
        std::fprintf(out, " | sm_clock=%dMHz", sys->sm_clock_mhz);
    }
    std::fputc('\n', out);

    // Actionable hint: distinguish instance throttle vs kernel headroom.
    if (sys && sys->gpu_power_w >= 0 && sys->gpu_power_limit_w > 0 &&
        (100.0 * sys->gpu_power_w / sys->gpu_power_limit_w) < 70.0 &&
        m.tops_pct > 0.0 && m.tops_pct < 33.0) {
        std::fprintf(out,
            "%shealth: HINT low power AND low TOPS -> instance is likely "
            "power-capped/shared; a full-power GPU is the biggest win.\n",
            prefix);
    } else if (m.tops_pct > 0.0 && m.tops_pct < 22.0) {
        std::fprintf(out,
            "%shealth: HINT low TOPS at healthy power -> tune batch/graph_batch/"
            "cluster_m or improve the GEMM kernel.\n",
            prefix);
    }
}

inline void print_hashrate_metrics_json(FILE* out,
                                        const HashrateMetrics& m,
                                        const MiningConfig& cfg,
                                        int elapsed_sec = 0,
                                        uint64_t total_iters = 0,
                                        int bench_seconds = 0,
                                        const char* git_sha = "",
                                        const char* miner_version = "",
                                        const SystemSnapshot* sys = nullptr) {
    std::fprintf(out,
        "{\"ts\":%lld,\"git_sha\":\"%s\",\"miner_version\":\"%s\","
        "\"bench_seconds\":%d,"
        "\"elapsed_sec\":%d,\"total_iters\":%llu,"
        "\"tmad_per_sec\":%.6f,\"pool_th_per_sec\":%.6f,"
        "\"protocol_hps\":%.6e,\"daf\":%llu,\"tops_pct\":%.4f,"
        "\"tiles_per_sec\":%.6e,\"iters_per_sec\":%.6f,"
        "\"batch_ms\":%.3f,\"graph_batch\":%d,"
        "\"m\":%d,\"n\":%d,\"k\":%d",
        static_cast<long long>(std::time(nullptr)), git_sha, miner_version,
        bench_seconds, elapsed_sec,
        static_cast<unsigned long long>(total_iters),
        m.tmad_per_sec, m.tmad_per_sec, m.protocol_hps,
        static_cast<unsigned long long>(cfg.difficulty_adjustment_factor()),
        m.tops_pct, m.tiles_per_sec, m.iters_per_sec,
        m.batch_ms, m.batch, m.m, m.n, m.k);
    if (sys && sys->valid) {
        std::fprintf(out,
            ",\"gpu_util_pct\":%d,\"gpu_mem_util_pct\":%d,"
            "\"gpu_temp_c\":%d,\"gpu_power_w\":%d,\"gpu_power_limit_w\":%d,"
            "\"vram_used_mb\":%llu,\"vram_total_mb\":%llu,"
            "\"cpu_util_pct\":%d,\"ram_used_mb\":%llu,\"ram_total_mb\":%llu",
            sys->gpu_util_pct, sys->gpu_mem_util_pct,
            sys->gpu_temp_c, sys->gpu_power_w, sys->gpu_power_limit_w,
            static_cast<unsigned long long>(sys->vram_used_mb),
            static_cast<unsigned long long>(sys->vram_total_mb),
            sys->cpu_util_pct,
            static_cast<unsigned long long>(sys->ram_used_mb),
            static_cast<unsigned long long>(sys->ram_total_mb));
    }
    std::fputc('}', out);
    std::fputc('\n', out);
}

} // namespace pearl
