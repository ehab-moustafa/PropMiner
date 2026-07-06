#pragma once

#include "pearl_types.h"

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
    int m = 0;
    int n = 0;
    int k = 0;
};

inline HashrateMetrics hashrate_metrics_from_rates(const MiningConfig& cfg,
                                                 double tmad_per_sec,
                                                 double protocol_hps,
                                                 double batch_ms,
                                                 int batch) {
    HashrateMetrics m{};
    m.tmad_per_sec = tmad_per_sec;
    m.protocol_hps = protocol_hps;
    m.batch_ms = batch_ms;
    m.batch = batch;
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
    const double tmad =
        static_cast<double>(cfg.m) * cfg.n * cfg.k * ips / 1e12;
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
                                        const HashrateMetrics& m) {
    char protocol_buf[32];
    format_scaled_rate(m.protocol_hps, protocol_buf, sizeof(protocol_buf));

    std::fprintf(out,
        "%sTMAD/s=%.2f (pool TH/s) | protocol=%s (%.3e H/s) | "
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
        std::fprintf(out, " | graph_batch=%d", m.batch);
    }
    if (m.m > 0 && m.n > 0 && m.k > 0) {
        std::fprintf(out, " | M=%d N=%d K=%d", m.m, m.n, m.k);
    }
    std::fputc('\n', out);
}

inline void print_hashrate_metrics_json(FILE* out,
                                        const HashrateMetrics& m,
                                        const MiningConfig& cfg,
                                        int elapsed_sec = 0,
                                        uint64_t total_iters = 0,
                                        int bench_seconds = 0,
                                        const char* git_sha = "",
                                        const char* miner_version = "") {
    std::fprintf(out,
        "{\"ts\":%lld,\"git_sha\":\"%s\",\"miner_version\":\"%s\","
        "\"bench_seconds\":%d,"
        "\"elapsed_sec\":%d,\"total_iters\":%llu,"
        "\"tmad_per_sec\":%.6f,\"pool_th_per_sec\":%.6f,"
        "\"protocol_hps\":%.6e,\"daf\":%llu,\"tops_pct\":%.4f,"
        "\"tiles_per_sec\":%.6e,\"iters_per_sec\":%.6f,"
        "\"batch_ms\":%.3f,\"graph_batch\":%d,"
        "\"m\":%d,\"n\":%d,\"k\":%d}\n",
        static_cast<long long>(std::time(nullptr)), git_sha, miner_version,
        bench_seconds, elapsed_sec,
        static_cast<unsigned long long>(total_iters),
        m.tmad_per_sec, m.tmad_per_sec, m.protocol_hps,
        static_cast<unsigned long long>(cfg.difficulty_adjustment_factor()),
        m.tops_pct, m.tiles_per_sec, m.iters_per_sec,
        m.batch_ms, m.batch, m.m, m.n, m.k);
}

} // namespace pearl
