#pragma once

#include "rtx5090_profile.h"

#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace pearl {

// RTX 5090 production tuning is controlled via environment variables.
// Defaults below match scripts/salad/kryptex.env.example — change env at runtime
// without rebuilding; change rtx5090_profile.h when updating shipped defaults.
// Production N cap default: kDefaultProdNCap (131072); PROPMINER_N_CAP=0 = uncapped.

inline bool env_is_set(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] != '\0';
}

inline int env_int_or(const char* name, int fallback) {
    if (const char* v = std::getenv(name); v && v[0]) {
        return std::max(1, std::atoi(v));
    }
    return fallback;
}

inline long long env_ll_or(const char* name, long long fallback) {
    if (const char* v = std::getenv(name); v && v[0]) {
        const long long n = std::atoll(v);
        if (n > 0) return n;
    }
    return fallback;
}

inline bool autotune_env_enabled() {
    const char* v = std::getenv("PROPMINER_AUTOTUNE");
    if (!v || !v[0]) return false;
    return std::strcmp(v, "0") != 0;
}

inline bool tune_cache_env_enabled() {
    const char* v = std::getenv("PROPMINER_USE_TUNE_CACHE");
    if (!v || !v[0]) return true;  // default on for mine mode
    return v[0] != '0';
}

inline bool mine_batch_env_set() { return env_is_set("PROPMINER_BATCH"); }
inline bool graph_batch_env_set() { return env_is_set("PROPMINER_GRAPH_BATCH"); }
inline bool cluster_m_env_set() { return env_is_set("PEARL_GEMM_CONSUMER_CLUSTER_M"); }
inline bool stratum_diff_env_set() { return env_is_set("PROPMINER_STRATUM_DIFF"); }
inline bool n_cap_env_set() { return env_is_set("PROPMINER_N_CAP"); }

// Default N=131072 (~10 GiB VRAM with Stratum K=4096). PROPMINER_N_CAP=0 = uncapped.
inline int resolve_n_cap() {
    if (const char* v = std::getenv("PROPMINER_N_CAP"); v && v[0]) {
        if (std::strcmp(v, "0") == 0) return 0;
        return std::max(1, std::atoi(v));
    }
    return Rtx5090Profile::kDefaultProdNCap;
}

inline bool manual_tuning_env_set() {
    return mine_batch_env_set() || graph_batch_env_set() || cluster_m_env_set();
}

inline int resolve_mine_batch() {
    return env_int_or("PROPMINER_BATCH", Rtx5090Profile::kDefaultMineBatch);
}

inline int resolve_graph_batch() {
    return env_int_or("PROPMINER_GRAPH_BATCH", Rtx5090Profile::kDefaultGraphBatch);
}

inline int resolve_cluster_m() {
    return env_int_or("PEARL_GEMM_CONSUMER_CLUSTER_M",
                      Rtx5090Profile::kProdDefaultClusterM);
}

inline long long resolve_stratum_share_diff() {
    return env_ll_or("PROPMINER_STRATUM_DIFF",
                     Rtx5090Profile::kDefaultStratumShareDiff);
}

inline double resolve_stratum_share_diff_double() {
    return static_cast<double>(resolve_stratum_share_diff());
}

// Canonical pattern expansion (2×64) vs header extract_indices (Kryptex default).
// Headless GPU PoW checks per-thread transcripts; extract mode matches what
// previously passed local verify + Kryptex pool accept. Set PROPMINER_HASH_TILE_PATTERN=1
// when the kernel aggregates the full committed hash tile.
inline bool hash_tile_use_pattern_expansion() {
    const char* v = std::getenv("PROPMINER_HASH_TILE_PATTERN");
    return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');
}

// True when mining the HeroMiners/Kryptex committed stratum shape (131072² @
// K=4096 R=256). One matmul at this shape can take many seconds on CUDA; ARC
// forces MatmulsPerPoll=1 and uses a 300s progress watchdog (not 9s).
inline bool is_canonical_stratum_shape(const MiningConfig& cfg) {
    return cfg.m >= Rtx5090Profile::kStratumLargeM &&
           cfg.n >= Rtx5090Profile::kStratumLargeN &&
           cfg.k == Rtx5090Profile::kStratumPoolK;
}

// Watchdog fires when elapsed > period_ms * 3. ARC default budget is 300s
// (AKOYA_MINE_WATCHDOG_TIMEOUT_SEC). PropMiner default was 3s*3=9s, which
// false-trips during one legitimate canonical-shape matmul.
inline int resolve_watchdog_period_ms(bool canonical_stratum_shape) {
    if (const char* v = std::getenv("PROPMINER_WATCHDOG_PERIOD_MS"); v && v[0]) {
        return std::max(1000, std::atoi(v));
    }
    return canonical_stratum_shape ? 100000 : 3000;  // 300s vs 9s stall budget
}

// ARC ComputeMpp: floor(10ms / iter_ms), clamped [1, 16]. At canonical shape
// without a measured iter_ms, use 1 (one matmul per poll).
inline int stratum_matmuls_per_poll(const MiningConfig& cfg, double iter_ms = 0.0) {
    if (!is_canonical_stratum_shape(cfg)) {
        return Rtx5090Profile::kDefaultMineBatch;
    }
    if (iter_ms > 0.0 && std::isfinite(iter_ms)) {
        int mpp = static_cast<int>(10.0 / iter_ms);
        if (mpp < 1) mpp = 1;
        if (mpp > Rtx5090Profile::kMaxMineBatch) {
            mpp = Rtx5090Profile::kMaxMineBatch;
        }
        return mpp;
    }
    return 1;
}

// graph_batch must divide batch evenly for the CUDA graph launch path.
inline void normalize_batch_and_graph(int& batch, int& graph_batch) {
    if (graph_batch > batch) {
        graph_batch = batch;
    }
    if (batch > 0 && graph_batch > 0 && (batch % graph_batch) != 0) {
        std::fprintf(stderr,
                     "[tuning] WARN: PROPMINER_BATCH=%d not divisible by "
                     "PROPMINER_GRAPH_BATCH=%d — graph path disabled (iter_batch only)\n",
                     batch, graph_batch);
    }
}

inline void log_resolved_tuning(const char* prefix = "[tuning]") {
    const int n_cap = resolve_n_cap();
    std::fprintf(stderr,
                 "%s n_cap=%d%s batch=%d%s graph_batch=%d%s cluster_m=%d%s "
                 "stratum_diff=%lld%s\n",
                 prefix,
                 n_cap,
                 n_cap_env_set() ? " (PROPMINER_N_CAP)"
                                 : (n_cap > 0 ? " (default)" : " (uncapped)"),
                 resolve_mine_batch(),
                 mine_batch_env_set() ? " (PROPMINER_BATCH)" : " (default)",
                 resolve_graph_batch(),
                 graph_batch_env_set() ? " (PROPMINER_GRAPH_BATCH)" : " (default)",
                 resolve_cluster_m(),
                 cluster_m_env_set() ? " (PEARL_GEMM_CONSUMER_CLUSTER_M)"
                                     : " (default)",
                 static_cast<long long>(resolve_stratum_share_diff()),
                 stratum_diff_env_set() ? " (PROPMINER_STRATUM_DIFF)"
                                        : " (default)");
}

}  // namespace pearl
