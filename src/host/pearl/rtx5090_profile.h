#pragma once

#include "cuda_compat.h"
#include "pearl_types.h"

#include <cstdint>
#include <numeric>

namespace pearl {

// Hard-coded physical profile for the NVIDIA GeForce RTX 5090 (GB202).
//
// Target compute capability: native Blackwell sm_120a (CC 12.0). NOT sm_90 Hopper,
// NOT sm_100a datacenter.  Grid sizing prefers the largest N that fits in VRAM
// while keeping CTA waves aligned to the 170 physical SMs.
struct Rtx5090Profile {
    static constexpr int kSMCount = 170;        // enabled SMs on RTX 5090
    static constexpr int kWarpsPerSM = 48;      // practical warp slots per SM
    static constexpr int kThreadsPerWarp = 32;
    static constexpr int kThreadsPerCTA = 256;  // 8 warps, matches consumer kernel

    // Consumer kernel tile.  128x256x128 is chosen so that the accumulators
    // (128*256*4B = 128 KiB) plus A/B smem tiles fit comfortably inside the
    // 164 KiB shared-memory budget per SM while keeping the MMA dimensions
    // aligned to Blackwell tcgen05 operand layouts.
    static constexpr int kTileM = 128;
    static constexpr int kTileN = 256;
    static constexpr int kTileK = 128;

    // Compile-time fallback shape (runtime may pick a larger N from VRAM).
    static constexpr int kDefaultM = 8192;
    static constexpr int kDefaultN = 32768;
    static constexpr int kDefaultK = 128;
    static constexpr int kDefaultR = 128;

    // Matmuls queued per GPU poll (cargo volume per execution).
    static constexpr int kDefaultMineBatch = 32;
    // CUDA graph capture depth (staging queue per graph launch).
    static constexpr int kDefaultGraphBatch = 8;
    static constexpr int kMaxMineBatch = 32;
    static constexpr int kDefaultBatch = kDefaultMineBatch;  // legacy alias

    // Consumer GeForce Blackwell (RTX 5090): cluster_m>1 can wedge on some driver
    // stacks; kernel falls back to standard launch on cudaLaunchKernelEx failure.
    // Safe default is 1 (no clustering); tune / PROPMINER_TUNE_CLUSTERS finds best.
    static constexpr int kProdDefaultClusterM = 1;

    // Stratum share difficulty when pool has not sent vardiff yet (also PROPMINER_STRATUM_DIFF).
    // Lower = easier shares, more frequent submits (Salad: helps before job supersede).
    static constexpr long long kDefaultStratumShareDiff = 32768;

    static constexpr int kMineBatchCandidates[] = {
        1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32,
    };

    // Default production N cap (Salad/Kryptex Stratum K=4096 VRAM headroom).
    // Set PROPMINER_N_CAP=0 for uncapped VRAM pick (up to 262144), or PROPMINER_N_CAP=N.
    static constexpr int kDefaultProdNCap = 131072;

    // Bench/tune/mine share production N (pick_n_for_vram). Override via PROPMINER_N_CAP.
    static constexpr int kLegacySmallN = 32768;
    static constexpr int kBenchDefaultSeconds = 300;
    static constexpr int kBenchGraceSeconds = 120;

    // Number of CTAs launched per GEMM = (M/kTileM) * (N/kTileN).
    static constexpr int tiles(int M, int N) {
        return (M / kTileM) * (N / kTileN);
    }

    // True if the grid has enough tiles to fill all SMs twice.
    static constexpr bool full_occupancy(int M, int N) {
        return tiles(M, N) >= kSMCount * 2;
    }

    // True when total CTAs divide evenly across 170 SMs (no tail wave).
    static constexpr bool wave_aligned(int M, int N) {
        return tiles(M, N) % kSMCount == 0;
    }

    // Smallest N >= min_n (tile-aligned) with wave_aligned(M, N).
    static int wave_aligned_n_at_least(int M, int min_n) {
        const int grid_x = M / kTileM;
        const int step_t = kSMCount / std::gcd(grid_x, kSMCount);
        int t = (min_n + kTileN - 1) / kTileN;
        if ((grid_x * t) % kSMCount != 0) {
            t = ((t + step_t - 1) / step_t) * step_t;
        }
        return t * kTileN;
    }

    // Conservative VRAM budget for production N selection (ping-pong C, resident B,
    // CUDA graphs, pearl workspace).  Strict < free_bytes.
    static bool shape_fits_vram(int M, int N, int K, size_t free_bytes) {
        if (free_bytes == 0) {
            // Assume 32 GB GDDR7 minus ~4 GiB driver/OS reserve when query unavailable.
            free_bytes = (28ULL << 30);
        }
        const int64_t a_ping_pong = 2 * int64_t(M) * K;
        const int64_t b_resident = int64_t(K) * N;
        const int64_t c_ping_pong = 2 * int64_t(M) * N * 2;  // two fp16 C halves
        const int64_t r = 128;
        const int64_t resident_b = b_resident + 6 * int64_t(N) * r;  // B + noise side
        int64_t bytes = a_ping_pong + b_resident + c_ping_pong + resident_b;
        bytes += bytes / 4;  // workspace, graphs, leaf CVs, alignment
        return static_cast<size_t>(bytes) < free_bytes;
    }

    // Prefer largest high-intensity N validated on 5090; fall back gracefully.
    // Wave alignment is logged at startup; a small tail wave is cheaper than
    // shrinking N (e.g. 262144 -> 65280).
    static int pick_n_for_vram(size_t free_bytes, int cap_n = 0, int K = kDefaultK) {
        static constexpr int kCandidates[] = {
            262144, 131072, 65536, 65280, 43520, 32768,
        };
        for (int n : kCandidates) {
            if (cap_n > 0 && n > cap_n) continue;
            if (n % kTileN != 0) continue;
            if (!shape_fits_vram(kDefaultM, n, K, free_bytes)) continue;
            return n;
        }
        return (cap_n > 0 && cap_n < kDefaultN) ? cap_n : kDefaultN;
    }

    // Kryptex / suprnova stratum sanity (§7.1): k >= 1024, k % 64 == 0,
    // k >= 16*rank. PropMiner kernels ship R=128; K=4096 satisfies rank=128.
    static constexpr int kStratumPoolK = 4096;
    static constexpr int kStratumPoolR = 128;
};

// Stratum pool PlainProof preamble must pass §7.1 (k >= 1024, etc.).
// Local Akoya gRPC path may use K=128; Kryptex rejects that as invalid proof.
inline MiningConfig stratum_pool_mining_config(size_t vram_budget_bytes = 0,
                                              int cap_n = 0) {
    MiningConfig cfg;
    cfg.m = Rtx5090Profile::kDefaultM;
    cfg.k = Rtx5090Profile::kStratumPoolK;
    cfg.r = Rtx5090Profile::kStratumPoolR;
    if (const char* kenv = std::getenv("PROPMINER_STRATUM_K"); kenv && kenv[0]) {
        const int req = std::atoi(kenv);
        if (req >= 2048 && req % 64 == 0) cfg.k = req;
    }
    if (const char* renv = std::getenv("PROPMINER_STRATUM_RANK"); renv && renv[0]) {
        const int req = std::atoi(renv);
        if (req == 64 || req == 128) cfg.r = req;
    }
    cfg.n = Rtx5090Profile::pick_n_for_vram(vram_budget_bytes, cap_n, cfg.k);
    cfg.bM = Rtx5090Profile::kTileM;
    cfg.bN = Rtx5090Profile::kTileN;
    cfg.bK = Rtx5090Profile::kTileK;
    return cfg;
}

// Compile-time guarantees that the fallback RTX 5090 shape saturates the chip.
static_assert(Rtx5090Profile::kDefaultM % Rtx5090Profile::kTileM == 0,
              "M must be a multiple of tile M");
static_assert(Rtx5090Profile::kDefaultN % Rtx5090Profile::kTileN == 0,
              "N must be a multiple of tile N");
static_assert(Rtx5090Profile::kDefaultK % Rtx5090Profile::kTileK == 0,
              "K must be a multiple of tile K");
static_assert(Rtx5090Profile::full_occupancy(Rtx5090Profile::kDefaultM,
                                             Rtx5090Profile::kDefaultN),
              "Default RTX 5090 shape must launch enough CTAs to fill all SMs");

// Return a MiningConfig pre-tuned for RTX 5090.
//
// vram_budget_bytes: free VRAM after driver reserve (0 = pick largest candidate).
// cap_n: when >0, limit N (default kDefaultProdNCap; PROPMINER_N_CAP=0 = uncapped).
inline MiningConfig rtx5090_mining_config(size_t vram_budget_bytes = 0,
                                          int cap_n = 0) {
    MiningConfig cfg;
    cfg.m = Rtx5090Profile::kDefaultM;
    cfg.n = Rtx5090Profile::pick_n_for_vram(vram_budget_bytes, cap_n);
    cfg.k = Rtx5090Profile::kDefaultK;
    cfg.r = Rtx5090Profile::kDefaultR;
    cfg.bM = Rtx5090Profile::kTileM;
    cfg.bN = Rtx5090Profile::kTileN;
    cfg.bK = Rtx5090Profile::kTileK;
    return cfg;
}

} // namespace pearl
