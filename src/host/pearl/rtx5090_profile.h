#pragma once

#include "cuda_compat.h"
#include "pearl_types.h"

namespace pearl {

// Hard-coded physical profile for the NVIDIA GeForce RTX 5090 (GB202).
//
// Target compute capability: native Blackwell sm_120 (NOT sm_90 Hopper, NOT
// sm_100a datacenter).  We deliberately avoid any datacenter bleed-through and
// size the grid so the kernel waves fill exactly the 170 physical SMs.
//
// Occupancy math for the default shape M=8192, N=32768, tile=128x256:
//   grid = (8192/128) * (32768/256) = 64 * 128 = 8192 CTAs.
//   With 170 SMs the kernel launches ~48 waves, giving every SM a persistent
//   stream of work and hiding all launch/PCIe latency behind execution.
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

    // Default matrix shape for ~12-14 GB VRAM footprint.
    static constexpr int kDefaultM = 8192;
    static constexpr int kDefaultN = 32768;
    static constexpr int kDefaultK = 128;
    static constexpr int kDefaultR = 128;

    // Batch size tuned to amortize launch overhead without excessive latency.
    static constexpr int kDefaultBatch = 20;

    // Number of CTAs launched per GEMM = (M/kTileM) * (N/kTileN).
    static constexpr int tiles(int M, int N) {
        return (M / kTileM) * (N / kTileN);
    }

    // True if the grid has enough tiles to fill all SMs twice.
    static constexpr bool full_occupancy(int M, int N) {
        return tiles(M, N) >= kSMCount * 2;
    }
};

// Compile-time guarantees that the default RTX 5090 shape saturates the chip.
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
// Important: this profile intentionally hardcodes RTX 5090 geometry instead of
// exposing runtime knobs.  For other NVIDIA GPUs the generic GpuTuner path is
// used; for a 5090 we want deterministic, optimal launch dimensions.
inline MiningConfig rtx5090_mining_config() {
    MiningConfig cfg;
    cfg.m = Rtx5090Profile::kDefaultM;
    cfg.n = Rtx5090Profile::kDefaultN;
    cfg.k = Rtx5090Profile::kDefaultK;
    cfg.r = Rtx5090Profile::kDefaultR;
    cfg.bM = Rtx5090Profile::kTileM;
    cfg.bN = Rtx5090Profile::kTileN;
    cfg.bK = Rtx5090Profile::kTileK;
    return cfg;
}

} // namespace pearl
