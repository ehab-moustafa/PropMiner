#pragma once

#include <cuda_runtime.h>
#include <cstdint>

/* ── LCG int7 noise generation (GPU device function) ─────────────────
 *
 * Deterministic PRNG using SplitMix64. Produces int7 values in [-63, +63].
 * Matches the reference implementation in Akoya-miner's pearl_gemm_capi_util.cu
 * and the C# host-side LcgInt7.cs.
 *
 * Designed to be inlined into the main mining kernel per-element, NOT
 * launched as a separate kernel. Each call generates a single int7 byte.
 */

__device__ __forceinline__ uint64_t splitmix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ── Generate a batch of int7 values from a seed ─────────────────────
 *
 * Fills `out[count]` with deterministic int7 bytes derived from (seed_lo, seed_hi).
 * `offset` allows skipping the first N bytes (for strided access).
 *
 * Implementation: one splitmix64 call produces 8 output bytes.
 */
template <int COUNT>
__device__ __forceinline__ void
lcg_int7_batch(int8_t out[COUNT], uint64_t seed_lo, uint64_t seed_hi, uint64_t offset = 0) {
    uint64_t base = splitmix64(seed_lo ^ splitmix64(seed_hi));

    constexpr int full_groups = COUNT / 8;
    constexpr int remainder   = COUNT % 8;

    /* Process full 8-byte groups */
#pragma unroll
    for (int g = 0; g < full_groups; ++g) {
        uint64_t z = splitmix64(base + offset + g);
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            uint32_t v = static_cast<uint32_t>((z >> (b * 8)) & 0xFFu);
            uint32_t r = v % 127u;
            out[g * 8 + b] = static_cast<int8_t>(static_cast<int32_t>(r) - 63);
        }
    }

    /* Trailing bytes */
#pragma unroll
    for (int b = 0; b < remainder; ++b) {
        uint64_t z = splitmix64(base + offset + full_groups);
        uint32_t v = static_cast<uint32_t>((z >> (b * 8)) & 0xFFu);
        uint32_t r = v % 127u;
        out[full_groups * 8 + b] = static_cast<int8_t>(static_cast<int32_t>(r) - 63);
    }
}

/* ── Generate a single int7 value (most common case for element-wise) ─ */

__device__ __forceinline__ int8_t
lcg_int7_single(uint64_t seed_lo, uint64_t seed_hi, uint64_t element_idx) {
    uint64_t base = splitmix64(seed_lo ^ splitmix64(seed_hi));
    uint64_t z = splitmix64(base + element_idx);
    uint32_t v = static_cast<uint32_t>((z >> ((element_idx % 8) * 8)) & 0xFFu);
    uint32_t r = v % 127u;
    return static_cast<int8_t>(static_cast<int32_t>(r) - 63);
}

/* ── Generate an int8 tile (for GEMM operand loading) ────────────────
 *
 * Generates a TILE_M × TILE_K or TILE_N × TILE_K tile of int7 values.
 * Thread-local: each thread generates its own portion using
 * (row, col) as the element index into the PRNG stream.
 */

template <int ROWS, int COLS>
__device__ __forceinline__ void
lcg_int7_tile(int8_t tile[ROWS][COLS],
              uint64_t seed_lo, uint64_t seed_hi,
              int row_offset, int col_offset) {
    uint64_t base = splitmix64(seed_lo ^ splitmix64(seed_hi));

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int c = 0; c < COLS; ++c) {
            uint64_t idx = static_cast<uint64_t>(row_offset + r) * COLS + col_offset + c;
            uint64_t group = idx / 8;
            int      byte_in_group = static_cast<int>(idx % 8);
            uint64_t z = splitmix64(base + group);
            uint32_t v = static_cast<uint32_t>((z >> (byte_in_group * 8)) & 0xFFu);
            uint32_t r_val = v % 127u;
            tile[r][c] = static_cast<int8_t>(static_cast<int32_t>(r_val) - 63);
        }
    }
}