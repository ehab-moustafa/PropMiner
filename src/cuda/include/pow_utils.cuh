#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "blake3.cuh"
#include "result_buffer.h"

/* ── PoW target checking & transcript accumulation ────────────────────
 *
 * XOR tree reduction uses lop3.b32 (3-input XOR) for maximum throughput.
 * TileHashAccumulator keeps transcript elements in registers during the
 * k-block loop, avoiding shared/global memory in the hot path.
 */

static constexpr int HASH_ACCUMULATE_ROTATION = 13;

/* ── PTX primitives (shared with blake3.cuh but used here independently) */

__device__ __forceinline__ uint32_t pow_xor3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;"
                 : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ uint32_t pow_rotl_xor(uint32_t x, uint32_t y) {
    uint32_t rotated;
    /* shf.l.wrap.b32 rotates left by immediate */
    asm volatile("shf.l.wrap.b32 %0, %1, %1, %2;"
                 : "=r"(rotated) : "r"(x), "n"(HASH_ACCUMULATE_ROTATION));
    return rotated ^ y;
}

/* ── XOR tree reduction for N uint32 elements ────────────────────────
 *
 * Reduces an array of N uint32_t values to a single uint32_t using
 * a ternary XOR tree (groups of 3 → 1 via lop3). For N not divisible
 * by 3, trailing elements pass through.
 */

template <int N>
__device__ __forceinline__ uint32_t xor_reduce(const uint32_t* __restrict__ data) {
    if constexpr (N == 1) return data[0];
    if constexpr (N == 2) return data[0] ^ data[1];
    if constexpr (N == 3) return pow_xor3(data[0], data[1], data[2]);
    if constexpr (N == 4) return data[0] ^ pow_xor3(data[1], data[2], data[3]);

    if constexpr (N <= 12) {
        /* Small enough to keep all intermediates in registers */
        constexpr int triplets = N / 3;
        constexpr int remainder = N % 3;
        uint32_t layer[triplets + remainder];
#pragma unroll
        for (int i = 0; i < triplets; ++i)
            layer[i] = pow_xor3(data[3 * i], data[3 * i + 1], data[3 * i + 2]);
#pragma unroll
        for (int i = 0; i < remainder; ++i)
            layer[triplets + i] = data[triplets * 3 + i];
        return xor_reduce<triplets + remainder>(layer);
    }

    /* Larger N: iterative two-pass reduction */
    constexpr int triplets = N / 3;
    constexpr int remainder = N % 3;
    uint32_t layer[triplets + remainder];
#pragma unroll
    for (int i = 0; i < triplets; ++i)
        layer[i] = pow_xor3(data[3 * i], data[3 * i + 1], data[3 * i + 2]);
#pragma unroll
    for (int i = 0; i < remainder; ++i)
        layer[triplets + i] = data[triplets * 3 + i];
    return xor_reduce<triplets + remainder>(layer);
}

/* ── XOR reduction of an int32 matrix tile via reinterpret ─────────── */

template <int TILE_M, int TILE_N>
__device__ __forceinline__ uint32_t
xor_reduce_tile(const int32_t c_reg[TILE_M / 4][TILE_N / 2]) {
    /* reinterpret as uint32 array — same bit width */
    constexpr int total = (TILE_M / 4) * (TILE_N / 2);
    const uint32_t* data = reinterpret_cast<const uint32_t*>(c_reg);
    return xor_reduce<total>(data);
}

/* ── TileHashAccumulator ─────────────────────────────────────────────
 *
 * Keeps a window of transcript elements in registers. Call accumulate()
 * periodically during the k-block loop, then writeback() at tile end.
 */

template <int ACCUMS_PER_TILE>
struct TileHashAccumulator {
    uint32_t m_tile_transcript[ACCUMS_PER_TILE];
    int      m_accum_idx = 0;

    __device__ __forceinline__ void reset(const uint32_t* transcript, int offset) {
        m_accum_idx = 0;
#pragma unroll
        for (int i = 0; i < ACCUMS_PER_TILE; ++i)
            m_tile_transcript[i] = transcript[offset + i];
    }

    __device__ __forceinline__ void accumulate(uint32_t hash) {
        if (m_accum_idx < ACCUMS_PER_TILE) {
            m_tile_transcript[m_accum_idx] = pow_rotl_xor(m_tile_transcript[m_accum_idx], hash);
            ++m_accum_idx;
        }
    }

    __device__ __forceinline__ void writeback(uint32_t* transcript, int offset) {
#pragma unroll
        for (int i = 0; i < ACCUMS_PER_TILE; ++i)
            transcript[offset + i] = m_tile_transcript[i];
    }
};

/* ── check_pow_target ────────────────────────────────────────────────
 *
 * Compresses the transcript with BLAKE3 (keyed) and checks if the hash
 * is below the difficulty target. Little-endian uint256 comparison:
 * hash < target means MSW is compared first.
 */

__device__ __forceinline__ bool
check_pow_target(const uint32_t transcript[B3_MSG_BLOCK_U32],
                 const uint32_t* __restrict__ pow_key,
                 const uint32_t* __restrict__ pow_target) {
    uint32_t chaining[B3_CHAINING_VALUE_U32];
#pragma unroll
    for (int i = 0; i < B3_CHAINING_VALUE_U32; ++i)
        chaining[i] = pow_key[i];

    blake3::compress_msg_block_u32(transcript, chaining, blake3::B3_PARAMS_KEYED_SINGLE);

    /* Compare chaining (hash) against target: hash < target (little-endian big-int) */
    bool found = true;
#pragma unroll
    for (int i = B3_CHAINING_VALUE_U32 - 1; i >= 0; --i) {
        if (chaining[i] > pow_target[i]) { found = false; break; }
        if (chaining[i] < pow_target[i])  break;
    }
    return found;
}

/* ── write_share_result ──────────────────────────────────────────────
 *
 * Atomically appends a share to the zero-copy result buffer.
 * Returns the slot index (for tile coordinate reporting) or -1 if full.
 */

__device__ __forceinline__ int
write_share_result(ShareResult* __restrict__ buffer,
                   uint32_t capacity,
                   const uint8_t  sigma[32],
                   uint64_t       nonce,
                   const uint8_t  hash[32],
                   int32_t        tile_row,
                   int32_t        tile_col,
                   uint32_t       target_nbits) {
    int slot = atomicAdd(&buffer->write_pos, 1);
    if (slot >= (int)capacity) {
        atomicExch(&buffer->write_pos, 0);
        return -1;
    }

    ShareResult* sr = &buffer[slot];
#pragma unroll
    for (int i = 0; i < 32; ++i) sr->sigma[i] = sigma[i];
    sr->nonce        = nonce;
#pragma unroll
    for (int i = 0; i < 32; ++i) sr->hash[i] = hash[i];
    sr->tile_row     = tile_row;
    sr->tile_col     = tile_col;
    sr->target_nbits = target_nbits;
    return slot;
}