/*
 * PropMiner — Persistent PearlHash Mining Kernel
 *
 * Architecture:
 *   One kernel launch per GPU, kernel runs forever (persistent).
 *   Thread blocks poll a device-side work queue for new jobs.
 *   Per nonce: LCG noise → INT8 GEMM (mma.sync) → XOR reduction → BLAKE3 → target check.
 *
 * Register optimization:
 *   - __launch_bounds__(256, 5) forces ≤64 registers/thread for 5 blocks/SM
 *   - BLAKE3 state uses named registers (s0..s15), NOT arrays
 *   - PTX inline assembly for rotates and 3-input XOR
 *
 * Memory:
 *   - Shared memory for matrix tiles with bank-conflict padding
 *   - Pinned host memory for result buffer (zero-copy readback)
 */

#include <cuda_runtime.h>
#include <cstdint>

#include "../include/propminer_config.h"
#include "../include/work_queue.h"
#include "../include/result_buffer.h"
#include "../include/blake3.cuh"
#include "../include/pow_utils.cuh"
#include "../include/noise_gen.cuh"

/* ── Kernel constants ──────────────────────────────────────────────── */

#define PM_BLOCK_X     256    // 8 warps
#define PM_WARP_SIZE   32
#define PM_WARPS       8      // 8 warps per block

// GEMM tile dimensions — 64x64 so shared memory fits (smem_A=4096, smem_B=4096, total=8192/block)
#define PM_TILE_M      64    // rows
#define PM_TILE_N      64    // cols
#define PM_TILE_K      64    // depth

// Tensor core atom: m16n8k32 (INT8→INT32)
#define PM_MMA_M       16
#define PM_MMA_N        8
#define PM_MMA_K       32

// Shared memory tile sizes
#define PM_SMEM_A_SIZE (PM_TILE_M * PM_TILE_K)
#define PM_SMEM_B_SIZE ((PM_TILE_N + 1) * PM_TILE_K)  // padded for bank conflicts

// K-blocks per tile
#define PM_K_BLOCKS    (PM_TILE_K / PM_MMA_K)  // 2

// Use canonical types from work_queue.h: WorkQueue, WorkItem

/* ── Shared memory ─────────────────────────────────────────────────── */

__shared__ int8_t  smem_A[PM_SMEM_A_SIZE];
__shared__ int8_t  smem_B[PM_SMEM_B_SIZE];

// Block-level work data: only thread 0 claims, then broadcasts via shared mem
__shared__ uint64_t block_seed_a_lo, block_seed_a_hi;
__shared__ uint64_t block_seed_b_lo, block_seed_b_hi;
__shared__ uint8_t  block_sigma[32];
__shared__ uint32_t block_target_nbits;
__shared__ uint64_t block_claimed_nonce;
__shared__ uint32_t block_has_work;

/* ── Pack 4 int8 bytes into one int32 for mma.sync ────────────────── */

__device__ __forceinline__ int pack_4int8_to_int32(int8_t a, int8_t b, int8_t c, int8_t d) {
    return (static_cast<int>(static_cast<uint8_t>(a))        ) |
           (static_cast<int>(static_cast<uint8_t>(b)) <<  8) |
           (static_cast<int>(static_cast<uint8_t>(c)) << 16) |
           (static_cast<int>(static_cast<uint8_t>(d)) << 24);
}

/* ── Main PERSISTENT KERNEL ──────────────────────────────────────────
 *
 * __launch_bounds__(256, 5): max 256 threads/block, min 5 blocks/SM
 * This forces the compiler to use ≤64 registers/thread, enabling
 * 5 thread blocks per SM for higher occupancy (~83% vs 50%).
 *
 * Each block processes work items from the shared queue in a loop.
 * Kernel runs until *shutdown_flag is set by the host.
 */

__global__ void __launch_bounds__(PM_BLOCK_X, 5)
propminer_persistent_kernel(
    WorkQueue*      queue,
    ResultBuffer*   result_buffer,
    const uint32_t* pow_target,
    const uint32_t* pow_key,
    uint64_t*       stats,
    volatile uint32_t* shutdown_flag)
{
    int tid = threadIdx.x;

    while (!*shutdown_flag) {
        // Only thread 0 claims a work item for the block
        if (tid == 0) {
            uint64_t tail = queue->tail;
            uint64_t head = queue->head;

            if (head >= tail) {
                block_has_work = 0;
            } else {
                uint64_t claimed = atomicAdd((unsigned long long*)&queue->head, 1);
                if (claimed >= tail) {
                    block_has_work = 0;
                } else {
                    // Memory fence: ensure we see the host's WorkItem write before tail increment
                    __threadfence_device();

                    uint64_t slot = claimed % queue->capacity;
                    WorkItem* work = reinterpret_cast<WorkItem*>
                        ((char*)queue + sizeof(WorkQueue) + slot * sizeof(WorkItem));

                    // Derive seeds from sigma + nonce
                    uint64_t s_a_lo = 0, s_a_hi = 0, s_b_lo = 0, s_b_hi = 0;
                    for (int i = 0; i < 8; i++) {
                        s_a_lo |= (uint64_t)work->sigma[i] << (i * 8);
                        s_a_hi |= (uint64_t)work->sigma[8 + i] << (i * 8);
                        s_b_lo |= (uint64_t)work->b_seed[i] << (i * 8);
                        s_b_hi |= (uint64_t)work->b_seed[8 + i] << (i * 8);
                    }

                    block_seed_a_lo = s_a_lo;
                    block_seed_a_hi = s_a_hi;
                    block_seed_b_lo = s_b_lo;
                    block_seed_b_hi = s_b_hi;
                    block_target_nbits = work->target_nbits;
                    block_claimed_nonce = claimed;
                    for (int i = 0; i < 32; i++)
                        block_sigma[i] = work->sigma[i];
                    block_has_work = 1;
                }
            }
        }
        __syncthreads();

        if (!block_has_work) {
            __syncthreads();
            continue;
        }

        // Use block-level shared variables
        uint64_t seed_a_lo = block_seed_a_lo;
        uint64_t seed_a_hi = block_seed_a_hi;
        uint64_t seed_b_lo = block_seed_b_lo;
        uint64_t seed_b_hi = block_seed_b_hi;
        uint64_t claimed = block_claimed_nonce;
        uint32_t work_target_nbits = block_target_nbits;
        uint8_t  work_sigma[32];
        for (int i = 0; i < 32; i++) work_sigma[i] = block_sigma[i];

        // ── Load A tile into shared memory ──────────────────────
        {
            uint64_t base_seed = splitmix64(seed_a_lo ^ splitmix64(seed_a_hi));
            for (int i = tid; i < PM_SMEM_A_SIZE; i += PM_BLOCK_X) {
                uint64_t idx = i;
                uint64_t group = idx / 8;
                int byte_in_group = static_cast<int>(idx % 8);
                uint64_t z = splitmix64(base_seed + group);
                uint32_t v = static_cast<uint32_t>((z >> (byte_in_group * 8)) & 0xFFu);
                uint32_t r = v % 127u;
                smem_A[i] = static_cast<int8_t>(static_cast<int32_t>(r) - 63);
            }
        }

        // ── Load B tile into shared memory (padded layout) ──────
        {
            uint64_t base_seed = splitmix64(seed_b_lo ^ splitmix64(seed_b_hi));
            for (int i = tid; i < PM_TILE_N * PM_TILE_K; i += PM_BLOCK_X) {
                uint64_t idx = i;
                uint64_t group = idx / 8;
                int byte_in_group = static_cast<int>(idx % 8);
                uint64_t z = splitmix64(base_seed + group);
                uint32_t v = static_cast<uint32_t>((z >> (byte_in_group * 8)) & 0xFFu);
                uint32_t r = v % 127u;
                int row = i / PM_TILE_K;
                int col = i % PM_TILE_K;
                smem_B[row * (PM_TILE_K + 1) + col] = static_cast<int8_t>(static_cast<int32_t>(r) - 63);
            }
        }

        __syncthreads();

        // ── GEMM: C = A × Bᵀ using INT8 Tensor Cores ────────────
        //
        // Each warp computes one 16×8 output tile.
        // 8 warps → 16 rows × 64 cols per block.
        //
        // Initialize 8 accumulator registers (m16n8k32 produces 8 outputs per warp)
        int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        int c4 = 0, c5 = 0, c6 = 0, c7 = 0;

        int warp_id = tid / PM_WARP_SIZE;
        int lane_id = tid % PM_WARP_SIZE;

        // K-dimension loop: process PM_TILE_K in chunks of PM_MMA_K
        for (int k = 0; k < PM_TILE_K; k += PM_MMA_K) {
            // Each warp loads its A fragment (16 rows × 32 cols = 512 int8)
            // from shared memory into register fragments
            int a_row = warp_id * PM_MMA_M;

            // Pack A tile rows into int32 fragments (4 int8 per register)
            // Each lane loads one row of 4 int8s = 1 register
            int a_frag[4] = {};  // 4 registers per warp for A
            if (lane_id < 4) {
                int row_off = lane_id * 4;  // each lane covers 4 rows of the 16-row tile
                a_frag[lane_id] = pack_4int8_to_int32(
                    smem_A[(a_row + row_off) * PM_TILE_K + k],
                    smem_A[(a_row + row_off) * PM_TILE_K + k + 1],
                    smem_A[(a_row + row_off) * PM_TILE_K + k + 2],
                    smem_A[(a_row + row_off) * PM_TILE_K + k + 3]);
            }

            // Pack B tile (transposed: B[col][k] → we need Bᵀ[k][col])
            int b_frag[2] = {};
            if (lane_id < 2) {
                int col_off = lane_id * 4;
                b_frag[lane_id] = pack_4int8_to_int32(
                    smem_B[col_off * (PM_TILE_K + 1) + k],
                    smem_B[col_off * (PM_TILE_K + 1) + k + 1],
                    smem_B[col_off * (PM_TILE_K + 1) + k + 2],
                    smem_B[col_off * (PM_TILE_K + 1) + k + 3]);
            }

            // Broadcast fragments to all lanes in the warp
            a_frag[0] = __shfl_sync(0xffffffff, a_frag[0], 0);
            a_frag[1] = __shfl_sync(0xffffffff, a_frag[1], 1);
            a_frag[2] = __shfl_sync(0xffffffff, a_frag[2], 2);
            a_frag[3] = __shfl_sync(0xffffffff, a_frag[3], 3);
            b_frag[0] = __shfl_sync(0xffffffff, b_frag[0], 0);
            b_frag[1] = __shfl_sync(0xffffffff, b_frag[1], 1);

            // Tensor core mma.sync: C += A × B (m16n8k32 INT8→INT32)
            // A: 4 registers, B: 2 registers, C: 8 registers (read-write via +r)
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13}, {%0,%1,%2,%3,%4,%5,%6,%7};"
                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3),
                  "+r"(c4), "+r"(c5), "+r"(c6), "+r"(c7)
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                  "r"(b_frag[0]), "r"(b_frag[1])
            );
        }

        // ── XOR reduction of GEMM accumulator ────────────────────
        uint32_t transcript[B3_MSG_BLOCK_U32] = {};

        // Initialize transcript from sigma
        for (int i = 0; i < B3_MSG_BLOCK_U32; i++) {
            int idx = (i * 4) % 32;
            transcript[i] = ((uint32_t)work_sigma[idx]       ) |
                            ((uint32_t)work_sigma[idx + 1] << 8)  |
                            ((uint32_t)work_sigma[idx + 2] << 16) |
                            ((uint32_t)work_sigma[idx + 3] << 24);
        }

        // XOR-reduce the 8 MMA accumulator outputs into transcript (m16n8k32 produces 8 values)
        uint32_t c_arr[] = {c0, c1, c2, c3, c4, c5, c6, c7};

        for (int i = 0; i < 8; i++) {
            transcript[i] = pow_rotl_xor(transcript[i], c_arr[i]);
        }

        // ── BLAKE3 keyed hash ────────────────────────────────────
        uint32_t chaining[B3_CHAINING_VALUE_U32];
        for (int i = 0; i < B3_CHAINING_VALUE_U32; i++)
            chaining[i] = pow_key[i];

        blake3::compress_msg_block_u32(transcript, chaining, blake3::B3_PARAMS_KEYED_SINGLE);

        // ── Target comparison ────────────────────────────────────
        bool found = true;
        for (int i = B3_CHAINING_VALUE_U32 - 1; i >= 0; i--) {
            if (chaining[i] > pow_target[i]) { found = false; break; }
            if (chaining[i] < pow_target[i])  break;
        }

        // ── Write share if found ─────────────────────────────────
        if (found) {
            uint32_t slot = atomicAdd(&result_buffer->write_pos, 1);
            if (slot >= result_buffer->capacity) {
                atomicExch(&result_buffer->write_pos, 0);
                continue;
            }
            ShareResult* results = reinterpret_cast<ShareResult*>
                ((char*)result_buffer + sizeof(ResultBuffer));
            ShareResult* sr = &results[slot];
            for (int i = 0; i < 32; i++) sr->sigma[i] = work_sigma[i];
            sr->nonce        = claimed;
            for (int i = 0; i < 32; i++) sr->hash[i] = static_cast<uint8_t>((chaining[i / 4] >> ((i % 4) * 8)) & 0xFFu);
            sr->tile_row     = warp_id * PM_MMA_M;
            sr->tile_col     = 0;
            sr->target_nbits = work_target_nbits;
            atomicAdd((unsigned long long*)&stats[0], 1);
        }

        // Update computed counter
        atomicAdd((unsigned long long*)&stats[2], 1);
    }
}