#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "propminer_config.h"

/* ── BLAKE3 GPU implementation using unrolled named registers ──────────────
 *
 * All 16 state variables are individual uint32_t registers, NOT arrays.
 * Rotate operations use PTX shf.l.wrap.b32. 3-input XOR uses lop3.b32.
 * 7 rounds: 6 with permutation + 1 final without permutation.
 */

namespace blake3 {

/* BLAKE3 IV constants in device constant memory */
__device__ __constant__ uint32_t B3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

/* Compress parameters */
struct __align__(16) CompressParams {
    uint64_t counter;
    uint32_t block_len;
    uint32_t flags;
};

/* Pre-initialized compress params for single-block keyed hash */
__device__ __constant__ CompressParams B3_PARAMS_KEYED_SINGLE = {
    .counter   = 0,
    .block_len = B3_MSG_BLOCK_BYTES,
    .flags     = B3_FLAGS_KEYED_SINGLE
};

/* ── PTX primitives ───────────────────────────────────────────────── */

__device__ __forceinline__ uint32_t b3_rotr(uint32_t x, uint32_t n) {
    uint32_t r;
    /* shf.l.wrap.b32: rotate left; for rotate-right by n, use (32-n) */
    asm volatile("shf.l.wrap.b32 %0, %1, %1, %2;"
                 : "=r"(r) : "r"(x), "r"(32u - n));
    return r;
}

__device__ __forceinline__ uint32_t b3_xor3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    /* lop3.b32 LUT 0x96 = a ^ b ^ c */
    asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;"
                 : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

/* ── One G operation (adds msg words mx, my) ──────────────────────── */
#define B3_G(a_var, b_var, c_var, d_var, mx, my) \
    do {                                          \
        (a_var) = (a_var) + (b_var) + (mx);       \
        (d_var) = b3_rotr((d_var) ^ (a_var), 16); \
        (c_var) = (c_var) + (d_var);              \
        (b_var) = b3_rotr((b_var) ^ (c_var), 12); \
        (a_var) = (a_var) + (b_var) + (my);       \
        (d_var) = b3_rotr((d_var) ^ (a_var), 8);  \
        (c_var) = (c_var) + (d_var);              \
        (b_var) = b3_rotr((b_var) ^ (c_var), 7);  \
    } while (0)

/* ── Full BLAKE3 round using 16 named state registers ─────────────── */
#define B3_ROUND(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15, \
                 b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15)  \
    do {                                                                   \
        B3_G(s0, s4, s8,  s12, b0,  b1);                                  \
        B3_G(s1, s5, s9,  s13, b2,  b3);                                  \
        B3_G(s2, s6, s10, s14, b4,  b5);                                  \
        B3_G(s3, s7, s11, s15, b6,  b7);                                  \
        B3_G(s0, s5, s10, s15, b8,  b9);                                  \
        B3_G(s1, s6, s11, s12, b10, b11);                                 \
        B3_G(s2, s7, s8,  s13, b12, b13);                                 \
        B3_G(s3, s4, s9,  s14, b14, b15);                                 \
    } while (0)

/* ── BLAKE3 sigma permutation of the message block ────────────────── */
#define B3_PERMUTE(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15) \
    do {                                                                    \
        uint32_t _t0  = (b0);  uint32_t _t1  = (b1);  uint32_t _t2  = (b2); \
        uint32_t _t3  = (b3);  uint32_t _t4  = (b4);  uint32_t _t5  = (b5); \
        uint32_t _t6  = (b6);  uint32_t _t7  = (b7);  uint32_t _t8  = (b8); \
        uint32_t _t9  = (b9);  uint32_t _t10 = (b10); uint32_t _t11 = (b11);\
        uint32_t _t12 = (b12); uint32_t _t13 = (b13); uint32_t _t14 = (b14);\
        uint32_t _t15 = (b15);                                                  \
        (b0)  = _t2;   (b1)  = _t6;   (b2)  = _t3;   (b3)  = _t10;            \
        (b4)  = _t7;   (b5)  = _t0;   (b6)  = _t4;   (b7)  = _t13;            \
        (b8)  = _t1;   (b9)  = _t11;  (b10) = _t12;  (b11) = _t5;             \
        (b12) = _t9;   (b13) = _t14;  (b14) = _t15;  (b15) = _t8;             \
    } while (0)

/* ── compress_msg_block_u32 ─────────────────────────────────────────
 *
 * Inputs:
 *   msg[16]  — 64-byte message block (little-endian u32s)
 *   chaining[8] — input chaining value (read-write; output on return)
 *   params    — counter / block_len / flags
 *
 * All 16 state + 16 message + 16 temp variables are named registers.
 */
__device__ __forceinline__ void
compress_msg_block_u32(const uint32_t msg[16],
                       uint32_t chaining[8],
                       const CompressParams& params) {
    /* Load message block into named registers */
    uint32_t b0  = msg[0],  b1  = msg[1],  b2  = msg[2],  b3  = msg[3];
    uint32_t b4  = msg[4],  b5  = msg[5],  b6  = msg[6],  b7  = msg[7];
    uint32_t b8  = msg[8],  b9  = msg[9],  b10 = msg[10], b11 = msg[11];
    uint32_t b12 = msg[12], b13 = msg[13], b14 = msg[14], b15 = msg[15];

    /* Initialize state: s0..s7 = chaining, s8..s15 = IV + params */
    uint32_t s0  = chaining[0], s1  = chaining[1], s2  = chaining[2], s3  = chaining[3];
    uint32_t s4  = chaining[4], s5  = chaining[5], s6  = chaining[6], s7  = chaining[7];
    s8  = B3_IV[0]; s9  = B3_IV[1]; s10 = B3_IV[2]; s11 = B3_IV[3];
    s12 = static_cast<uint32_t>(params.counter);
    s13 = static_cast<uint32_t>(params.counter >> 32);
    s14 = params.block_len;
    s15 = params.flags;

    /* 6 rounds with permutation */
    B3_ROUND(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,
             b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15);
    B3_PERMUTE(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15);

    B3_ROUND(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,
             b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15);
    B3_PERMUTE(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15);

    B3_ROUND(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,
             b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15);
    B3_PERMUTE(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15);

    B3_ROUND(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,
             b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15);
    B3_PERMUTE(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15);

    B3_ROUND(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,
             b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15);
    B3_PERMUTE(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15);

    B3_ROUND(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,
             b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15);
    B3_PERMUTE(b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15);

    /* Final round without permutation */
    B3_ROUND(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,
             b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15);

    /* Output: chaining[i] = s[i] ^ s[i+8] */
    chaining[0]  = s0  ^ s8;
    chaining[1]  = s1  ^ s9;
    chaining[2]  = s2  ^ s10;
    chaining[3]  = s3  ^ s11;
    chaining[4]  = s4  ^ s12;
    chaining[5]  = s5  ^ s13;
    chaining[6]  = s6  ^ s14;
    chaining[7]  = s7  ^ s15;
}

}  // namespace blake3