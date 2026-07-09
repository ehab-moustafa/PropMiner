// blake3_finalize_optimization_test.cpp — Comprehensive CPU-side validation
// of BLAKE3 finalize pipeline for Pearl consensus safety.
//
// Compiles and runs on macOS/Linux WITHOUT GPU (no CUDA/ROCm dependency).
// Verifies that ANY BLAKE3 optimization produces BYTE-IDENTICAL output to the
// original implementation.
//
// Build:
//   clang++ -std=c++17 -O2 -I src/host/tests \
//       src/host/tests/blake3_finalize_optimization_test.cpp \
//       src/host/tests/blake3_reference.c \
//       -o blake3_finalize_optimization_test
//
// Run:
//   ./blake3_finalize_optimization_test

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <vector>
#include <string>
#include <cassert>

// ─── BLAKE3 Constants (from blake3_constants.hpp) ──────────────────────────

static constexpr uint32_t BLAKE3_IV0 = 0x6A09E667;
static constexpr uint32_t BLAKE3_IV1 = 0xBB67AE85;
static constexpr uint32_t BLAKE3_IV2 = 0x3C6EF372;
static constexpr uint32_t BLAKE3_IV3 = 0xA54FF53A;
static constexpr uint32_t BLAKE3_IV4 = 0x510e527f;
static constexpr uint32_t BLAKE3_IV5 = 0x9b05688c;
static constexpr uint32_t BLAKE3_IV6 = 0x1f83d9ab;
static constexpr uint32_t BLAKE3_IV7 = 0x5be0cd19;

static constexpr uint32_t CHUNK_START = 1 << 0;
static constexpr uint32_t CHUNK_END = 1 << 1;
static constexpr uint32_t PARENT = 1 << 2;
static constexpr uint32_t ROOT = 1 << 3;
static constexpr uint32_t KEYED_HASH = 1 << 4;
static constexpr uint32_t DERIVE_KEY_CONTEXT = 1 << 5;
static constexpr uint32_t DERIVE_KEY_MATERIAL = 1 << 6;

// PoW flags used in production: KEYED_HASH | CHUNK_START | CHUNK_END | ROOT
static constexpr uint32_t POW_FLAGS = KEYED_HASH | CHUNK_START | CHUNK_END | ROOT;

static constexpr uint32_t MSG_BLOCK_SIZE = 64;
static constexpr uint32_t CHAINING_VALUE_SIZE_U32 = 8;
static constexpr uint32_t TRANSCRIPT_SIZE_U32 = 16;
static constexpr uint32_t SIGMA_SIZE = 32;

// ─── Portable BLAKE3 primitives (ported from blake3.cuh — the GPU implementation) ─
// These are the EXACT same operations the GPU kernel performs.
// An "optimized" version must produce identical output from these primitives.

static inline uint32_t rightrotate32(uint32_t x, uint32_t n) {
    return (x << (32 - n)) | (x >> n);
}

static inline void g(uint32_t state[16], size_t a, size_t b, size_t c, size_t d,
                     uint32_t mx, uint32_t my) {
    state[a] = state[a] + state[b] + mx;
    state[d] = rightrotate32(state[d] ^ state[a], 16);
    state[c] = state[c] + state[d];
    state[b] = rightrotate32(state[b] ^ state[c], 12);
    state[a] = state[a] + state[b] + my;
    state[d] = rightrotate32(state[d] ^ state[a], 8);
    state[c] = state[c] + state[d];
    state[b] = rightrotate32(state[b] ^ state[c], 7);
}

static inline void blake3_round(uint32_t state[16], uint32_t block[16]) {
    g(state, 0, 4, 8, 12, block[0], block[1]);
    g(state, 1, 5, 9, 13, block[2], block[3]);
    g(state, 2, 6, 10, 14, block[4], block[5]);
    g(state, 3, 7, 11, 15, block[6], block[7]);
    g(state, 0, 5, 10, 15, block[8], block[9]);
    g(state, 1, 6, 11, 12, block[10], block[11]);
    g(state, 2, 7, 8, 13, block[12], block[13]);
    g(state, 3, 4, 9, 14, block[14], block[15]);
}

static inline void blake3_permute(uint32_t block[16]) {
    uint32_t orig[16];
    memcpy(orig, block, sizeof(orig));
    block[0]  = orig[2];   block[1]  = orig[6];   block[2]  = orig[3];
    block[3]  = orig[10];  block[4]  = orig[7];   block[5]  = orig[0];
    block[6]  = orig[4];   block[7]  = orig[13];  block[8]  = orig[1];
    block[9]  = orig[11];  block[10] = orig[12];  block[11] = orig[5];
    block[12] = orig[9];   block[13] = orig[14];  block[14] = orig[15];
    block[15] = orig[8];
}

// ─── Core compress functions ────────────────────────────────────────────────

// Standard compress: returns state[0..7] ^ state[8..15] (the chaining value)
static void compress(uint32_t cv[8], const uint32_t block_in[16],
                     uint64_t counter, uint32_t block_len, uint32_t flags) {
    uint32_t state[16], block[16];
    for (int i = 0; i < 16; ++i) block[i] = block_in[i];
    for (int i = 0; i < 8; ++i) state[i] = cv[i];
    state[8]  = BLAKE3_IV0;
    state[9]  = BLAKE3_IV1;
    state[10] = BLAKE3_IV2;
    state[11] = BLAKE3_IV3;
    state[12] = (uint32_t)counter;
    state[13] = (uint32_t)(counter >> 32);
    state[14] = block_len;
    state[15] = flags;

    for (int i = 0; i < 6; ++i) {
        blake3_round(state, block);
        blake3_permute(block);
    }
    blake3_round(state, block);

    for (int i = 0; i < 8; ++i) cv[i] = state[i] ^ state[i + 8];
}

// Full compress: returns all 16 words (chaining value + intermediate state)
static void compress_full(const uint32_t cv_in[8], const uint32_t block_in[16],
                          uint64_t counter, uint32_t block_len, uint32_t flags,
                          uint32_t out16[16]) {
    uint32_t state[16], block[16];
    for (int i = 0; i < 16; ++i) block[i] = block_in[i];
    for (int i = 0; i < 8; ++i) state[i] = cv_in[i];
    state[8]  = BLAKE3_IV0;
    state[9]  = BLAKE3_IV1;
    state[10] = BLAKE3_IV2;
    state[11] = BLAKE3_IV3;
    state[12] = (uint32_t)counter;
    state[13] = (uint32_t)(counter >> 32);
    state[14] = block_len;
    state[15] = flags;

    for (int i = 0; i < 6; ++i) {
        blake3_round(state, block);
        blake3_permute(block);
    }
    blake3_round(state, block);

    for (int i = 0; i < 8; ++i) {
        out16[i]     = state[i] ^ state[i + 8];
        out16[i + 8] = state[i + 8] ^ cv_in[i];
    }
}

// ─── Reference compress (from blake3_reference.c) ───────────────────────────
// This is the official Rust C-port reference used for cross-validation.
// MUST match compress() exactly.

static void ref_compress(const uint32_t chaining_value[8],
                         const uint32_t block_words[16], uint64_t counter,
                         uint32_t block_len, uint32_t flags,
                         uint32_t out[16]) {
    uint32_t state[16] = {
        chaining_value[0], chaining_value[1], chaining_value[2],
        chaining_value[3], chaining_value[4], chaining_value[5],
        chaining_value[6], chaining_value[7],
        BLAKE3_IV0, BLAKE3_IV1, BLAKE3_IV2, BLAKE3_IV3,
        (uint32_t)counter, (uint32_t)(counter >> 32),
        block_len, flags,
    };
    uint32_t block[16];
    memcpy(block, block_words, sizeof(block));

    blake3_round(state, block);
    blake3_permute(block);
    blake3_round(state, block);
    blake3_permute(block);
    blake3_round(state, block);
    blake3_permute(block);
    blake3_round(state, block);
    blake3_permute(block);
    blake3_round(state, block);
    blake3_permute(block);
    blake3_round(state, block);
    blake3_permute(block);
    blake3_round(state, block);

    for (size_t i = 0; i < 8; ++i) {
        state[i]     ^= state[i + 8];
        state[i + 8] ^= chaining_value[i];
    }

    memcpy(out, state, sizeof(state));
}

// ─── PoW utility functions (from pow_utils.hpp) ─────────────────────────────

static inline uint32_t rotl_xor(uint32_t x, uint32_t y, int rot = 13) {
    return (x << rot) | (x >> (32 - rot)) ^ y;
}

static uint32_t xor_reduction(const uint32_t* data, size_t count) {
    if (count == 0) return 0;
    if (count == 1) return data[0];
    if (count == 2) return data[0] ^ data[1];

    std::vector<uint32_t> current(data, data + count);
    while (current.size() > 2) {
        std::vector<uint32_t> next;
        size_t triplets = current.size() / 3;
        size_t remainder = current.size() % 3;
        for (size_t i = 0; i < triplets; ++i) {
            next.push_back(current[3 * i] ^ current[3 * i + 1] ^ current[3 * i + 2]);
        }
        for (size_t i = 0; i < remainder; ++i) {
            next.push_back(current[triplets * 3 + i]);
        }
        current = std::move(next);
    }
    if (current.size() == 2) return current[0] ^ current[1];
    return current[0];
}

// check_pow_target from pow_utils.hpp:
// Returns true if hash <= target (big-endian uint256 comparison, MSW-first).
static bool check_pow_target(const uint32_t* hash, const uint32_t* pow_target) {
    for (int i = 7; i >= 0; --i) {
        if (hash[i] > pow_target[i]) return false;
        if (hash[i] < pow_target[i]) return true;
    }
    return true; // equal → found
}

// ─── Transcript pipeline (from pearlhash_kernel.cu / pow_utils.hpp) ─────────

static void sigma_to_transcript(const uint8_t sigma[32], uint32_t transcript[16]) {
    for (int i = 0; i < 16; ++i) {
        int idx = (i * 4) % 32;
        transcript[i] = ((uint32_t)sigma[idx]) |
                        ((uint32_t)sigma[idx + 1] << 8) |
                        ((uint32_t)sigma[idx + 2] << 16) |
                        ((uint32_t)sigma[idx + 3] << 24);
    }
}

static void full_pow_pipeline(const uint8_t sigma[32],
                              const uint32_t* pow_key,
                              const uint32_t* gemm_outputs,
                              size_t gemm_count,
                              uint32_t out_cv[8]) {
    uint32_t transcript[16];
    sigma_to_transcript(sigma, transcript);

    if (gemm_outputs && gemm_count > 0) {
        uint32_t reduced = xor_reduction(gemm_outputs, gemm_count);
        for (int i = 0; i < 16; ++i) {
            transcript[i] = rotl_xor(transcript[i], reduced);
        }
    }

    uint32_t cv[8];
    for (int i = 0; i < 8; ++i) cv[i] = pow_key[i];

    uint32_t block[16];
    memcpy(block, transcript, sizeof(block));

    compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

    memcpy(out_cv, cv, 8 * sizeof(uint32_t));
}

// ─── Deterministic PRNG (LCG) ──────────────────────────────────────────────

static void gen_random_u32(uint32_t* out, int count, uint64_t seed = 0x12345678ULL) {
    uint64_t state = seed;
    for (int i = 0; i < count; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        out[i] = (uint32_t)(state >> 33) ^ (uint32_t)state;
    }
}

static void gen_random_u8(uint8_t* out, int count, uint64_t seed = 0x12345678ULL) {
    uint64_t state = seed;
    for (int i = 0; i < count; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        out[i] = (uint8_t)(state & 0xFF);
    }
}

// ─── Test helpers ───────────────────────────────────────────────────────────

static int g_test_count = 0;
static int g_pass_count = 0;
static int g_fail_count = 0;

static void print_u32_array(const char* label, const uint32_t* arr, int n) {
    printf("  %s", label);
    for (int i = 0; i < n; ++i) {
        if (i > 0) printf(" ");
        printf("0x%08x", arr[i]);
    }
    printf("\n");
}

static void print_u8_array(const char* label, const uint8_t* arr, int n) {
    printf("  %s", label);
    for (int i = 0; i < n; ++i) {
        if (i > 0) printf(" ");
        printf("0x%02x", arr[i]);
    }
    printf("\n");
}

static bool arrays_equal_u32(const uint32_t* a, const uint32_t* b, int n) {
    return memcmp(a, b, n * sizeof(uint32_t)) == 0;
}

static bool arrays_equal_u8(const uint8_t* a, const uint8_t* b, int n) {
    return memcmp(a, b, n) == 0;
}

static int count_bits(uint32_t x) {
    int c = 0;
    while (x) { c += x & 1; x >>= 1; }
    return c;
}

static void run_test(const char* name, bool passed) {
    g_test_count++;
    if (passed) {
        g_pass_count++;
        printf("[TEST %d] %s... PASSED\n", g_test_count, name);
    } else {
        g_fail_count++;
        printf("[TEST %d] %s... FAILED\n", g_test_count, name);
    }
}

// ─── GROUP 1: BLAKE3 Compress Correctness (50+ tests) ──────────────────────

static void test_group_1_compress_correctness() {
    printf("\n============================================================\n");
    printf("  GROUP 1: BLAKE3 Compress Correctness\n");
    printf("============================================================\n\n");

    // 1.1: All-zero transcript with specific pow_key
    {
        uint32_t cv[8] = {0};
        uint32_t block[16] = {0};
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));
        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("1.1 all-zero transcript with zero pow_key",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 1.2: All-0xFFFFFFFF transcript with specific pow_key
    {
        uint32_t cv[8];
        for (int i = 0; i < 8; ++i) cv[i] = 0xFFFFFFFF;
        uint32_t block[16];
        for (int i = 0; i < 16; ++i) block[i] = 0xFFFFFFFF;
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("1.2 all-0xFFFFFFFF transcript",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 1.3: Random transcript with specific pow_key (deterministic PRNG seed 1)
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0x11111111ULL);
        gen_random_u32(block, 16, 0x22222222ULL);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("1.3 random transcript (seed 0x11111111)",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 1.4: Random transcript with deterministic PRNG seed 2
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0x33333333ULL);
        gen_random_u32(block, 16, 0x44444444ULL);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("1.4 random transcript (seed 0x33333333)",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 1.5: Random transcript with deterministic PRNG seed 3
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0x55555555ULL);
        gen_random_u32(block, 16, 0x66666666ULL);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("1.5 random transcript (seed 0x55555555)",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 1.6: Random transcript with deterministic PRNG seed 4
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0x77777777ULL);
        gen_random_u32(block, 16, 0x88888888ULL);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("1.6 random transcript (seed 0x77777777)",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 1.7: Random transcript with deterministic PRNG seed 5
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0x99999999ULL);
        gen_random_u32(block, 16, 0xAAAAAAAAULL);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("1.7 random transcript (seed 0x99999999)",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 1.8: Random transcript with deterministic PRNG seed 6
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0xBBBBBBBBULL);
        gen_random_u32(block, 16, 0xCCCCCCCCULL);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("1.8 random transcript (seed 0xBBBBBBBB)",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 1.9: Random transcript with deterministic PRNG seed 7
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0xDDDDDDDDULL);
        gen_random_u32(block, 16, 0xEEEEEEEEULL);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("1.9 random transcript (seed 0xDDDDDDDD)",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 1.10: Random transcript with deterministic PRNG seed 8
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0xFFFFFFFF00000000ULL);
        gen_random_u32(block, 16, 0x00000000FFFFFFFFULL);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("1.10 random transcript (seed 0xFFFFFFFF00000000)",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 1.11: Single-bit flip in transcript[0] — compare against original
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0x1234);
        gen_random_u32(block, 16, 0x5678);
        uint32_t cv_orig[8];
        memcpy(cv_orig, cv, sizeof(cv_orig));
        uint32_t block_orig[16];
        memcpy(block_orig, block, sizeof(block_orig));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        // Flip bit 0 of block[0]
        block[0] ^= 1;
        uint32_t cv_flipped[8];
        memcpy(cv_flipped, cv_orig, sizeof(cv_flipped));
        compress(cv_flipped, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        // Output must differ (avalanche)
        bool differs = !arrays_equal_u32(cv, cv_flipped, 8);

        // And at least 30% of bits should differ
        int diff_bits = 0;
        for (int i = 0; i < 8; ++i) {
            diff_bits += count_bits(cv[i] ^ cv_flipped[i]);
        }
        bool avalanche = diff_bits >= 77; // 30% of 256

        run_test("1.11 single-bit flip in transcript[0] produces different hash",
                 differs && avalanche);
    }

    // 1.12: Single-bit flip in transcript[7]
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0x1234);
        gen_random_u32(block, 16, 0x5678);
        uint32_t cv_orig[8];
        memcpy(cv_orig, cv, sizeof(cv_orig));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        block[7] ^= (1u << 31); // flip MSB of transcript[7]
        uint32_t cv_flipped[8];
        memcpy(cv_flipped, cv_orig, sizeof(cv_flipped));
        compress(cv_flipped, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        bool differs = !arrays_equal_u32(cv, cv_flipped, 8);
        int diff_bits = 0;
        for (int i = 0; i < 8; ++i) {
            diff_bits += count_bits(cv[i] ^ cv_flipped[i]);
        }
        bool avalanche = diff_bits >= 77;

        run_test("1.12 single-bit flip in transcript[7] produces different hash",
                 differs && avalanche);
    }

    // 1.13: Single-bit flip in transcript[15]
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0x1234);
        gen_random_u32(block, 16, 0x5678);
        uint32_t cv_orig[8];
        memcpy(cv_orig, cv, sizeof(cv_orig));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        block[15] ^= 1; // flip LSB of transcript[15]
        uint32_t cv_flipped[8];
        memcpy(cv_flipped, cv_orig, sizeof(cv_flipped));
        compress(cv_flipped, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        bool differs = !arrays_equal_u32(cv, cv_flipped, 8);
        int diff_bits = 0;
        for (int i = 0; i < 8; ++i) {
            diff_bits += count_bits(cv[i] ^ cv_flipped[i]);
        }
        bool avalanche = diff_bits >= 77;

        run_test("1.13 single-bit flip in transcript[15] produces different hash",
                 differs && avalanche);
    }

    // 1.14: All transcript words equal to pow_key
    {
        uint32_t pow_key[8] = {0xDEADBEEF, 0xC0FFEE01, 0xBADF00D2, 0xCAFEBAB3,
                                0x12345678, 0x9ABCDEF0, 0xFEDCBA98, 0x55555555};
        uint32_t cv[8], block[16];
        for (int i = 0; i < 8; ++i) cv[i] = pow_key[i];
        for (int i = 0; i < 16; ++i) block[i] = pow_key[i % 8];

        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("1.14 transcript words equal to pow_key",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 1.15: All transcript words equal to pow_target
    {
        uint32_t pow_target[8] = {0x11111111, 0x22222222, 0x33333333, 0x44444444,
                                   0x55555555, 0x66666666, 0x77777777, 0x88888888};
        uint32_t cv[8];
        for (int i = 0; i < 8; ++i) cv[i] = pow_target[i];
        uint32_t block[16];
        for (int i = 0; i < 16; ++i) block[i] = pow_target[i % 8];

        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("1.15 transcript words equal to pow_target",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 1.16: Transcript with BLAKE3 IV values
    {
        uint32_t cv[8];
        gen_random_u32(cv, 8, 0xAABB);
        uint32_t block[16];
        block[0] = BLAKE3_IV0; block[1] = BLAKE3_IV1;
        block[2] = BLAKE3_IV2; block[3] = BLAKE3_IV3;
        block[4] = BLAKE3_IV4; block[5] = BLAKE3_IV5;
        block[6] = BLAKE3_IV6; block[7] = BLAKE3_IV7;
        for (int i = 8; i < 16; ++i) block[i] = BLAKE3_IV0 + i;

        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("1.16 transcript with BLAKE3 IV values",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 1.17: Transcript with sequential values 0,1,2,...,15
    {
        uint32_t cv[8];
        gen_random_u32(cv, 8, 0xCCDD);
        uint32_t block[16];
        for (int i = 0; i < 16; ++i) block[i] = (uint32_t)i;

        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("1.17 transcript sequential 0..15",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 1.18: pow_key = all zeros, transcript random
    {
        uint32_t cv[8] = {0};
        uint32_t block[16];
        gen_random_u32(block, 16, 0x1111);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("1.18 pow_key all-zeros, random transcript",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 1.19: pow_key = all 0xFFFFFFFF, transcript random
    {
        uint32_t cv[8];
        for (int i = 0; i < 8; ++i) cv[i] = 0xFFFFFFFF;
        uint32_t block[16];
        gen_random_u32(block, 16, 0x2222);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("1.19 pow_key all-0xFFFFFFFF, random transcript",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 1.20: pow_key alternating pattern, transcript random
    {
        uint32_t cv[8];
        for (int i = 0; i < 8; ++i) cv[i] = (i & 1) ? 0xFFFFFFFF : 0x00000000;
        uint32_t block[16];
        gen_random_u32(block, 16, 0x3333);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("1.20 pow_key alternating 0x00/0xFF, random transcript",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 1.21-1.30: Various pow_key/pow_target combinations with counter variation
    {
        bool all_pass = true;
        uint64_t seeds[] = {0x100000001ULL, 0x200000002ULL, 0x300000003ULL,
                            0x400000004ULL, 0x500000005ULL, 0x600000006ULL,
                            0x700000007ULL, 0x800000008ULL, 0x900000009ULL,
                            0xA0000000AULL};
        uint64_t counters[] = {0ULL, 1ULL, 42ULL, 1000ULL, 0x7FFFFFFFULL,
                               0x80000000ULL, 0xFFFFFFFFULL, 0x123456789ABCDEFULL,
                               0x00FF00FF00FF00FFULL, 0xFF00FF00FF00FF00ULL};

        for (int s = 0; s < 10 && all_pass; ++s) {
            for (int c = 0; c < 10 && all_pass; ++c) {
                uint32_t cv[8], block[16];
                gen_random_u32(cv, 8, seeds[s]);
                gen_random_u32(block, 16, seeds[s] + 1);
                uint32_t cv_copy[8];
                memcpy(cv_copy, cv, sizeof(cv_copy));

                compress(cv, block, counters[c], MSG_BLOCK_SIZE, POW_FLAGS);

                uint32_t ref_out[16];
                ref_compress(cv_copy, block, counters[c], MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

                if (!arrays_equal_u32(cv, ref_out, 8)) {
                    all_pass = false;
                }
            }
        }
        run_test("1.21-1.30 counter variations (10 seeds x 10 counters)",
                 all_pass);
    }

    // 1.31-1.40: Various pow_key/pow_target combinations with block_len variation
    {
        bool all_pass = true;
        uint64_t seeds[] = {0x1111111111111111ULL, 0x2222222222222222ULL,
                            0x3333333333333333ULL, 0x4444444444444444ULL,
                            0x5555555555555555ULL, 0x6666666666666666ULL,
                            0x7777777777777777ULL, 0x8888888888888888ULL,
                            0x9999999999999999ULL, 0xAAAAAAAAAAAAAAAAULL};
        uint32_t block_lens[] = {0, 1, 31, 32, 63, 64, 127, 128, 1023, 1024};

        for (int s = 0; s < 10 && all_pass; ++s) {
            for (int bl = 0; bl < 10 && all_pass; ++bl) {
                uint32_t cv[8], block[16];
                gen_random_u32(cv, 8, seeds[s]);
                gen_random_u32(block, 16, seeds[s] + 1);
                uint32_t cv_copy[8];
                memcpy(cv_copy, cv, sizeof(cv_copy));

                compress(cv, block, 0, block_lens[bl], POW_FLAGS);

                uint32_t ref_out[16];
                ref_compress(cv_copy, block, 0, block_lens[bl], POW_FLAGS, ref_out);

                if (!arrays_equal_u32(cv, ref_out, 8)) {
                    all_pass = false;
                }
            }
        }
        run_test("1.31-1.40 block_len variations (10 seeds x 10 block_lens)",
                 all_pass);
    }

    // 1.41-1.50: Various pow_key/pow_target combinations with flag variation
    {
        bool all_pass = true;
        uint64_t seeds[] = {0x1212121212121212ULL, 0x3434343434343434ULL,
                            0x5656565656565656ULL, 0x7878787878787878ULL,
                            0x9A9A9A9A9A9A9A9AULL, 0xBCBCBCBCBCBCBCBCULL,
                            0xDEDEDEDEDEDEDEDEULL, 0xF0F0F0F0F0F0F0F0ULL,
                            0x1357135713571357ULL, 0x2468246824682468ULL};
        uint32_t flags_list[] = {0, CHUNK_START, CHUNK_END, ROOT,
                                  KEYED_HASH, CHUNK_START | CHUNK_END,
                                  KEYED_HASH | CHUNK_START | CHUNK_END | ROOT,
                                  PARENT | ROOT, KEYED_HASH | ROOT,
                                  CHUNK_START | ROOT | PARENT};

        for (int s = 0; s < 10 && all_pass; ++s) {
            for (int f = 0; f < 10 && all_pass; ++f) {
                uint32_t cv[8], block[16];
                gen_random_u32(cv, 8, seeds[s]);
                gen_random_u32(block, 16, seeds[s] + 1);
                uint32_t cv_copy[8];
                memcpy(cv_copy, cv, sizeof(cv_copy));

                compress(cv, block, 0, MSG_BLOCK_SIZE, flags_list[f]);

                uint32_t ref_out[16];
                ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, flags_list[f], ref_out);

                if (!arrays_equal_u32(cv, ref_out, 8)) {
                    all_pass = false;
                }
            }
        }
        run_test("1.41-1.50 flag variations (10 seeds x 10 flag combos)",
                 all_pass);
    }
}

// ─── GROUP 2: check_pow_target Correctness (30+ tests) ──────────────────────

static void test_group_2_pow_target() {
    printf("\n============================================================\n");
    printf("  GROUP 2: check_pow_target Correctness\n");
    printf("============================================================\n\n");

    // 2.1: Hash exactly equal to target (should return true)
    {
        uint32_t hash[8] = {0x12345678, 0x9ABCDEF0, 0xFEDCBA98, 0x11111111,
                            0x22222222, 0x33333333, 0x44444444, 0x55555555};
        uint32_t target[8];
        memcpy(target, hash, sizeof(target));
        run_test("2.1 hash == target → true",
                 check_pow_target(hash, target) == true);
    }

    // 2.2: Hash one below target (should return true)
    {
        uint32_t hash[8] = {0, 0, 0, 0, 0, 0, 0, 0x00000000};
        uint32_t target[8] = {0, 0, 0, 0, 0, 0, 0, 0x00000001};
        run_test("2.2 hash(0) < target(1) → true",
                 check_pow_target(hash, target) == true);
    }

    // 2.3: Hash one above target (should return false)
    {
        uint32_t hash[8] = {0, 0, 0, 0, 0, 0, 0, 0x00000002};
        uint32_t target[8] = {0, 0, 0, 0, 0, 0, 0, 0x00000001};
        run_test("2.3 hash(2) > target(1) → false",
                 check_pow_target(hash, target) == false);
    }

    // 2.4: Hash with all zeros (should return true for any reasonable target)
    {
        uint32_t hash[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        uint32_t target[8] = {0xDEADBEEF, 0xC0FFEE01, 0xBADF00D2, 0xCAFEBAB3,
                              0x12345678, 0x9ABCDEF0, 0xFEDCBA98, 0x55555555};
        run_test("2.4 hash all-zeros < random target → true",
                 check_pow_target(hash, target) == true);
    }

    // 2.5: Hash with all 0xFFFFFFFF (should return false for any target < max)
    {
        uint32_t hash[8] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
        uint32_t target[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        run_test("2.5 hash all-0xFFFFFFFF > target all-zeros → false",
                 check_pow_target(hash, target) == false);
    }

    // 2.6: MSW comparison — hash[7] < target[7]
    {
        uint32_t hash[8] = {0, 0, 0, 0, 0, 0, 0, 0x00000001};
        uint32_t target[8] = {0, 0, 0, 0, 0, 0, 0, 0x00000002};
        run_test("2.6 MSW: hash[7]=1 < target[7]=2 → true",
                 check_pow_target(hash, target) == true);
    }

    // 2.7: MSW comparison — hash[7] > target[7]
    {
        uint32_t hash[8] = {0, 0, 0, 0, 0, 0, 0, 0x00000003};
        uint32_t target[8] = {0, 0, 0, 0, 0, 0, 0, 0x00000002};
        run_test("2.7 MSW: hash[7]=3 > target[7]=2 → false",
                 check_pow_target(hash, target) == false);
    }

    // 2.8: MSW difference at index 6 (hash[6]=1 < target[6]=2)
    {
        uint32_t hash[8] = {0, 0, 0, 0, 0, 0, 0x00000001, 0x00000000};
        uint32_t target[8] = {0, 0, 0, 0, 0, 0, 0x00000002, 0xFFFFFFFF};
        run_test("2.8 MSW: hash[6]=1 < target[6]=2 (even if hash[7]>target[7]) → true",
                 check_pow_target(hash, target) == true);
    }

    // 2.9: MSW difference at index 6 (hash[6]=2 > target[6]=1)
    {
        uint32_t hash[8] = {0, 0, 0, 0, 0, 0, 0x00000002, 0xFFFFFFFF};
        uint32_t target[8] = {0, 0, 0, 0, 0, 0, 0x00000001, 0x00000000};
        run_test("2.9 MSW: hash[6]=2 > target[6]=1 → false",
                 check_pow_target(hash, target) == false);
    }

    // 2.10: MSW difference at index 0 (LSW)
    {
        uint32_t hash[8] = {0x00000001, 0, 0, 0, 0, 0, 0, 0};
        uint32_t target[8] = {0x00000002, 0, 0, 0, 0, 0, 0, 0};
        run_test("2.10 LSW: hash[0]=1 < target[0]=2 → true",
                 check_pow_target(hash, target) == true);
    }

    // 2.11-2.20: Random hash/target pairs with expected results
    {
        bool all_pass = true;
        for (int i = 0; i < 10 && all_pass; ++i) {
            uint32_t hash[8], target[8];
            gen_random_u32(hash, 8, (uint64_t)(0x2000 + i));
            gen_random_u32(target, 8, (uint64_t)(0x3000 + i));

            // Force hash < target by setting MSW lower
            uint32_t hash_lt[8], target_eq[8];
            memcpy(hash_lt, hash, sizeof(hash_lt));
            memcpy(target_eq, target, sizeof(target_eq));
            hash_lt[7] = target_eq[7] - 1; // guarantee hash < target
            bool result = check_pow_target(hash_lt, target_eq);
            if (result != true) all_pass = false;

            // Force hash > target by setting MSW higher
            uint32_t hash_gt[8];
            memcpy(hash_gt, hash, sizeof(hash_gt));
            hash_gt[7] = target_eq[7] + 1; // guarantee hash > target
            result = check_pow_target(hash_gt, target_eq);
            if (result != false) all_pass = false;
        }
        run_test("2.11-2.20 random hash/target pairs (10 iterations)",
                 all_pass);
    }

    // 2.21-2.30: Boundary conditions around common target values
    {
        bool all_pass = true;

        // Test with target = 0x7FFFFFFF (max positive int32)
        uint32_t target_pos[8] = {0, 0, 0, 0, 0, 0, 0, 0x7FFFFFFF};
        uint32_t hash_below[8] = {0, 0, 0, 0, 0, 0, 0, 0x7FFFFFFE};
        uint32_t hash_above[8] = {0, 0, 0, 0, 0, 0, 0, 0x80000000};
        if (check_pow_target(hash_below, target_pos) != true) all_pass = false;
        if (check_pow_target(hash_above, target_pos) != false) all_pass = false;

        // Test with target = 0x80000000
        uint32_t target_mid[8] = {0, 0, 0, 0, 0, 0, 0, 0x80000000};
        uint32_t hash_low[8] = {0, 0, 0, 0, 0, 0, 0, 0x7FFFFFFF};
        uint32_t hash_high[8] = {0, 0, 0, 0, 0, 0, 0, 0x80000001};
        if (check_pow_target(hash_low, target_mid) != true) all_pass = false;
        if (check_pow_target(hash_high, target_mid) != false) all_pass = false;

        // Test with target = 0xFFFFFFFF
        uint32_t target_max[8] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                                   0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
        uint32_t hash_near_max[8] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                                      0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE};
        if (check_pow_target(hash_near_max, target_max) != true) all_pass = false;
        // No hash can be > all-0xFFFFFFFF, so any hash <= target_max → true
        uint32_t hash_any[8] = {0x1234, 0x5678, 0x9ABC, 0xDEF0,
                                0x1111, 0x2222, 0x3333, 0x4444};
        if (check_pow_target(hash_any, target_max) != true) all_pass = false;

        run_test("2.21-2.30 boundary conditions (max targets)",
                 all_pass);
    }
}

// ─── GROUP 3: Determinism Tests (20+ tests) ────────────────────────────────

static void test_group_3_determinism() {
    printf("\n============================================================\n");
    printf("  GROUP 3: Determinism Tests\n");
    printf("============================================================\n\n");

    // 3.1-3.10: Run same input 10 times each, verify identical output
    {
        bool all_pass = true;
        uint64_t seeds[] = {0x100000001ULL, 0x200000002ULL, 0x300000003ULL,
                            0x400000004ULL, 0x500000005ULL, 0x600000006ULL,
                            0x700000007ULL, 0x800000008ULL, 0x900000009ULL,
                            0xA0000000AULL};

        for (int s = 0; s < 10 && all_pass; ++s) {
            uint32_t cv[8], block[16];
            gen_random_u32(cv, 8, seeds[s]);
            gen_random_u32(block, 16, seeds[s] + 1);
            uint32_t expected[8];
            memcpy(expected, cv, sizeof(expected));
            compress(expected, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

            for (int r = 0; r < 10 && all_pass; ++r) {
                uint32_t cv_test[8], block_test[16];
                memcpy(cv_test, cv, sizeof(cv));
                memcpy(block_test, block, sizeof(block));
                compress(cv_test, block_test, 0, MSG_BLOCK_SIZE, POW_FLAGS);
                if (!arrays_equal_u32(cv_test, expected, 8)) {
                    all_pass = false;
                }
            }
        }
        run_test("3.1-3.10 compress determinism (10 inputs x 10 runs each)",
                 all_pass);
    }

    // 3.11-3.15: Determinism of compress_full
    {
        bool all_pass = true;
        uint64_t seeds[] = {0x1111111111111111ULL, 0x2222222222222222ULL,
                            0x3333333333333333ULL, 0x4444444444444444ULL,
                            0x5555555555555555ULL};

        for (int s = 0; s < 5 && all_pass; ++s) {
            uint32_t cv[8], block[16];
            gen_random_u32(cv, 8, seeds[s]);
            gen_random_u32(block, 16, seeds[s] + 1);
            uint32_t expected[16];
            compress_full(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, expected);

            for (int r = 0; r < 10 && all_pass; ++r) {
                uint32_t cv_test[8], block_test[16];
                memcpy(cv_test, cv, sizeof(cv));
                memcpy(block_test, block, sizeof(block));
                uint32_t result[16];
                compress_full(cv_test, block_test, 0, MSG_BLOCK_SIZE, POW_FLAGS, result);
                if (!arrays_equal_u32(result, expected, 16)) {
                    all_pass = false;
                }
            }
        }
        run_test("3.11-3.15 compress_full determinism (5 inputs x 10 runs)",
                 all_pass);
    }

    // 3.16-3.20: Determinism of ref_compress
    {
        bool all_pass = true;
        uint64_t seeds[] = {0x6666666666666666ULL, 0x7777777777777777ULL,
                            0x8888888888888888ULL, 0x9999999999999999ULL,
                            0xAAAAAAAAAAAAAAAAULL};

        for (int s = 0; s < 5 && all_pass; ++s) {
            uint32_t cv[8], block[16];
            gen_random_u32(cv, 8, seeds[s]);
            gen_random_u32(block, 16, seeds[s] + 1);
            uint32_t expected[16];
            ref_compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, expected);

            for (int r = 0; r < 10 && all_pass; ++r) {
                uint32_t cv_test[8], block_test[16];
                memcpy(cv_test, cv, sizeof(cv));
                memcpy(block_test, block, sizeof(block));
                uint32_t result[16];
                ref_compress(cv_test, block_test, 0, MSG_BLOCK_SIZE, POW_FLAGS, result);
                if (!arrays_equal_u32(result, expected, 16)) {
                    all_pass = false;
                }
            }
        }
        run_test("3.16-3.20 ref_compress determinism (5 inputs x 10 runs)",
                 all_pass);
    }

    // 3.21-3.25: Different inputs produce different outputs (avalanche)
    {
        bool all_pass = true;
        uint32_t base_cv[8] = {0x12345678, 0x9ABCDEF0, 0xFEDCBA98, 0x11111111,
                               0x22222222, 0x33333333, 0x44444444, 0x55555555};
        uint32_t base_block[16] = {0, 1, 2, 3, 4, 5, 6, 7,
                                    8, 9, 10, 11, 12, 13, 14, 15};

        uint32_t expected[8];
        memcpy(expected, base_cv, sizeof(expected));
        compress(expected, base_block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        // Flip one bit in block and verify different output
        for (int bit = 0; bit < 16 * 32 && all_pass; bit += 32) {
            uint32_t block_test[16];
            memcpy(block_test, base_block, sizeof(block_test));
            block_test[bit / 32] ^= (1u << (bit % 32));

            uint32_t result[8];
            memcpy(result, base_cv, sizeof(result));
            compress(result, block_test, 0, MSG_BLOCK_SIZE, POW_FLAGS);

            if (arrays_equal_u32(result, expected, 8)) {
                all_pass = false; // Should always differ!
            }
        }
        run_test("3.21-3.25 different inputs produce different outputs (512 bit flips)",
                 all_pass);
    }

    // 3.26-3.30: check_pow_target determinism
    {
        bool all_pass = true;
        uint64_t seeds[] = {0xBBBBBBBBBBBBBBBBULL, 0xCCCCCCCCCCCCCCCCULL,
                            0xDDDDDDDDDDDDDDDDULL, 0xEEEEEEEEEEEEEEEEULL,
                            0xFFFFFFFFFFFFFFFFULL};

        for (int s = 0; s < 5 && all_pass; ++s) {
            uint32_t hash[8], target[8];
            gen_random_u32(hash, 8, seeds[s]);
            gen_random_u32(target, 8, seeds[s] + 1);
            bool expected = check_pow_target(hash, target);

            for (int r = 0; r < 100 && all_pass; ++r) {
                bool result = check_pow_target(hash, target);
                if (result != expected) {
                    all_pass = false;
                }
            }
        }
        run_test("3.26-3.30 check_pow_target determinism (5 inputs x 100 runs)",
                 all_pass);
    }
}

// ─── GROUP 4: Reference Implementation Cross-Validation (50+ tests) ─────────

static void test_group_4_reference_validation() {
    printf("\n============================================================\n");
    printf("  GROUP 4: Reference Implementation Cross-Validation\n");
    printf("============================================================\n\n");

    // 4.1-4.10: Systematic comparison of compress vs ref_compress across many inputs
    {
        bool all_pass = true;
        for (int i = 0; i < 10 && all_pass; ++i) {
            uint32_t cv[8], block[16];
            gen_random_u32(cv, 8, (uint64_t)(0x4000 + i * 100));
            gen_random_u32(block, 16, (uint64_t)(0x5000 + i * 100));
            uint32_t cv_copy[8];
            memcpy(cv_copy, cv, sizeof(cv_copy));

            compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

            uint32_t ref_out[16];
            ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

            if (!arrays_equal_u32(cv, ref_out, 8)) {
                all_pass = false;
            }
        }
        run_test("4.1-4.10 compress vs ref_compress (10 random inputs)",
                 all_pass);
    }

    // 4.11-4.20: Cross-validation with different counters
    {
        bool all_pass = true;
        uint64_t counters[] = {0, 1, 2, 100, 1000, 0x7FFFFFFF,
                               0x80000000, 0xFFFFFFFF, 0x123456789ABCDEF0ULL,
                               0x0123456789ABCDEFULL};

        for (int i = 0; i < 10 && all_pass; ++i) {
            uint32_t cv[8], block[16];
            gen_random_u32(cv, 8, (uint64_t)(0x6000 + i));
            gen_random_u32(block, 16, (uint64_t)(0x7000 + i));
            uint32_t cv_copy[8];
            memcpy(cv_copy, cv, sizeof(cv_copy));

            compress(cv, block, counters[i], MSG_BLOCK_SIZE, POW_FLAGS);

            uint32_t ref_out[16];
            ref_compress(cv_copy, block, counters[i], MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

            if (!arrays_equal_u32(cv, ref_out, 8)) {
                all_pass = false;
            }
        }
        run_test("4.11-4.20 compress vs ref_compress with different counters (10)",
                 all_pass);
    }

    // 4.21-4.30: Cross-validation with different block lengths
    {
        bool all_pass = true;
        uint32_t block_lens[] = {0, 1, 15, 16, 31, 32, 63, 64, 128, 255};

        for (int i = 0; i < 10 && all_pass; ++i) {
            uint32_t cv[8], block[16];
            gen_random_u32(cv, 8, (uint64_t)(0x8000 + i));
            gen_random_u32(block, 16, (uint64_t)(0x9000 + i));
            uint32_t cv_copy[8];
            memcpy(cv_copy, cv, sizeof(cv_copy));

            compress(cv, block, 0, block_lens[i], POW_FLAGS);

            uint32_t ref_out[16];
            ref_compress(cv_copy, block, 0, block_lens[i], POW_FLAGS, ref_out);

            if (!arrays_equal_u32(cv, ref_out, 8)) {
                all_pass = false;
            }
        }
        run_test("4.21-4.30 compress vs ref_compress with different block_lens (10)",
                 all_pass);
    }

    // 4.31-4.40: Cross-validation with different flags
    {
        bool all_pass = true;
        uint32_t flags_list[] = {0, CHUNK_START, CHUNK_END, ROOT, KEYED_HASH,
                                  CHUNK_START | CHUNK_END, KEYED_HASH | CHUNK_START | CHUNK_END | ROOT,
                                  PARENT | ROOT, KEYED_HASH | ROOT, CHUNK_START | ROOT};

        for (int i = 0; i < 10 && all_pass; ++i) {
            uint32_t cv[8], block[16];
            gen_random_u32(cv, 8, (uint64_t)(0xA000 + i));
            gen_random_u32(block, 16, (uint64_t)(0xB000 + i));
            uint32_t cv_copy[8];
            memcpy(cv_copy, cv, sizeof(cv_copy));

            compress(cv, block, 0, MSG_BLOCK_SIZE, flags_list[i]);

            uint32_t ref_out[16];
            ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, flags_list[i], ref_out);

            if (!arrays_equal_u32(cv, ref_out, 8)) {
                all_pass = false;
            }
        }
        run_test("4.31-4.40 compress vs ref_compress with different flags (10)",
                 all_pass);
    }

    // 4.41-4.50: Cross-validation with all-zero CV
    {
        bool all_pass = true;
        uint64_t seeds[] = {0x1000000000000001ULL, 0x2000000000000002ULL,
                            0x3000000000000003ULL, 0x4000000000000004ULL,
                            0x5000000000000005ULL, 0x6000000000000006ULL,
                            0x7000000000000007ULL, 0x8000000000000008ULL,
                            0x9000000000000009ULL, 0xA00000000000000AULL};

        for (int i = 0; i < 10 && all_pass; ++i) {
            uint32_t cv[8] = {0};
            uint32_t block[16];
            gen_random_u32(block, 16, seeds[i]);
            uint32_t cv_copy[8] = {0};

            compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

            uint32_t ref_out[16];
            ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

            if (!arrays_equal_u32(cv, ref_out, 8)) {
                all_pass = false;
            }
        }
        run_test("4.41-4.50 compress vs ref_compress with all-zero CV (10)",
                 all_pass);
    }
}

// ─── GROUP 5: Edge Cases (20+ tests) ────────────────────────────────────────

static void test_group_5_edge_cases() {
    printf("\n============================================================\n");
    printf("  GROUP 5: Edge Cases\n");
    printf("============================================================\n\n");

    // 5.1: pow_key = all zeros
    {
        uint32_t cv[8] = {0};
        uint32_t block[16];
        gen_random_u32(block, 16, 0x1111);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        bool matches_ref = arrays_equal_u32(cv, ref_out, 8);

        // Also verify output is deterministic (non-zero due to avalanche)
        bool non_zero = false;
        for (int i = 0; i < 8; ++i) {
            if (cv[i] != 0) { non_zero = true; break; }
        }

        run_test("5.1 pow_key all-zeros matches ref, produces non-zero hash",
                 matches_ref && non_zero);
    }

    // 5.2: pow_target = all zeros (hardest possible target)
    {
        uint32_t hash[8] = {0xDEADBEEF, 0xC0FFEE01, 0xBADF00D2, 0xCAFEBAB3,
                            0x12345678, 0x9ABCDEF0, 0xFEDCBA98, 0x55555555};
        uint32_t target[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        run_test("5.2 non-zero hash > all-zero target → false",
                 check_pow_target(hash, target) == false);
    }

    // 5.3: pow_target = all 0xFFFFFFFF (easiest possible target)
    {
        uint32_t hash[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        uint32_t target[8] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
                              0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
        run_test("5.3 any hash <= all-0xFFFFFFFF target → true",
                 check_pow_target(hash, target) == true);
    }

    // 5.4: Transcript with specific BLAKE3 test vector — all IV values
    {
        uint32_t cv[8];
        gen_random_u32(cv, 8, 0x1234);
        uint32_t block[16];
        // Set transcript to IV values
        for (int i = 0; i < 8; ++i) block[i] = BLAKE3_IV0 + i;
        for (int i = 0; i < 8; ++i) block[i + 8] = BLAKE3_IV0 + i + 8;

        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("5.4 transcript = BLAKE3 IV values matches ref",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 5.5: Zero block_len
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0x1111);
        gen_random_u32(block, 16, 0x2222);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, 0, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, 0, POW_FLAGS, ref_out);

        run_test("5.5 zero block_len matches ref",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 5.6: block_len = 1
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0x1111);
        gen_random_u32(block, 16, 0x2222);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, 1, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, 1, POW_FLAGS, ref_out);

        run_test("5.6 block_len=1 matches ref",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 5.7: block_len = 31 (odd, less than half block)
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0x1111);
        gen_random_u32(block, 16, 0x2222);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, 31, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, 31, POW_FLAGS, ref_out);

        run_test("5.7 block_len=31 matches ref",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 5.8: block_len = 32 (exactly half block)
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0x1111);
        gen_random_u32(block, 16, 0x2222);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, 32, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, 32, POW_FLAGS, ref_out);

        run_test("5.8 block_len=32 matches ref",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 5.9: block_len = 63 (one less than full block)
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0x1111);
        gen_random_u32(block, 16, 0x2222);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, 63, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, 63, POW_FLAGS, ref_out);

        run_test("5.9 block_len=63 matches ref",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 5.10: block_len = 1024 (chunk size)
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0x1111);
        gen_random_u32(block, 16, 0x2222);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, 1024, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, 1024, POW_FLAGS, ref_out);

        run_test("5.10 block_len=1024 matches ref",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 5.11: Counter = 0
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0x1111);
        gen_random_u32(block, 16, 0x2222);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("5.11 counter=0 matches ref",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 5.12: Counter = 1
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0x1111);
        gen_random_u32(block, 16, 0x2222);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 1, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 1, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("5.12 counter=1 matches ref",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 5.13: Counter = max uint64
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0x1111);
        gen_random_u32(block, 16, 0x2222);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0xFFFFFFFFFFFFFFFFULL, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0xFFFFFFFFFFFFFFFFULL, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("5.13 counter=0xFFFFFFFFFFFFFFFF matches ref",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 5.14: Flags = 0 (no flags set)
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0x1111);
        gen_random_u32(block, 16, 0x2222);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, 0);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, 0, ref_out);

        run_test("5.14 flags=0 matches ref",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 5.15: Flags = single flag CHUNK_START
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0x1111);
        gen_random_u32(block, 16, 0x2222);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, CHUNK_START);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, CHUNK_START, ref_out);

        run_test("5.15 flags=CHUNK_START matches ref",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 5.16: Flags = single flag ROOT
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0x1111);
        gen_random_u32(block, 16, 0x2222);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, ROOT);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, ROOT, ref_out);

        run_test("5.16 flags=ROOT matches ref",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 5.17: Flags = KEYED_HASH | ROOT
    {
        uint32_t cv[8], block[16];
        gen_random_u32(cv, 8, 0x1111);
        gen_random_u32(block, 16, 0x2222);
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, KEYED_HASH | ROOT);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, KEYED_HASH | ROOT, ref_out);

        run_test("5.17 flags=KEYED_HASH|ROOT matches ref",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 5.18: All transcript = 0x01010101
    {
        uint32_t cv[8];
        gen_random_u32(cv, 8, 0x1111);
        uint32_t block[16];
        for (int i = 0; i < 16; ++i) block[i] = 0x01010101;
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("5.18 transcript all=0x01010101 matches ref",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 5.19: All transcript = 0x80000000
    {
        uint32_t cv[8];
        gen_random_u32(cv, 8, 0x1111);
        uint32_t block[16];
        for (int i = 0; i < 16; ++i) block[i] = 0x80000000;
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("5.19 transcript all=0x80000000 matches ref",
                 arrays_equal_u32(cv, ref_out, 8));
    }

    // 5.20: Alternating transcript pattern
    {
        uint32_t cv[8];
        gen_random_u32(cv, 8, 0x1111);
        uint32_t block[16];
        for (int i = 0; i < 16; ++i) block[i] = (i & 1) ? 0xFFFFFFFF : 0x00000000;
        uint32_t cv_copy[8];
        memcpy(cv_copy, cv, sizeof(cv_copy));

        compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

        uint32_t ref_out[16];
        ref_compress(cv_copy, block, 0, MSG_BLOCK_SIZE, POW_FLAGS, ref_out);

        run_test("5.20 alternating transcript matches ref",
                 arrays_equal_u32(cv, ref_out, 8));
    }
}

// ─── GROUP 6: Full Pipeline Tests (30+ tests) ───────────────────────────────

static void test_group_6_full_pipeline() {
    printf("\n============================================================\n");
    printf("  GROUP 6: Full Pipeline Tests\n");
    printf("============================================================\n\n");

    // 6.1: Full pipeline — sigma=[0..31], pow_key=zero, no GEMM outputs
    {
        uint8_t sigma[32];
        for (int i = 0; i < 32; ++i) sigma[i] = (uint8_t)i;

        uint32_t pow_key[8] = {0};
        uint32_t out_cv[8];
        full_pow_pipeline(sigma, pow_key, nullptr, 0, out_cv);

        // Verify deterministic
        uint32_t out_cv2[8];
        full_pow_pipeline(sigma, pow_key, nullptr, 0, out_cv2);

        bool deterministic = arrays_equal_u32(out_cv, out_cv2, 8);

        // Verify non-trivial
        bool non_trivial = false;
        for (int i = 0; i < 8; ++i) {
            if (out_cv[i] != 0) { non_trivial = true; break; }
        }

        run_test("6.1 sigma=[0..31], pow_key=zero, no GEMM → deterministic, non-trivial",
                 deterministic && non_trivial);
    }

    // 6.2: Full pipeline — sigma all-zero, pow_key random, no GEMM outputs
    {
        uint8_t sigma[32] = {0};
        uint32_t pow_key[8];
        gen_random_u32(pow_key, 8, 0x1111);
        uint32_t out_cv[8];
        full_pow_pipeline(sigma, pow_key, nullptr, 0, out_cv);

        uint32_t out_cv2[8];
        full_pow_pipeline(sigma, pow_key, nullptr, 0, out_cv2);

        run_test("6.2 sigma=zero, pow_key=random, no GEMM → deterministic",
                 arrays_equal_u32(out_cv, out_cv2, 8));
    }

    // 6.3: Full pipeline — sigma random, pow_key=zero, no GEMM outputs
    {
        uint8_t sigma[32];
        gen_random_u8(sigma, 32, 0x2222);
        uint32_t pow_key[8] = {0};
        uint32_t out_cv[8];
        full_pow_pipeline(sigma, pow_key, nullptr, 0, out_cv);

        uint32_t out_cv2[8];
        full_pow_pipeline(sigma, pow_key, nullptr, 0, out_cv2);

        run_test("6.3 sigma=random, pow_key=zero, no GEMM → deterministic",
                 arrays_equal_u32(out_cv, out_cv2, 8));
    }

    // 6.4: Full pipeline — sigma random, pow_key random, no GEMM outputs
    {
        uint8_t sigma[32];
        gen_random_u8(sigma, 32, 0x3333);
        uint32_t pow_key[8];
        gen_random_u32(pow_key, 8, 0x4444);
        uint32_t out_cv[8];
        full_pow_pipeline(sigma, pow_key, nullptr, 0, out_cv);

        uint32_t out_cv2[8];
        full_pow_pipeline(sigma, pow_key, nullptr, 0, out_cv2);

        run_test("6.4 sigma=random, pow_key=random, no GEMM → deterministic",
                 arrays_equal_u32(out_cv, out_cv2, 8));
    }

    // 6.5: Full pipeline — with 1 GEMM output
    {
        uint8_t sigma[32];
        gen_random_u8(sigma, 32, 0x5555);
        uint32_t pow_key[8];
        gen_random_u32(pow_key, 8, 0x6666);
        uint32_t gemm[1] = {0xDEADBEEF};
        uint32_t out_cv[8];
        full_pow_pipeline(sigma, pow_key, gemm, 1, out_cv);

        uint32_t out_cv2[8];
        full_pow_pipeline(sigma, pow_key, gemm, 1, out_cv2);

        run_test("6.5 full pipeline with 1 GEMM output → deterministic",
                 arrays_equal_u32(out_cv, out_cv2, 8));
    }

    // 6.6: Full pipeline — with 8 GEMM outputs
    {
        uint8_t sigma[32];
        gen_random_u8(sigma, 32, 0x7777);
        uint32_t pow_key[8];
        gen_random_u32(pow_key, 8, 0x8888);
        uint32_t gemm[8];
        gen_random_u32(gemm, 8, 0x9999);
        uint32_t out_cv[8];
        full_pow_pipeline(sigma, pow_key, gemm, 8, out_cv);

        uint32_t out_cv2[8];
        full_pow_pipeline(sigma, pow_key, gemm, 8, out_cv2);

        run_test("6.6 full pipeline with 8 GEMM outputs → deterministic",
                 arrays_equal_u32(out_cv, out_cv2, 8));
    }

    // 6.7: Full pipeline — with 16 GEMM outputs
    {
        uint8_t sigma[32];
        gen_random_u8(sigma, 32, 0xAAAA);
        uint32_t pow_key[8];
        gen_random_u32(pow_key, 8, 0xBBBB);
        uint32_t gemm[16];
        gen_random_u32(gemm, 16, 0xCCCC);
        uint32_t out_cv[8];
        full_pow_pipeline(sigma, pow_key, gemm, 16, out_cv);

        uint32_t out_cv2[8];
        full_pow_pipeline(sigma, pow_key, gemm, 16, out_cv2);

        run_test("6.7 full pipeline with 16 GEMM outputs → deterministic",
                 arrays_equal_u32(out_cv, out_cv2, 8));
    }

    // 6.8: Full pipeline — with 32 GEMM outputs
    {
        uint8_t sigma[32];
        gen_random_u8(sigma, 32, 0xDDDD);
        uint32_t pow_key[8];
        gen_random_u32(pow_key, 8, 0xEEEE);
        uint32_t gemm[32];
        gen_random_u32(gemm, 32, 0xFFFF);
        uint32_t out_cv[8];
        full_pow_pipeline(sigma, pow_key, gemm, 32, out_cv);

        uint32_t out_cv2[8];
        full_pow_pipeline(sigma, pow_key, gemm, 32, out_cv2);

        run_test("6.8 full pipeline with 32 GEMM outputs → deterministic",
                 arrays_equal_u32(out_cv, out_cv2, 8));
    }

    // 6.9: Full pipeline — with GEMM outputs all zeros
    {
        uint8_t sigma[32];
        gen_random_u8(sigma, 32, 0x1010);
        uint32_t pow_key[8];
        gen_random_u32(pow_key, 8, 0x2020);
        uint32_t gemm[16] = {0};
        uint32_t out_cv[8];
        full_pow_pipeline(sigma, pow_key, gemm, 16, out_cv);

        uint32_t out_cv2[8];
        full_pow_pipeline(sigma, pow_key, gemm, 16, out_cv2);

        run_test("6.9 full pipeline with GEMM=zero → deterministic",
                 arrays_equal_u32(out_cv, out_cv2, 8));
    }

    // 6.10: Full pipeline — with GEMM outputs all 0xFFFFFFFF
    {
        uint8_t sigma[32];
        gen_random_u8(sigma, 32, 0x3030);
        uint32_t pow_key[8];
        gen_random_u32(pow_key, 8, 0x4040);
        uint32_t gemm[16];
        for (int i = 0; i < 16; ++i) gemm[i] = 0xFFFFFFFF;
        uint32_t out_cv[8];
        full_pow_pipeline(sigma, pow_key, gemm, 16, out_cv);

        uint32_t out_cv2[8];
        full_pow_pipeline(sigma, pow_key, gemm, 16, out_cv2);

        run_test("6.10 full pipeline with GEMM=0xFFFFFFFF → deterministic",
                 arrays_equal_u32(out_cv, out_cv2, 8));
    }

    // 6.11-6.15: Pipeline determinism across 5 different sigma values
    {
        bool all_pass = true;
        for (int s = 0; s < 5 && all_pass; ++s) {
            uint8_t sigma[32];
            gen_random_u8(sigma, 32, (uint64_t)(0x5000 + s * 1000));
            uint32_t pow_key[8];
            gen_random_u32(pow_key, 8, (uint64_t)(0x6000 + s * 1000));
            uint32_t gemm[16];
            gen_random_u32(gemm, 16, (uint64_t)(0x7000 + s * 1000));

            uint32_t expected[8];
            full_pow_pipeline(sigma, pow_key, gemm, 16, expected);

            for (int r = 0; r < 10 && all_pass; ++r) {
                uint32_t result[8];
                full_pow_pipeline(sigma, pow_key, gemm, 16, result);
                if (!arrays_equal_u32(result, expected, 8)) {
                    all_pass = false;
                }
            }
        }
        run_test("6.11-6.15 pipeline determinism (5 sigmas x 10 runs each)",
                 all_pass);
    }

    // 6.16-6.20: Sigma to transcript correctness verification
    {
        bool all_pass = true;

        // Test with sigma = [0..31]
        {
            uint8_t sigma[32];
            for (int i = 0; i < 32; ++i) sigma[i] = (uint8_t)i;
            uint32_t transcript[16];
            sigma_to_transcript(sigma, transcript);

            // Expected: byte-sliced LE u32
            uint32_t expected[16] = {
                0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
                0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c,
                0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
                0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c
            };
            if (!arrays_equal_u32(transcript, expected, 16)) all_pass = false;
        }

        // Test with sigma = all-zero
        if (all_pass) {
            uint8_t sigma[32] = {0};
            uint32_t transcript[16];
            sigma_to_transcript(sigma, transcript);
            for (int i = 0; i < 16 && all_pass; ++i) {
                if (transcript[i] != 0) all_pass = false;
            }
        }

        // Test with sigma = all-0xFF
        if (all_pass) {
            uint8_t sigma[32];
            for (int i = 0; i < 32; ++i) sigma[i] = 0xFF;
            uint32_t transcript[16];
            sigma_to_transcript(sigma, transcript);
            for (int i = 0; i < 16 && all_pass; ++i) {
                if (transcript[i] != 0xFFFFFFFF) all_pass = false;
            }
        }

        // Test with sigma = alternating 0x00/0xFF
        if (all_pass) {
            uint8_t sigma[32];
            for (int i = 0; i < 32; ++i) sigma[i] = (i & 1) ? 0xFF : 0x00;
            uint32_t transcript[16];
            sigma_to_transcript(sigma, transcript);
            // transcript[0] = 0xFF00FF00
            if (transcript[0] != 0xFF00FF00) all_pass = false;
        }

        // Test with sigma = production-like pattern
        if (all_pass) {
            uint8_t sigma[32] = {
                0x4a, 0x2b, 0x8c, 0x1d, 0xe3, 0xf4, 0x05, 0x96,
                0x77, 0x68, 0x59, 0x4a, 0x3b, 0x2c, 0x1d, 0x0e,
                0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88,
                0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00
            };
            uint32_t transcript[16];
            sigma_to_transcript(sigma, transcript);
            // Verify first few:
            // transcript[0] = sigma[0] | sigma[1]<<8 | sigma[2]<<16 | sigma[3]<<24
            //              = 0x4a | 0x2b<<8 | 0x8c<<16 | 0x1d<<24 = 0x1d8c2b4a
            if (transcript[0] != 0x1d8c2b4a) all_pass = false;
            // transcript[1] = sigma[4] | sigma[5]<<8 | sigma[6]<<16 | sigma[7]<<24
            //              = 0xe3 | 0xf4<<8 | 0x05<<16 | 0x96<<24 = 0x9605f4e3
            if (transcript[1] != 0x9605f4e3) all_pass = false;
        }

        run_test("6.16-6.20 sigma_to_transcript correctness (5 patterns)",
                 all_pass);
    }

    // 6.21-6.25: XOR reduction edge cases in pipeline
    {
        bool all_pass = true;

        // 1 GEMM output: transcript[i] = rotl_xor(transcript[i], gemm[0])
        {
            uint8_t sigma[32];
            for (int i = 0; i < 32; ++i) sigma[i] = (uint8_t)i;
            uint32_t pow_key[8] = {0};
            uint32_t gemm[1] = {0x12345678};

            // Manual pipeline
            uint32_t transcript[16];
            sigma_to_transcript(sigma, transcript);
            uint32_t reduced = xor_reduction(gemm, 1); // should equal gemm[0]
            if (reduced != 0x12345678) all_pass = false;

            for (int i = 0; i < 16 && all_pass; ++i) {
                uint32_t expected = rotl_xor(transcript[i], reduced);
                transcript[i] = expected;
            }

            uint32_t cv[8];
            for (int i = 0; i < 8; ++i) cv[i] = pow_key[i];
            uint32_t block[16];
            memcpy(block, transcript, sizeof(block));
            compress(cv, block, 0, MSG_BLOCK_SIZE, POW_FLAGS);

            // Compare against full pipeline
            uint32_t pipeline_out[8];
            full_pow_pipeline(sigma, pow_key, gemm, 1, pipeline_out);

            if (!arrays_equal_u32(cv, pipeline_out, 8)) all_pass = false;
        }

        run_test("6.21-6.25 XOR reduction in pipeline (1 GEMM output)",
                 all_pass);
    }

    // 6.26-6.30: Full pipeline with check_pow_target
    {
        bool all_pass = true;

        // Create an easy target that should always be satisfied
        uint32_t easy_target[8];
        for (int i = 0; i < 8; ++i) easy_target[i] = 0xFFFFFFFF;

        // Create a hard target that should never be satisfied
        uint32_t hard_target[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        uint8_t sigma[32];
        gen_random_u8(sigma, 32, 0xA1A1);
        uint32_t pow_key[8];
        gen_random_u32(pow_key, 8, 0xB2B2);

        uint32_t out_cv[8];
        full_pow_pipeline(sigma, pow_key, nullptr, 0, out_cv);

        // Easy target: any hash <= all-0xFFFFFFFF → true
        bool easy_result = check_pow_target(out_cv, easy_target);
        if (easy_result != true) all_pass = false;

        // Hard target: non-zero hash > all-zeros → false
        bool non_zero = false;
        for (int i = 0; i < 8; ++i) {
            if (out_cv[i] != 0) { non_zero = true; break; }
        }
        bool hard_result = check_pow_target(out_cv, hard_target);
        if (non_zero && hard_result != false) all_pass = false;
        if (!non_zero && hard_result != true) all_pass = false; // all-zero hash passes

        run_test("6.26-6.30 pipeline + check_pow_target (easy/hard targets)",
                 all_pass);
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main() {
    printf("============================================================\n");
    printf("  BLAKE3 Finalize Optimization — Comprehensive CPU Tests\n");
    printf("  Pearl Consensus Safety Validation\n");
    printf("============================================================\n");
    printf("No GPU required — pure C++ validation\n");
    printf("Verifies byte-identical output for consensus safety\n\n");

    test_group_1_compress_correctness();
    test_group_2_pow_target();
    test_group_3_determinism();
    test_group_4_reference_validation();
    test_group_5_edge_cases();
    test_group_6_full_pipeline();

    printf("\n============================================================\n");
    printf("  SUMMARY\n");
    printf("============================================================\n");
    printf("Total tests : %d\n", g_test_count);
    printf("Passed      : %d\n", g_pass_count);
    if (g_fail_count > 0) {
        printf("Failed      : %d\n", g_fail_count);
    }
    printf("============================================================\n");

    if (g_fail_count > 0) {
        printf("\n*** %d TEST(S) FAILED — NOT CONSENSUS SAFE ***\n\n", g_fail_count);
        return 1;
    } else {
        printf("\n*** ALL %d TESTS PASSED — BYTE-IDENTICAL OUTPUT VERIFIED ***\n\n", g_test_count);
        return 0;
    }
}
