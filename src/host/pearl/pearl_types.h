#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <vector>

#include "cuda_compat.h"

namespace pearl {

// Periodic pattern for the committed hash-tile shape.
// Mirrors C# Akoya.Crypto.PeriodicPattern.
struct PeriodicPattern {
    uint32_t stride0 = 1, length0 = 1;
    uint32_t stride1 = 1, length1 = 1;
    uint32_t stride2 = 1, length2 = 1;

    static constexpr size_t kSerializedSize = 6;

    // Identity pattern: size 1 (single element).
    static PeriodicPattern identity() {
        return PeriodicPattern{};
    }

    // Build from a sorted list of zero-rooted indices, matching C# FromIndices.
    static PeriodicPattern from_indices(const std::vector<uint32_t>& idx);

    // Default rows pattern for H100-style kernels: {0, 8}.
    static PeriodicPattern default_rows();

    // Default cols pattern for H100-style kernels: 64 values {8r, 8r+1} for r in 0..31.
    static PeriodicPattern default_cols();

    // Serialise to 6 bytes (factor-1, length-1 per dim).
    std::array<uint8_t, kSerializedSize> to_bytes() const;

    uint32_t size() const { return length0 * length1 * length2; }
};

// Pearl mining configuration. Default shape targets H100/RTX 5090 class GPUs.
struct MiningConfig {
    int32_t m = 8192;
    int32_t n = 32768;
    int32_t k = 128;

    // Sparse-noise rank used by the proof-time noise generator.
    // Matches the protocol default of 64.
    int32_t r = 64;

    // Committed hash-tile periodic pattern. MUST match the kernel's actual tile.
    PeriodicPattern rows_pattern;
    PeriodicPattern cols_pattern;

    // GEMM CTA tile shape (must match kernel instantiation).
    // H100 kernels are fixed at 128x256x128; consumer kernels ignore these
    // values but keep them canonical.
    int32_t bM = 128;
    int32_t bN = 256;
    int32_t bK = 128;
    int32_t cM = 1;
    int32_t cN = 1;

    // tensor_hash launch shape.
    uint32_t tensor_hash_threads = 128;
    uint32_t tensor_hash_stages  = 2;
    uint32_t tensor_hash_leaves  = 512;

    MiningConfig() {
        rows_pattern = PeriodicPattern::default_rows();
        cols_pattern = PeriodicPattern::default_cols();
    }

    // 52-byte protocol serialization. MUST match C# MiningConfiguration.ToBytes.
    std::array<uint8_t, 52> to_bytes() const;

    // Dot-product length used for per-tile target scaling.
    uint32_t dot_product_length() const;

    // Difficulty adjustment factor = rows.size * cols.size * dot_product_length.
    uint64_t difficulty_adjustment_factor() const;

    // MiningConfig with smaller defaults for low-VRAM cards.
    static MiningConfig conservative();

    // Pick a shape that fits into the given free VRAM budget while keeping the
    // compute intensity high.  budget_bytes = 0 means "use defaults".
    static MiningConfig auto_shape_for_gpu(const cudaDeviceProp& prop,
                                            size_t budget_bytes = 0);

    uint32_t tensor_hash_num_blocks(int64_t matrix_bytes) const {
        uint32_t threads = tensor_hash_threads;
        uint32_t bytes_per_block = threads * 1024;
        return static_cast<uint32_t>((matrix_bytes + bytes_per_block - 1) / bytes_per_block);
    }
};

struct Job {
    std::array<uint8_t, 32> sigma{};
    std::array<uint8_t, 32> job_key{};   // BLAKE3(sigma || config_bytes)
    std::array<uint8_t, 32> b_seed{};
    uint32_t target_nbits = 0;
    uint32_t audit_k = 0;
    uint64_t block_height = 0;
    std::array<uint8_t, 16> job_id{};
    MiningConfig config{};
};

struct ShareFound {
    Job job;
    uint64_t nonce = 0;
    uint32_t tile_row = 0;
    uint32_t tile_col = 0;
    int mma_tile_m = 0;
    int mma_tile_n = 0;
    std::vector<uint32_t> a_row_indices;
    std::vector<uint32_t> b_col_indices;
    std::array<uint8_t, 32> hash_b{};     // BHash from GPU
    std::vector<uint8_t> a_slice;              // opened A rows (rows.size() * k)
    std::vector<uint8_t> a_opened_leaf_data; // opened leaves (leaf_indices.size() * 1024)
    std::vector<uint8_t> a_leaf_cvs;       // full A leaf CV table from GPU
    std::array<uint8_t, 32> claimed_hash{};
};

} // namespace pearl
