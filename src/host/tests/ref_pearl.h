#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <vector>

#include "../pearl/pearl_types.h"

namespace pearl {
namespace ref {

// Pure C++ reference implementation of Pearl primitives used in share
// verification.  Mirrors the Rust pearl-mining-capi and C# Akoya.Crypto code.
// This lets tests run without libpearl_mining_capi.so.

// Merkle tree and proof helpers using keyed BLAKE3.
struct RefMerkleProof {
    std::vector<std::array<uint8_t, 1024>> leaf_data;
    std::vector<uint32_t> leaf_indices;
    uint32_t total_leaves = 0;
    std::array<uint8_t, 32> root{};
    std::vector<std::array<uint8_t, 32>> siblings;
};

class RefMerkleTree {
public:
    RefMerkleTree() = default;
    RefMerkleTree(const uint8_t* data, size_t len, const uint8_t key[32]);

    const std::array<uint8_t, 32>& root() const { return root_; }
    uint32_t total_leaves() const { return static_cast<uint32_t>(layers_[0].size()); }

    RefMerkleProof proof_for_rows(
        const std::vector<uint32_t>& row_indices, size_t row_width) const;
    RefMerkleProof proof_for_indices(const std::vector<uint32_t>& indices) const;
    std::vector<uint8_t> audit_paths(const std::vector<uint32_t>& leaf_indices) const;

    static std::vector<uint32_t> leaf_indices_for_rows(
        const std::vector<uint32_t>& row_indices, size_t row_width);

private:
    std::array<uint8_t, 32> key_{};
    std::vector<std::vector<std::array<uint8_t, 32>>> layers_;
    std::vector<uint8_t> data_;
    std::array<uint8_t, 32> root_{};
};

// BSeed expansion: BLAKE3 XOF with int7 mapping.
std::vector<uint8_t> bseed_expand(const uint8_t bseed[32], size_t total_bytes);
std::vector<uint8_t> bseed_expand_range(
    const uint8_t bseed[32], uint64_t byte_offset, size_t len);

// Noise generation and jackpot hashing.
struct NoiseSeedPair {
    std::array<uint8_t, 32> b_noise_seed{};
    std::array<uint8_t, 32> a_noise_seed{};
};
NoiseSeedPair derive_noise_seeds(
    const uint8_t job_key[32], const uint8_t hashA[32], const uint8_t hashB[32]);

std::vector<int8_t> generate_uniform_random_matrix(
    const uint8_t seed_label[32], const uint8_t a_noise_seed[32],
    const std::vector<uint32_t>& row_indices, int num_cols);

struct PermutationPair { uint32_t first; uint32_t second; };
std::vector<PermutationPair> generate_permutation_matrix(
    const uint8_t seed_label[32], const uint8_t noise_seed[32],
    int k, int noise_rank);

std::vector<int8_t> apply_sparse_permutation(
    const std::vector<PermutationPair>& perm, const std::vector<int8_t>& vec);

// In-place sparse permutation into a pre-allocated output buffer.
// `out_len` must equal `perm.size()`.
void apply_sparse_permutation_into(
    const std::vector<PermutationPair>& perm, const int8_t* in,
    int8_t* out, size_t out_len);

std::array<uint8_t, 32> compute_claimed_hash(
    const MiningConfig& cfg,
    const std::vector<uint32_t>& a_row_indices,
    const std::vector<uint32_t>& b_col_indices,
    const std::vector<uint8_t>& a_slice,
    const uint8_t bseed[32],
    const uint8_t job_key[32],
    const uint8_t hashA[32],
    const uint8_t hashB[32]);

std::vector<uint32_t> derive_audit_indices(
    const uint8_t claimed_hash[32],
    const uint8_t b_seed[32],
    uint32_t audit_k,
    uint32_t total_leaves);

// Reference ShareFound structure used for deterministic CPU verification tests.
struct RefSyntheticShare {
    MiningConfig cfg;
    std::array<uint8_t, 32> sigma{};
    std::array<uint8_t, 32> bseed{};
    std::array<uint8_t, 32> job_key{};
    std::vector<uint32_t> a_rows;
    std::vector<uint32_t> b_cols;
    std::vector<uint8_t> a_data;
    RefMerkleTree a_tree;
    RefMerkleTree b_tree;
};

RefSyntheticShare make_synthetic_share(int m, int n, int k, int r);

// Concatenate 1024-byte leaves selected by row indices (each row starts at leaf boundary).
std::vector<uint8_t> read_leaves(
    const uint8_t* data, const std::vector<uint32_t>& leaf_indices);

} // namespace ref
} // namespace pearl
