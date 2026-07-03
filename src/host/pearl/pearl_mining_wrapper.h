#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <memory>
#include <vector>

#include "pearl_mining_capi.h"

namespace pearl {

using pearl_mining_capi_tree_t = void;

// RAII owned buffers returned by the Rust API.
struct OwnedProof {
    std::vector<uint8_t> root;           // 32 bytes
    std::vector<uint8_t> leaf_data;      // leaf_count * 1024 bytes
    std::vector<uint32_t> leaf_indices;
    std::vector<uint8_t> siblings;       // sibling_count * 32 bytes
    uint32_t total_leaves = 0;

    size_t leaf_count() const { return leaf_data.size() / 1024; }
    size_t sibling_count() const { return siblings.size() / 32; }
};

// RAII wrapper around a pearl_mining_capi_tree_t*.
class MerkleTree {
public:
    enum class Type { Host, BSeed };

    explicit MerkleTree(pearl_mining_capi_tree_t* t, Type type);
    ~MerkleTree();
    MerkleTree(const MerkleTree&) = delete;
    MerkleTree& operator=(const MerkleTree&) = delete;
    MerkleTree(MerkleTree&& other) noexcept;
    MerkleTree& operator=(MerkleTree&& other) noexcept;

    pearl_mining_capi_tree_t* get() const { return tree_; }
    const uint8_t* root() const { return root_.data(); }
    uint32_t total_leaves() const { return total_leaves_; }
    Type type() const { return type_; }

private:
    pearl_mining_capi_tree_t* tree_;
    Type type_;
    std::array<uint8_t, 32> root_;
    uint32_t total_leaves_ = 0;

    friend class MiningCapi;
};

// Wrapper for the Rust mining C API.
class MiningCapi {
public:
    MiningCapi();

    // Direct keyed BLAKE3 hash.
    void blake3_keyed(const uint8_t* data, size_t len,
                      const uint8_t key[32], uint8_t out[32]) const;

    // Standard (unkeyed) BLAKE3 hash — required for jobKey/noise-seed compatibility.
    void blake3_hash(const uint8_t* data, size_t len, uint8_t out[32]) const;

    // BLAKE3 XOF output. out_len bytes written.
    void blake3_xof(const uint8_t* data, size_t len,
                    uint8_t* out, size_t out_len) const;

    // Host-side B expansion + tree build (fallback when no BSeed leaf CVs).
    MerkleTree build_b_host_tree(
        const uint8_t bseed[32],
        uint64_t n,
        uint64_t k,
        const uint8_t job_key[32]) const;

    // BSeed-backed tree from GPU tensor_hash leaf CVs (preferred).
    MerkleTree build_bseed_tree_from_leaf_cvs(
        const std::vector<uint8_t>& leaf_cvs,
        const uint8_t bseed[32],
        const uint8_t job_key[32],
        uint64_t num_rows,
        uint64_t row_width) const;

    // Host-side A tree build + proof (fallback).
    OwnedProof a_root_and_proof(
        const uint8_t* data,
        uint64_t rows,
        uint64_t cols,
        const uint8_t job_key[32],
        const std::vector<uint32_t>& row_indices) const;

    // Fast A proof from GPU leaf CVs + opened leaf chunks.
    OwnedProof a_proof_from_leaf_cvs(
        const std::vector<uint8_t>& leaf_cvs,
        const std::vector<uint8_t>& leaf_data,
        const uint8_t job_key[32],
        uint64_t num_rows,
        uint64_t row_width,
        const std::vector<uint32_t>& row_indices) const;

    // Extract B proof from a pre-built handle.
    OwnedProof proof_for_handle(
        const MerkleTree& tree,
        const std::vector<uint32_t>& indices) const;

    // Audit paths (returns sibling bytes).
    std::vector<uint8_t> audit_paths_for_handle(
        const MerkleTree& tree,
        const std::vector<uint32_t>& leaf_indices) const;

    // BSeed range expansion on host.
    std::vector<uint8_t> bseed_expand_range(
        const uint8_t bseed[32],
        uint64_t byte_offset,
        uint64_t n) const;

    // Verify a proof (useful for tests).
    bool verify_proof(const OwnedProof& proof,
                      const uint8_t job_key[32],
                      const uint8_t expected_root[32]) const;
};

} // namespace pearl
