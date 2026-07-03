#include "pearl_mining_wrapper.h"

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>

namespace pearl {

namespace {
    void check_err(int rc, char* err) {
        if (rc == 0) {
            if (err) pearl_capi_free_string(err);
            return;
        }
        std::string msg = "pearl_mining_capi call failed";
        if (err) {
            msg += ": ";
            msg += err;
            pearl_capi_free_string(err);
        }
        throw std::runtime_error(msg);
    }
}

MerkleTree::MerkleTree(pearl_mining_capi_tree_t* t, Type type)
    : tree_(t), type_(type) {
    if (!t) throw std::invalid_argument("null tree handle");
}
MerkleTree::~MerkleTree() {
    if (!tree_) return;
    if (type_ == Type::BSeed)
        pearl_capi_bseed_merkle_tree_free(tree_);
    else
        pearl_capi_merkle_tree_free(tree_);
}
MerkleTree::MerkleTree(MerkleTree&& other) noexcept
    : tree_(other.tree_), type_(other.type_),
      root_(other.root_), total_leaves_(other.total_leaves_) {
    other.tree_ = nullptr;
}
MerkleTree& MerkleTree::operator=(MerkleTree&& other) noexcept {
    if (this != &other) {
        if (tree_) {
            if (type_ == Type::BSeed)
                pearl_capi_bseed_merkle_tree_free(tree_);
            else
                pearl_capi_merkle_tree_free(tree_);
        }
        tree_ = other.tree_;
        type_ = other.type_;
        root_ = other.root_;
        total_leaves_ = other.total_leaves_;
        other.tree_ = nullptr;
    }
    return *this;
}

MiningCapi::MiningCapi() {
    if (pearl_capi_version() < 3) {
        throw std::runtime_error("libpearl_mining_capi ABI version too old (need >= 3 for unkeyed BLAKE3/XOF)");
    }
}

void MiningCapi::blake3_keyed(const uint8_t* data, size_t len,
                              const uint8_t key[32], uint8_t out[32]) const {
    char* err = nullptr;
    int rc = pearl_capi_blake3_keyed(data, len, key, out, &err);
    check_err(rc, err);
}

void MiningCapi::blake3_hash(const uint8_t* data, size_t len, uint8_t out[32]) const {
    char* err = nullptr;
    int rc = pearl_capi_blake3_hash(data, len, out, &err);
    check_err(rc, err);
}

void MiningCapi::blake3_xof(const uint8_t* data, size_t len,
                            uint8_t* out, size_t out_len) const {
    char* err = nullptr;
    int rc = pearl_capi_blake3_xof(data, len, out, out_len, &err);
    check_err(rc, err);
}

MerkleTree MiningCapi::build_b_host_tree(
    const uint8_t bseed[32],
    uint64_t n,
    uint64_t k,
    const uint8_t job_key[32]) const {
    size_t total = n * k;
    std::vector<uint8_t> b(total);
    char* err = nullptr;
    int rc = pearl_capi_bseed_expand_raw(bseed, n, k, b.data(), b.size(), &err);
    check_err(rc, err);

    void* handle = nullptr;
    uint32_t total_leaves = 0;
    std::array<uint8_t, 32> root{};
    rc = pearl_capi_merkle_build_tree(
        b.data(), b.size(),
        job_key, static_cast<size_t>(k),
        &handle, root.data(), &total_leaves, &err);
    check_err(rc, err);

    MerkleTree tree(static_cast<pearl_mining_capi_tree_t*>(handle), MerkleTree::Type::Host);
    tree.root_ = root;
    tree.total_leaves_ = total_leaves;
    return tree;
}

MerkleTree MiningCapi::build_bseed_tree_from_leaf_cvs(
    const std::vector<uint8_t>& leaf_cvs,
    const uint8_t bseed[32],
    const uint8_t job_key[32],
    uint64_t num_rows,
    uint64_t row_width) const {
    void* handle = nullptr;
    uint32_t total_leaves = 0;
    std::array<uint8_t, 32> root{};
    char* err = nullptr;
    int rc = pearl_capi_bseed_merkle_build_tree_from_leaf_cvs(
        leaf_cvs.data(), leaf_cvs.size(),
        bseed,
        job_key,
        static_cast<size_t>(num_rows),
        static_cast<size_t>(row_width),
        &handle, root.data(), &total_leaves, &err);
    check_err(rc, err);

    MerkleTree tree(static_cast<pearl_mining_capi_tree_t*>(handle), MerkleTree::Type::BSeed);
    tree.root_ = root;
    tree.total_leaves_ = total_leaves;
    return tree;
}

OwnedProof MiningCapi::a_root_and_proof(
    const uint8_t* data,
    uint64_t rows,
    uint64_t cols,
    const uint8_t job_key[32],
    const std::vector<uint32_t>& row_indices) const {
    OwnedProof p;
    char* err = nullptr;
    size_t leaf_count = 0, leaf_indices_len = 0, sibling_count = 0;
    uint32_t total_leaves = 0;
    std::array<uint8_t, 32> root{};
    uint8_t* leaf_data_ptr = nullptr;
    uint32_t* indices_ptr = nullptr;
    uint8_t* siblings_ptr = nullptr;

    int rc = pearl_capi_merkle_root_and_proof(
        data, static_cast<size_t>(rows * cols),
        job_key,
        row_indices.data(), row_indices.size(),
        static_cast<size_t>(cols),
        root.data(),
        &total_leaves,
        &leaf_data_ptr, &leaf_count,
        &indices_ptr, &leaf_indices_len,
        &siblings_ptr, &sibling_count,
        &err);
    check_err(rc, err);

    p.total_leaves = total_leaves;
    p.root.assign(root.begin(), root.end());
    if (leaf_data_ptr && leaf_count) {
        p.leaf_data.assign(leaf_data_ptr, leaf_data_ptr + leaf_count * 1024);
        pearl_capi_free_buffer(leaf_data_ptr, leaf_count * 1024);
    }
    if (indices_ptr && leaf_indices_len) {
        p.leaf_indices.assign(indices_ptr, indices_ptr + leaf_indices_len);
        pearl_capi_free_u32_buffer(indices_ptr, leaf_indices_len);
    }
    if (siblings_ptr && sibling_count) {
        p.siblings.assign(siblings_ptr, siblings_ptr + sibling_count * 32);
        pearl_capi_free_buffer(siblings_ptr, sibling_count * 32);
    }
    return p;
}

OwnedProof MiningCapi::a_proof_from_leaf_cvs(
    const std::vector<uint8_t>& leaf_cvs,
    const std::vector<uint8_t>& leaf_data,
    const uint8_t job_key[32],
    uint64_t num_rows,
    uint64_t row_width,
    const std::vector<uint32_t>& row_indices) const {
    OwnedProof p;
    char* err = nullptr;
    uint32_t total_leaves = 0;
    size_t leaf_indices_len = 0, sibling_count = 0;
    std::array<uint8_t, 32> root{};
    uint32_t* indices_ptr = nullptr;
    uint8_t* siblings_ptr = nullptr;

    int rc = pearl_capi_merkle_proof_from_leaf_cvs(
        leaf_cvs.data(), leaf_cvs.size(),
        leaf_data.data(), leaf_data.size(),
        job_key,
        row_indices.data(), row_indices.size(),
        static_cast<size_t>(num_rows),
        static_cast<size_t>(row_width),
        root.data(),
        &total_leaves,
        &indices_ptr, &leaf_indices_len,
        &siblings_ptr, &sibling_count,
        &err);
    check_err(rc, err);

    p.total_leaves = total_leaves;
    p.root.assign(root.begin(), root.end());
    p.leaf_data = leaf_data;
    if (indices_ptr && leaf_indices_len) {
        p.leaf_indices.assign(indices_ptr, indices_ptr + leaf_indices_len);
        pearl_capi_free_u32_buffer(indices_ptr, leaf_indices_len);
    }
    if (siblings_ptr && sibling_count) {
        p.siblings.assign(siblings_ptr, siblings_ptr + sibling_count * 32);
        pearl_capi_free_buffer(siblings_ptr, sibling_count * 32);
    }
    return p;
}

OwnedProof MiningCapi::proof_for_handle(
    const MerkleTree& tree,
    const std::vector<uint32_t>& indices) const {
    OwnedProof p;
    char* err = nullptr;
    size_t leaf_count = 0, leaf_indices_len = 0, sibling_count = 0;
    uint8_t* leaf_data_ptr = nullptr;
    uint32_t* indices_ptr = nullptr;
    uint8_t* siblings_ptr = nullptr;

    int rc = 0;
    if (tree.type() == MerkleTree::Type::BSeed) {
        rc = pearl_capi_bseed_merkle_proof_for_handle(
            tree.get(),
            indices.data(), indices.size(),
            &leaf_data_ptr, &leaf_count,
            &indices_ptr, &leaf_indices_len,
            &siblings_ptr, &sibling_count,
            &err);
    } else {
        rc = pearl_capi_merkle_proof_for_handle(
            tree.get(),
            indices.data(), indices.size(),
            &leaf_data_ptr, &leaf_count,
            &indices_ptr, &leaf_indices_len,
            &siblings_ptr, &sibling_count,
            &err);
    }
    check_err(rc, err);

    if (leaf_data_ptr && leaf_count) {
        p.leaf_data.assign(leaf_data_ptr, leaf_data_ptr + leaf_count * 1024);
        pearl_capi_free_buffer(leaf_data_ptr, leaf_count * 1024);
    }
    if (indices_ptr && leaf_indices_len) {
        p.leaf_indices.assign(indices_ptr, indices_ptr + leaf_indices_len);
        pearl_capi_free_u32_buffer(indices_ptr, leaf_indices_len);
    }
    if (siblings_ptr && sibling_count) {
        p.siblings.assign(siblings_ptr, siblings_ptr + sibling_count * 32);
        pearl_capi_free_buffer(siblings_ptr, sibling_count * 32);
    }
    p.total_leaves = tree.total_leaves();
    p.root.assign(tree.root(), tree.root() + 32);
    return p;
}

std::vector<uint8_t> MiningCapi::audit_paths_for_handle(
    const MerkleTree& tree,
    const std::vector<uint32_t>& leaf_indices) const {
    char* err = nullptr;
    size_t bytes = 0;
    uint8_t* ptr = nullptr;
    int rc = 0;
    if (tree.type() == MerkleTree::Type::BSeed) {
        rc = pearl_capi_bseed_merkle_audit_paths_for_handle(
            tree.get(),
            leaf_indices.data(), leaf_indices.size(),
            &ptr, &bytes, &err);
    } else {
        rc = pearl_capi_merkle_audit_paths_for_handle(
            tree.get(),
            leaf_indices.data(), leaf_indices.size(),
            &ptr, &bytes, &err);
    }
    check_err(rc, err);
    std::vector<uint8_t> out;
    if (ptr && bytes) {
        out.assign(ptr, ptr + bytes);
        pearl_capi_free_buffer(ptr, bytes);
    }
    return out;
}

std::vector<uint8_t> MiningCapi::bseed_expand_range(
    const uint8_t bseed[32],
    uint64_t byte_offset,
    uint64_t n) const {
    std::vector<uint8_t> out(n);
    char* err = nullptr;
    int rc = pearl_capi_bseed_expand_range_raw(
        bseed, byte_offset, out.data(), out.size(), &err);
    check_err(rc, err);
    return out;
}

bool MiningCapi::verify_proof(const OwnedProof& proof,
                              const uint8_t job_key[32],
                              const uint8_t expected_root[32]) const {
    char* err = nullptr;
    int rc = pearl_capi_merkle_verify_proof(
        proof.leaf_data.data(),
        proof.leaf_indices.data(),
        proof.leaf_count(),
        proof.total_leaves,
        proof.siblings.data(),
        proof.sibling_count(),
        job_key,
        expected_root,
        &err);
    if (err) pearl_capi_free_string(err);
    return rc == 0;
}

} // namespace pearl
