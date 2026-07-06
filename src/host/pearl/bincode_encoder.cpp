#include "bincode_encoder.h"

#include <cstring>

namespace pearl {

void BincodeEncoder::write_u64(std::vector<uint8_t>& out, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<uint8_t>(v >> (8 * i)));
    }
}

void BincodeEncoder::write_bytes(std::vector<uint8_t>& out, const uint8_t* data, size_t len) {
    out.insert(out.end(), data, data + len);
}

void BincodeEncoder::write_merkle_proof(std::vector<uint8_t>& out, const OwnedProof& proof) {
    const size_t leaf_count = proof.leaf_count();
    write_u64(out, leaf_count);
    for (size_t i = 0; i < leaf_count; ++i) {
        const uint8_t* leaf = proof.leaf_data.data() + i * 1024;
        write_u64(out, 1024);
        write_bytes(out, leaf, 1024);
    }

    write_u64(out, proof.leaf_indices.size());
    for (uint32_t idx : proof.leaf_indices) {
        write_u64(out, idx);
    }

    write_u64(out, proof.total_leaves);

    if (proof.root.size() >= 32) {
        write_bytes(out, proof.root.data(), 32);
    } else {
        uint8_t zero[32] = {};
        write_bytes(out, zero, 32);
    }

    const size_t sib_count = proof.sibling_count();
    write_u64(out, sib_count);
    for (size_t i = 0; i < sib_count; ++i) {
        write_bytes(out, proof.siblings.data() + i * 32, 32);
    }
}

void BincodeEncoder::write_matrix_merkle_proof(std::vector<uint8_t>& out,
                                               const OwnedProof& proof,
                                               const std::vector<uint32_t>& indices) {
    write_merkle_proof(out, proof);
    write_u64(out, indices.size());
    for (uint32_t idx : indices) {
        write_u64(out, idx);
    }
}

std::vector<uint8_t> BincodeEncoder::encode_plain_proof(
    const MiningConfig& cfg,
    const OwnedProof& a_proof,
    const OwnedProof& b_proof,
    const std::vector<uint32_t>& a_row_indices,
    const std::vector<uint32_t>& b_col_indices) {
    std::vector<uint8_t> out;
    out.reserve(4096);

    write_u64(out, static_cast<uint64_t>(cfg.m));
    write_u64(out, static_cast<uint64_t>(cfg.n));
    write_u64(out, static_cast<uint64_t>(cfg.k));
    write_u64(out, static_cast<uint64_t>(cfg.r));

    write_matrix_merkle_proof(out, a_proof, a_row_indices);
    write_matrix_merkle_proof(out, b_proof, b_col_indices);
    return out;
}

}  // namespace pearl
