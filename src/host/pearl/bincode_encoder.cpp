#include "bincode_encoder.h"

#include <cstdlib>
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

void BincodeEncoder::write_merkle_proof(std::vector<uint8_t>& out,
                                        const OwnedProof& proof,
                                        const uint8_t root[32]) {
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
    write_bytes(out, root, 32);

    const size_t sib_count = proof.sibling_count();
    write_u64(out, sib_count);
    for (size_t i = 0; i < sib_count; ++i) {
        write_bytes(out, proof.siblings.data() + i * 32, 32);
    }
}

void BincodeEncoder::write_matrix_merkle_proof(std::vector<uint8_t>& out,
                                               const OwnedProof& proof,
                                               const uint8_t root[32],
                                               const std::vector<uint32_t>& indices) {
    write_merkle_proof(out, proof, root);
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
    const std::vector<uint32_t>& b_col_indices,
    const uint8_t hash_a[32],
    const uint8_t hash_b[32],
    int cert_version) {
    std::vector<uint8_t> out;
    out.reserve(4096);

    write_u64(out, static_cast<uint64_t>(cfg.m));
    write_u64(out, static_cast<uint64_t>(cfg.n));
    write_u64(out, static_cast<uint64_t>(cfg.k));
    write_u64(out, static_cast<uint64_t>(cfg.r));

    // ARC StratumSession.WriteMatrixMerkleProof uses share.HashA/HashB as wire roots.
    write_matrix_merkle_proof(out, a_proof, hash_a, a_row_indices);
    write_matrix_merkle_proof(out, b_proof, hash_b, b_col_indices);
    // Kryptex object-notify cert_version=2 is a proof *version* label, not the
    // MoE-fork PlainProof shape. ARC BincodeSerializer has no moe suffix; only
    // append when PROPMINER_PLAINPROOF_MOE_SUFFIX=1 (future MoE pools).
    (void)cert_version;
    if (const char* env = std::getenv("PROPMINER_PLAINPROOF_MOE_SUFFIX");
        env && (env[0] == '1' || env[0] == 'y' || env[0] == 'Y')) {
        out.push_back(0x00);  // moe: Option::None
    }
    return out;
}

}  // namespace pearl
