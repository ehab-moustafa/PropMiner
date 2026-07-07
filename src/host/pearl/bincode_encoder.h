#pragma once

#include <cstdint>
#include <vector>

#include "pearl_mining_wrapper.h"
#include "pearl_types.h"

namespace pearl {

// Bincode PlainProof encoder for Stratum mining.submit (ARC-compatible).
class BincodeEncoder {
public:
    // cert_version >= 2 (Kryptex MoE fork): append moe: Option::None (0x00).
    static std::vector<uint8_t> encode_plain_proof(
        const MiningConfig& cfg,
        const OwnedProof& a_proof,
        const OwnedProof& b_proof,
        const std::vector<uint32_t>& a_row_indices,
        const std::vector<uint32_t>& b_col_indices,
        const uint8_t hash_a[32],
        const uint8_t hash_b[32],
        int cert_version = 1);

private:
    static void write_u64(std::vector<uint8_t>& out, uint64_t v);
    static void write_bytes(std::vector<uint8_t>& out, const uint8_t* data, size_t len);
    static void write_merkle_proof(std::vector<uint8_t>& out,
                                   const OwnedProof& proof,
                                   const uint8_t root[32]);
    static void write_matrix_merkle_proof(std::vector<uint8_t>& out,
                                          const OwnedProof& proof,
                                          const uint8_t root[32],
                                          const std::vector<uint32_t>& indices);
};

}  // namespace pearl
