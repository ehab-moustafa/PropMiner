#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#include "pearl_mining_wrapper.h"
#include "pearl_types.h"

namespace pearl {

// Minimal runtime protobuf wire encoder (no protoc dependency).
// Implements only the Pearl V2 messages required for ShareSubmission.
class ProtobufEncoder {
public:
    // Encode a ShareSubmission message.
    static std::vector<uint8_t> encode_share_submission(
        const ShareFound& share,
        const MiningConfig& cfg,
        const OwnedProof& a_proof,
        const OwnedProof& b_proof,
        const std::vector<uint8_t>& audit_siblings);

private:
    std::vector<uint8_t> buf_;

    static std::vector<uint8_t> encode_merkle_proof(const OwnedProof& proof);
    static std::vector<uint8_t> encode_audit_proof(const std::vector<uint8_t>& siblings);

    void write_varint(uint64_t v);
    void write_tag(uint32_t field, uint32_t wire);
    void write_bytes_field(uint32_t field, const uint8_t* data, size_t len);
    void write_string_field(uint32_t field, const std::string& s);
    void write_uint32_field(uint32_t field, uint32_t v);
    void write_int32_field(uint32_t field, int32_t v);
    void write_embedded_field(uint32_t field, const std::vector<uint8_t>& msg);

    static void append_bytes(std::vector<uint8_t>& out,
                              const uint8_t* data, size_t len);
};

} // namespace pearl
