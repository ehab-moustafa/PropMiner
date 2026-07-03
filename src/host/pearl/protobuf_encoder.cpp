#include "protobuf_encoder.h"

#include <cstring>

#include "pearl_types.h"

namespace pearl {

void ProtobufEncoder::write_varint(uint64_t v) {
    while (v >= 0x80) {
        buf_.push_back(static_cast<uint8_t>(v | 0x80));
        v >>= 7;
    }
    buf_.push_back(static_cast<uint8_t>(v));
}

void ProtobufEncoder::write_tag(uint32_t field, uint32_t wire) {
    write_varint((static_cast<uint64_t>(field) << 3) | wire);
}

void ProtobufEncoder::write_bytes_field(uint32_t field, const uint8_t* data, size_t len) {
    write_tag(field, 2);
    write_varint(len);
    append_bytes(buf_, data, len);
}

void ProtobufEncoder::write_embedded_field(uint32_t field, const std::vector<uint8_t>& msg) {
    write_tag(field, 2);
    write_varint(msg.size());
    append_bytes(buf_, msg.data(), msg.size());
}

void ProtobufEncoder::write_uint32_field(uint32_t field, uint32_t v) {
    write_tag(field, 5);
    for (int i = 0; i < 4; ++i) {
        buf_.push_back(static_cast<uint8_t>(v >> (8 * i)));
    }
}

void ProtobufEncoder::write_int32_field(uint32_t field, int32_t v) {
    // zigzag encoding for sint32 would need field decl; proto3 int32 is varint.
    write_tag(field, 0);
    write_varint(static_cast<uint64_t>(static_cast<uint32_t>(v)));
}

std::vector<uint8_t> ProtobufEncoder::encode_merkle_proof(const OwnedProof& proof) {
    ProtobufEncoder e;
    for (size_t i = 0; i < proof.leaf_count(); ++i) {
        e.write_bytes_field(1, proof.leaf_data.data() + i * 1024, 1024);
    }
    for (uint32_t idx : proof.leaf_indices) {
        e.write_uint32_field(2, idx);
    }
    e.write_uint32_field(3, proof.total_leaves);
    for (size_t i = 0; i < proof.sibling_count(); ++i) {
        e.write_bytes_field(4, proof.siblings.data() + i * 32, 32);
    }
    return e.buf_;
}

std::vector<uint8_t> ProtobufEncoder::encode_audit_proof(const std::vector<uint8_t>& siblings) {
    ProtobufEncoder e;
    e.write_bytes_field(1, siblings.data(), siblings.size());
    return e.buf_;
}

std::vector<uint8_t> ProtobufEncoder::encode_share_submission(
    const ShareFound& share,
    const MiningConfig& cfg,
    const OwnedProof& a_proof,
    const OwnedProof& b_proof,
    const std::vector<uint8_t>& audit_siblings) {

    ProtobufEncoder e;
    auto cfg_bytes = cfg.to_bytes();

    e.write_bytes_field(2, share.job.sigma.data(), share.job.sigma.size());
    e.write_bytes_field(3, cfg_bytes.data(), cfg_bytes.size());
    e.write_bytes_field(4, a_proof.root.data(), 32);
    e.write_bytes_field(5, share.hash_b.data(), 32);
    e.write_bytes_field(6, share.a_slice.data(), share.a_slice.size());

    // B slice: expand selected columns from BSeed.
    std::vector<uint8_t> b_slice(share.b_col_indices.size() * static_cast<size_t>(cfg.k));
    for (size_t i = 0; i < share.b_col_indices.size(); ++i) {
#if !defined(PROP_MINER_DISABLE_RUST_CRYPTO)
        uint32_t col = share.b_col_indices[i];
        uint64_t byte_offset = static_cast<uint64_t>(col) * cfg.k;
        MiningCapi mining;
        auto row = mining.bseed_expand_range(share.job.b_seed.data(), byte_offset, cfg.k);
        std::memcpy(b_slice.data() + i * cfg.k, row.data(), cfg.k);
#else
        (void)cfg;
        auto row = std::vector<uint8_t>(cfg.k, 0);
        std::memcpy(b_slice.data() + i * cfg.k, row.data(), cfg.k);
#endif
    }
    e.write_bytes_field(7, b_slice.data(), b_slice.size());

    auto a_proof_bytes = encode_merkle_proof(a_proof);
    e.write_embedded_field(8, a_proof_bytes);

    auto b_proof_bytes = encode_merkle_proof(b_proof);
    e.write_embedded_field(9, b_proof_bytes);

    e.write_bytes_field(10, share.claimed_hash.data(), 32);
    e.write_uint32_field(11, share.job.target_nbits);
    e.write_int32_field(12, static_cast<int32_t>(share.tile_row));
    e.write_int32_field(13, static_cast<int32_t>(share.tile_col));
    e.write_uint32_field(14, static_cast<uint32_t>(cfg.m));
    e.write_uint32_field(15, static_cast<uint32_t>(cfg.n));
    e.write_bytes_field(16, share.job.b_seed.data(), share.job.b_seed.size());

    if (!audit_siblings.empty()) {
        auto audit_bytes = encode_audit_proof(audit_siblings);
        e.write_embedded_field(17, audit_bytes);
    }

    return e.buf_;
}

void ProtobufEncoder::append_bytes(std::vector<uint8_t>& out,
                                    const uint8_t* data, size_t len) {
    out.insert(out.end(), data, data + len);
}

} // namespace pearl
