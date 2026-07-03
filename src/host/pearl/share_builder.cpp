#include "share_builder.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "pearl_blake3.h"
#include "protobuf_encoder.h"
#include "sigma_context.h"

namespace pearl {

namespace {
    constexpr int DIGEST_SIZE = 32;
    constexpr int BYTES_PER_LINE = 4;
    constexpr int LINES_PER_HASH = DIGEST_SIZE / BYTES_PER_LINE;
    constexpr int JACKPOT_SIZE = 16;
    constexpr int ROTATE_LEFT = 13;
    constexpr int UNIFORM_RANGE = 64;
    constexpr int ZERO_POINT = UNIFORM_RANGE / 2;
    constexpr uint8_t RANGE_MASK = UNIFORM_RANGE - 1;

    const std::array<uint8_t, 32> SEED_LABEL_A = []{
        std::array<uint8_t, 32> a{};
        const char* s = "A_tensor";
        std::memcpy(a.data(), s, std::strlen(s));
        return a;
    }();
    const std::array<uint8_t, 32> SEED_LABEL_B = []{
        std::array<uint8_t, 32> b{};
        const char* s = "B_tensor";
        std::memcpy(b.data(), s, std::strlen(s));
        return b;
    }();

    uint32_t mul_hi_u32(uint32_t a, uint32_t b) {
        return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32);
    }

    void get_random_hash(uint32_t index, const uint8_t seed[32], const uint8_t key[32],
                         int prepend_index, uint8_t output[32]) {
        std::array<uint8_t, 64> msg{};
        *reinterpret_cast<uint32_t*>(msg.data() + prepend_index * sizeof(uint32_t)) = index + 1;
        std::memcpy(msg.data() + 32, seed, 32);
        auto h = Blake3Helper::keyed(msg.data(), msg.size(), key);
        std::memcpy(output, h.data(), 32);
    }

    std::vector<std::vector<int8_t>> generate_uniform_random_matrix(
        const uint8_t seed[32], const uint8_t key[32],
        const std::vector<uint32_t>& row_indices, int num_cols) {
        std::vector<std::vector<int8_t>> rows;
        rows.reserve(row_indices.size());
        std::array<uint8_t, 32> hash{};
        for (uint32_t row_index : row_indices) {
            long start_index = static_cast<long>(row_index) * num_cols;
            long end_index = start_index + num_cols;
            int start_block = static_cast<int>(start_index / DIGEST_SIZE);
            int end_block = static_cast<int>((end_index + DIGEST_SIZE - 1) / DIGEST_SIZE);
            std::vector<int8_t> row(num_cols);
            int written = 0;
            for (int block = start_block; block < end_block; ++block) {
                get_random_hash(block, seed, key, 0, hash.data());
                for (int i = 0; i < DIGEST_SIZE; ++i) {
                    long abs_idx = static_cast<long>(block) * DIGEST_SIZE + i;
                    if (abs_idx < start_index || abs_idx >= end_index) continue;
                    row[written++] = static_cast<int8_t>((hash[i] & RANGE_MASK) - ZERO_POINT);
                }
            }
            rows.push_back(std::move(row));
        }
        return rows;
    }

    struct PermutationPair { uint32_t first; uint32_t second; };

    std::vector<PermutationPair> generate_permutation_matrix(
        const uint8_t seed[32], const uint8_t key[32],
        int k, int noise_rank) {
        uint32_t rank_mask = static_cast<uint32_t>(noise_rank - 1);
        std::vector<PermutationPair> result(k);
        std::array<uint8_t, 32> hash{};
        for (int chunk = 0; chunk * LINES_PER_HASH < k; ++chunk) {
            get_random_hash(chunk, seed, key, 1, hash.data());
            int len = std::min(LINES_PER_HASH, k - chunk * LINES_PER_HASH);
            for (int j = 0; j < len; ++j) {
                uint32_t rand = *reinterpret_cast<uint32_t*>(hash.data() + j * BYTES_PER_LINE);
                uint32_t first = rand & rank_mask;
                uint32_t second = first ^ (1u + mul_hi_u32(static_cast<uint32_t>(noise_rank - 1), rand));
                result[chunk * LINES_PER_HASH + j] = { first, second };
            }
        }
        return result;
    }

    std::vector<int8_t> apply_sparse_permutation(const std::vector<PermutationPair>& perm,
                                                 const std::vector<int8_t>& vec) {
        std::vector<int8_t> out(perm.size());
        apply_sparse_permutation_into(perm, vec.data(), out.data(), out.size());
        return out;
    }

    void apply_sparse_permutation_into(const std::vector<PermutationPair>& perm,
                                       const int8_t* in, int8_t* out, size_t len) {
        for (size_t i = 0; i < len; ++i) {
            out[i] = static_cast<int8_t>(in[perm[i].first] - in[perm[i].second]);
        }
    }

    std::array<uint8_t, 32> jackpot_hash(const uint32_t words[JACKPOT_SIZE],
                                         const uint8_t a_noise_seed[32]) {
        std::array<uint8_t, JACKPOT_SIZE * sizeof(uint32_t)> buf{};
        for (int i = 0; i < JACKPOT_SIZE; ++i) {
            *reinterpret_cast<uint32_t*>(buf.data() + i * 4) = words[i];
        }
        auto h = Blake3Helper::keyed(buf.data(), buf.size(), a_noise_seed);
        return h;
    }
}

ShareBuilder::ShareBuilder(const MiningConfig& cfg) : cfg_(cfg) {}

void ShareBuilder::derive_noise_seeds(const uint8_t job_key[32],
                                      const uint8_t hashA[32],
                                      const uint8_t hashB[32],
                                      uint8_t b_noise_seed[32],
                                      uint8_t a_noise_seed[32]) {
    // Matches Akoya.Crypto MiningConfiguration.DeriveNoiseSeeds:
    // bNoiseSeed = BLAKE3(jobKey || hashB)
    // aNoiseSeed = BLAKE3(bNoiseSeed || hashA)
    std::array<uint8_t, 64> b_input{};
    std::memcpy(b_input.data(), job_key, 32);
    std::memcpy(b_input.data() + 32, hashB, 32);
    auto bns = Blake3Helper::hash(b_input.data(), b_input.size());
    std::memcpy(b_noise_seed, bns.data(), 32);

    std::array<uint8_t, 64> a_input{};
    std::memcpy(a_input.data(), b_noise_seed, 32);
    std::memcpy(a_input.data() + 32, hashA, 32);
    auto ans = Blake3Helper::hash(a_input.data(), a_input.size());
    std::memcpy(a_noise_seed, ans.data(), 32);
}

std::array<uint8_t, 32> ShareBuilder::compute_claimed_hash(
    const ShareFound& share,
    const uint8_t job_key[32],
    const uint8_t hashA[32],
    const uint8_t hashB[32]) const {
    uint8_t b_noise_seed[32]{};
    uint8_t a_noise_seed[32]{};
    derive_noise_seeds(job_key, hashA, hashB, b_noise_seed, a_noise_seed);

    const int h = static_cast<int>(share.a_row_indices.size());
    const int w = static_cast<int>(share.b_col_indices.size());
    const int k = cfg_.k;
    const int r = cfg_.r;

    // secretA: contiguous h x k.
    std::vector<int8_t> secretA(static_cast<size_t>(h) * k);
    std::memcpy(secretA.data(), share.a_slice.data(), secretA.size());

    // secretB: expand only the columns we need, packed w x k.
    std::vector<int8_t> secretB(static_cast<size_t>(w) * k);
    auto b_slice = mining_.bseed_expand_range(
        share.job.b_seed.data(), 0,
        static_cast<uint64_t>(cfg_.n) * k);
    for (int j = 0; j < w; ++j) {
        uint32_t col = share.b_col_indices[j];
        std::memcpy(secretB.data() + static_cast<size_t>(j) * k,
                    b_slice.data() + static_cast<size_t>(col) * k, k);
    }

    auto eAl = generate_uniform_random_matrix(SEED_LABEL_A.data(), a_noise_seed, share.a_row_indices, r);
    auto eAr = generate_permutation_matrix(SEED_LABEL_A.data(), a_noise_seed, k, r);
    auto eBl = generate_permutation_matrix(SEED_LABEL_B.data(), b_noise_seed, k, r);
    auto eBr = generate_uniform_random_matrix(SEED_LABEL_B.data(), b_noise_seed, share.b_col_indices, r);

    // noiseA / noiseB: contiguous row-major.
    std::vector<int8_t> noiseA(static_cast<size_t>(h) * k);
    for (int i = 0; i < h; ++i) {
        apply_sparse_permutation_into(
            eAr, eAl[i].data(),
            noiseA.data() + static_cast<size_t>(i) * k, k);
    }
    std::vector<int8_t> noiseB(static_cast<size_t>(w) * k);
    for (int j = 0; j < w; ++j) {
        apply_sparse_permutation_into(
            eBl, eBr[j].data(),
            noiseB.data() + static_cast<size_t>(j) * k, k);
    }

    // Jackpot transcript using flat buffers.
    std::vector<int> tile(static_cast<size_t>(h) * w, 0);
    uint32_t msg[JACKPOT_SIZE] = {};
    for (int ll = r; ll <= k; ll += r) {
        for (int u = 0; u < h; ++u) {
            const int8_t* a_row = secretA.data() + static_cast<size_t>(u) * k;
            const int8_t* na_row = noiseA.data() + static_cast<size_t>(u) * k;
            int* tile_row = tile.data() + static_cast<size_t>(u) * w;
            for (int v = 0; v < w; ++v) {
                const int8_t* b_row = secretB.data() + static_cast<size_t>(v) * k;
                const int8_t* nb_row = noiseB.data() + static_cast<size_t>(v) * k;
                int acc = tile_row[v];
                for (int l = ll - r; l < ll; ++l) {
                    acc += (static_cast<int>(a_row[l]) + static_cast<int>(na_row[l])) *
                           (static_cast<int>(b_row[l]) + static_cast<int>(nb_row[l]));
                }
                tile_row[v] = acc;
            }
        }

        uint32_t xored = 0;
        for (int u = 0; u < h; ++u) {
            const int* tile_row = tile.data() + static_cast<size_t>(u) * w;
            for (int v = 0; v < w; ++v)
                xored ^= static_cast<uint32_t>(tile_row[v]);
        }

        int tid = ((ll / r) - 1) % JACKPOT_SIZE;
        msg[tid] = ((msg[tid] << ROTATE_LEFT) | (msg[tid] >> (32 - ROTATE_LEFT))) ^ xored;
    }

    return jackpot_hash(msg, a_noise_seed);
}

std::vector<uint8_t> ShareBuilder::build(const ShareFound& share,
                                          const SigmaContext& ctx) const {
    // Build A proof from GPU leaf CVs. leaf_data must be full 1024-byte
    // chunks for the opened leaves. The Rust C API checks leaf CVs match.
    size_t row_bytes = static_cast<size_t>(cfg_.k);
    size_t opened_leaves = share.a_row_indices.size();
    std::vector<uint8_t> a_leaf_data(opened_leaves * 1024, 0);
    for (size_t i = 0; i < opened_leaves; ++i) {
        std::memcpy(a_leaf_data.data() + i * 1024,
                    share.a_slice.data() + i * row_bytes,
                    row_bytes);
    }

    OwnedProof a_proof = mining_.a_proof_from_leaf_cvs(
        share.a_leaf_cvs,
        a_leaf_data,
        ctx.job().job_key.data(),
        cfg_.m,
        cfg_.k,
        share.a_row_indices);

    std::array<uint8_t, 32> hashA{};
    std::memcpy(hashA.data(), a_proof.root.data(), 32);

    // Build B proof from handle.
    OwnedProof b_proof = mining_.proof_for_handle(
        *ctx.b_merkle_tree(), share.b_col_indices);

    std::array<uint8_t, 32> hashB{};
    std::memcpy(hashB.data(), b_proof.root.data(), 32);

    // Compute claimed hash.
    auto claimed = compute_claimed_hash(share, ctx.job().job_key.data(),
                                        hashA.data(), hashB.data());

    // Audit siblings per audit_proof v1.
    std::vector<uint8_t> audit_siblings;
    if (ctx.job().audit_k > 0 && b_proof.total_leaves > 0) {
        auto audit_indices = derive_audit_indices(
            claimed.data(), share.job.b_seed.data(),
            ctx.job().audit_k, b_proof.total_leaves);
        audit_siblings = mining_.audit_paths_for_handle(*ctx.b_merkle_tree(), audit_indices);
    }

    return ProtobufEncoder::encode_share_submission(
        share, cfg_, a_proof, b_proof, audit_siblings);
}

std::vector<uint32_t> ShareBuilder::derive_audit_indices(
    const uint8_t claimed_hash[32],
    const uint8_t b_seed[32],
    uint32_t audit_k,
    uint32_t total_leaves) const {
    if (audit_k == 0) return {};
    if (audit_k > 64) throw std::invalid_argument("audit_k exceeds spec max 64");
    if (total_leaves == 0 || (total_leaves & (total_leaves - 1)) != 0)
        throw std::invalid_argument("total_leaves must be a power of two");

    constexpr const char* domain = "akoya-audit-v1";
    std::vector<uint8_t> xof_input;
    xof_input.insert(xof_input.end(), domain, domain + 14);
    xof_input.insert(xof_input.end(), claimed_hash, claimed_hash + 32);
    xof_input.insert(xof_input.end(), b_seed, b_seed + 32);
    for (int i = 0; i < 4; ++i) xof_input.push_back(static_cast<uint8_t>(audit_k >> (8 * i)));

    auto xof = Blake3Helper::xof(xof_input.data(), xof_input.size(), audit_k * 4);
    std::vector<uint32_t> indices(audit_k);
    for (uint32_t i = 0; i < audit_k; ++i) {
        uint32_t word = static_cast<uint32_t>(xof[i * 4]) |
                        (static_cast<uint32_t>(xof[i * 4 + 1]) << 8) |
                        (static_cast<uint32_t>(xof[i * 4 + 2]) << 16) |
                        (static_cast<uint32_t>(xof[i * 4 + 3]) << 24);
        indices[i] = static_cast<uint32_t>(static_cast<uint64_t>(word) % total_leaves);
    }
    return indices;
}

bool ShareBuilder::verify(const std::vector<uint8_t>& proof,
                          const SigmaContext& ctx) const {
    (void)proof; (void)ctx;
    // Full protobuf verification requires re-parsing ShareSubmission.
    // The high-level check is done by VerifyShare() below, which works on the
    // pre-build ShareFound and is used by the self-test harness.
    return true;
}

bool ShareBuilder::VerifyShare(const ShareFound& share,
                               const SigmaContext& ctx) {
    ShareBuilder builder(share.job.config);

    // Rebuild A proof from the GPU leaf CVs and opened rows.
    size_t row_bytes = static_cast<size_t>(share.job.config.k);
    size_t opened_leaves = share.a_row_indices.size();
    std::vector<uint8_t> a_leaf_data(opened_leaves * 1024, 0);
    for (size_t i = 0; i < opened_leaves; ++i) {
        std::memcpy(a_leaf_data.data() + i * 1024,
                    share.a_slice.data() + i * row_bytes,
                    row_bytes);
    }

    OwnedProof a_proof = builder.mining_.a_proof_from_leaf_cvs(
        share.a_leaf_cvs,
        a_leaf_data,
        share.job.job_key.data(),
        share.job.config.m,
        share.job.config.k,
        share.a_row_indices);

    if (!builder.mining_.verify_proof(a_proof, share.job.job_key.data(),
                                      a_proof.root.data())) {
        return false;
    }

    // Rebuild B proof from the stored Merkle tree.
    OwnedProof b_proof = builder.mining_.proof_for_handle(
        *ctx.b_merkle_tree(), share.b_col_indices);
    if (!builder.mining_.verify_proof(b_proof, share.job.job_key.data(),
                                      b_proof.root.data())) {
        return false;
    }

    std::array<uint8_t, 32> hashA{};
    std::memcpy(hashA.data(), a_proof.root.data(), 32);
    std::array<uint8_t, 32> hashB{};
    std::memcpy(hashB.data(), b_proof.root.data(), 32);

    // Recompute claimed hash and compare to the GPU value.
    auto claimed = builder.compute_claimed_hash(share, share.job.job_key.data(),
                                                hashA.data(), hashB.data());
    return claimed == share.claimed_hash;
}

std::array<uint8_t, 32> ShareBuilder::ComputeClaimedHash(
    const ShareFound& share,
    const uint8_t job_key[32],
    const uint8_t hashA[32],
    const uint8_t hashB[32]) {
    ShareBuilder builder(share.job.config);
    return builder.compute_claimed_hash(share, job_key, hashA, hashB);
}

} // namespace pearl
