#include "ref_pearl.h"
#include "ref_blake3.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace pearl {
namespace ref {

namespace {

uint32_t mul_hi_u32(uint32_t a, uint32_t b) {
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32);
}

uint64_t splitmix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z ^= (z >> 30);
    z *= 0xBF58476D1CE4E5B9ULL;
    z ^= (z >> 27);
    z *= 0x94D049BB133111EBULL;
    z ^= (z >> 31);
    return z;
}

// Output signed int7 bytes in [-63, 63].
void lcg_int7_fill(std::vector<int8_t>& dst, uint64_t seedLo, uint64_t seedHi) {
    uint64_t base_seed = splitmix64(seedLo ^ splitmix64(seedHi));
    size_t n = dst.size();
    size_t n8 = n / 8;
    for (size_t i = 0; i < n8; ++i) {
        uint64_t z = splitmix64(base_seed + i);
        for (int b = 0; b < 8; ++b) {
            uint32_t v = static_cast<uint32_t>((z >> (b * 8)) & 0xFFULL);
            uint32_t r = v % 127u;
            dst[i * 8 + b] = static_cast<int8_t>(static_cast<int>(r) - 63);
        }
    }
    size_t tail_off = n8 * 8;
    size_t tail_len = n - tail_off;
    if (tail_len > 0) {
        uint64_t z = splitmix64(base_seed + n8);
        for (size_t b = 0; b < tail_len; ++b) {
            uint32_t v = static_cast<uint32_t>((z >> (b * 8)) & 0xFFULL);
            uint32_t r = v % 127u;
            dst[tail_off + b] = static_cast<int8_t>(static_cast<int>(r) - 63);
        }
    }
}

std::array<uint8_t, 32> blake3_key_from_seed(const uint8_t seed_label[32], const uint8_t noise_seed[32]) {
    std::array<uint8_t, 64> input{};
    std::memcpy(input.data(), seed_label, 32);
    std::memcpy(input.data() + 32, noise_seed, 32);
    return Blake3Ref::keyed_hash(input.data(), input.size(), seed_label);
}

std::array<uint8_t, 32> blake3_key_for_label(const char* label, const uint8_t noise_seed[32]) {
    std::array<uint8_t, 32> seed_label{};
    size_t len = std::strlen(label);
    std::memcpy(seed_label.data(), label, std::min<size_t>(len, 32));
    return blake3_key_from_seed(seed_label.data(), noise_seed);
}

} // namespace

std::vector<uint8_t> read_leaves(
    const uint8_t* data, const std::vector<uint32_t>& leaf_indices) {
    std::vector<uint8_t> out;
    out.reserve(leaf_indices.size() * 1024);
    for (uint32_t li : leaf_indices) {
        const uint8_t* leaf = data + static_cast<size_t>(li) * 1024;
        out.insert(out.end(), leaf, leaf + 1024);
    }
    return out;
}

std::vector<uint8_t> bseed_expand(const uint8_t bseed[32], size_t total_bytes) {
    return Blake3Ref::xof(bseed, 32, total_bytes);
}

std::vector<uint8_t> bseed_expand_range(
    const uint8_t bseed[32], uint64_t byte_offset, size_t len) {
    // XOF of bseed; request enough to cover offset+len.
    size_t needed = byte_offset + len;
    auto full = bseed_expand(bseed, needed);
    return std::vector<uint8_t>(full.begin() + byte_offset, full.begin() + needed);
}

RefMerkleTree::RefMerkleTree(const uint8_t* data, size_t len, const uint8_t key[32]) {
    if (len % 1024 != 0) throw std::invalid_argument("data length must be a multiple of 1024");
    key_ = {};
    std::memcpy(key_.data(), key, 32);
    data_.assign(data, data + len);
    size_t num_leaves = len / 1024;
    if (num_leaves == 0) num_leaves = 1;
    layers_.clear();
    layers_.emplace_back();
    layers_[0].reserve(num_leaves);
    for (size_t i = 0; i < num_leaves; ++i) {
        layers_[0].push_back(Blake3Ref::chunk_cv(data + i * 1024, 1024, static_cast<uint64_t>(i), key));
    }
    while (layers_.back().size() > 1) {
        const auto& prev = layers_.back();
        std::vector<std::array<uint8_t, 32>> next;
        size_t count = prev.size();
        for (size_t i = 0; i < count; i += 2) {
            const auto& left = prev[i];
            const auto& right = prev[i + 1 < count ? i + 1 : i];
            next.push_back(Blake3Ref::parent_cv(left.data(), right.data(), key));
        }
        layers_.push_back(std::move(next));
    }
    root_ = layers_.back()[0];
}

std::vector<uint32_t> RefMerkleTree::leaf_indices_for_rows(
    const std::vector<uint32_t>& row_indices, size_t row_width) {
    std::vector<uint32_t> out;
    out.reserve(row_indices.size());
    for (uint32_t r : row_indices) out.push_back(static_cast<uint32_t>(r * row_width));
    return out;
}

RefMerkleProof RefMerkleTree::proof_for_rows(
    const std::vector<uint32_t>& row_indices, size_t row_width) const {
    return proof_for_indices(leaf_indices_for_rows(row_indices, row_width));
}

RefMerkleProof RefMerkleTree::proof_for_indices(const std::vector<uint32_t>& indices) const {
    RefMerkleProof proof;
    proof.total_leaves = static_cast<uint32_t>(layers_[0].size());
    proof.leaf_indices = indices;
    proof.leaf_data.resize(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        uint32_t li = indices[i];
        std::memcpy(proof.leaf_data[i].data(), data_.data() + static_cast<size_t>(li) * 1024, 1024);
    }
    std::vector<uint32_t> current = indices;
    for (const auto& layer : layers_) {
        std::vector<uint32_t> next;
        for (uint32_t idx : current) {
            uint32_t sibling = (idx ^ 1u);
            if (sibling >= layer.size()) sibling = idx;
            proof.siblings.push_back(layer[sibling]);
            next.push_back(idx / 2);
        }
        current = std::move(next);
        if (layer.size() <= 1) break;
    }
    proof.root = root_;
    return proof;
}

std::vector<uint8_t> RefMerkleTree::audit_paths(const std::vector<uint32_t>& leaf_indices) const {
    std::vector<uint8_t> out;
    std::vector<uint32_t> current = leaf_indices;
    for (const auto& layer : layers_) {
        for (uint32_t idx : current) {
            uint32_t sibling = (idx ^ 1u);
            if (sibling >= layer.size()) sibling = idx;
            out.insert(out.end(), layer[sibling].begin(), layer[sibling].end());
        }
        std::vector<uint32_t> next;
        for (uint32_t idx : current) next.push_back(idx / 2);
        current = std::move(next);
        if (layer.size() <= 1) break;
    }
    return out;
}

NoiseSeedPair derive_noise_seeds(
    const uint8_t job_key[32], const uint8_t hashA[32], const uint8_t hashB[32]) {
    // Matches Akoya.Crypto MiningConfiguration.DeriveNoiseSeeds:
    // bNoiseSeed = BLAKE3(jobKey || hashB)
    // aNoiseSeed = BLAKE3(bNoiseSeed || hashA)
    std::array<uint8_t, 64> b_input{};
    std::memcpy(b_input.data(), job_key, 32);
    std::memcpy(b_input.data() + 32, hashB, 32);
    NoiseSeedPair p;
    p.b_noise_seed = Blake3Ref::hash(b_input.data(), b_input.size());

    std::array<uint8_t, 64> a_input{};
    std::memcpy(a_input.data(), p.b_noise_seed.data(), 32);
    std::memcpy(a_input.data() + 32, hashA, 32);
    p.a_noise_seed = Blake3Ref::hash(a_input.data(), a_input.size());
    return p;
}

std::vector<int8_t> generate_uniform_random_matrix(
    const uint8_t seed_label[32], const uint8_t a_noise_seed[32],
    const std::vector<uint32_t>& row_indices, int num_cols) {
    auto key = blake3_key_from_seed(seed_label, a_noise_seed);
    std::vector<int8_t> out(static_cast<size_t>(row_indices.size()) * num_cols);
    for (size_t i = 0; i < row_indices.size(); ++i) {
        uint64_t seed_lo = 0, seed_hi = 0;
        std::memcpy(&seed_lo, key.data(), 8);
        std::memcpy(&seed_hi, key.data() + 8, 8);
        seed_lo ^= row_indices[i];
        std::vector<int8_t> row(num_cols);
        lcg_int7_fill(row, seed_lo, seed_hi);
        std::memcpy(out.data() + i * num_cols, row.data(), num_cols);
    }
    return out;
}

std::vector<PermutationPair> generate_permutation_matrix(
    const uint8_t seed_label[32], const uint8_t noise_seed[32],
    int k, int noise_rank) {
    auto key = blake3_key_from_seed(seed_label, noise_seed);
    uint32_t rank_mask = static_cast<uint32_t>(noise_rank - 1);
    std::vector<PermutationPair> result(k);
    for (int chunk = 0; chunk * 8 < k; ++chunk) {
        uint8_t msg[64] = {};
        *reinterpret_cast<uint32_t*>(msg + 4) = static_cast<uint32_t>(chunk) + 1;
        std::memcpy(msg + 32, seed_label, 32);
        auto hash = Blake3Ref::keyed_hash(msg, sizeof(msg), key.data());
        int len = std::min(8, k - chunk * 8);
        for (int j = 0; j < len; ++j) {
            uint32_t rand = *reinterpret_cast<const uint32_t*>(hash.data() + j * 4);
            uint32_t first = rand & rank_mask;
            uint32_t second = first ^ (1u + mul_hi_u32(static_cast<uint32_t>(noise_rank - 1), rand));
            result[chunk * 8 + j] = { first, second };
        }
    }
    return result;
}

std::vector<int8_t> apply_sparse_permutation(
    const std::vector<PermutationPair>& perm, const std::vector<int8_t>& vec) {
    std::vector<int8_t> out(perm.size());
    apply_sparse_permutation_into(perm, vec.data(), out.data(), out.size());
    return out;
}

void apply_sparse_permutation_into(
    const std::vector<PermutationPair>& perm, const int8_t* in,
    int8_t* out, size_t out_len) {
    for (size_t i = 0; i < out_len; ++i) {
        out[i] = static_cast<int8_t>(in[perm[i].first] - in[perm[i].second]);
    }
}

std::vector<uint32_t> derive_audit_indices(
    const uint8_t claimed_hash[32],
    const uint8_t b_seed[32],
    uint32_t audit_k,
    uint32_t total_leaves) {
    if (audit_k == 0) return {};
    if (total_leaves == 0 || (total_leaves & (total_leaves - 1)) != 0) return {};
    const char domain[] = "akoya-audit-v1";
    std::array<uint8_t, 82> input{};
    std::memcpy(input.data(), domain, 14);
    std::memcpy(input.data() + 14, claimed_hash, 32);
    std::memcpy(input.data() + 46, b_seed, 32);
    input[78] = static_cast<uint8_t>(audit_k);
    input[79] = static_cast<uint8_t>(audit_k >> 8);
    input[80] = static_cast<uint8_t>(audit_k >> 16);
    input[81] = static_cast<uint8_t>(audit_k >> 24);
    auto xof = Blake3Ref::xof(input.data(), input.size(), audit_k * 4);
    std::vector<uint32_t> out;
    out.reserve(audit_k);
    for (uint32_t i = 0; i < audit_k; ++i) {
        uint32_t v = 0;
        std::memcpy(&v, xof.data() + i * 4, 4);
        out.push_back(static_cast<uint32_t>(static_cast<uint64_t>(v) % total_leaves));
    }
    return out;
}

std::array<uint8_t, 32> compute_claimed_hash(
    const MiningConfig& cfg,
    const std::vector<uint32_t>& a_row_indices,
    const std::vector<uint32_t>& b_col_indices,
    const std::vector<uint8_t>& a_slice,
    const uint8_t bseed[32],
    const uint8_t job_key[32],
    const uint8_t hashA[32],
    const uint8_t hashB[32]) {
    auto noise = derive_noise_seeds(job_key, hashA, hashB);

    const int h = static_cast<int>(a_row_indices.size());
    const int w = static_cast<int>(b_col_indices.size());
    const int k = cfg.k;
    const int r = cfg.r;

    // secretA: contiguous h x k.
    std::vector<int8_t> secretA(static_cast<size_t>(h) * k);
    std::memcpy(secretA.data(), a_slice.data(), secretA.size());

    // secretB: expand only the columns we need, packed w x k, row-major.
    std::vector<int8_t> secretB(static_cast<size_t>(w) * k);
    std::vector<uint8_t> b_full;
    b_full.resize(static_cast<size_t>(k) * cfg.n);
    Blake3Ref::xof_into(bseed, 32, b_full.data(), b_full.size());
    for (int j = 0; j < w; ++j) {
        uint32_t col = b_col_indices[j];
        std::memcpy(secretB.data() + static_cast<size_t>(j) * k,
                    b_full.data() + static_cast<size_t>(col) * k, k);
    }

    // Noise matrices exactly matching C# NoiseGenerator / JackpotComputer.
    auto a_label = blake3_key_for_label("A_tensor", noise.a_noise_seed.data());
    auto eAl = generate_uniform_random_matrix(a_label.data(), noise.a_noise_seed.data(), a_row_indices, r);
    auto eAr = generate_permutation_matrix(a_label.data(), noise.a_noise_seed.data(), k, r);

    auto b_label = blake3_key_for_label("B_tensor", noise.b_noise_seed.data());
    auto eBl = generate_permutation_matrix(b_label.data(), noise.b_noise_seed.data(), k, r);
    auto eBr = generate_uniform_random_matrix(b_label.data(), noise.b_noise_seed.data(), b_col_indices, r);

    // noiseA / noiseB: contiguous h x k and w x k, row-major.
    std::vector<int8_t> noiseA(static_cast<size_t>(h) * k);
    for (int i = 0; i < h; ++i) {
        apply_sparse_permutation_into(eAr,
            eAl.data() + static_cast<size_t>(i) * r,
            noiseA.data() + static_cast<size_t>(i) * k, k);
    }
    std::vector<int8_t> noiseB(static_cast<size_t>(w) * k);
    for (int j = 0; j < w; ++j) {
        apply_sparse_permutation_into(eBl,
            eBr.data() + static_cast<size_t>(j) * r,
            noiseB.data() + static_cast<size_t>(j) * k, k);
    }

    // Jackpot transcript using flat buffers.
    std::vector<int> tile(static_cast<size_t>(h) * w, 0);
    uint32_t msg[16] = {};
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

        int tid = ((ll / r) - 1) % 16;
        msg[tid] = ((msg[tid] << 13) | (msg[tid] >> 19)) ^ xored;
    }

    std::array<uint8_t, 64> buf{};
    for (int i = 0; i < 16; ++i) {
        buf[i * 4]     = static_cast<uint8_t>(msg[i]);
        buf[i * 4 + 1] = static_cast<uint8_t>(msg[i] >> 8);
        buf[i * 4 + 2] = static_cast<uint8_t>(msg[i] >> 16);
        buf[i * 4 + 3] = static_cast<uint8_t>(msg[i] >> 24);
    }
    return Blake3Ref::keyed_hash(buf.data(), buf.size(), noise.a_noise_seed.data());
}

RefSyntheticShare make_synthetic_share(int m, int n, int k, int r) {
    RefSyntheticShare sh;
    sh.cfg.m = m; sh.cfg.n = n; sh.cfg.k = k; sh.cfg.r = r;
    sh.cfg.rows_pattern = PeriodicPattern::default_rows();
    sh.cfg.cols_pattern = PeriodicPattern::default_cols();
    sh.sigma.fill(0xab);
    sh.bseed.fill(0xcd);
    sh.job_key = Blake3Ref::hash(sh.sigma.data(), sh.sigma.size());

    // A matrix: int7 via LCG.
    sh.a_data.assign(static_cast<size_t>(m) * k, 0);
    uint64_t seed_lo = 0, seed_hi = 0;
    std::memcpy(&seed_lo, sh.job_key.data(), 8);
    std::memcpy(&seed_hi, sh.job_key.data() + 8, 8);
    std::vector<int8_t> a_signed(sh.a_data.size());
    lcg_int7_fill(a_signed, seed_lo, seed_hi);
    for (size_t i = 0; i < sh.a_data.size(); ++i) sh.a_data[i] = static_cast<uint8_t>(a_signed[i]);

    sh.a_rows = {0, 8};
    sh.b_cols = {0, 1};

    sh.a_tree = RefMerkleTree(sh.a_data.data(), sh.a_data.size(), sh.job_key.data());
    auto b_full = bseed_expand(sh.bseed.data(), static_cast<size_t>(k) * n);
    sh.b_tree = RefMerkleTree(b_full.data(), b_full.size(), sh.job_key.data());
    return sh;
}

} // namespace ref
} // namespace pearl
