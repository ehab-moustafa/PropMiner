#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "pearl/host_signal_header.h"
#include "pearl/job_key.h"
#include "pearl/pearl_types.h"
#include "pearl/rtx5090_profile.h"
#include "pearl/proto/mining_v2.h"
#include "pearl/protobuf_encoder.h"

#include "tests/ref_blake3.h"
#include "tests/ref_pearl.h"

#ifndef PROP_MINER_HOST_ONLY_TESTS
#define PROP_MINER_HOST_ONLY_TESTS 0
#endif

#if !PROP_MINER_HOST_ONLY_TESTS
#include "pearl/pearl_blake3.h"
#include "pearl/pearl_mining_wrapper.h"
#include "pearl/share_builder.h"
#include "pearl/sigma_context.h"
#endif

namespace pearl {

static int g_failures = 0;

#define EXPECT(cond)                                            \
    do {                                                        \
        if (!(cond)) {                                          \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ \
                      << " " << #cond << std::endl;             \
            ++g_failures;                                       \
        }                                                       \
    } while (0)

static void test_mining_config_to_bytes() {
    MiningConfig cfg;
    cfg.k = 128;
    cfg.r = 64;
    cfg.rows_pattern = PeriodicPattern::default_rows();
    cfg.cols_pattern = PeriodicPattern::default_cols();
    auto bytes = cfg.to_bytes();
    EXPECT(bytes.size() == 52);
    EXPECT(bytes[0] == 128 && bytes[1] == 0 && bytes[2] == 0 && bytes[3] == 0);
    EXPECT(bytes[4] == 64 && bytes[5] == 0);
    EXPECT(bytes[6] == 0 && bytes[7] == 0);
}

static void test_job_key_derivation() {
    Job job;
    job.sigma.fill(0xab);
    job.config = MiningConfig::conservative();
    auto key = derive_job_key(job.sigma, job.config);
    auto key2 = derive_job_key(job.sigma, job.config);
    EXPECT(key == key2);
    auto key3 = derive_job_key(job.sigma, MiningConfig());
    EXPECT(key != key3);
}

static void test_reference_blake3() {
    const std::array<uint8_t, 32> empty_hash = {
        0xaf, 0x13, 0x49, 0xb9, 0xf5, 0xf9, 0xa1, 0xa6,
        0xa0, 0x40, 0x4d, 0xea, 0x36, 0xdc, 0xc9, 0x49,
        0x9b, 0xcb, 0x25, 0xc9, 0xad, 0xc1, 0x12, 0xb7,
        0xcc, 0x9a, 0x93, 0xca, 0xe4, 0x1f, 0x32, 0x62};
    auto h_ref = ref::Blake3Ref::hash(nullptr, 0);
    EXPECT(h_ref == empty_hash);

    const char msg[] = "hello";
    auto out_ref = ref::Blake3Ref::xof(
        reinterpret_cast<const uint8_t*>(msg), strlen(msg), 64);
    EXPECT(out_ref.size() == 64);
    auto h_msg = ref::Blake3Ref::hash(
        reinterpret_cast<const uint8_t*>(msg), strlen(msg));
    EXPECT(std::memcmp(out_ref.data(), h_msg.data(), 32) == 0);

    std::array<uint8_t, 32> key{};
    key.fill(0x42);
    auto kh = ref::Blake3Ref::keyed_hash(
        reinterpret_cast<const uint8_t*>(msg), strlen(msg), key.data());
    EXPECT(!std::all_of(kh.begin(), kh.end(), [](uint8_t b){ return b == 0; }));

#if !PROP_MINER_HOST_ONLY_TESTS
    auto h = Blake3Helper::hash(nullptr, 0);
    EXPECT(h == empty_hash);
    auto out = Blake3Helper::xof(
        reinterpret_cast<const uint8_t*>(msg), strlen(msg), 64);
    EXPECT(out == out_ref);
#endif
}

static void test_reference_bseed_expand() {
    std::array<uint8_t, 32> bseed{};
    bseed.fill(0x11);
    auto expanded = ref::bseed_expand(bseed.data(), 1024);
    EXPECT(expanded.size() == 1024);

    auto range = ref::bseed_expand_range(bseed.data(), 100, 200);
    EXPECT(range.size() == 200);
    EXPECT(std::memcmp(range.data(), expanded.data() + 100, 200) == 0);
}

static void test_reference_merkle_tree() {
    std::array<uint8_t, 32> key{};
    key.fill(0x22);
    std::vector<uint8_t> data(4 * 1024);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<uint8_t>(i);

    ref::RefMerkleTree tree(data.data(), data.size(), key.data());
    EXPECT(tree.total_leaves() == 4);
    EXPECT(!std::all_of(tree.root().begin(), tree.root().end(), [](uint8_t b){ return b == 0; }));

    auto proof = tree.proof_for_indices({0, 3});
    EXPECT(proof.leaf_indices.size() == 2);
    EXPECT(proof.siblings.size() > 0);
}

static void test_reference_audit_index_derive() {
    std::array<uint8_t, 32> claimed{};
    claimed.fill(0x33);
    std::array<uint8_t, 32> bseed{};
    bseed.fill(0x44);
    auto idx = ref::derive_audit_indices(claimed.data(), bseed.data(), 8, 1024);
    EXPECT(idx.size() == 8);
    for (uint32_t i : idx) EXPECT(i < 1024);
}

static void test_reference_claimed_hash_deterministic() {
    auto share = ref::make_synthetic_share(256, 512, 128, 64);
    auto a_slice = ref::read_leaves(share.a_data.data(), share.a_rows);
    auto h1 = ref::compute_claimed_hash(
        share.cfg, share.a_rows, share.b_cols, a_slice,
        share.bseed.data(), share.job_key.data(),
        share.a_tree.root().data(), share.b_tree.root().data());
    auto h2 = ref::compute_claimed_hash(
        share.cfg, share.a_rows, share.b_cols, a_slice,
        share.bseed.data(), share.job_key.data(),
        share.a_tree.root().data(), share.b_tree.root().data());
    EXPECT(h1 == h2);
    EXPECT(!std::all_of(h1.begin(), h1.end(), [](uint8_t b){ return b == 0; }));
}

static void test_host_signal_header_index_extraction() {
    // Build a synthetic header matching the documented layout.
    // Status = 1 (trigger), tile coord (1, 2), 16 registers/thread.
    std::vector<uint8_t> hdr(1024, 0);
    auto put32 = [&](int off, uint32_t v) {
        hdr[off]     = static_cast<uint8_t>(v);
        hdr[off + 1] = static_cast<uint8_t>(v >> 8);
        hdr[off + 2] = static_cast<uint8_t>(v >> 16);
        hdr[off + 3] = static_cast<uint8_t>(v >> 24);
    };
    put32(0, 1);                 // status
    put32(40, 1);                // tile_row_coord
    put32(44, 2);                // tile_col_coord
    put32(592, 128);             // mma_tile_m
    put32(596, 256);             // mma_tile_n
    hdr[64] = 16;                // num_registers_per_thread (low byte)
    hdr[65] = 0;
    // Per-thread rows: 0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60
    // Per-thread cols: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    for (int i = 0; i < 16; ++i) {
        hdr[66 + i] = static_cast<uint8_t>(i * 4);
        hdr[322 + i] = static_cast<uint8_t>(i);
    }

    HostSignalHeader h(hdr);
    EXPECT(h.status() == 1);
    EXPECT(h.tile_row_coord() == 1);
    EXPECT(h.tile_col_coord() == 2);
    EXPECT(h.mma_tile_m() == 128);
    EXPECT(h.mma_tile_n() == 256);
    EXPECT(h.num_registers_per_thread() == 16);

    std::vector<uint32_t> rows, cols;
    h.extract_indices(rows, cols);
    EXPECT(rows.size() == 16);
    EXPECT(cols.size() == 16);
    EXPECT(rows[0] == 128);
    EXPECT(rows[15] == 128 + 60);
    EXPECT(cols[0] == 512);
    EXPECT(cols[15] == 512 + 15);
}

static void test_mining_config_conservative() {
    auto cfg = MiningConfig::conservative();
    EXPECT(cfg.m == 2048);
    EXPECT(cfg.n == 2048);
    EXPECT(cfg.k == 2048);
    EXPECT(cfg.r == 64);
    EXPECT(cfg.difficulty_adjustment_factor() > 0);
}

static void test_rtx5090_wave_alignment() {
    EXPECT(Rtx5090Profile::wave_aligned(8192, 65280));
    EXPECT(Rtx5090Profile::wave_aligned(8192, 43520));
    EXPECT(!Rtx5090Profile::wave_aligned(8192, 32768));
    EXPECT(Rtx5090Profile::wave_aligned_n_at_least(8192, 32768) >= 43520);
    // 32 GB minus reserve: production mine should pick N=262144.
    constexpr size_t k32GbMinusReserve = (28ULL << 30);
    auto prod_cfg = rtx5090_mining_config(k32GbMinusReserve, 0);
    EXPECT(prod_cfg.m == 8192);
    EXPECT(prod_cfg.n == 262144);
    EXPECT(Rtx5090Profile::tiles(prod_cfg.m, prod_cfg.n) == 65536);
    auto cfg = rtx5090_mining_config(0);
    EXPECT(cfg.m == 8192);
    EXPECT(cfg.n >= 32768);
    EXPECT(Rtx5090Profile::tiles(cfg.m, cfg.n) >= Rtx5090Profile::kSMCount * 2);
    auto bench_cfg = rtx5090_mining_config(0, Rtx5090Profile::kBenchMaxN);
    EXPECT(bench_cfg.n == Rtx5090Profile::kBenchMaxN);
}

static void test_share_found_serialization_sanity() {
    ShareFound share;
    share.job.config = MiningConfig::conservative();
    share.nonce = 0xdeadbeef;
    share.tile_row = 7;
    share.tile_col = 9;
    share.mma_tile_m = 128;
    share.mma_tile_n = 256;
    share.a_row_indices = {0, 8};
    share.b_col_indices = {0, 1};
    EXPECT(share.a_row_indices.size() == 2);
    EXPECT(share.b_col_indices.size() == 2);
}

#if !PROP_MINER_HOST_ONLY_TESTS
static void test_merkle_proof_roundtrip() {
    MiningCapi capi;
    std::array<uint8_t, 32> bseed{};
    std::array<uint8_t, 32> job_key{};
    bseed[0] = 1;
    job_key[0] = 2;

    MerkleTree tree = capi.build_b_host_tree(bseed.data(), 64, 32, job_key.data());
    EXPECT(tree.total_leaves() > 0);

    std::vector<uint32_t> indices = {0, 7, 15};
    OwnedProof proof = capi.proof_for_handle(tree, indices);
    EXPECT(proof.leaf_count() == indices.size());
    EXPECT(proof.sibling_count() > 0);

    bool ok = capi.verify_proof(proof, job_key.data(), tree.root());
    EXPECT(ok);
}
#endif

static void test_protobuf_register_request_roundtrip() {
    proto::RegisterRequest req;
    req.wallet_address = "krxTest";
    req.worker_name = "worker";
    req.miner_version = "propminer/2.0";
    req.protocol_version = 2;
    req.k = 128;

    auto encoded = req.encode();
    EXPECT(!encoded.empty());

    proto::RegisterRequest decoded;
    EXPECT(decoded.decode(encoded));
    EXPECT(decoded.wallet_address == req.wallet_address);
    EXPECT(decoded.worker_name == req.worker_name);
    EXPECT(decoded.protocol_version == req.protocol_version);
    EXPECT(decoded.k == req.k);
}

static void test_share_submission_encoding_sanity() {
    ShareFound share;
    share.job.sigma.fill(0x11);
    share.job.job_key.fill(0x22);
    share.job.b_seed.fill(0x33);
    share.job.config = MiningConfig::conservative();
    share.nonce = 12345;
    share.tile_row = 0;
    share.tile_col = 1;
    share.a_row_indices = {0, 8};
    share.b_col_indices = {0, 1};
    share.hash_b.fill(0x44);
    share.a_slice.assign(share.a_row_indices.size() * share.job.config.k, 0x55);
    share.a_leaf_cvs.assign(32, 0x66);
    share.claimed_hash.fill(0x77);

    OwnedProof a_proof;
    a_proof.root.assign(32, 0x88);
    a_proof.total_leaves = 4;
    a_proof.leaf_data.assign(2 * 1024, 0x99);
    a_proof.leaf_indices = {0, 1};
    a_proof.siblings.assign(64, 0xaa);

    OwnedProof b_proof;
    b_proof.root.assign(32, 0xbb);
    b_proof.total_leaves = 4;
    b_proof.leaf_data.assign(2 * 1024, 0xcc);
    b_proof.leaf_indices = {0, 1};
    b_proof.siblings.assign(64, 0xdd);

    std::vector<uint8_t> audit_siblings(64, 0xee);

    auto encoded = ProtobufEncoder::encode_share_submission(
        share, share.job.config, a_proof, b_proof, audit_siblings);
    EXPECT(!encoded.empty());
    EXPECT(encoded.size() > 100);
}

#if !PROP_MINER_HOST_ONLY_TESTS
static void test_share_claimed_hash_roundtrip() {
    MiningConfig cfg;
    cfg.m = 2048;
    cfg.n = 4096;
    cfg.k = 128;
    cfg.r = 128;

    Job job;
    job.sigma.fill(0xab);
    job.b_seed.fill(0xcd);
    job.config = cfg;
    job.job_key = derive_job_key(job.sigma, cfg);

    MiningCapi capi;

    MerkleTree b_tree = capi.build_b_host_tree(
        job.b_seed.data(), cfg.n, cfg.k, job.job_key.data());

    std::vector<uint8_t> a_data(static_cast<size_t>(cfg.m) * cfg.k);
    for (size_t i = 0; i < a_data.size(); ++i) a_data[i] = static_cast<uint8_t>((i * 7 + 3) & 0x7f);
    std::vector<uint32_t> rows = {0, 8};
    OwnedProof a_proof = capi.a_root_and_proof(
        a_data.data(), cfg.m, cfg.k, job.job_key.data(), rows);

    std::vector<uint32_t> cols = {0, 1};
    OwnedProof b_proof = capi.proof_for_handle(b_tree, cols);

    std::vector<uint8_t> a_slice(rows.size() * cfg.k);
    for (size_t i = 0; i < rows.size(); ++i) {
        std::memcpy(a_slice.data() + i * cfg.k,
                    a_data.data() + static_cast<size_t>(rows[i]) * cfg.k,
                    cfg.k);
    }

    size_t a_leaves = (a_data.size() + 1023) / 1024;
    std::vector<uint8_t> a_leaf_cvs(a_leaves * 32, 0);

    ShareFound share;
    share.job = job;
    share.nonce = 1;
    share.a_row_indices = rows;
    share.b_col_indices = cols;
    share.a_slice = a_slice;
    share.a_leaf_cvs = a_leaf_cvs;
    std::memcpy(share.hash_b.data(), b_proof.root.data(), 32);

    share.claimed_hash = ShareBuilder::ComputeClaimedHash(
        share, job.job_key.data(), a_proof.root.data(), b_proof.root.data());

    SigmaContext ctx(job, cfg);
    ctx.set_b_merkle_tree(std::move(b_tree));

    bool ok = ShareBuilder::VerifyShare(share, ctx);
    EXPECT(ok);
}
#endif

} // namespace pearl

int main() {
    using namespace pearl;
    std::cerr << "[tests] Running PropMiner host-side correctness tests..." << std::endl;
    test_mining_config_to_bytes();
    test_job_key_derivation();
    test_reference_blake3();
    test_reference_bseed_expand();
    test_reference_merkle_tree();
    test_reference_audit_index_derive();
    test_reference_claimed_hash_deterministic();
    test_host_signal_header_index_extraction();
    test_mining_config_conservative();
    test_rtx5090_wave_alignment();
    test_share_found_serialization_sanity();
#if !PROP_MINER_HOST_ONLY_TESTS
    test_merkle_proof_roundtrip();
#endif
    test_protobuf_register_request_roundtrip();
    test_share_submission_encoding_sanity();
#if !PROP_MINER_HOST_ONLY_TESTS
    test_share_claimed_hash_roundtrip();
#endif

    if (g_failures == 0) {
        std::cerr << "[tests] All tests passed." << std::endl;
        return 0;
    }
    std::cerr << "[tests] " << g_failures << " test(s) failed." << std::endl;
    return 1;
}
