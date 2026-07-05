// Micro-benchmarks for hashrate-sensitive host-side paths.
// These run without a GPU and exercise the reference crypto / encoding hot paths
// so future regressions are caught in CI.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "pearl/host_signal_header.h"
#include "pearl/job_key.h"
#include "pearl/pearl_types.h"
#include "pearl/proto/mining_v2.h"
#include "pearl/protobuf_encoder.h"

#include "tests/ref_blake3.h"
#include "tests/ref_pearl.h"

namespace pearl {

using Clock = std::chrono::steady_clock;
using DurationMs = std::chrono::duration<double, std::milli>;

template <typename F>
static double bench_ms(int iterations, F&& f) {
    // Warmup
    for (int i = 0; i < 3; ++i) f();
    auto t0 = Clock::now();
    for (int i = 0; i < iterations; ++i) f();
    auto t1 = Clock::now();
    return DurationMs(t1 - t0).count();
}

static void benchmark_blake3_throughput() {
    constexpr int iterations = 100000;
    std::array<uint8_t, 32> key{};
    key.fill(0x42);
    std::vector<uint8_t> msg(64);
    for (size_t i = 0; i < msg.size(); ++i) msg[i] = static_cast<uint8_t>(i);

    double ms = bench_ms(iterations, [&]() {
        auto h = ref::Blake3Ref::hash(msg.data(), msg.size());
        (void)h;
    });
    double bytes = static_cast<double>(iterations) * msg.size();
    double mbps = bytes / (ms / 1000.0) / (1024.0 * 1024.0);
    std::printf("blake3_hash_64B        : %.3f ms  %.2f MB/s\n", ms, mbps);
}

static void benchmark_job_key_derivation() {
    constexpr int iterations = 100000;
    std::array<uint8_t, 32> sigma{};
    sigma.fill(0xab);
    MiningConfig cfg = MiningConfig::conservative();
    double ms = bench_ms(iterations, [&]() {
        auto k = derive_job_key(sigma.data(), sigma.size(), cfg);
        (void)k;
    });
    std::printf("derive_job_key         : %.3f ms  %.2f kops/s\n",
                ms, iterations / ms);
}

static void benchmark_bseed_expand() {
    constexpr int iterations = 1000;
    std::array<uint8_t, 32> bseed{};
    bseed.fill(0x11);
    double ms = bench_ms(iterations, [&]() {
        auto data = ref::bseed_expand(bseed.data(), 1024 * 1024);
        (void)data;
    });
    double bytes = static_cast<double>(iterations) * 1024 * 1024;
    double mbps = bytes / (ms / 1000.0) / (1024.0 * 1024.0);
    std::printf("bseed_expand_1MiB      : %.3f ms  %.2f MB/s\n", ms, mbps);
}

static void benchmark_merkle_proof() {
    constexpr int iterations = 1000;
    std::array<uint8_t, 32> key{};
    key.fill(0x22);
    std::vector<uint8_t> data(64 * 1024);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<uint8_t>(i);
    ref::RefMerkleTree tree(data.data(), data.size(), key.data());

    double ms = bench_ms(iterations, [&]() {
        auto p = tree.proof_for_indices({0, 7, 15, 31});
        (void)p;
    });
    std::printf("merkle_proof_4_leaves  : %.3f ms  %.2f kops/s\n",
                ms, iterations / ms);
}

static void benchmark_claimed_hash() {
    constexpr int iterations = 1000;
    auto share = ref::make_synthetic_share(256, 512, 128, 64);
    auto a_slice = ref::read_leaves(share.a_data.data(), share.a_rows);

    double ms = bench_ms(iterations, [&]() {
        auto h = ref::compute_claimed_hash(
            share.cfg, share.a_rows, share.b_cols, a_slice,
            share.bseed.data(), share.job_key.data(),
            share.a_tree.root().data(), share.b_tree.root().data());
        (void)h;
    });
    std::printf("compute_claimed_hash   : %.3f ms  %.2f kops/s\n",
                ms, iterations / ms);
}

static void benchmark_protobuf_share_encoding() {
    constexpr int iterations = 50000;
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

    double ms = bench_ms(iterations, [&]() {
        auto encoded = ProtobufEncoder::encode_share_submission(
            share, share.job.config, a_proof, b_proof, audit_siblings);
        (void)encoded;
    });
    std::printf("encode_share_submission: %.3f ms  %.2f kops/s\n",
                ms, iterations / ms);
}

} // namespace pearl

int main() {
    std::cerr << "[benchmarks] PropMiner host-side micro-benchmarks" << std::endl;
    pearl::benchmark_blake3_throughput();
    pearl::benchmark_job_key_derivation();
    pearl::benchmark_bseed_expand();
    pearl::benchmark_merkle_proof();
    pearl::benchmark_claimed_hash();
    pearl::benchmark_protobuf_share_encoding();
    return 0;
}
