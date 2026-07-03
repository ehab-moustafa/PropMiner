// Rigorous statistical micro-benchmarks for host-side hot paths.
// Reports median, p90, p95, p99, mean, stddev, and throughput across
// multiple independent runs.  Used to validate before/after performance
// changes without requiring a CUDA-capable GPU.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
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
using DurationUs = std::chrono::duration<double, std::micro>;

struct Stats {
    double min = 0, max = 0, mean = 0, median = 0;
    double p90 = 0, p95 = 0, p99 = 0;
    double stddev = 0;
};

static Stats analyze(const std::vector<double>& samples_us) {
    Stats s;
    auto v = samples_us;
    std::sort(v.begin(), v.end());
    s.min = v.front();
    s.max = v.back();
    s.median = v[v.size() / 2];

    auto pct = [&](double p) {
        size_t idx = static_cast<size_t>(p * (v.size() - 1));
        return v[idx];
    };
    s.p90 = pct(0.90);
    s.p95 = pct(0.95);
    s.p99 = pct(0.99);

    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    s.mean = sum / v.size();
    double sq = 0.0;
    for (double x : v) sq += (x - s.mean) * (x - s.mean);
    s.stddev = std::sqrt(sq / v.size());
    return s;
}

static void print_stats(const char* name, const Stats& s, double ops_per_call) {
    std::printf("%-24s med=%8.2f us  p90=%8.2f  p95=%8.2f  p99=%8.2f  "
                "avg=%8.2f  std=%6.2f  throughput=%.2f ops/s\n",
                name, s.median, s.p90, s.p95, s.p99, s.mean, s.stddev,
                1e6 / s.median * ops_per_call);
}

template <typename F>
static Stats benchmark(int warmup, int iterations, int repeats, F&& f) {
    std::vector<double> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r) {
        for (int w = 0; w < warmup; ++w) f();
        auto t0 = Clock::now();
        for (int i = 0; i < iterations; ++i) f();
        auto t1 = Clock::now();
        double us = DurationUs(t1 - t0).count() / iterations;
        samples.push_back(us);
    }
    return analyze(samples);
}

static void bench_blake3_hash() {
    std::vector<uint8_t> msg(64);
    for (size_t i = 0; i < msg.size(); ++i) msg[i] = static_cast<uint8_t>(i);
    auto s = benchmark(5, 10000, 30, [&]() {
        auto h = ref::Blake3Ref::hash(msg.data(), msg.size());
        (void)h;
    });
    print_stats("blake3_hash_64B", s, 1.0);
}

static void bench_job_key() {
    std::array<uint8_t, 32> sigma{};
    sigma.fill(0xab);
    auto cfg = MiningConfig::conservative();
    auto s = benchmark(5, 10000, 30, [&]() {
        auto k = derive_job_key(sigma.data(), cfg);
        (void)k;
    });
    print_stats("derive_job_key", s, 1.0);
}

static void bench_bseed_expand() {
    std::array<uint8_t, 32> bseed{};
    bseed.fill(0x11);
    auto s = benchmark(3, 100, 20, [&]() {
        auto data = ref::bseed_expand(bseed.data(), 1024 * 1024);
        (void)data;
    });
    print_stats("bseed_expand_1MiB", s, 1.0);
}

static void bench_merkle_proof() {
    std::array<uint8_t, 32> key{};
    key.fill(0x22);
    std::vector<uint8_t> data(64 * 1024);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<uint8_t>(i);
    ref::RefMerkleTree tree(data.data(), data.size(), key.data());
    auto s = benchmark(3, 1000, 20, [&]() {
        auto p = tree.proof_for_indices({0, 7, 15, 31});
        (void)p;
    });
    print_stats("merkle_proof_4_leaves", s, 1.0);
}

static void bench_claimed_hash() {
    auto share = ref::make_synthetic_share(256, 512, 128, 64);
    auto a_slice = ref::read_leaves(share.a_data.data(), share.a_rows);
    auto s = benchmark(3, 500, 20, [&]() {
        auto h = ref::compute_claimed_hash(
            share.cfg, share.a_rows, share.b_cols, a_slice,
            share.bseed.data(), share.job_key.data(),
            share.a_tree.root().data(), share.b_tree.root().data());
        (void)h;
    });
    print_stats("compute_claimed_hash", s, 1.0);
}

static void bench_encode_share() {
    ShareFound share;
    share.job.config = MiningConfig::conservative();
    share.job.sigma.fill(0x11);
    share.job.job_key.fill(0x22);
    share.job.b_seed.fill(0x33);
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

    auto s = benchmark(5, 10000, 30, [&]() {
        auto enc = ProtobufEncoder::encode_share_submission(
            share, share.job.config, a_proof, b_proof, audit_siblings);
        (void)enc;
    });
    print_stats("encode_share_submission", s, 1.0);
}

} // namespace pearl

int main() {
    std::cerr << "[rigorous benchmarks] PropMiner host-side hot paths (us/op)\n";
    pearl::bench_blake3_hash();
    pearl::bench_job_key();
    pearl::bench_bseed_expand();
    pearl::bench_merkle_proof();
    pearl::bench_claimed_hash();
    pearl::bench_encode_share();
    return 0;
}
