#include "pearl_types.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace pearl {

static void u32_to_le(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v);
    p[1] = static_cast<uint8_t>(v >> 8);
    p[2] = static_cast<uint8_t>(v >> 16);
    p[3] = static_cast<uint8_t>(v >> 24);
}

static void u16_to_le(uint8_t* p, uint16_t v) {
    p[0] = static_cast<uint8_t>(v);
    p[1] = static_cast<uint8_t>(v >> 8);
}

PeriodicPattern PeriodicPattern::from_indices(const std::vector<uint32_t>& pattern) {
    if (pattern.empty()) throw std::invalid_argument("Pattern cannot be empty");
    for (size_t i = 1; i < pattern.size(); ++i) {
        if (pattern[i] <= pattern[i - 1])
            throw std::invalid_argument("Pattern must be sorted and have no duplicates");
    }
    if (pattern[0] != 0) throw std::invalid_argument("Pattern must start at 0");

    std::vector<uint32_t> p(pattern.begin(), pattern.end());
    struct Pair { uint32_t stride; uint32_t length; };
    std::vector<Pair> shape;

    while (p.size() > 1) {
        bool found = false;
        for (size_t period = 1; period < p.size(); ++period) {
            if (p.size() % period != 0) continue;
            uint32_t s = p[period];
            bool periodic = true;
            for (size_t i = 0; i + period < p.size(); ++i) {
                if (p[i] + s != p[i + period]) { periodic = false; break; }
            }
            if (periodic) {
                shape.push_back({s, static_cast<uint32_t>(p.size() / period)});
                p.erase(p.begin() + static_cast<std::ptrdiff_t>(period), p.end());
                found = true;
                break;
            }
        }
        if (!found) throw std::invalid_argument("Pattern is not periodic");
    }

    std::reverse(shape.begin(), shape.end());
    uint32_t trailing = shape.empty() ? 1u : shape.back().stride * shape.back().length;
    while (shape.size() < 3) shape.push_back({trailing, 1});

    PeriodicPattern out;
    out.stride0 = shape[0].stride; out.length0 = shape[0].length;
    out.stride1 = shape[1].stride; out.length1 = shape[1].length;
    out.stride2 = shape[2].stride; out.length2 = shape[2].length;
    return out;
}

PeriodicPattern PeriodicPattern::default_rows() {
    return from_indices({0, 8});
}

PeriodicPattern PeriodicPattern::default_cols() {
    std::vector<uint32_t> cols(64);
    for (int r = 0; r < 32; ++r) {
        cols[2 * r]     = static_cast<uint32_t>(8 * r);
        cols[2 * r + 1] = static_cast<uint32_t>(8 * r + 1);
    }
    return from_indices(cols);
}

std::array<uint8_t, PeriodicPattern::kSerializedSize> PeriodicPattern::to_bytes() const {
    std::array<uint8_t, kSerializedSize> data{};
    uint32_t min_stride = 1;
    const uint32_t strides[3] = {stride0, stride1, stride2};
    const uint32_t lengths[3] = {length0, length1, length2};
    for (int i = 0; i < 3; ++i) {
        uint32_t factor = strides[i] / min_stride;
        data[2 * i]     = static_cast<uint8_t>(factor - 1);
        data[2 * i + 1] = static_cast<uint8_t>(lengths[i] - 1);
        min_stride = strides[i] * lengths[i];
    }
    return data;
}

std::array<uint8_t, 52> MiningConfig::to_bytes() const {
    std::array<uint8_t, 52> buf{};
    u32_to_le(buf.data() + 0, static_cast<uint32_t>(k));
    u16_to_le(buf.data() + 4, static_cast<uint16_t>(r));
    u16_to_le(buf.data() + 6, 0); // MmaType::Int7xInt7ToInt32
    auto rows = rows_pattern.to_bytes();
    auto cols = cols_pattern.to_bytes();
    std::memcpy(buf.data() + 8, rows.data(), 6);
    std::memcpy(buf.data() + 14, cols.data(), 6);
    // bytes [20..52) reserved, already zero.
    return buf;
}

uint32_t MiningConfig::dot_product_length() const {
    // Int7xInt7ToInt32 quantum.
    uint32_t quantum = 128;
    return (static_cast<uint32_t>(k) / quantum) * quantum;
}

uint64_t MiningConfig::difficulty_adjustment_factor() const {
    return static_cast<uint64_t>(rows_pattern.size()) *
           static_cast<uint64_t>(cols_pattern.size()) *
           static_cast<uint64_t>(dot_product_length());
}

MiningConfig MiningConfig::conservative() {
    MiningConfig cfg;
    cfg.m = 2048;
    cfg.n = 2048;
    cfg.k = 2048;
    cfg.r = 64;
    cfg.bM = 128;
    cfg.bN = 128;
    cfg.bK = 128;
    return cfg;
}

MiningConfig MiningConfig::auto_shape_for_gpu(const cudaDeviceProp& prop,
                                              size_t budget_bytes) {
    MiningConfig cfg;

    size_t free = budget_bytes;
    if (free == 0) {
        size_t total = 0;
        cudaMemGetInfo(&free, &total);
        // Leave a generous safety margin for the driver / OS.
        free = free > (512ULL << 20) ? free - (512ULL << 20) : free;
    }

    const int major = prop.major;
    const int minor = prop.minor;
    const bool is_ada_plus = (major > 8) || (major == 8 && minor >= 9); // Ada / Hopper / Blackwell
    const bool is_ampere = (major == 8);

    // Rough memory model: A (M*K int8) + B (K*N int8) + C (M*N int32) + leaf CVs
    // plus working buffers.  We require the C matrix (M*N*4 bytes) plus ~25% headroom.
    auto fits = [&](int M, int N, int K) -> bool {
        int64_t a = int64_t(M) * K;
        int64_t b = int64_t(K) * N;
        int64_t c = int64_t(M) * N;
        int64_t bytes = a + b + 4 * c;
        // Add ~25% for leaf CVs, noise, and alignment.
        bytes = bytes + bytes / 4;
        return static_cast<size_t>(bytes) < free;
    };

    // Default targets by architecture class.
    struct Shape { int m, n, k; };
    std::vector<Shape> candidates;
    if (is_ada_plus) {
        candidates = {{8192, 32768, 128}, {8192, 16384, 128}, {4096, 32768, 128},
                      {4096, 16384, 128}, {4096, 8192, 128}};
    } else if (is_ampere) {
        candidates = {{4096, 16384, 128}, {4096, 8192, 128}, {2048, 16384, 128},
                      {2048, 8192, 128}};
    } else {
        candidates = {{2048, 8192, 128}, {2048, 4096, 128}, {1024, 4096, 128}};
    }

    for (const auto& s : candidates) {
        if (fits(s.m, s.n, s.k)) {
            cfg.m = s.m;
            cfg.n = s.n;
            cfg.k = s.k;
            break;
        }
    }

    if (!fits(cfg.m, cfg.n, cfg.k)) {
        // Fallback to conservative shape if nothing fits.
        cfg = conservative();
    }

    // Tile shapes tuned per architecture.  Consumer kernels hard-code BM=128,
    // BN=256 for proof-canonical layout, but keep bK aligned with the kernel's
    // preferred smem K-tile.  Blackwell defaults to kBK=128; Ampere to 64.
    if (is_ada_plus) {
        cfg.bM = 128; cfg.bN = 256; cfg.bK = 128;
        cfg.r = 128;
    } else if (is_ampere) {
        cfg.bM = 128; cfg.bN = 128; cfg.bK = 64;
    } else {
        cfg.bM = 128; cfg.bN = 128; cfg.bK = 128;
    }

    return cfg;
}

void MiningConfig::warn_if_cluster_m_mismatch(int cluster_m) {
    if (cluster_m <= 1) return;
    const auto rows = PeriodicPattern::default_rows();
    const auto cols = PeriodicPattern::default_cols();
    std::fprintf(stderr,
        "[config] PEARL_GEMM_CONSUMER_CLUSTER_M=%d with default periodic "
        "pattern (rows=%u cols=%u). If shares fail claimed_hash_mismatch, "
        "set CLUSTER_M=1 or align rows/cols_pattern with the kernel tile.\n",
        cluster_m, rows.size(), cols.size());
}

} // namespace pearl
