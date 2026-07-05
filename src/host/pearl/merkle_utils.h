#pragma once

#include <cstdint>
#include <set>
#include <vector>

namespace pearl {

// Mirror pearl_blake3::MerkleTree::compute_leaf_indices_from_rows.
// Each 1024-byte BLAKE3 leaf covers row_width bytes contiguously in row-major A.
inline std::vector<uint32_t> compute_leaf_indices_from_rows(
        const std::vector<uint32_t>& row_indices,
        size_t row_width) {
    constexpr size_t CHUNK_LEN = 1024;
    std::set<uint32_t> uniq;
    for (uint32_t row : row_indices) {
        const size_t r = row;
        const size_t first = (r * row_width) / CHUNK_LEN;
        const size_t last = ((r + 1) * row_width - 1) / CHUNK_LEN;
        for (size_t li = first; li <= last; ++li) {
            uniq.insert(static_cast<uint32_t>(li));
        }
    }
    return {uniq.begin(), uniq.end()};
}

}  // namespace pearl
