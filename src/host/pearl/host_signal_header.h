#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace pearl {

// C++ mirror of the C++ HostSignalHeader struct.
// Layout must match pearl_gemm_capi host_signal_header.hpp.
// Layout matches the C++ HostSignalHeader in pearl_gemm_capi.
class HostSignalHeader {
public:
    explicit HostSignalHeader(const std::vector<uint8_t>& header);

    int status() const;
    uint32_t tile_row_coord() const;
    uint32_t tile_col_coord() const;
    int mma_tile_m() const;
    int mma_tile_n() const;
    uint16_t num_registers_per_thread() const;

    // Extract sorted unique A row and B col indices within the winning tile.
    void extract_indices(std::vector<uint32_t>& a_rows,
                         std::vector<uint32_t>& b_cols) const;

private:
    const std::vector<uint8_t>& h_;
    int le32(int off) const;
};

} // namespace pearl
