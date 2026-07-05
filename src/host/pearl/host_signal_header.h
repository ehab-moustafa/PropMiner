#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace pearl {

// C++ mirror of the HostSignalHeader struct in pearl_gemm_capi.
class HostSignalHeader {
public:
    explicit HostSignalHeader(const std::vector<uint8_t>& header);
    HostSignalHeader(const uint8_t* data, size_t size);

    int status() const;
    uint32_t tile_row_coord() const;
    uint32_t tile_col_coord() const;
    int mma_tile_m() const;
    int mma_tile_n() const;
    uint16_t num_registers_per_thread() const;

    void extract_indices(std::vector<uint32_t>& a_rows,
                         std::vector<uint32_t>& b_cols) const;

private:
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    int le32(int off) const;
};

} // namespace pearl
