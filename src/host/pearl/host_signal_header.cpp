#include "host_signal_header.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace pearl {

namespace {
    constexpr int OFF_STATUS = 0;
    constexpr int OFF_TILE_COORD = 40;
    constexpr int OFF_NUM_REGISTERS_PER_THREAD = 64;
    constexpr int OFF_THREAD_ROWS = 66;
    constexpr int OFF_THREAD_COLS = 322;
    constexpr int OFF_MMA_TILE_SIZE = 592;
    constexpr int MAX_REGS = 256;
}

HostSignalHeader::HostSignalHeader(const std::vector<uint8_t>& header) : h_(header) {}

int HostSignalHeader::le32(int off) const {
    if (off + 4 > static_cast<int>(h_.size())) return 0;
    uint32_t v = 0;
    std::memcpy(&v, h_.data() + off, 4);
    return static_cast<int>(v);
}

int HostSignalHeader::status() const { return le32(OFF_STATUS); }

uint32_t HostSignalHeader::tile_row_coord() const {
    return static_cast<uint32_t>(le32(OFF_TILE_COORD + 0));
}

uint32_t HostSignalHeader::tile_col_coord() const {
    return static_cast<uint32_t>(le32(OFF_TILE_COORD + 4));
}

int HostSignalHeader::mma_tile_m() const {
    return le32(OFF_MMA_TILE_SIZE + 0);
}

int HostSignalHeader::mma_tile_n() const {
    return le32(OFF_MMA_TILE_SIZE + 4);
}

uint16_t HostSignalHeader::num_registers_per_thread() const {
    if (OFF_NUM_REGISTERS_PER_THREAD + 2 > static_cast<int>(h_.size())) return 0;
    uint16_t v = 0;
    std::memcpy(&v, h_.data() + OFF_NUM_REGISTERS_PER_THREAD, 2);
    return v;
}

void HostSignalHeader::extract_indices(std::vector<uint32_t>& a_rows,
                                       std::vector<uint32_t>& b_cols) const {
    uint16_t n = num_registers_per_thread();
    if (n == 0 || n > MAX_REGS) {
        throw std::runtime_error("invalid num_registers_per_thread");
    }

    std::vector<uint8_t> rbuf(n), cbuf(n);
    for (int i = 0; i < n; ++i) {
        rbuf[i] = h_[OFF_THREAD_ROWS + i];
        cbuf[i] = h_[OFF_THREAD_COLS + i];
    }
    std::sort(rbuf.begin(), rbuf.end());
    std::sort(cbuf.begin(), cbuf.end());
    rbuf.erase(std::unique(rbuf.begin(), rbuf.end()), rbuf.end());
    cbuf.erase(std::unique(cbuf.begin(), cbuf.end()), cbuf.end());

    uint32_t row_off = tile_row_coord() * static_cast<uint32_t>(mma_tile_m());
    uint32_t col_off = tile_col_coord() * static_cast<uint32_t>(mma_tile_n());

    a_rows.clear();
    b_cols.clear();
    for (auto r : rbuf) a_rows.push_back(row_off + r);
    for (auto c : cbuf) b_cols.push_back(col_off + c);
}

} // namespace pearl
