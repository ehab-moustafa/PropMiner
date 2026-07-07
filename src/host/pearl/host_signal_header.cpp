#include "host_signal_header.h"

#include <algorithm>
#include <array>
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
    constexpr int OFF_TARGET = 604;
    constexpr int MAX_REGS = 256;
}

HostSignalHeader::HostSignalHeader(const std::vector<uint8_t>& header)
    : data_(header.data()), size_(header.size()) {}

HostSignalHeader::HostSignalHeader(const uint8_t* data, size_t size)
    : data_(data), size_(size) {}

int HostSignalHeader::le32(int off) const {
    if (!data_ || off + 4 > static_cast<int>(size_)) return 0;
    uint32_t v = 0;
    std::memcpy(&v, data_ + off, 4);
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
    if (!data_ || OFF_NUM_REGISTERS_PER_THREAD + 2 > static_cast<int>(size_)) return 0;
    uint16_t v = 0;
    std::memcpy(&v, data_ + OFF_NUM_REGISTERS_PER_THREAD, 2);
    return v;
}

std::array<uint32_t, 8> HostSignalHeader::header_pow_target() const {
    std::array<uint32_t, 8> words{};
    if (!data_ || OFF_TARGET + static_cast<int>(words.size()) * 4 >
                      static_cast<int>(size_)) {
        return words;
    }
    for (size_t i = 0; i < words.size(); ++i) {
        std::memcpy(&words[i], data_ + OFF_TARGET + static_cast<int>(i) * 4, 4);
    }
    return words;
}

uint8_t HostSignalHeader::min_register_row() const {
    const uint16_t n = num_registers_per_thread();
    if (n == 0 || n > MAX_REGS) return 0;
    uint8_t min_r = 255;
    for (int i = 0; i < n; ++i)
        min_r = std::min(min_r, data_[OFF_THREAD_ROWS + i]);
    return min_r;
}

uint8_t HostSignalHeader::min_register_col() const {
    const uint16_t n = num_registers_per_thread();
    if (n == 0 || n > MAX_REGS) return 0;
    uint8_t min_c = 255;
    for (int i = 0; i < n; ++i)
        min_c = std::min(min_c, data_[OFF_THREAD_COLS + i]);
    return min_c;
}

void HostSignalHeader::extract_indices(std::vector<uint32_t>& a_rows,
                                       std::vector<uint32_t>& b_cols) const {
    uint16_t n = num_registers_per_thread();
    if (n == 0 || n > MAX_REGS) {
        throw std::runtime_error("invalid num_registers_per_thread");
    }

    std::vector<uint8_t> rbuf(n), cbuf(n);
    for (int i = 0; i < n; ++i) {
        rbuf[i] = data_[OFF_THREAD_ROWS + i];
        cbuf[i] = data_[OFF_THREAD_COLS + i];
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
