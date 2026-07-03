#pragma once

#include <stdint.h>

/* ── Share result written by GPU into pinned (zero-copy) memory ─────
 *
 * The host reads shares via atomic index exchange:
 *   idx = atomicExch(&result_buffer->write_pos, 0)
 * Then reads up to idx results from the ring.
 */

struct __align__(32) ShareResult {
    uint8_t  sigma[32];
    uint64_t nonce;
    uint8_t  hash[32];
    int32_t  tile_row;
    int32_t  tile_col;
    uint32_t target_nbits;
    uint32_t padding;
};

struct __align__(64) ResultBuffer {
    volatile uint32_t write_pos;   /* GPU writes → host resets to 0 after read */
    uint32_t          capacity;
    uint32_t          _pad[6];
};