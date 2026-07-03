#pragma once

#include <stdint.h>

/* ── Device-side work queue (ring buffer) ───────────────────────────
 *
 * The host pushes work items into the queue; kernel thread blocks
 * poll queue->tail and consume from queue->head. Both indices are
 * volatile so the GPU sees host writes without a full fence.
 */

struct __align__(64) WorkQueue {
    volatile uint64_t head;       /* consumer (GPU) advances */
    volatile uint64_t tail;       /* producer (host) advances */
    uint64_t          capacity;   /* ring buffer slot count */
    volatile uint32_t active;     /* 1 = new work available, 0 = idle */
    uint32_t          _pad[15];   /* cache-line pad */
};

/* Flexible array: items follow the header at runtime */

/* ── Work item (one mining job) ───────────────────────────────────── */

struct __align__(32) WorkItem {
    uint64_t nonce_base;       /* starting nonce for this batch */
    uint32_t nonce_count;      /* how many nonces to process */
    uint32_t padding;
    uint8_t  sigma[32];        /* current block header bytes */
    uint32_t target_nbits;     /* vardiff target */
    uint32_t padding2;
    uint8_t  b_seed[32];       /* B matrix seed */
    uint32_t block_height;
    uint32_t padding3;
};

/* Host-side helpers */
static inline uint64_t work_queue_slots(const WorkQueue* q) {
    return q->tail - q->head;
}

static inline int work_queue_empty(const WorkQueue* q) {
    return q->head == q->tail;
}

static inline WorkItem* work_queue_peek(const WorkQueue* q, void* items_base) {
    if (work_queue_empty(q)) return NULL;
    WorkItem* items = (WorkItem*)items_base;
    return &items[q->head % q->capacity];
}