#pragma once

#include <stdint.h>
#include <stddef.h>

#include "propminer_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Result types ─────────────────────────────────────────────────── */

#define PROP_SUCCESS              0
#define PROP_ERR_INVALID_ARGS   -1
#define PROP_ERR_CUDA           -2
#define PROP_ERR_OUT_OF_MEMORY  -3
#define PROP_ERR_POOL           -4
#define PROP_ERR_TIMEOUT        -5

/* ── Share result ─────────────────────────────────────────────────── */

typedef struct {
    uint8_t  sigma[32];
    uint64_t nonce;
    uint8_t  hash[32];
    int32_t  tile_row;
    int32_t  tile_col;
    uint32_t target_nbits;
} PropShare;

/* ── Hashrate stats (returned by get_hashrate) ────────────────────── */

typedef struct {
    uint64_t shares_submitted;
    uint64_t shares_accepted;
    uint64_t shares_rejected;
    double   nonce_hashrate;   /* nonces per second */
} PropStats;

/* ── Opaque miner handle ──────────────────────────────────────────── */

typedef struct PropMiner PropMiner;

/* ── Public API ───────────────────────────────────────────────────── */

/**
 * Initialize the miner. Allocates GPU memory, compiles/loads kernels,
 * creates CUDA context(s). Call once before any other API call.
 *
 * Returns PROP_SUCCESS on success, negative error code otherwise.
 */
int propminer_init(PropMiner** out, const PropMinerConfig* config);

/**
 * Start the persistent mining loop. This blocks the calling thread
 * until propminer_stop() is called or an error occurs.
 *
 * Returns PROP_SUCCESS on success.
 */
int propminer_start(PropMiner* miner);

/**
 * Signal the mining kernel to stop and wait for graceful shutdown.
 * Call from a separate thread or via signal handler.
 */
void propminer_stop(PropMiner* miner);

/**
 * Query current hashrate and share statistics.
 * Thread-safe; may be called while mining is active.
 */
void propminer_get_hashrate(const PropMiner* miner, PropStats* out);

/**
 * Destroy the miner and release all resources.
 */
void propminer_destroy(PropMiner* miner);

/**
 * Retrieve the most recent share from the result buffer.
 * Returns PROP_SUCCESS if a share is available, PROP_ERR_TIMEOUT if empty.
 */
int propminer_pop_share(PropMiner* miner, PropShare* out);

#ifdef __cplusplus
}
#endif