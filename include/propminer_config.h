#pragma once

#include <stdint.h>
#include <stdio.h>

/* ── PearlHash protocol constants ─────────────────────────────────── */

#define PEARL_TILE_M            128
#define PEARL_TILE_N            128
#define PEARL_TILE_K            128

/* GEMM tile shape for INT8→INT32 tensor cores (m16n8k32 atom) */
#define MMA_M                   16
#define MMA_N                    8
#define MMA_K                   32

/* Threads per block — 256 for 8 warps (4 warps per m16n8 output tile) */
#define WARP_SIZE               32
#define THREADS_PER_BLOCK       256

/* BLAKE3 parameters */
#define B3_CHAINING_VALUE_BYTES 32
#define B3_CHAINING_VALUE_U32   8
#define B3_MSG_BLOCK_BYTES      64
#define B3_MSG_BLOCK_U32        16
#define B3_ROUNDS               7   /* 6 with permutation + 1 final */

/* BLAKE3 flags */
#define B3_FLAG_CHUNK_START     (1u << 0)
#define B3_FLAG_CHUNK_END       (1u << 1)
#define B3_FLAG_PARENT          (1u << 2)
#define B3_FLAG_ROOT            (1u << 3)
#define B3_FLAG_KEYED_HASH      (1u << 4)

/* Flags used for the single-block keyed hash (noise + PoW final compress) */
#define B3_FLAGS_KEYED_SINGLE   (B3_FLAG_KEYED_HASH | B3_FLAG_CHUNK_START | \
                                  B3_FLAG_CHUNK_END | B3_FLAG_ROOT)

/* ── Miner configuration defaults ─────────────────────────────────── */

#define DEFAULT_POOL_URL        "prl.kryptex.network:7048"
#define DEFAULT_WALLET          ""
#define DEFAULT_WORKER_NAME     "PropMiner"
#define DEFAULT_GPUS            "all"
#define DEFAULT_INTENSITY       128    /* nonce range per work item */
#define DEFAULT_SHARE_TARGET    48     /* nbits for local share detection */

#define WORK_QUEUE_CAPACITY     64     /* ring buffer slots */
#define RESULT_BUFFER_SIZE      1024   /* max pending shares */
#define SHARE_POLL_INTERVAL_US  50     /* microseconds between share polls */

#define MAX_GPUS                16

/* ── Command-line config struct ───────────────────────────────────── */

typedef struct {
    char    pool_url[256];
    char    wallet[128];
    char    worker_name[64];
    int     gpu_indices[MAX_GPUS];
    int     gpu_count;
    int     intensity;
    uint32_t share_target_nbits;
    int     verbose;
    int     speed_test;   /* 1 = run benchmark and exit */
} PropMinerConfig;

/* Initialize with defaults */
static inline void propminer_config_defaults(PropMinerConfig* cfg) {
    snprintf(cfg->pool_url, sizeof(cfg->pool_url), "%s", DEFAULT_POOL_URL);
    cfg->wallet[0] = '\0';
    snprintf(cfg->worker_name, sizeof(cfg->worker_name), "%s", DEFAULT_WORKER_NAME);
    cfg->gpu_count = -1;       /* -1 = all visible GPUs */
    cfg->intensity = DEFAULT_INTENSITY;
    cfg->share_target_nbits = DEFAULT_SHARE_TARGET;
    cfg->verbose = 0;
    cfg->speed_test = 0;
}