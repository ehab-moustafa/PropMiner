#include "propminer.h"
#include "multiplexer.h"
#include "cuda_driver.h"
#include <cstring>

extern "C" {

int propminer_init(PropMiner** out, const PropMinerConfig* config) {
    if (!out || !config) return PROP_ERR_INVALID_ARGS;
    // Initialize CUDA
    if (cuda_driver_init() != 0) return PROP_ERR_CUDA;
    // TODO: Wire through Multiplexer when C API is needed
    *out = nullptr;
    return PROP_SUCCESS;
}

int propminer_start(PropMiner* miner) {
    return PROP_SUCCESS;
}

void propminer_stop(PropMiner* miner) {
}

void propminer_get_hashrate(const PropMiner* miner, PropStats* out) {
    if (out) memset(out, 0, sizeof(*out));
}

void propminer_destroy(PropMiner* miner) {
}

int propminer_pop_share(PropMiner* miner, PropShare* out) {
    return PROP_ERR_TIMEOUT;
}

} // extern "C"