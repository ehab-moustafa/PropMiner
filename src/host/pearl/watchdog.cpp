#include "watchdog.h"

#include <chrono>
#include <cstdio>

#if !defined(PROP_MINER_HOST_ONLY_TESTS)
#include <cuda_runtime.h>
#endif

namespace pearl {

namespace {
    int64_t now_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
}

Watchdog::Watchdog() = default;

Watchdog::~Watchdog() {
    stop();
}

void Watchdog::start(RecoveryCallback cb, int period_ms) {
    callback_ = std::move(cb);
    period_ms_ = period_ms;
    last_heartbeat_ms_ = now_ms();
    running_ = true;
    thread_ = std::thread(&Watchdog::thread_func, this);
}

void Watchdog::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
}

void Watchdog::heartbeat() {
    last_heartbeat_ms_ = now_ms();
}

void Watchdog::thread_func() {
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(period_ms_));
        if (!running_.load()) break;

        int64_t last = last_heartbeat_ms_.load();
        int64_t elapsed = now_ms() - last;

#if !defined(PROP_MINER_HOST_ONLY_TESTS)
        // Also probe the CUDA driver for a stuck context.
        cudaError_t err = cudaSetDevice(0);  // probe current context
        bool cuda_stuck = (err == cudaErrorLaunchTimeout ||
                           err == cudaErrorUnknown ||
                           err == cudaErrorLaunchFailure);
#else
        bool cuda_stuck = false;
#endif

        if (elapsed > period_ms_ * 3 || cuda_stuck) {
            std::fprintf(stderr,
                "[watchdog] Worker stall detected (elapsed=%lld ms). "
                "Triggering recovery...\n",
                static_cast<long long>(elapsed));
            triggered_ = true;
            if (callback_) {
                try {
                    callback_();
                } catch (...) {
                    std::fprintf(stderr,
                        "[watchdog] Recovery callback threw; will retry\n");
                }
            }
            last_heartbeat_ms_ = now_ms();
        }
    }
}

} // namespace pearl
