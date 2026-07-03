#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace pearl {

// Monitors a CUDA worker stream for TDR/launch timeout and triggers a recovery
// callback.  The callback is responsible for destroying the old context, re-
// creating it, and resuming mining from the last known good nonce.
class Watchdog {
public:
    using RecoveryCallback = std::function<void()>;

    Watchdog();
    ~Watchdog();

    // Start the watchdog thread.  `period_ms` is the poll interval.
    void start(RecoveryCallback cb, int period_ms = 3000);

    // Stop the watchdog thread.
    void stop();

    // Mark that the worker is still alive.  The watchdog expects this to be
    // called frequently (e.g., after every successful kernel poll).
    void heartbeat();

    // True if recovery has been triggered.
    bool triggered() const { return triggered_.load(); }

    Watchdog(const Watchdog&) = delete;
    Watchdog& operator=(const Watchdog&) = delete;

private:
    void thread_func();

    std::atomic<bool> running_{false};
    std::atomic<bool> triggered_{false};
    std::atomic<int64_t> last_heartbeat_ms_{0};
    RecoveryCallback callback_;
    int period_ms_ = 3000;
    std::thread thread_;
};

} // namespace pearl
