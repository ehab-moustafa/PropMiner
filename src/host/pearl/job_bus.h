#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>

#include "pearl_types.h"
#include "sigma_context.h"

namespace pearl {

// Thread-safe single-producer/multi-consumer job distribution.
// The orchestrator publishes a new SigmaContext; workers drain the current
// version atomically and install it locally.
class JobBus {
public:
    struct Entry {
        std::shared_ptr<SigmaContext> ctx;
        uint64_t version = 0;
    };

    // Publish a new job. Replaces any previous job.
    void publish(std::shared_ptr<SigmaContext> ctx);

    // Consumers call this to get the latest published entry (non-blocking).
    Entry drain_latest();

private:
    std::mutex mtx_;
    std::shared_ptr<SigmaContext> ctx_;
    uint64_t version_ = 0;
};

} // namespace pearl
