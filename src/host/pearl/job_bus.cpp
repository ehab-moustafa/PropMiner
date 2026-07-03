#include "job_bus.h"

namespace pearl {

void JobBus::publish(std::shared_ptr<SigmaContext> ctx) {
    std::lock_guard<std::mutex> lk(mtx_);
    ctx_ = std::move(ctx);
    ++version_;
}

JobBus::Entry JobBus::drain_latest() {
    std::lock_guard<std::mutex> lk(mtx_);
    return {ctx_, version_};
}

} // namespace pearl
