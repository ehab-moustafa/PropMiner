#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace pearl {

// CPU-side seed generation ring buffer.
//
// Design constraints:
//   - Performs ZERO hashing / NoisyGEMM math.  It only produces 64-bit seed
//     values and small metadata packets ahead of time so the GPU never stalls
//     waiting for the host.
//   - Runs on a dedicated host thread (the "matrix assembly line").
//   - Uses lock-free SPSC semantics where possible to avoid CPU overhead.
//   - The produced seeds are consumed by GpuWorker and uploaded to the device
//     via cudaMemcpyAsync on a non-blocking copy stream.
class SeedGenerator {
public:
    struct Packet {
        uint64_t seed_lo = 0;
        uint64_t seed_hi = 0;
        uint32_t batch_id = 0;
    };

    explicit SeedGenerator(uint64_t seed_base);
    ~SeedGenerator();

    // Start the background generator thread.
    void start(size_t ring_size = 1024);

    // Stop the generator thread.
    void stop();

    // Consumer side: pop the next `count` packets.  Returns the number actually
    // available; caller may use fewer if the ring is not full yet.
    size_t pop(Packet* out, size_t count);

    // Number of packets currently ready to consume.
    size_t ready_count() const;

    SeedGenerator(const SeedGenerator&) = delete;
    SeedGenerator& operator=(const SeedGenerator&) = delete;

private:
    void thread_func();

    uint64_t seed_base_ = 0;
    std::atomic<uint64_t> write_seq_{0};
    std::atomic<uint64_t> read_seq_{0};
    std::vector<Packet> ring_;
    size_t ring_mask_ = 0;

    std::atomic<bool> running_{false};
    std::thread thread_;
};

} // namespace pearl
