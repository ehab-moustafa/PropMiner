#include "seed_generator.h"

#include <bit>
#include <chrono>
#include <cstdio>

namespace pearl {

namespace {
    // xorshift64* — fast, high-quality 64-bit PRNG for seed generation only.
    // This is NOT used for any hashing or matrix math; it merely produces the
    // nonce/seed values that feed the GPU-side LCG and noise generators.
    uint64_t xorshift64star(uint64_t& state) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 0x2545F4914F6CDD1DULL;
    }

    uint64_t round_up_pow2(uint64_t v) {
        if (v == 0) return 1;
        return 1ULL << (64 - std::countl_zero(v - 1));
    }
}

SeedGenerator::SeedGenerator(uint64_t seed_base) : seed_base_(seed_base) {}

SeedGenerator::~SeedGenerator() {
    stop();
}

void SeedGenerator::start(size_t ring_size) {
    if (running_.exchange(true)) return;
    size_t sz = round_up_pow2(ring_size);
    ring_.resize(sz);
    ring_mask_ = sz - 1;
    write_seq_ = 0;
    read_seq_ = 0;
    thread_ = std::thread(&SeedGenerator::thread_func, this);
}

void SeedGenerator::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
}

size_t SeedGenerator::ready_count() const {
    uint64_t w = write_seq_.load(std::memory_order_acquire);
    uint64_t r = read_seq_.load(std::memory_order_relaxed);
    return static_cast<size_t>(w - r);
}

size_t SeedGenerator::pop(Packet* out, size_t count) {
    uint64_t r = read_seq_.load(std::memory_order_relaxed);
    uint64_t w = write_seq_.load(std::memory_order_acquire);
    size_t available = static_cast<size_t>(w - r);
    size_t take = count < available ? count : available;
    for (size_t i = 0; i < take; ++i) {
        out[i] = ring_[(r + i) & ring_mask_];
    }
    read_seq_.store(r + take, std::memory_order_release);
    return take;
}

void SeedGenerator::thread_func() {
    // Seed the PRNG from the base so different GPUs get disjoint sequences.
    uint64_t prng_state = seed_base_ ^ 0x9E3779B97F4A7C15ULL;
    uint64_t seq = 0;

    while (running_.load(std::memory_order_relaxed)) {
        uint64_t w = write_seq_.load(std::memory_order_relaxed);
        uint64_t r = read_seq_.load(std::memory_order_acquire);
        if (w - r >= ring_mask_) {
            // Ring full; back off without spinning hard.
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            continue;
        }

        Packet p;
        p.seed_lo = xorshift64star(prng_state);
        p.seed_hi = xorshift64star(prng_state);
        p.batch_id = static_cast<uint32_t>(seq++);
        ring_[w & ring_mask_] = p;
        write_seq_.store(w + 1, std::memory_order_release);
    }
}

} // namespace pearl
