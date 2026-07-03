#pragma once

#include "cuda_compat.h"

#include <cstdint>
#include <cstddef>
#include <vector>

namespace pearl {

// Double-buffered host->device upload pipeline for the B matrix.
// While the GPU kernel consumes buffer [i], the host asynchronously fills and
// uploads the next B chunk into buffer [i^1].  Uses pinned host memory and
// cudaMemcpyAsync on a dedicated copy stream.
//
// CPU isolation guarantee: this class performs no hashing, no matrix math, and
// no noise generation.  It only moves pre-computed bytes from host to device.
class AsyncBStream {
public:
    AsyncBStream();
    ~AsyncBStream();

    // Allocate pinned host buffers and device buffers.  `bytes_per_buffer` must
    // be the total size of one B upload chunk (usually N*K bytes).
    bool allocate(int device_index, size_t bytes_per_buffer);

    // Release all resources.
    void destroy();

    // Return a pinned host pointer for the *next* buffer to fill.
    uint8_t* host_write_ptr();

    // Queue an async upload of the current host buffer to the matching device
    // buffer.  The caller must ensure the kernel consuming the *previous*
    // device buffer has completed (e.g., via events).
    bool upload_async(size_t bytes);

    // Return the device pointer currently ready for the kernel.
    uint8_t* device_read_ptr();

    // Swap ping/pong buffers after the current upload/read are complete.
    void swap();

    cudaStream_t stream() const { return stream_; }

    AsyncBStream(const AsyncBStream&) = delete;
    AsyncBStream& operator=(const AsyncBStream&) = delete;

private:
    int device_index_ = -1;
    size_t bytes_per_buffer_ = 0;
    int ping_ = 0;

    uint8_t* host_[2] = {nullptr, nullptr};
    uint8_t* device_[2] = {nullptr, nullptr};
    cudaStream_t stream_ = nullptr;
    bool allocated_ = false;
};

} // namespace pearl
