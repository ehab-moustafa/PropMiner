#include "async_b_stream.h"

#include <cstring>
#include <stdexcept>

#if !defined(PROP_MINER_HOST_ONLY_TESTS)
#include <cuda_runtime.h>
#endif

namespace pearl {

AsyncBStream::AsyncBStream() = default;

AsyncBStream::~AsyncBStream() {
    destroy();
}

bool AsyncBStream::allocate(int device_index, size_t bytes_per_buffer) {
#if defined(PROP_MINER_HOST_ONLY_TESTS)
    (void)device_index; (void)bytes_per_buffer;
    return false;
#else
    if (allocated_) return false;
    device_index_ = device_index;
    bytes_per_buffer_ = bytes_per_buffer;

    cudaError_t e = cudaSetDevice(device_index_);
    if (e != cudaSuccess) return false;

    e = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (e != cudaSuccess) { stream_ = nullptr; return false; }

    for (int i = 0; i < 2; ++i) {
        e = cudaHostAlloc(&host_[i], bytes_per_buffer_, cudaHostAllocPortable);
        if (e != cudaSuccess) { destroy(); return false; }
        e = cudaMalloc(&device_[i], bytes_per_buffer_);
        if (e != cudaSuccess) { destroy(); return false; }
        std::memset(host_[i], 0, bytes_per_buffer_);
    }

    allocated_ = true;
    ping_ = 0;
    return true;
#endif
}

void AsyncBStream::destroy() {
#if !defined(PROP_MINER_HOST_ONLY_TESTS)
    if (!allocated_) return;
    if (stream_) {
        cudaStreamSynchronize(stream_);
    }
    for (int i = 0; i < 2; ++i) {
        if (host_[i]) { cudaFreeHost(host_[i]); host_[i] = nullptr; }
        if (device_[i]) { cudaFree(device_[i]); device_[i] = nullptr; }
    }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
#endif
    allocated_ = false;
    bytes_per_buffer_ = 0;
    device_index_ = -1;
    ping_ = 0;
}

uint8_t* AsyncBStream::host_write_ptr() {
    return allocated_ ? host_[ping_] : nullptr;
}

bool AsyncBStream::upload_async(size_t bytes) {
#if defined(PROP_MINER_HOST_ONLY_TESTS)
    (void)bytes;
    return false;
#else
    if (!allocated_ || !stream_) return false;
    if (bytes > bytes_per_buffer_) return false;
    cudaError_t e = cudaMemcpyAsync(device_[ping_], host_[ping_], bytes,
                                    cudaMemcpyHostToDevice, stream_);
    return e == cudaSuccess;
#endif
}

uint8_t* AsyncBStream::device_read_ptr() {
    return allocated_ ? device_[ping_ ^ 1] : nullptr;
}

void AsyncBStream::swap() {
    ping_ ^= 1;
}

} // namespace pearl
