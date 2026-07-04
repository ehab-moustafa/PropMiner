#pragma once

#include <cuda.h>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace propminer {

/* Opaque handle to per-GPU CUDA state */
struct CudaGpu {
    int         device;
    CUstream    stream;
    CUmodule    module;
    CUfunction  kernel;

    // Device memory
    CUdeviceptr work_queue_ptr;
    CUdeviceptr result_buf_ptr;
    CUdeviceptr matrix_data_ptr;
    CUdeviceptr stats_ptr;
    CUdeviceptr shutdown_ptr;
    CUdeviceptr pow_target_ptr;
    CUdeviceptr pow_key_ptr;

    // Pinned (page-locked) host memory for zero-copy readback
    void*       pinned_result_buf;
    void*       pinned_stats;
    uint32_t    pinned_shutdown;

    // Device info
    int         compute_major;
    int         compute_minor;
    size_t      total_mem;
    char        name[256];
};

/*
 * Initialize CUDA Driver API. Call once at startup.
 * Returns 0 on success, negative on failure.
 */
int cuda_driver_init();

/*
 * Get the number of visible CUDA devices.
 */
int cuda_driver_device_count();

/*
 * Initialize a single GPU: create context, load kernel module, allocate memory.
 * Returns 0 on success.
 */
int cuda_driver_init_gpu(CudaGpu* gpu, int device_index,
                          size_t work_queue_size,
                          size_t result_buf_size);

/*
 * Destroy a GPU: free memory, unload module, destroy context.
 */
void cuda_driver_destroy_gpu(CudaGpu* gpu);

/*
 * Launch the persistent mining kernel on the given GPU.
 * grid_x: number of thread blocks in x dimension
 * block_x: threads per block (256 for PearlHash)
 */
int cuda_driver_launch_kernel(CudaGpu* gpu,
                               uint32_t grid_x,
                               uint32_t block_x);

/*
 * Push a work item into the device-side work queue.
 * Blocks if the queue is full.
 */
int cuda_driver_push_work(CudaGpu* gpu,
                           const void* work_item,  // pointer to WorkItem
                           size_t item_size);

/*
 * Synchronize the GPU stream.
 */
int cuda_driver_sync(CudaGpu* gpu);

/*
 * Copy device memory to pinned host memory (for result readback).
 */
int cuda_driver_copy_to_host(CudaGpu* gpu,
                               void* dst,
                               CUdeviceptr src,
                               size_t bytes);

/*
 * Copy host memory to device.
 */
int cuda_driver_copy_to_device(CudaGpu* gpu,
                                 CUdeviceptr dst,
                                 const void* src,
                                 size_t bytes);

/*
 * Signal shutdown to the persistent kernel.
 */
int cuda_driver_signal_shutdown(CudaGpu* gpu);

/*
 * Load kernel from a compiled cubin file.
 * Falls back to loading from embedded fatbin if file not found.
 */
int cuda_driver_load_module(CudaGpu* gpu, const char* cubin_path);

} // namespace propminer