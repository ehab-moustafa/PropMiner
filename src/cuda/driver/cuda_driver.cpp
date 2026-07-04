#include "cuda_driver.h"
#include "propminer_config.h"
#include "../include/work_queue.h"
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace propminer {

// Helper to check CUDA Driver API errors
#define CU_CHECK(call)                                            \
    do {                                                          \
        CUresult _err = (call);                                   \
        if (_err != CUDA_SUCCESS) {                               \
            const char* _msg = nullptr;                           \
            cuGetErrorString(_err, &_msg);                        \
            fprintf(stderr, "[cuda] %s:%d %s -> %s (%d)\n",      \
                    __FILE__, __LINE__, #call, _msg, _err);       \
            return -1;                                            \
        }                                                         \
    } while (0)

int cuda_driver_init() {
    // Runtime API initializes implicitly. Force a lightweight runtime init
    // so that a primary context exists before any driver API calls.
    cudaError_t err = cudaFree(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[cuda] runtime init failed: %s (%d)\n",
                cudaGetErrorString(err), static_cast<int>(err));
        return -1;
    }
    return 0;
}

int cuda_driver_device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        const char* msg = cudaGetErrorString(err);
        fprintf(stderr, "[cuda] cudaGetDeviceCount failed: %s (%d)\n", msg, static_cast<int>(err));
        return 0;
    }
    return count;
}

int cuda_driver_init_gpu(CudaGpu* gpu, int device_index,
                          size_t work_queue_size,
                          size_t result_buf_size) {
    // Use Runtime API to set the device. This creates the implicit primary
    // context on WSL2 hosts where cuCtxCreate fails.
    cudaError_t rt_err = cudaSetDevice(device_index);
    if (rt_err != cudaSuccess) {
        fprintf(stderr, "[cuda] cudaSetDevice(%d) failed: %s (%d)\n",
                device_index, cudaGetErrorString(rt_err), static_cast<int>(rt_err));
        return -1;
    }
    gpu->device = device_index;
    gpu->ctx = nullptr;  // implicit primary context

    // Get device attributes via Runtime API (WSL2-safe)
    cudaDeviceProp prop;
    cudaError_t prop_err = cudaGetDeviceProperties(&prop, gpu->device);
    if (prop_err == cudaSuccess) {
        gpu->compute_major = prop.major;
        gpu->compute_minor = prop.minor;
        gpu->total_mem = prop.totalGlobalMem;
        strncpy(gpu->name, prop.name, sizeof(gpu->name) - 1);
        gpu->name[sizeof(gpu->name) - 1] = '\0';
    } else {
        gpu->compute_major = 0;
        gpu->compute_minor = 0;
        gpu->total_mem = 0;
        gpu->name[0] = '\0';
    }

    fprintf(stderr, "[cuda] GPU %d: %s (sm_%d%d, %zu MB)\n",
            device_index, gpu->name,
            gpu->compute_major, gpu->compute_minor,
            gpu->total_mem / (1024 * 1024));

    // Create stream
    CU_CHECK(cuStreamCreate(&gpu->stream, 0));

    // Allocate device memory for work queue
    // WorkQueue header + capacity * WorkItem
    size_t queue_alloc = sizeof(WorkQueue) + work_queue_size;
    CU_CHECK(cuMemAlloc(&gpu->work_queue_ptr, queue_alloc));

    // Allocate device memory for result buffer
    CU_CHECK(cuMemAlloc(&gpu->result_buf_ptr, result_buf_size));

    // Allocate device memory for stats counters [accepted, rejected, computed]
    CU_CHECK(cuMemAlloc(&gpu->stats_ptr, 3 * sizeof(uint64_t)));

    // Allocate device memory for shutdown flag
    CU_CHECK(cuMemAlloc(&gpu->shutdown_ptr, sizeof(uint32_t)));

    // Allocate device memory for PoW target (8 x uint32)
    CU_CHECK(cuMemAlloc(&gpu->pow_target_ptr, 8 * sizeof(uint32_t)));

    // Allocate device memory for PoW key (8 x uint32)
    CU_CHECK(cuMemAlloc(&gpu->pow_key_ptr, 8 * sizeof(uint32_t)));

    // Allocate pinned host memory for zero-copy result readback
    CU_CHECK(cuMemAllocHost(&gpu->pinned_result_buf, result_buf_size));
    memset(gpu->pinned_result_buf, 0, result_buf_size);

    // Allocate pinned host memory for stats
    CU_CHECK(cuMemAllocHost(&gpu->pinned_stats, 3 * sizeof(uint64_t)));
    memset(gpu->pinned_stats, 0, 3 * sizeof(uint64_t));

    gpu->pinned_shutdown = 0;

    // Zero out device stats
    uint64_t zero_stats[3] = {0, 0, 0};
    CU_CHECK(cuMemcpyHtoD(gpu->stats_ptr, zero_stats, 3 * sizeof(uint64_t)));

    // Zero out device shutdown flag
    CU_CHECK(cuMemcpyHtoD(gpu->shutdown_ptr, &gpu->pinned_shutdown, sizeof(uint32_t)));

    // Initialize work queue header: head=0, tail=0, capacity, active=0
    uint64_t queue_head = 0;
    uint64_t queue_tail = 0;
    uint64_t queue_capacity = WORK_QUEUE_CAPACITY;
    uint32_t queue_active = 0;
    CU_CHECK(cuMemcpyHtoD(gpu->work_queue_ptr, &queue_head, sizeof(uint64_t)));
    CU_CHECK(cuMemcpyHtoD(gpu->work_queue_ptr + 8, &queue_tail, sizeof(uint64_t)));
    CU_CHECK(cuMemcpyHtoD(gpu->work_queue_ptr + 16, &queue_capacity, sizeof(uint64_t)));
    CU_CHECK(cuMemcpyHtoD(gpu->work_queue_ptr + 24, &queue_active, sizeof(uint32_t)));

    // Load kernel module — try multiple paths
    char cubin_path[512];
    bool loaded = false;

    // Try 1: Same directory as the build output
    snprintf(cubin_path, sizeof(cubin_path), "pearlhash_kernel_sm%d%d.cubin",
             gpu->compute_major, gpu->compute_minor);
    if (cuda_driver_load_module(gpu, cubin_path) == 0) loaded = true;

    // Try 2: Build subdirectory
    if (!loaded) {
        snprintf(cubin_path, sizeof(cubin_path), "build/pearlhash_kernel_sm%d%d.cubin",
                 gpu->compute_major, gpu->compute_minor);
        if (cuda_driver_load_module(gpu, cubin_path) == 0) loaded = true;
    }

    // Try 3: Generic cubin
    if (!loaded) {
        snprintf(cubin_path, sizeof(cubin_path), "pearlhash_kernel.cubin");
        if (cuda_driver_load_module(gpu, cubin_path) == 0) loaded = true;
    }

    // Try 4: Architecture fallback (e.g., sm_90 for sm_90)
    if (!loaded) {
        snprintf(cubin_path, sizeof(cubin_path), "pearlhash_kernel_sm%d0.cubin",
                 gpu->compute_major);
        if (cuda_driver_load_module(gpu, cubin_path) == 0) loaded = true;
    }

    if (!loaded) {
        fprintf(stderr, "[cuda] No kernel module found for sm_%d%d\n",
                gpu->compute_major, gpu->compute_minor);
        cuda_driver_destroy_gpu(gpu);
        return -1;
    }

    // Get kernel function
    CU_CHECK(cuModuleGetFunction(&gpu->kernel, gpu->module,
                                  "propminer_persistent_kernel"));

    return 0;
}

void cuda_driver_destroy_gpu(CudaGpu* gpu) {
    if (gpu->kernel) {
        // Kernel is part of module, destroyed with it
        gpu->kernel = nullptr;
    }
    if (gpu->module) {
        cuModuleUnload(gpu->module);
        gpu->module = nullptr;
    }
    if (gpu->stream) {
        cuStreamDestroy(gpu->stream);
        gpu->stream = nullptr;
    }
    if (gpu->work_queue_ptr)   cuMemFree(gpu->work_queue_ptr);
    if (gpu->result_buf_ptr)   cuMemFree(gpu->result_buf_ptr);
    if (gpu->matrix_data_ptr)  cuMemFree(gpu->matrix_data_ptr);
    if (gpu->stats_ptr)        cuMemFree(gpu->stats_ptr);
    if (gpu->shutdown_ptr)     cuMemFree(gpu->shutdown_ptr);
    if (gpu->pow_target_ptr)   cuMemFree(gpu->pow_target_ptr);
    if (gpu->pow_key_ptr)      cuMemFree(gpu->pow_key_ptr);

    gpu->work_queue_ptr = 0;
    gpu->result_buf_ptr = 0;
    gpu->stats_ptr = 0;
    gpu->shutdown_ptr = 0;
    gpu->pow_target_ptr = 0;
    gpu->pow_key_ptr = 0;

    if (gpu->pinned_result_buf) {
        cuMemFreeHost(gpu->pinned_result_buf);
        gpu->pinned_result_buf = nullptr;
    }
    if (gpu->pinned_stats) {
        cuMemFreeHost(gpu->pinned_stats);
        gpu->pinned_stats = nullptr;
    }

    // With Runtime API the primary context is implicit; do not destroy it.
    gpu->ctx = nullptr;
}

int cuda_driver_launch_kernel(CudaGpu* gpu,
                               uint32_t grid_x,
                               uint32_t block_x) {
    void* kernel = reinterpret_cast<void*>(gpu->kernel);

    // Kernel parameters:
    // 1. WorkQueue* queue
    // 2. ResultBuffer* result_buffer
    // 3. uint32_t* pow_target
    // 4. uint32_t* pow_key
    // 5. uint64_t* stats [accepted, rejected, computed]
    // 6. uint32_t* shutdown_flag

    void* params[] = {
        &gpu->work_queue_ptr,    // queue
        &gpu->result_buf_ptr,    // result_buffer
        &gpu->pow_target_ptr,    // pow_target
        &gpu->pow_key_ptr,       // pow_key
        &gpu->stats_ptr,         // stats
        &gpu->shutdown_ptr       // shutdown_flag
    };

    CUresult err = cuLaunchKernel(gpu->kernel,
                                   grid_x, 1, 1,   // grid dim
                                   block_x, 1, 1,  // block dim
                                   0,              // shared memory bytes
                                   gpu->stream,    // stream
                                   params,         // kernel params
                                   nullptr);       // extra

    if (err != CUDA_SUCCESS) {
        const char* msg = nullptr;
        cuGetErrorString(err, &msg);
        fprintf(stderr, "[cuda] cuLaunchKernel failed: %s (%d)\n", msg, err);
        return -1;
    }

    return 0;
}

int cuda_driver_push_work(CudaGpu* gpu,
                           const void* work_item,
                           size_t item_size) {
    // Read current tail from device
    uint64_t tail = 0;
    CUresult err = cuMemcpyDtoH(&tail,
                                 gpu->work_queue_ptr + 8,
                                 sizeof(uint64_t));
    if (err != CUDA_SUCCESS) return -1;

    // Check if queue is full
    uint64_t head = 0;
    err = cuMemcpyDtoH(&head, gpu->work_queue_ptr, sizeof(uint64_t));
    if (err != CUDA_SUCCESS) return -1;

    if ((tail - head) >= WORK_QUEUE_CAPACITY) {
        // Queue full, skip
        return 0;
    }

    // Copy work item to queue slot
    size_t item_offset = sizeof(WorkQueue) + (tail % WORK_QUEUE_CAPACITY) * item_size;
    err = cuMemcpyHtoDAsync(gpu->work_queue_ptr + item_offset,
                             work_item,
                             item_size,
                             gpu->stream);
    if (err != CUDA_SUCCESS) return -1;

    // Increment tail
    uint64_t new_tail = tail + 1;
    err = cuMemcpyHtoDAsync(gpu->work_queue_ptr + 8,
                             &new_tail,
                             sizeof(uint64_t),
                             gpu->stream);
    if (err != CUDA_SUCCESS) return -1;

    // Set active flag (offset 24 in WorkQueue: head=0, tail=8, capacity=16, active=24)
    uint32_t active = 1;
    err = cuMemcpyHtoDAsync(gpu->work_queue_ptr + 24,
                             &active,
                             sizeof(uint32_t),
                             gpu->stream);

    return err == CUDA_SUCCESS ? 0 : -1;
}

int cuda_driver_sync(CudaGpu* gpu) {
    CUresult err = cuStreamSynchronize(gpu->stream);
    return err == CUDA_SUCCESS ? 0 : -1;
}

int cuda_driver_copy_to_host(CudaGpu* gpu,
                               void* dst,
                               CUdeviceptr src,
                               size_t bytes) {
    CUresult err = cuMemcpyDtoHAsync(dst, src, bytes, gpu->stream);
    return err == CUDA_SUCCESS ? 0 : -1;
}

int cuda_driver_copy_to_device(CudaGpu* gpu,
                                 CUdeviceptr dst,
                                 const void* src,
                                 size_t bytes) {
    CUresult err = cuMemcpyHtoDAsync(dst, src, bytes, gpu->stream);
    return err == CUDA_SUCCESS ? 0 : -1;
}

int cuda_driver_signal_shutdown(CudaGpu* gpu) {
    uint32_t flag = 1;
    gpu->pinned_shutdown = 1;
    CUresult err = cuMemcpyHtoD(gpu->shutdown_ptr, &flag, sizeof(uint32_t));
    return err == CUDA_SUCCESS ? 0 : -1;
}

int cuda_driver_load_module(CudaGpu* gpu, const char* cubin_path) {
    // Check if file exists
    std::ifstream file(cubin_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        fprintf(stderr, "[cuda] Module file not found: %s\n", cubin_path);
        return -1;
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Load file into host memory
    std::vector<char> buffer(file_size);
    if (!file.read(buffer.data(), file_size)) {
        fprintf(stderr, "[cuda] Failed to read module file: %s\n", cubin_path);
        return -1;
    }
    file.close();

    // Load module into CUDA context
    CUresult err = cuModuleLoadDataEx(&gpu->module, buffer.data(), 0, nullptr, nullptr);
    if (err != CUDA_SUCCESS) {
        const char* msg = nullptr;
        cuGetErrorString(err, &msg);
        fprintf(stderr, "[cuda] cuModuleLoadDataEx failed: %s (%d)\n", msg, err);
        return -1;
    }

    fprintf(stderr, "[cuda] Loaded module: %s (%zu bytes)\n", cubin_path, file_size);
    return 0;
}

} // namespace propminer