#pragma once

// Compatibility shim for CUDA driver/runtime types used in PropMiner.
// On a normal Linux CUDA build, <cuda.h> provides CUdeviceptr/CUstream/CUresult.
// On systems where only the runtime API is available we fall back to aliases.
// On non-CUDA systems (e.g. macOS host testing) we provide stub types.

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#if __has_include(<cuda.h>)
#include <cuda.h>
#endif
#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#endif

#if __has_include(<cuda.h>) && __has_include(<cuda_runtime.h>)
// Full CUDA toolkit: both driver and runtime headers available.
#elif __has_include(<cuda_runtime.h>)
// Runtime-only toolkit.
using CUdeviceptr = uintptr_t;
using CUstream = cudaStream_t;
using CUdevice = int;
using CUcontext = void*;
using CUresult = cudaError_t;
constexpr CUresult CUDA_SUCCESS = cudaSuccess;
constexpr CUresult CUDA_ERROR_UNKNOWN = cudaErrorUnknown;
#else
// Stub mode: no CUDA toolkit available.
using CUdeviceptr = uintptr_t;
using CUstream = void*;
using CUdevice = int;
using CUcontext = void*;
using CUresult = int;
constexpr CUresult CUDA_SUCCESS = 0;
constexpr CUresult CUDA_ERROR_UNKNOWN = 999;
struct cudaDeviceProp {
    char name[256];
    int major;
    int minor;
    size_t totalGlobalMem;
    int multiProcessorCount;
    int maxThreadsPerMultiProcessor;
    int maxThreadsPerBlock;
    int warpSize;
    int regsPerBlock;
    int sharedMemPerBlock;
    int sharedMemPerBlockOptin;
    size_t memPitch;
    int maxRegsPerBlock;
    int clockRate;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
};
struct cudaUUID_t { char bytes[16]; };
using cudaStream_t = void*;
using cudaEvent_t = void*;
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorNotReady = 34;
constexpr unsigned int cudaStreamNonBlocking = 1;
constexpr unsigned int cudaEventDisableTiming = 2;
constexpr unsigned int cudaHostAllocPortable = 1;
inline cudaError_t cudaFreeHost(void* p) { (void)p; return cudaSuccess; }
inline cudaError_t cudaHostAlloc(void** p, size_t n, unsigned int) {
    *p = std::malloc(n);
    return *p ? cudaSuccess : 2; // cudaErrorMemoryAllocation
}
inline const char* cudaGetErrorString(cudaError_t) { return "unknown"; }
inline int cudaGetDeviceCount(int* n) { *n = 0; return 0; }
inline int cudaSetDevice(int) { return 0; }
inline int cudaMemGetInfo(size_t* free, size_t* total) {
    if (free) *free = 0;
    if (total) *total = 0;
    return 0;
}
inline int cudaDeviceGetName(char* name, int len, int) {
    const char* s = "StubDevice";
    int i = 0;
    while (i < len - 1 && s[i]) { name[i] = s[i]; ++i; }
    if (len > 0) name[i] = '\0';
    return 0;
}
inline int cudaDeviceGetUuid(cudaUUID_t*, int) { return 0; }
inline int cudaGetDeviceProperties(cudaDeviceProp* prop, int) {
    std::memset(prop, 0, sizeof(*prop));
    prop->major = 0; prop->minor = 0;
    return 0;
}
inline int cudaStreamCreateWithFlags(cudaStream_t* s, unsigned int) {
    *s = nullptr;
    return 0;
}
inline int cudaStreamDestroy(cudaStream_t s) { (void)s; return 0; }
inline int cudaStreamSynchronize(cudaStream_t s) { (void)s; return 0; }
inline int cudaEventCreateWithFlags(cudaEvent_t* e, unsigned int) {
    *e = nullptr;
    return 0;
}
inline int cudaEventDestroy(cudaEvent_t e) { (void)e; return 0; }
inline int cudaEventRecord(cudaEvent_t e, cudaStream_t s) { (void)e; (void)s; return 0; }
inline int cudaEventQuery(cudaEvent_t e) { (void)e; return 0; }
inline int cudaEventSynchronize(cudaEvent_t e) { (void)e; return 0; }
constexpr int CU_CTX_SCHED_AUTO = 0;
constexpr unsigned int CU_STREAM_NON_BLOCKING = 1;
#endif

#if !__has_include(<cuda.h>)
// Provide minimal driver-like helpers when only the runtime API (or stubs) is present.
inline CUresult cuInit(unsigned int) { return CUDA_SUCCESS; }
inline CUresult cuDeviceGet(CUdevice*, int) { return CUDA_SUCCESS; }
inline CUresult cuCtxCreate(CUcontext*, unsigned int, CUdevice) { return CUDA_SUCCESS; }
inline CUresult cuCtxDestroy(CUcontext) { return CUDA_SUCCESS; }
inline CUresult cuStreamCreate(CUstream*, unsigned int) { return CUDA_SUCCESS; }
inline CUresult cuStreamDestroy(CUstream) { return CUDA_SUCCESS; }
inline CUresult cuMemAlloc(CUdeviceptr* p, size_t n) {
#if __has_include(<cuda_runtime.h>)
    void* ptr = nullptr;
    cudaError_t e = cudaMalloc(&ptr, n);
    *p = reinterpret_cast<CUdeviceptr>(ptr);
    return e;
#else
    *p = 0;
    (void)n;
    return CUDA_ERROR_UNKNOWN;
#endif
}
inline CUresult cuMemFree(CUdeviceptr p) {
#if __has_include(<cuda_runtime.h>)
    return cudaFree(reinterpret_cast<void*>(p));
#else
    (void)p;
    return CUDA_ERROR_UNKNOWN;
#endif
}
inline CUresult cuMemcpyHtoDAsync(CUdeviceptr d, const void* s, size_t n, CUstream stream) {
#if __has_include(<cuda_runtime.h>)
    return cudaMemcpyAsync(reinterpret_cast<void*>(d), s, n, cudaMemcpyHostToDevice, stream);
#else
    (void)d; (void)s; (void)n; (void)stream;
    return CUDA_ERROR_UNKNOWN;
#endif
}
inline CUresult cuMemcpyDtoHAsync(void* d, CUdeviceptr s, size_t n, CUstream stream) {
#if __has_include(<cuda_runtime.h>)
    return cudaMemcpyAsync(d, reinterpret_cast<void*>(s), n, cudaMemcpyDeviceToHost, stream);
#else
    (void)d; (void)s; (void)n; (void)stream;
    return CUDA_ERROR_UNKNOWN;
#endif
}
inline CUresult cuMemsetD8Async(CUdeviceptr d, unsigned char v, size_t n, CUstream stream) {
#if __has_include(<cuda_runtime.h>)
    return cudaMemsetAsync(reinterpret_cast<void*>(d), v, n, stream);
#else
    (void)d; (void)v; (void)n; (void)stream;
    return CUDA_ERROR_UNKNOWN;
#endif
}
inline CUresult cuStreamSynchronize(CUstream stream) {
#if __has_include(<cuda_runtime.h>)
    return cudaStreamSynchronize(stream);
#else
    (void)stream;
    return CUDA_SUCCESS;
#endif
}
inline CUresult cuDeviceGetCount(int* n) {
#if __has_include(<cuda_runtime.h>)
    return cudaGetDeviceCount(n);
#else
    *n = 0;
    return CUDA_SUCCESS;
#endif
}
inline CUresult cuCtxGetDevice(CUdevice* d) {
#if __has_include(<cuda_runtime.h>)
    return cudaGetDevice(d);
#else
    *d = 0;
    return CUDA_SUCCESS;
#endif
}
enum CUdevice_attribute_stub { CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 0,
                               CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 1 };
inline CUresult cuDeviceGetAttribute(int* v, CUdevice_attribute_stub, CUdevice) {
    *v = 0;
    return CUDA_SUCCESS;
}
inline CUresult cuGetErrorString(CUresult, const char** s) {
    *s = "unknown";
    return CUDA_SUCCESS;
}
struct CUuuid_st { char bytes[16]; };
using CUuuid = CUuuid_st;
inline CUresult cuDeviceGetName(char* name, int len, CUdevice) {
    const char* s = "StubDevice";
    int i = 0;
    while (i < len - 1 && s[i]) { name[i] = s[i]; ++i; }
    if (len > 0) name[i] = '\0';
    return CUDA_SUCCESS;
}
inline CUresult cuDeviceGetUuid(CUuuid_st* uuid, CUdevice) {
    std::memset(uuid, 0, sizeof(*uuid));
    return CUDA_SUCCESS;
}
#endif
