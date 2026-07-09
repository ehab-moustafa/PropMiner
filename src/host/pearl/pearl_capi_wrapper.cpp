#include "pearl_capi_wrapper.h"

#include <cuda_runtime_api.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace pearl {

namespace {
// Verbose pearl-gemm C API checker. On failure prints the failing function,
// its return code, and the last CUDA error, then throws. This makes remote
// WSL2/Salad debugging self-service: the logs identify the exact call that died.
inline void pearl_check(const char* fn, int rc) {
    if (rc >= 0) return;
    cudaError_t last = cudaGetLastError();
    fprintf(stderr,
            "[pearl-capi] %s failed with rc=%d; last CUDA error: %s (%d)\n",
            fn, rc, cudaGetErrorString(last), static_cast<int>(last));
    if (rc == -62 || rc == -63) {
        fprintf(stderr,
                "[pearl-capi] hint: CUDA graph validation replay failed — "
                "mining will fall back to iter_batch if gpu_worker catches this.\n");
    } else if (rc >= -50 && rc <= -44) {
        fprintf(stderr,
                "[pearl-capi] hint: graph capture failed at rc=%d — see "
                "[pearl-gemm] lines above; try PEARL_GEMM_DEBUG=1.\n", rc);
    }
    throw std::runtime_error(std::string(fn) + " failed (rc=" +
                             std::to_string(rc) + ")");
}
} // namespace

Workspace::Workspace(void* ws) : ws_(ws) {
    if (!ws_) throw std::invalid_argument("null workspace");
}
Workspace::~Workspace() {
    if (ws_) pearl_capi_workspace_free(ws_, nullptr);
}
Workspace::Workspace(Workspace&& other) noexcept : ws_(other.ws_) {
    other.ws_ = nullptr;
}
Workspace& Workspace::operator=(Workspace&& other) noexcept {
    if (this != &other) {
        if (ws_) pearl_capi_workspace_free(ws_, nullptr);
        ws_ = other.ws_;
        other.ws_ = nullptr;
    }
    return *this;
}

GemmCapi::GemmCapi() {
    constexpr int kExpectedAbiVersion = 2;
    if (pearl_capi_abi_version() != kExpectedAbiVersion) {
        throw std::runtime_error(
            "libpearl_gemm_capi ABI version mismatch: expected " +
            std::to_string(kExpectedAbiVersion));
    }
}

int GemmCapi::abi_version() const { return pearl_capi_abi_version(); }
const char* GemmCapi::build_profile() const { return pearl_capi_build_profile(); }
const char* GemmCapi::build_knobs() const { return pearl_capi_build_knobs(); }
const char* GemmCapi::active_kernel_name() const {
    return pearl_capi_active_kernel_name();
}
int GemmCapi::validate_kernel_selection() const {
    return pearl_capi_validate_kernel_selection();
}
bool GemmCapi::supports_sm(int major, int minor) const {
    return pearl_capi_supports_sm(major, minor);
}

int GemmCapi::device_count() {
    int n = 0;
    cudaError_t r = cudaGetDeviceCount(&n);
    if (r != cudaSuccess) {
        fprintf(stderr, "[cuda] cudaGetDeviceCount failed: %d (%s)\n", static_cast<int>(r), cudaGetErrorString(r));
        n = 0;
    }
    return n;
}

int GemmCapi::current_sm_major() {
    int dev = 0;
    cudaError_t r = cudaGetDevice(&dev);
    if (r != cudaSuccess) return 0;
    cudaDeviceProp prop{};
    r = cudaGetDeviceProperties(&prop, dev);
    if (r != cudaSuccess) return 0;
    return prop.major;
}

int GemmCapi::current_sm_minor() {
    int dev = 0;
    cudaError_t r = cudaGetDevice(&dev);
    if (r != cudaSuccess) return 0;
    cudaDeviceProp prop{};
    r = cudaGetDeviceProperties(&prop, dev);
    if (r != cudaSuccess) return 0;
    return prop.minor;
}

Workspace GemmCapi::alloc_workspace(int32_t m, int32_t n, int32_t k, int32_t r,
                                    CUstream stream) {
    void* ws = nullptr;
    // noise_A scratch is used every iter inside noisy_gemm; noise_B scratch is
    // used once per σ inside install_B.  Keep both around so the hot path never
    // pays for cudaMallocAsync and σ-refresh can reuse the same handle.
    int rc = pearl_capi_workspace_alloc(m, n, k, r, 1, 1, &ws, stream);
    if (rc != 0 || !ws) {
        pearl_check("pearl_capi_workspace_alloc", rc);
        throw std::runtime_error("pearl_capi_workspace_alloc returned null");
    }
    return Workspace(ws);
}

void GemmCapi::install_params(void* ws, const PearlCapiWorkspaceParams& p) const {
    pearl_check("pearl_capi_workspace_install_params",
                pearl_capi_workspace_install_params(ws, &p));
}

int GemmCapi::iter_batch(void* ws,
                         CUstream stream,
                         uint64_t seed_lo_start,
                         void* const* host_signal_headers,
                         int32_t count) const {
    int triggers = pearl_capi_iter_batch(ws, seed_lo_start,
                                         host_signal_headers, count, stream);
    pearl_check("pearl_capi_iter_batch", triggers);
    return triggers;
}

void GemmCapi::iter_batch_graph_prepare(void* ws,
                                        CUstream stream,
                                        void* const* host_signal_headers,
                                        int32_t count) const {
    pearl_check("pearl_capi_iter_batch_graph_prepare",
                pearl_capi_iter_batch_graph_prepare(ws, host_signal_headers, count, stream));
}

void GemmCapi::iter_batch_graph_launch(void* ws,
                                       CUstream stream,
                                       uint64_t seed_lo_start) const {
    pearl_check("pearl_capi_iter_batch_graph_launch",
                pearl_capi_iter_batch_graph_launch(ws, seed_lo_start, stream));
}

void GemmCapi::iter_batch_graph_prepare_ex(void* ws,
                                           CUstream stream,
                                           void* const* host_signal_headers,
                                           int32_t count,
                                           void* seed_lo_dev) const {
    pearl_check("pearl_capi_iter_batch_graph_prepare_ex",
                pearl_capi_iter_batch_graph_prepare_ex(ws, host_signal_headers, count,
                                                       seed_lo_dev, stream));
}

void GemmCapi::iter_batch_graph_launch_ex(void* ws, CUstream stream) const {
    pearl_check("pearl_capi_iter_batch_graph_launch_ex",
                pearl_capi_iter_batch_graph_launch_ex(ws, stream));
}

void GemmCapi::bseed_expand_and_tensor_hash_leaf_cvs(const uint8_t* bseed,
                                                     void* data, uint32_t data_size,
                                                     uint8_t* out, const uint8_t* key,
                                                     uint32_t num_blocks,
                                                     uint32_t threads,
                                                     uint32_t stages,
                                                     uint32_t leaves,
                                                     uint8_t* roots,
                                                     uint8_t* leaf_cvs,
                                                     int device_id,
                                                     CUstream stream) const {
    pearl_check("pearl_capi_bseed_expand_and_tensor_hash_leaf_cvs",
                pearl_capi_bseed_expand_and_tensor_hash_leaf_cvs(
                    bseed, reinterpret_cast<uint8_t*>(data), data_size, out, key,
                    num_blocks, threads, stages, leaves, roots, leaf_cvs,
                    device_id, stream));
}

void GemmCapi::commitment_hash_from_merkle_roots(const uint8_t* A_merkle_root,
                                                 const uint8_t* B_merkle_root,
                                                 const uint8_t* key,
                                                 uint8_t* A_commitment_hash,
                                                 uint8_t* B_commitment_hash,
                                                 int device_id,
                                                 CUstream stream) const {
    pearl_check("pearl_capi_commitment_hash_from_merkle_roots",
                pearl_capi_commitment_hash_from_merkle_roots(
                    A_merkle_root, B_merkle_root, key,
                    A_commitment_hash, B_commitment_hash,
                    device_id, stream));
}

void GemmCapi::lcg_int7_fill(void* dst, int64_t n,
                             uint64_t seed_lo, uint64_t seed_hi,
                             CUstream stream) const {
    pearl_check("pearl_capi_lcg_int7_fill",
                pearl_capi_lcg_int7_fill(dst, n, seed_lo, seed_hi, stream));
}

void GemmCapi::bseed_expand_raw_device(const uint8_t* bseed, void* dst,
                                       int64_t n, CUstream stream) const {
    pearl_check("pearl_capi_bseed_expand_raw_device",
                pearl_capi_bseed_expand_raw_device(bseed, dst, n, stream));
}

void GemmCapi::tensor_hash_leaf_cvs(const uint8_t* data, uint32_t data_size,
                                    uint8_t* out, const uint8_t* key,
                                    uint32_t num_blocks, uint32_t threads,
                                    uint32_t stages, uint32_t leaves,
                                    uint8_t* roots, uint8_t* leaf_cvs,
                                    int device_id, CUstream stream) const {
    pearl_check("pearl_capi_tensor_hash_leaf_cvs",
                pearl_capi_tensor_hash_leaf_cvs(
                    data, data_size, out, key,
                    num_blocks, threads, stages, leaves,
                    roots, leaf_cvs,
                    device_id, stream));
}

int GemmCapi::host_signal_header_size() const {
    return pearl_capi_get_host_signal_header_size();
}

int GemmCapi::host_signal_sync_size() const {
    return pearl_capi_get_host_signal_sync_size();
}

} // namespace pearl
