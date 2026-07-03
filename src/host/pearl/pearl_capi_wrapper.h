#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "cuda_compat.h"
#include "pearl_gemm_capi.h"

namespace pearl {

// RAII wrapper around a pearl_capi_workspace_t (opaque void*).
class Workspace {
public:
    explicit Workspace(void* ws);
    ~Workspace();
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    Workspace(Workspace&& other) noexcept;
    Workspace& operator=(Workspace&& other) noexcept;

    void* get() const { return ws_; }

private:
    void* ws_;
};

// Thin wrapper for the CUDA GEMM C API.
class GemmCapi {
public:
    GemmCapi();

    int abi_version() const;
    const char* build_profile() const;
    bool supports_sm(int major, int minor) const;

    static int device_count();
    static int current_sm_major();
    static int current_sm_minor();

    // Workspace.
    Workspace alloc_workspace(int32_t m, int32_t n, int32_t k, int32_t r,
                              CUstream stream);

    // Install constant params for a sigma.
    void install_params(void* ws, const PearlCapiWorkspaceParams& p) const;

    // Per-nonce hot path (batch).  Returns number of triggers found.
    int iter_batch(void* ws,
                   CUstream stream,
                   uint64_t seed_lo_start,
                   void* const* host_signal_headers,
                   int32_t count) const;

    // CUDA graph variants of the hot path.
    void iter_batch_graph_prepare(void* ws,
                                  CUstream stream,
                                  void* const* host_signal_headers,
                                  int32_t count) const;
    void iter_batch_graph_launch(void* ws,
                                 CUstream stream,
                                 uint64_t seed_lo_start) const;

    // Extended graph path with caller-owned device-side seed pointer.
    // The caller uploads seed values via cudaMemcpyAsync on a separate copy
    // stream; the captured graph reads them directly from `seed_lo_dev`.
    void iter_batch_graph_prepare_ex(void* ws,
                                     CUstream stream,
                                     void* const* host_signal_headers,
                                     int32_t count,
                                     void* seed_lo_dev) const;
    void iter_batch_graph_launch_ex(void* ws, CUstream stream) const;

    // Helpers used during sigma-install.
    void bseed_expand_and_tensor_hash_leaf_cvs(const uint8_t* bseed,
                                               void* data, uint32_t data_size,
                                               uint8_t* out, const uint8_t* key,
                                               uint32_t num_blocks,
                                               uint32_t threads,
                                               uint32_t stages,
                                               uint32_t leaves,
                                               uint8_t* roots,
                                               uint8_t* leaf_cvs,
                                               int device_id,
                                               CUstream stream) const;

    void commitment_hash_from_merkle_roots(const uint8_t* A_merkle_root,
                                           const uint8_t* B_merkle_root,
                                           const uint8_t* key,
                                           uint8_t* A_commitment_hash,
                                           uint8_t* B_commitment_hash,
                                           int device_id,
                                           CUstream stream) const;

    // Direct int7 fill / BSeed expansion when needed.
    void lcg_int7_fill(void* dst, int64_t n,
                       uint64_t seed_lo, uint64_t seed_hi,
                       CUstream stream) const;

    void bseed_expand_raw_device(const uint8_t* bseed, void* dst,
                                 int64_t n, CUstream stream) const;

    // tensor_hash with leaf CV output.
    void tensor_hash_leaf_cvs(const uint8_t* data, uint32_t data_size,
                              uint8_t* out, const uint8_t* key,
                              uint32_t num_blocks, uint32_t threads,
                              uint32_t stages, uint32_t leaves,
                              uint8_t* roots, uint8_t* leaf_cvs,
                              int device_id, CUstream stream) const;

    int host_signal_header_size() const;
    int host_signal_sync_size() const;
};

} // namespace pearl
