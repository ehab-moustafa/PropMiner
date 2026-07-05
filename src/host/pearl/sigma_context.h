#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "cuda_compat.h"
#include "pearl_capi_wrapper.h"
#include "pearl_mining_wrapper.h"
#include "pearl_types.h"

namespace pearl {

// Per-σ resident device buffers shared by both ping/pong halves.
// These are allocated once and reused across nonces.
class ResidentBState {
public:
    ResidentBState() = default;
    ~ResidentBState();
    ResidentBState(const ResidentBState&) = delete;
    ResidentBState& operator=(const ResidentBState&) = delete;

    // Allocate device buffers for the configured dimensions.
    void allocate(const MiningConfig& cfg, CUstream stream);
    void free(CUstream stream);

    bool allocated() const { return b_ != 0; }

    // Device pointers (CUdeviceptr stored as uintptr_t to avoid CUDA type ambiguity).
    CUdeviceptr key() const { return key_; }
    CUdeviceptr b() const { return b_; }
    CUdeviceptr b_hash() const { return b_hash_; }
    CUdeviceptr roots() const { return roots_; }
    CUdeviceptr ebr() const { return ebr_; }
    CUdeviceptr ebr_fp16() const { return ebr_fp16_; }
    CUdeviceptr ebl_r() const { return ebl_r_; }
    CUdeviceptr ebl_k() const { return ebl_k_; }
    CUdeviceptr earx_bpeb() const { return earx_bpeb_; }
    CUdeviceptr bpeb() const { return bpeb_; }
    CUdeviceptr b_scales() const { return b_scales_; }
    CUdeviceptr leaf_cvs() const { return leaf_cvs_; }

    size_t b_bytes() const { return b_bytes_; }
    size_t leaf_cv_bytes() const { return leaf_cv_bytes_; }

private:
    CUdeviceptr key_ = 0;
    CUdeviceptr b_ = 0;
    CUdeviceptr b_hash_ = 0;
    CUdeviceptr roots_ = 0;
    CUdeviceptr ebr_ = 0;
    CUdeviceptr ebr_fp16_ = 0;
    CUdeviceptr ebl_r_ = 0;
    CUdeviceptr ebl_k_ = 0;
    CUdeviceptr earx_bpeb_ = 0;
    CUdeviceptr bpeb_ = 0;
    CUdeviceptr b_scales_ = 0;
    CUdeviceptr leaf_cvs_ = 0;

    size_t b_bytes_ = 0;
    size_t leaf_cv_bytes_ = 0;
    bool b_uploaded_ = false;
};

// Per-σ context: holds the job, resident B state, B Merkle tree, and target.
class SigmaContext {
public:
    SigmaContext() = default;
    SigmaContext(const Job& job, const MiningConfig& cfg);

    const Job& job() const { return job_; }
    const MiningConfig& config() const { return cfg_; }

    // Install resident B on device. Must be called with CUDA context current.
    // Safe to call multiple times; the second call is a no-op.
    // `workspace` is the per-σ pearl_capi_workspace handle (for noise_B scratch).
    // `device_id` is the CUDA device ordinal used for property lookups.
    void install(CUstream stream, void* workspace, int device_id);

    bool installed() const { return installed_; }

    ResidentBState& resident() { return resident_; }
    const ResidentBState& resident() const { return resident_; }

    MerkleTree* b_merkle_tree() { return b_tree_.get(); }
    const MerkleTree* b_merkle_tree() const { return b_tree_.get(); }

    // Test helper: inject a pre-built host-side B tree.
    void set_b_merkle_tree(MerkleTree&& tree);

    uint64_t sigma_seed() const { return sigma_seed_; }

    void set_target_nbits(uint32_t nbits) { job_.target_nbits = nbits; }

private:
    Job job_;
    MiningConfig cfg_;
    ResidentBState resident_;
    std::unique_ptr<MerkleTree> b_tree_;
    uint64_t sigma_seed_ = 0;
    bool installed_ = false;

    // Guards install() so two GpuWorker halves can share one SigmaContext safely.
    std::mutex install_mtx_;

    GemmCapi gemm_;
    MiningCapi mining_;
};

} // namespace pearl
