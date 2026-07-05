#include "sigma_context.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "cuda_compat.h"
#include "job_key.h"
#include "pearl_blake3.h"

namespace pearl {

namespace {
    constexpr size_t K32 = 32;
    constexpr size_t K1024 = 1024;

    void check_cuda(CUresult r, const char* msg) {
        if (r == CUDA_SUCCESS) return;
        const char* s = nullptr;
        cuGetErrorString(r, &s);
        fprintf(stderr, "[sigma] %s: %s (%d)\n", msg, s ? s : "unknown", static_cast<int>(r));
        throw std::runtime_error(std::string(msg) + ": " + (s ? s : "unknown"));
    }
}

// The C API requires pearl_capi_free_buffer with a length; we don't use that
// in RAII paths. Resident buffers use cuMemFree.

ResidentBState::~ResidentBState() {
    // Cannot free without stream; leak warning. GpuWorker frees explicitly.
}

void ResidentBState::allocate(const MiningConfig& cfg, CUstream stream) {
    free(stream);

    b_bytes_ = static_cast<size_t>(cfg.n) * cfg.k;
    size_t b_hash_bytes = K32;
    size_t roots_bytes = K32 * 8; // scratch
    size_t ebr_bytes = static_cast<size_t>(cfg.n) * cfg.r;
    size_t ebr_fp16_bytes = ebr_bytes * 2;
    size_t ebl_r_bytes = static_cast<size_t>(cfg.k) * cfg.r;
    size_t ebl_k_bytes = static_cast<size_t>(cfg.r) * cfg.k;
    size_t earx_bpeb_bytes = static_cast<size_t>(cfg.n) * cfg.r * 2;
    size_t bpeb_bytes = static_cast<size_t>(cfg.n) * cfg.k;
    size_t b_scales_bytes = static_cast<size_t>(cfg.n) * sizeof(float);
    size_t total_leaves = (b_bytes_ + K1024 - 1) / K1024;
    leaf_cv_bytes_ = total_leaves * K32;

    check_cuda(cuMemAlloc(&key_, K32), "b key alloc");
    check_cuda(cuMemAlloc(&b_, b_bytes_), "b alloc");
    check_cuda(cuMemAlloc(&b_hash_, b_hash_bytes), "b_hash alloc");
    check_cuda(cuMemAlloc(&roots_, roots_bytes), "roots alloc");
    check_cuda(cuMemAlloc(&ebr_, ebr_bytes), "ebr alloc");
    check_cuda(cuMemAlloc(&ebr_fp16_, ebr_fp16_bytes), "ebr_fp16 alloc");
    check_cuda(cuMemAlloc(&ebl_r_, ebl_r_bytes), "ebl_r alloc");
    check_cuda(cuMemAlloc(&ebl_k_, ebl_k_bytes), "ebl_k alloc");
    check_cuda(cuMemAlloc(&earx_bpeb_, earx_bpeb_bytes), "earx_bpeb alloc");
    check_cuda(cuMemAlloc(&bpeb_, bpeb_bytes), "bpeb alloc");
    check_cuda(cuMemAlloc(&b_scales_, b_scales_bytes), "b_scales alloc");
    check_cuda(cuMemAlloc(&leaf_cvs_, leaf_cv_bytes_), "leaf_cvs alloc");

    check_cuda(cuMemsetD8Async(key_, 0, K32, stream), "b key zero");
}

void ResidentBState::free(CUstream stream) {
    (void)stream;
    auto f = [](CUdeviceptr& p) { if (p) { cuMemFree(p); p = 0; } };
    f(key_);
    f(b_);
    f(b_hash_);
    f(roots_);
    f(ebr_);
    f(ebr_fp16_);
    f(ebl_r_);
    f(ebl_k_);
    f(earx_bpeb_);
    f(bpeb_);
    f(b_scales_);
    f(leaf_cvs_);
}

SigmaContext::SigmaContext(const Job& job, const MiningConfig& cfg)
    : job_(job), cfg_(cfg), sigma_seed_(0) {
    if (job_.sigma.size() != K32) {
        throw std::invalid_argument("sigma must be 32 bytes");
    }
    if (job_.b_seed.size() != K32) {
        throw std::invalid_argument("b_seed must be 32 bytes");
    }
    if (job_.job_key.size() != K32) {
        throw std::invalid_argument("job_key must be 32 bytes");
    }
    // Verify the job_key matches the canonical V2 derivation. If the caller
    // passed a pre-derived key it must be exact; otherwise we re-derive it.
    auto expected = derive_job_key(job_.sigma, cfg_);
    if (job_.job_key != expected) {
        // Overwrite with the canonical key so proofs match the pool's verifier.
        job_.job_key = expected;
    }
    // sigma_seed = first 8 bytes of sigma (little-endian), matching C# usage.
    sigma_seed_ = *reinterpret_cast<const uint64_t*>(job_.sigma.data());
}

void SigmaContext::install(CUstream stream, void* workspace, int device_id) {
    std::lock_guard<std::mutex> lk(install_mtx_);
    if (installed_) return;

    // Ensure the implicit primary context for the target device is current on
    // this thread before making driver API allocations. On WSL2, cudaSetDevice
    // alone does not always make the context current.
    if (device_id >= 0) {
        cudaError_t e = cudaSetDevice(device_id);
        if (e == cudaSuccess) {
            (void)cudaFree(0);
        }
    }

    resident_.allocate(cfg_, stream);

    // Upload job key.
    check_cuda(cuMemcpyHtoDAsync(resident_.key(), job_.job_key.data(), K32, stream),
               "key h2d");

    // Expand BSeed on device and tensor-hash it, producing BHash + leaf CVs.
    try {
        gemm_.bseed_expand_and_tensor_hash_leaf_cvs(
            job_.b_seed.data(),
            reinterpret_cast<void*>(resident_.b()),
            static_cast<uint32_t>(resident_.b_bytes()),
            reinterpret_cast<uint8_t*>(resident_.b_hash()),
            reinterpret_cast<const uint8_t*>(resident_.key()),
            cfg_.tensor_hash_num_blocks(resident_.b_bytes()),
            cfg_.tensor_hash_threads,
            cfg_.tensor_hash_stages,
            cfg_.tensor_hash_leaves,
            reinterpret_cast<uint8_t*>(resident_.roots()),
            reinterpret_cast<uint8_t*>(resident_.leaf_cvs()),
            device_id,
            stream);
    } catch (const std::exception& e) {
        cudaError_t last = cudaGetLastError();
        fprintf(stderr, "[sigma] bseed_expand failed: %s\n", e.what());
        fprintf(stderr, "[sigma] last CUDA error: %s (%d)\n",
                cudaGetErrorString(last), static_cast<int>(last));
        throw;
    }

    // Asynchronously copy leaf CVs to host on a side stream; the Merkle tree
    // build then happens while noise_B runs on the main stream.
    std::vector<uint8_t> leaf_cvs_host(resident_.leaf_cv_bytes());
    CUstream copy_stream = nullptr;
    cudaStream_t copy_stream_rt = nullptr;
    cudaError_t ce = cudaStreamCreateWithFlags(&copy_stream_rt, cudaStreamNonBlocking);
    if (ce != cudaSuccess) throw std::runtime_error("cudaStreamCreateWithFlags failed");
    copy_stream = reinterpret_cast<CUstream>(copy_stream_rt);
    check_cuda(cuMemcpyDtoHAsync(leaf_cvs_host.data(), resident_.leaf_cvs(),
                                 resident_.leaf_cv_bytes(), copy_stream),
               "leaf_cvs d2h async");

    // Build noise_B side: noise_gen(B-side only) + noise_B.
    // Use the device-resident key copy (already uploaded above); like
    // tensor_hash, the noise-generation kernel reads key data from GPU memory.
    const uint8_t* dev_key = reinterpret_cast<const uint8_t*>(resident_.key());
    int rc = pearl_capi_noise_gen(
        cfg_.r, 0, cfg_.n, cfg_.k,
        nullptr, nullptr,
        nullptr, nullptr,
        reinterpret_cast<void*>(resident_.ebl_r()),
        reinterpret_cast<void*>(resident_.ebl_k()),
        reinterpret_cast<void*>(resident_.ebr()),
        reinterpret_cast<void*>(resident_.ebr_fp16()),
        dev_key,
        dev_key,
        stream);
    if (rc != 0) {
        cudaError_t last = cudaGetLastError();
        fprintf(stderr,
                "[sigma] pearl_capi_noise_gen (B-side) failed rc=%d; last CUDA error: %s (%d)\n",
                rc, cudaGetErrorString(last), static_cast<int>(last));
        throw std::runtime_error("pearl_capi_noise_gen (B-side) failed");
    }

    PearlCapiNoiseBParams nb{};
    nb.n = cfg_.n;
    nb.k = cfg_.k;
    nb.r = cfg_.r;
    nb.B = reinterpret_cast<void*>(resident_.b());
    nb.EAR_K_major = nullptr;
    nb.EBL_R_major = reinterpret_cast<void*>(resident_.ebl_r());
    nb.EBR = reinterpret_cast<void*>(resident_.ebr());
    nb.EARxBpEB = reinterpret_cast<void*>(resident_.earx_bpeb());
    nb.BpEB = reinterpret_cast<void*>(resident_.bpeb());
    nb.workspace = workspace;
    rc = pearl_capi_noise_B(&nb, stream);
    if (rc != 0) {
        cudaError_t last = cudaGetLastError();
        fprintf(stderr,
                "[sigma] pearl_capi_noise_B failed rc=%d; last CUDA error: %s (%d)\n",
                rc, cudaGetErrorString(last), static_cast<int>(last));
        throw std::runtime_error("pearl_capi_noise_B failed");
    }

    // Wait for both the copy and the device work to finish, then build the tree.
    check_cuda(cuStreamSynchronize(stream), "sync install stream");
    cudaStreamSynchronize(copy_stream_rt);
    cudaStreamDestroy(copy_stream_rt);

    b_tree_ = std::make_unique<MerkleTree>(
        mining_.build_bseed_tree_from_leaf_cvs(
            leaf_cvs_host,
            job_.b_seed.data(),
            job_.job_key.data(),
            cfg_.n,
            cfg_.k));

    installed_ = true;
}

void SigmaContext::set_b_merkle_tree(MerkleTree&& tree) {
    b_tree_ = std::make_unique<MerkleTree>(std::move(tree));
}

} // namespace pearl
