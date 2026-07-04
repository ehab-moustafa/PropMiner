#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "cuda_compat.h"
#include "pearl_capi_wrapper.h"
#include "pearl_types.h"
#include "seed_generator.h"
#include "sigma_context.h"

namespace pearl {

// Callback interface for found shares.
class IShareSink {
public:
    virtual ~IShareSink() = default;
    virtual void submit(const ShareFound& share) = 0;
};

// Per-GPU mining worker. Runs a ping/pong double-buffered iter loop.
class GpuWorker {
public:
    GpuWorker(int device_index, int gpu_index,
              const MiningConfig& cfg,
              IShareSink* sink);
    ~GpuWorker();

    GpuWorker(const GpuWorker&) = delete;
    GpuWorker& operator=(const GpuWorker&) = delete;

    // Start / stop the worker thread.
    void start();
    void stop();

    // Publish a new sigma context (thread-safe).
    void set_sigma(std::shared_ptr<SigmaContext> ctx);

    // Update target nbits mid-sigma (vardiff).
    void set_target_nbits(uint32_t nbits);

    // Set matmuls per poll (batch size). Tuned by benchmark.
    void set_matmuls_per_poll(int mpp);

    double hashrate() const;
    uint64_t total_iters() const { return total_iters_.load(); }

private:
    struct HalfBuffers {
        CUdeviceptr a = 0;
        CUdeviceptr a_hash = 0;
        CUdeviceptr roots = 0;
        CUdeviceptr commit_a = 0;
        CUdeviceptr commit_b = 0;
        CUdeviceptr eal = 0;
        CUdeviceptr eal_fp16 = 0;
        CUdeviceptr ear_r = 0;
        CUdeviceptr ear_k = 0;
        CUdeviceptr ax_ebl_fp16 = 0;
        CUdeviceptr apea = 0;
        CUdeviceptr a_scales = 0;
        CUdeviceptr c = 0;
        CUdeviceptr sync = 0;
        CUdeviceptr pow_target = 0;
        std::vector<void*> host_headers;
        std::vector<std::vector<uint8_t>> host_header_storage;
        size_t header_size = 0;
        CUstream stream = nullptr;
        void* workspace = nullptr;
        bool graph_ready = false;
        int graph_batch_count = 0;

        // A leaf CVs for fast share reconstruction.
        CUdeviceptr a_leaf_cvs = 0;
        size_t a_leaf_cv_bytes = 0;

        // Timing / non-blocking wait.
        cudaEvent_t batch_done_event = nullptr;

        // Caller-owned device-side seed pointer for the extended graph path.
        // Allows cudaMemcpyAsync seed upload on a separate copy stream while
        // the compute stream is busy with the previous batch.
        CUdeviceptr seed_dev = 0;
        void* seed_dev_ptr = nullptr;

        void allocate(const MiningConfig& cfg, int device_id, CUstream s);
        void free();
    };

    void run();
    void install_sigma(SigmaContext& ctx, HalfBuffers& half);
    void prepare_graph(HalfBuffers& half);
    void queue_batch(HalfBuffers& half, uint64_t seed_lo_start, int count);

    // Upload the seed for the next batch on the dedicated copy stream while
    // the current batch runs on the compute stream (PCIe Gen5 conveyor belt).
    void upload_next_seed_async(HalfBuffers& half, uint64_t seed_lo);

    int sync_and_scan(HalfBuffers& half, int batch);
    bool wait_for_batch(HalfBuffers& half, int timeout_ms);
    bool handle_trigger(HalfBuffers& half, const SigmaContext& ctx,
                        const std::vector<uint8_t>& header,
                        uint64_t nonce);

    void check_cuda(CUresult r, const char* msg);

    int device_index_;
    int gpu_index_;
    MiningConfig cfg_;
    IShareSink* sink_;

    int device_ = 0;
    CUstream merkle_copy_stream_ = nullptr;

    // Dedicated copy stream for the PCIe Gen5 conveyor belt.  Seeds are
    // uploaded here while the Tensor Cores run on ping_/pong_.stream.
    CUstream seed_copy_stream_ = nullptr;
    cudaEvent_t seed_copy_done_event_ = nullptr;

    HalfBuffers ping_;
    HalfBuffers pong_;

    // CPU-side seed generation ring.  Performs 0% hashing math.
    std::unique_ptr<SeedGenerator> seed_gen_;

    std::atomic<bool> running_{false};
    std::atomic<bool> stop_flag_{false};
    std::thread thread_;

    std::mutex sigma_mtx_;
    std::shared_ptr<SigmaContext> sigma_;
    std::atomic<uint32_t> target_nbits_{0};
    std::atomic<int> matmuls_per_poll_{8};

    std::atomic<uint64_t> total_iters_{0};
    std::atomic<double> hashrate_{0.0};
    std::atomic<double> last_iter_ms_{0.0};

    // Per-GPU nonce-space base.  The high 16 bits encode gpu_index so different
    // GPUs never collide; low 32 bits are incremented each batch.
    uint64_t seed_base_ = 0;

    GemmCapi gemm_;
    MiningCapi mining_;
};

} // namespace pearl
