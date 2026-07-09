#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "cuda_compat.h"
#include "pearl_capi_wrapper.h"
#include "pearl_types.h"
#include "rtx5090_profile.h"
#include "sigma_context.h"

namespace pearl {

class Watchdog;

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

    // Live pool share target (may differ from device pow_target mid-batch).
    uint32_t target_nbits() const { return target_nbits_.load(); }

    // Matmuls per poll (total batch size).
    void set_matmuls_per_poll(int mpp);
    // CUDA graph capture depth (sub-launches when batch > graph_batch).
    void set_graph_batch(int gb);

    int graph_batch() const { return graph_batch_.load(); }

    double hashrate() const;
    double tmads_per_sec() const { return tmads_per_sec_.load(); }
    double last_iter_ms() const { return last_iter_ms_.load(); }
    int matmuls_per_poll() const { return matmuls_per_poll_.load(); }
    uint64_t total_iters() const { return total_iters_.load(); }
    int device_index() const { return device_index_; }

    // Pipeline instrumentation (wait_until_half_free stalls).
    uint64_t half_wait_ms_total() const { return half_wait_ms_total_.load(); }
    uint32_t half_wait_count() const { return half_wait_count_.load(); }
    uint64_t half_wait_ms_max() const { return half_wait_ms_max_.load(); }
    bool triple_buffer_active() const { return triple_buffer_active_; }

    void set_watchdog(Watchdog* wd) { watchdog_ = wd; }

    // Orchestrator thermal pause (PeakMiner-style): worker spins idle without
    // tearing down CUDA context.
    void set_paused(bool paused) { pause_flag_.store(paused); }
    bool paused() const { return pause_flag_.load(); }

    // Orchestrator health monitor: abort in-flight batch wait so stall recovery
    // triggers before the full PROPMINER_STALL_RESTART_MS timeout.
    void request_batch_abort() { batch_abort_requested_.store(true); }
    void clear_batch_abort() { batch_abort_requested_.store(false); }

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

        // Timing / non-blocking wait (cudaEventElapsedTime needs default flags).
        cudaEvent_t batch_start_event = nullptr;
        cudaEvent_t batch_done_event = nullptr;
        cudaEvent_t sub_batch_done_event = nullptr;

        // Caller-owned device-side seed pointer for the extended graph path.
        // Allows cudaMemcpyAsync seed upload on a separate copy stream while
        // the compute stream is busy with the previous batch.
        CUdeviceptr seed_dev = 0;
        void* seed_dev_ptr = nullptr;
        uint64_t* pinned_seed_host = nullptr;  // per-half pinned seed (no shared race)
        uint64_t batch_seed_start = 0;  // seed_lo at graph launch for this batch
        // nbits encoded in half.pow_target (updated by upload_pow_target).
        uint32_t pow_target_nbits = 0;
        // nbits active when batch was queued (pinned for share verify).
        uint32_t batch_mined_target_nbits = 0;

        // Pinned PCIe staging for rare share-D2H paths (GPU isolation).
        uint8_t* pinned_leaf_cvs = nullptr;
        uint8_t* pinned_a_slice = nullptr;
        uint8_t* pinned_opened_leaves = nullptr;
        uint8_t* pinned_hash_a = nullptr;
        uint8_t* pinned_hash_b = nullptr;
        uint8_t* pinned_commit_a = nullptr;
        uint8_t* pinned_commit_b = nullptr;
        size_t pinned_a_slice_bytes = 0;
        size_t pinned_opened_leaves_bytes = 0;

        // Share-GPU deferral: non-zero while side thread holds this half for proof prep.
        std::atomic<int> share_jobs_pending{0};

        void allocate(const MiningConfig& cfg, int batch, int device_id, CUstream s);
        void free();
    };

    void run();
    void ensure_half_workspace(HalfBuffers& half);
    void install_sigma(SigmaContext& ctx, HalfBuffers& half);
    // Worker-thread-only tail of install_sigma: bind resident B + this half's A
    // buffers into the half workspace, upload the PoW target, and capture the
    // CUDA graph. Assumes ctx.install() has already populated ctx.resident().
    void bind_sigma_to_half(SigmaContext& ctx, HalfBuffers& half);
    void prepare_graph(HalfBuffers& half);
    void capture_graphs_for_halves();
    void queue_batch(HalfBuffers& half, uint64_t seed_lo_start, int count);

    // Upload the seed for the next batch on the dedicated copy stream while
    // the current batch runs on the compute stream (PCIe Gen5 conveyor belt).
    void upload_next_seed_async(HalfBuffers& half, uint64_t seed_lo);

    // Upload seed_lo to half.seed_dev_ptr on the compute stream (must precede
    // iter_batch_graph_launch_ex). The captured-graph H2D path freezes the seed
    // at capture time and breaks multi sub-batch launches (gpu_cpu_jackpot_mismatch).
    void upload_seed_for_graph(HalfBuffers& half, uint64_t seed_lo);

    int sync_and_scan(HalfBuffers& half, int batch);
    bool wait_for_batch(HalfBuffers& half, int timeout_ms);
    bool wait_half_stream(HalfBuffers& half);
    bool handle_trigger(HalfBuffers& half,
                        const std::shared_ptr<SigmaContext>& ctx,
                        const std::vector<uint8_t>& header,
                        uint64_t nonce);

    struct ShareTriggerJob {
        HalfBuffers* half = nullptr;
        std::vector<uint8_t> header;
        uint64_t nonce = 0;
        std::shared_ptr<SigmaContext> sigma_ctx;
        uint32_t target_nbits = 0;
    };

    static bool defer_share_gpu_enabled();
    void enqueue_share_trigger(HalfBuffers& half, uint64_t nonce,
                               const std::vector<uint8_t>& header,
                               const std::shared_ptr<SigmaContext>& ctx);
    bool process_share_trigger(const ShareTriggerJob& job);
    bool process_share_trigger_impl(const ShareTriggerJob& job);
    void share_gpu_loop();
    void wait_until_half_free(HalfBuffers& half);
    void start_share_gpu_thread();
    void stop_share_gpu_thread();

    // Async job installation (PROPMINER_ASYNC_JOB_INSTALL). Background thread
    // runs ctx->install() on staging resources while the mining loop keeps
    // running the current sigma; the worker thread then swaps via
    // bind_sigma_to_half once the install is ready.
    static bool async_job_install_enabled();
    void start_async_install_thread();
    void stop_async_install_thread();
    void async_install_loop();
    // Hand a newly requested sigma to the background installer (deduped on ptr).
    void submit_async_install(const std::shared_ptr<SigmaContext>& ctx);
    // Return a fully-installed sigma ready to swap in (or nullptr), clearing it.
    std::shared_ptr<SigmaContext> take_async_ready();
    // True (and clears the marker) if the background install of `ctx` failed, so
    // the worker can fall back to a synchronous install instead of stalling.
    bool async_install_failed(const std::shared_ptr<SigmaContext>& ctx);
    // Conservative free-VRAM check before a background install. `need_workspace`
    // adds the one-time staging workspace to the estimate. Returns false (skip
    // async, use sync) when headroom is tight so the async path can never OOM.
    bool async_vram_headroom_ok(bool need_workspace);

    // Triple half-buffer (PROPMINER_TRIPLE_BUFFER). Third compute workspace so
    // share rebuild on one half leaves two free for GEMM rotation.
    static bool triple_buffer_enabled();
    bool triple_vram_headroom_ok() const;
    const char* half_tag(const HalfBuffers& half) const;
    void upload_pow_target_all_halves(uint32_t nbits);
    void drain_all_halves_for_sigma();
    void sync_all_compute_streams();

    // Scan batch for all PoW hits (status==1). Caller must sync stream first.
    std::vector<int> scan_winners(HalfBuffers& half, int batch);

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
    HalfBuffers third_;
    bool triple_buffer_active_ = false;

    // Pipeline stall instrumentation (wait_until_half_free wall time).
    std::atomic<uint64_t> half_wait_ms_total_{0};
    std::atomic<uint32_t> half_wait_count_{0};
    std::atomic<uint64_t> half_wait_ms_max_{0};

    std::atomic<bool> running_{false};
    std::atomic<bool> stop_flag_{false};
    std::atomic<bool> pause_flag_{false};
    std::atomic<bool> batch_abort_requested_{false};
    std::thread thread_;

    void upload_pow_target(HalfBuffers& half, uint32_t nbits);
    void upload_pow_target_both_halves(uint32_t nbits);

    std::mutex sigma_mtx_;
    std::shared_ptr<SigmaContext> sigma_;
    std::atomic<uint32_t> target_nbits_{0};
    std::atomic<bool> target_dirty_{false};
    std::atomic<int> matmuls_per_poll_{Rtx5090Profile::kDefaultMineBatch};
    std::atomic<int> graph_batch_{Rtx5090Profile::kDefaultGraphBatch};

    std::atomic<uint64_t> total_iters_{0};
    std::atomic<double> hashrate_{0.0};
    std::atomic<double> tmads_per_sec_{0.0};
    std::atomic<double> last_iter_ms_{0.0};

    Watchdog* watchdog_ = nullptr;

    bool logged_first_queue_ = false;
    bool logged_first_hashrate_ = false;

    // Per-GPU nonce-space base.
    // GPUs never collide; low 32 bits are incremented each batch.
    uint64_t seed_base_ = 0;

    GemmCapi gemm_;
    MiningCapi mining_;

    // Optional side thread for deferred share GPU work (PROPMINER_DEFER_SHARE_GPU=1).
    std::mutex share_mtx_;
    std::condition_variable share_cv_;
    std::condition_variable share_done_cv_;
    std::queue<ShareTriggerJob> share_queue_;
    std::atomic<bool> share_stop_{false};
    std::thread share_thread_;
    bool share_thread_started_ = false;

    // Async job installation (PROPMINER_ASYNC_JOB_INSTALL). Staging resources are
    // owned by the background installer thread and reused across installs.
    void* install_workspace_ = nullptr;
    CUstream install_stream_ = nullptr;
    CUstream install_copy_stream_ = nullptr;
    std::thread async_install_thread_;
    std::mutex async_mtx_;
    std::condition_variable async_cv_;
    std::shared_ptr<SigmaContext> async_pending_;   // requested, not yet installed
    std::shared_ptr<SigmaContext> async_ready_;     // installed, ready to swap
    std::shared_ptr<SigmaContext> async_failed_;    // install threw; sync fallback
    std::shared_ptr<SigmaContext> async_requested_; // worker-thread dedup marker
    std::atomic<bool> async_stop_{false};
    bool async_install_thread_started_ = false;
};

} // namespace pearl
