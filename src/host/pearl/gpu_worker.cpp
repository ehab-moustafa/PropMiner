#include "gpu_worker.h"
#include "hashrate_metrics.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

#include <cuda_runtime.h>

#if defined(__linux__)
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#endif

#include "host_signal_header.h"
#include "merkle_utils.h"
#include "pearl_blake3.h"
#include "pow_target_utils.h"
#include "share_builder.h"
#include "env_flags.h"
#include "env_tuning.h"
#include "share_trace.h"
#include "system_telemetry.h"
#include "watchdog.h"

// For cuda runtime helpers (events, graph).  CUstream and cudaStream_t are
// interchangeable pointers in the CUDA driver/runtime interop layer.
#define CUDA_CHECK(call)                                        \
    do {                                                        \
        cudaError_t e = (call);                                 \
        if (e != cudaSuccess) {                                 \
            check_cuda(CUDA_ERROR_UNKNOWN, cudaGetErrorString(e)); \
        }                                                       \
    } while (0)

namespace pearl {

namespace {
    constexpr size_t K32 = 32;
    constexpr size_t K1024 = 1024;

    uint64_t now_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // Hot-path batch wait: tight cudaEventQuery spin, then yield (no sleep).
    // Returns true when the batch completes. Returns false if `max_wait_ms` > 0
    // and the batch has not completed within that wall time — signalling a
    // wedged GPU stream (100% util, no progress) that the caller must recover
    // from by restarting the process. `max_wait_ms` <= 0 spins indefinitely.
    bool spin_wait_batch_event(cudaEvent_t event, Watchdog* watchdog,
                               int64_t max_wait_ms,
                               const std::atomic<bool>* abort) {
        constexpr int kSpinTight = 4096;
        constexpr int kYieldEvery = 64;
        int spins = 0;
        bool logged_half = false;
        const auto start = std::chrono::steady_clock::now();
        while (true) {
            if (abort && abort->load(std::memory_order_acquire)) {
                return false;
            }
            cudaError_t e = cudaEventQuery(event);
            if (e == cudaSuccess) return true;
            if (e != cudaErrorNotReady) {
                throw std::runtime_error(
                    std::string("cudaEventQuery: ") + cudaGetErrorString(e));
            }
            if (max_wait_ms > 0 && (spins & 0x3FFF) == 0) {
                const auto waited = std::chrono::duration_cast<
                    std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start).count();
                if (!logged_half && waited > max_wait_ms / 2) {
                    logged_half = true;
                    std::fprintf(stderr,
                        "[gpu] WARN: batch wait exceeded %lld ms (50%% of %lld ms "
                        "stall limit) — possible wedge forming\n",
                        static_cast<long long>(waited),
                        static_cast<long long>(max_wait_ms));
                    std::fflush(stderr);
                }
                if (waited > max_wait_ms) return false;
            }
            if (watchdog && (spins & 0x3FFF) == 0) {
                watchdog->heartbeat();
            }
            if (spins < kSpinTight) {
#if defined(__x86_64__) || defined(_M_X64)
                __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(_M_ARM64)
                __asm__ __volatile__("yield");
#endif
            } else if ((spins - kSpinTight) % kYieldEvery == 0) {
                std::this_thread::yield();
            }
            ++spins;
        }
    }
}

void GpuWorker::HalfBuffers::allocate(const MiningConfig& cfg, int batch, int device_id, CUstream s) {
    stream = s;
    size_t a_bytes = static_cast<size_t>(cfg.m) * cfg.k;
    // Pure-miner mode: C is never materialised in HBM (transcript stays in
    // registers).  Passing nullptr to the C API is safe — the kernel guards
    // with `if (C_gmem != nullptr)` and SkipCStore=true elides the epilogue.
    // Skipping this allocation frees 8-12 GiB for larger N dimensions.
    size_t eal_bytes = static_cast<size_t>(cfg.m) * cfg.r;
    size_t ear_r_bytes = static_cast<size_t>(cfg.k) * cfg.r;
    size_t ear_k_bytes = static_cast<size_t>(cfg.r) * cfg.k;
    size_t ax_ebl_bytes = static_cast<size_t>(cfg.m) * cfg.r * 2;
    size_t apea_bytes = static_cast<size_t>(cfg.m) * cfg.k;
    size_t a_scales_bytes = static_cast<size_t>(cfg.m) * sizeof(float);

    auto check = [](CUresult r, const char* msg) {
        if (r == CUDA_SUCCESS) return;
        const char* s = nullptr; cuGetErrorString(r, &s);
        throw std::runtime_error(std::string(msg) + ": " + (s ? s : "unknown"));
    };

    check(cuMemAlloc(&a, a_bytes), "a alloc");
    check(cuMemAlloc(&a_hash, K32), "a_hash alloc");
    check(cuMemAlloc(&roots, 256), "roots alloc");
    check(cuMemAlloc(&commit_a, K32), "commit_a alloc");
    check(cuMemAlloc(&commit_b, K32), "commit_b alloc");
    check(cuMemAlloc(&eal, eal_bytes), "eal alloc");
    check(cuMemAlloc(&eal_fp16, eal_bytes * 2), "eal_fp16 alloc");
    check(cuMemAlloc(&ear_r, ear_r_bytes), "ear_r alloc");
    check(cuMemAlloc(&ear_k, ear_k_bytes), "ear_k alloc");
    check(cuMemAlloc(&ax_ebl_fp16, ax_ebl_bytes), "ax_ebl_fp16 alloc");
    check(cuMemAlloc(&apea, apea_bytes), "apea alloc");
    check(cuMemAlloc(&a_scales, a_scales_bytes), "a_scales alloc");
    // c = nullptr (pure-miner: transcript in registers, C never materialised)
    check(cuMemAlloc(&sync, 256), "sync alloc");
    // Transcript buffer is allocated by the C API workspace (not here).
    check(cuMemAlloc(&pow_target, 32), "pow_target alloc");

    // Device-side seed pointer for the extended graph path.  One 8-byte seed
    // value per half-buffer; the CPU uploads it via cudaMemcpyAsync on a
    // separate copy stream while the other half runs the GEMM graph.
    check(cuMemAlloc(&seed_dev, sizeof(uint64_t)), "seed_dev alloc");
    seed_dev_ptr = reinterpret_cast<void*>(seed_dev);

    size_t a_leaves = (a_bytes + K1024 - 1) / K1024;
    a_leaf_cv_bytes = a_leaves * K32;
    check(cuMemAlloc(&a_leaf_cvs, a_leaf_cv_bytes), "a_leaf_cvs alloc");

    // Pre-create timing events so the hot path never allocates.
    if (!batch_start_event) {
        cudaError_t e = cudaEventCreate(&batch_start_event);
        if (e != cudaSuccess) batch_start_event = nullptr;
    }
    if (!batch_done_event) {
        cudaError_t e = cudaEventCreate(&batch_done_event);
        if (e != cudaSuccess) batch_done_event = nullptr;
    }
    if (!sub_batch_done_event) {
        cudaError_t e = cudaEventCreate(&sub_batch_done_event);
        if (e != cudaSuccess) sub_batch_done_event = nullptr;
    }

    auto check_rt = [](cudaError_t e, const char* msg) {
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
        }
    };

    check_rt(cudaHostAlloc(reinterpret_cast<void**>(&pinned_seed_host),
                           sizeof(uint64_t), cudaHostAllocDefault),
             "half pinned_seed_host");

    constexpr size_t kMaxShareRows = 32;
    pinned_a_slice_bytes = kMaxShareRows * static_cast<size_t>(cfg.k);
    pinned_opened_leaves_bytes = kMaxShareRows * 1024;
    check_rt(cudaHostAlloc(reinterpret_cast<void**>(&pinned_leaf_cvs),
                           a_leaf_cv_bytes, cudaHostAllocDefault),
             "pinned_leaf_cvs");
    check_rt(cudaHostAlloc(reinterpret_cast<void**>(&pinned_a_slice),
                           pinned_a_slice_bytes, cudaHostAllocDefault),
             "pinned_a_slice");
    check_rt(cudaHostAlloc(reinterpret_cast<void**>(&pinned_opened_leaves),
                           pinned_opened_leaves_bytes, cudaHostAllocDefault),
             "pinned_opened_leaves");
    check_rt(cudaHostAlloc(reinterpret_cast<void**>(&pinned_hash_a), 32,
                           cudaHostAllocDefault),
             "pinned_hash_a");
    check_rt(cudaHostAlloc(reinterpret_cast<void**>(&pinned_hash_b), 32,
                           cudaHostAllocDefault),
             "pinned_hash_b");
    check_rt(cudaHostAlloc(reinterpret_cast<void**>(&pinned_commit_a), 32,
                           cudaHostAllocDefault),
             "pinned_commit_a");
    check_rt(cudaHostAlloc(reinterpret_cast<void**>(&pinned_commit_b), 32,
                           cudaHostAllocDefault),
             "pinned_commit_b");

    (void)device_id;
}

void GpuWorker::HalfBuffers::free() {
    auto f = [](CUdeviceptr& p) { if (p) { cuMemFree(p); p = 0; } };
    f(a); f(a_hash); f(roots); f(commit_a); f(commit_b);
    f(eal); f(eal_fp16); f(ear_r); f(ear_k); f(ax_ebl_fp16);
    f(apea); f(a_scales); f(c); f(sync); f(pow_target); f(a_leaf_cvs);
    f(seed_dev);
    seed_dev_ptr = nullptr;
    if (pinned_seed_host) {
        cudaFreeHost(pinned_seed_host);
        pinned_seed_host = nullptr;
    }
    if (workspace) {
        pearl_capi_workspace_free(workspace, stream);
        workspace = nullptr;
    }
    if (batch_start_event) {
        cudaEventDestroy(batch_start_event);
        batch_start_event = nullptr;
    }
    if (batch_done_event) {
        cudaEventDestroy(batch_done_event);
        batch_done_event = nullptr;
    }
    if (sub_batch_done_event) {
        cudaEventDestroy(sub_batch_done_event);
        sub_batch_done_event = nullptr;
    }
    if (stream) { cuStreamDestroy(stream); stream = nullptr; }
    for (void* h : host_headers) {
        if (h) cudaFreeHost(h);
    }
    host_headers.clear();
    host_header_storage.clear();
    if (pinned_leaf_cvs) { cudaFreeHost(pinned_leaf_cvs); pinned_leaf_cvs = nullptr; }
    if (pinned_a_slice) { cudaFreeHost(pinned_a_slice); pinned_a_slice = nullptr; }
    if (pinned_opened_leaves) {
        cudaFreeHost(pinned_opened_leaves);
        pinned_opened_leaves = nullptr;
    }
    if (pinned_hash_a) { cudaFreeHost(pinned_hash_a); pinned_hash_a = nullptr; }
    if (pinned_hash_b) { cudaFreeHost(pinned_hash_b); pinned_hash_b = nullptr; }
    if (pinned_commit_a) { cudaFreeHost(pinned_commit_a); pinned_commit_a = nullptr; }
    if (pinned_commit_b) { cudaFreeHost(pinned_commit_b); pinned_commit_b = nullptr; }
}

GpuWorker::GpuWorker(int device_index, int gpu_index,
                     const MiningConfig& cfg, IShareSink* sink)
    : device_index_(device_index), gpu_index_(gpu_index),
      cfg_(cfg), sink_(sink) {
    // Use Runtime API to bind to the device. WSL2 containers reject
    // cuCtxCreate but accept cudaSetDevice.
    CUDA_CHECK(cudaSetDevice(device_index_));
    // Force the implicit primary context to be created and made current for
    // this thread. Without this, subsequent driver API calls (cuMemAlloc etc.)
    // fail with CUDA_ERROR_INVALID_CONTEXT on WSL2.
    CUDA_CHECK(cudaFree(0));
    device_ = device_index_;
    // implicit primary context; no explicit CUcontext
    check_cuda(cuStreamCreate(&merkle_copy_stream_, CU_STREAM_NON_BLOCKING), "merkle stream");
    check_cuda(cuStreamCreate(&ping_.stream, CU_STREAM_NON_BLOCKING), "ping stream");
    check_cuda(cuStreamCreate(&pong_.stream, CU_STREAM_NON_BLOCKING), "pong stream");
    check_cuda(cuStreamCreate(&seed_copy_stream_, CU_STREAM_NON_BLOCKING), "seed copy stream");

    // Tune L2 cache fetch granularity for large, sequential GEMM traffic on
    // the RTX 5090.  128-byte lines maximize GDDR7 bus utilization for the
    // M=8192,N=32768,K=128 workload.
    cudaError_t l2e = cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 128);
    if (l2e != cudaSuccess) {
        // Non-fatal; older drivers may ignore this limit.
        (void)l2e;
    }

    CUDA_CHECK(cudaEventCreateWithFlags(&seed_copy_done_event_,
                                        cudaEventDisableTiming));

    ping_.allocate(cfg_, Rtx5090Profile::kDefaultMineBatch, device_index_, ping_.stream);
    pong_.allocate(cfg_, Rtx5090Profile::kDefaultMineBatch, device_index_, pong_.stream);

    if (triple_buffer_enabled() && defer_share_gpu_enabled() &&
        triple_vram_headroom_ok()) {
        try {
            check_cuda(cuStreamCreate(&third_.stream, CU_STREAM_NON_BLOCKING),
                       "third stream");
            third_.allocate(cfg_, Rtx5090Profile::kDefaultMineBatch, device_index_, third_.stream);
            triple_buffer_active_ = true;
            size_t free_bytes = 0, total_bytes = 0;
            cudaMemGetInfo(&free_bytes, &total_bytes);
            std::fprintf(stderr,
                "[gpu] triple-buffer enabled gpu=%d N=%d K=%d free_vram=%.1f/%.1fGiB\n",
                device_index_, cfg_.n, cfg_.k,
                free_bytes / (1024.0 * 1024.0 * 1024.0),
                total_bytes / (1024.0 * 1024.0 * 1024.0));
        } catch (...) {
            third_.free();
            triple_buffer_active_ = false;
            std::fprintf(stderr,
                "[gpu] triple-buffer disabled gpu=%d reason=alloc_failed\n",
                device_index_);
        }
    } else if (triple_buffer_enabled()) {
        const char* reason = !defer_share_gpu_enabled()
                                 ? "defer_share_off"
                                 : "insufficient_vram";
        std::fprintf(stderr,
            "[gpu] triple-buffer disabled gpu=%d reason=%s N=%d K=%d\n",
            device_index_, reason, cfg_.n, cfg_.k);
    }

    // Per-GPU nonce-space partition: top 16 bits = gpu_index, next 16 bits =
    // time-based entropy.  This lets multiple GPUs mine disjoint ranges without
    // central coordination.
    seed_base_ = (static_cast<uint64_t>(gpu_index_) << 48) |
                 (static_cast<uint64_t>(now_ms() & 0xFFFF) << 32);
}

GpuWorker::~GpuWorker() {
    stop();
    if (triple_buffer_active_) third_.free();
    ping_.free();
    pong_.free();
    if (seed_copy_done_event_) { cudaEventDestroy(seed_copy_done_event_); seed_copy_done_event_ = nullptr; }
    if (seed_copy_stream_) { cuStreamDestroy(seed_copy_stream_); seed_copy_stream_ = nullptr; }
    if (merkle_copy_stream_) cuStreamDestroy(merkle_copy_stream_);
    // Primary context is implicit with Runtime API; do not destroy it.
}

void GpuWorker::check_cuda(CUresult r, const char* msg) {
    if (r == CUDA_SUCCESS) return;
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    throw std::runtime_error(std::string(msg) + ": " + (s ? s : "unknown"));
}

void GpuWorker::start() {
    stop_flag_ = false;
    pause_flag_ = false;
    running_ = true;
    start_share_gpu_thread();
    start_async_install_thread();
    thread_ = std::thread(&GpuWorker::run, this);
}

void GpuWorker::stop() {
    stop_flag_ = true;
    if (thread_.joinable()) thread_.join();
    stop_share_gpu_thread();
    stop_async_install_thread();
    running_ = false;
}

bool GpuWorker::defer_share_gpu_enabled() {
    // Default ON: share rebuild runs on the side thread so the GPU keeps
    // mining. Set PROPMINER_DEFER_SHARE_GPU=0 to restore inline handling.
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("PROPMINER_DEFER_SHARE_GPU");
        cached = (env && env[0] == '0') ? 0 : 1;
    }
    return cached == 1;
}

void GpuWorker::start_share_gpu_thread() {
    if (!defer_share_gpu_enabled() || share_thread_started_) return;
    share_stop_.store(false);
    share_thread_ = std::thread(&GpuWorker::share_gpu_loop, this);
    share_thread_started_ = true;
}

void GpuWorker::stop_share_gpu_thread() {
    if (!share_thread_started_) return;
    {
        std::lock_guard<std::mutex> lk(share_mtx_);
        share_stop_.store(true);
    }
    share_cv_.notify_all();
    if (share_thread_.joinable()) share_thread_.join();
    share_thread_started_ = false;
    {
        std::lock_guard<std::mutex> lk(share_mtx_);
        while (!share_queue_.empty()) share_queue_.pop();
    }
    ping_.share_jobs_pending.store(0);
    pong_.share_jobs_pending.store(0);
    third_.share_jobs_pending.store(0);
}

void GpuWorker::wait_until_half_free(HalfBuffers& half) {
    if (half.share_jobs_pending.load(std::memory_order_acquire) == 0) return;
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lk(share_mtx_);
    share_done_cv_.wait(lk, [&] {
        return half.share_jobs_pending.load(std::memory_order_acquire) == 0;
    });
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    if (ms > 0) {
        half_wait_ms_total_.fetch_add(static_cast<uint64_t>(ms),
                                      std::memory_order_relaxed);
        half_wait_count_.fetch_add(1, std::memory_order_relaxed);
        uint64_t cur_max = half_wait_ms_max_.load(std::memory_order_relaxed);
        while (static_cast<uint64_t>(ms) > cur_max &&
               !half_wait_ms_max_.compare_exchange_weak(
                   cur_max, static_cast<uint64_t>(ms),
                   std::memory_order_relaxed)) {}
    }
}

void GpuWorker::enqueue_share_trigger(HalfBuffers& half, uint64_t nonce,
                                      const std::vector<uint8_t>& header,
                                      const std::shared_ptr<SigmaContext>& ctx) {
    ShareTriggerJob job;
    job.half = &half;
    job.header = header;
    job.nonce = nonce;
    job.target_nbits = half.batch_mined_target_nbits
        ? half.batch_mined_target_nbits
        : target_nbits_.load();
    job.sigma_ctx = ctx;
    if (!job.sigma_ctx) {
        share_log("rebuild-drop", "nonce=" + std::to_string(nonce) +
                                  " reason=missing_sigma_context");
        return;
    }

    share_trace("rebuild-queued",
                "nonce=" + std::to_string(nonce) +
                " gpu=" + std::to_string(device_index_) +
                " batch_seed_start=" + u64_hex(half.batch_seed_start) +
                " defer_gpu=1");

    {
        std::lock_guard<std::mutex> lk(share_mtx_);
        half.share_jobs_pending.fetch_add(1, std::memory_order_relaxed);
        share_queue_.push(std::move(job));
    }
    share_cv_.notify_one();
}

void GpuWorker::share_gpu_loop() {
    CUDA_CHECK(cudaSetDevice(device_index_));
    while (true) {
        ShareTriggerJob job;
        {
            std::unique_lock<std::mutex> lk(share_mtx_);
            share_cv_.wait(lk, [&] {
                return share_stop_.load() || !share_queue_.empty();
            });
            if (share_stop_.load() && share_queue_.empty()) break;
            job = std::move(share_queue_.front());
            share_queue_.pop();
        }
        if (job.half && job.sigma_ctx) {
            try {
                (void)process_share_trigger(job);
            } catch (const std::exception& ex) {
                share_log("rebuild-fail",
                          "nonce=" + std::to_string(job.nonce) +
                          " reason=exception err=" + ex.what());
            } catch (...) {
                share_log("rebuild-fail",
                          "nonce=" + std::to_string(job.nonce) +
                          " reason=unknown_exception");
            }
        }
        if (job.half) {
            std::lock_guard<std::mutex> lk(share_mtx_);
            const int remaining =
                job.half->share_jobs_pending.fetch_sub(1) - 1;
            if (remaining == 0) share_done_cv_.notify_all();
        }
    }
}

void GpuWorker::set_sigma(std::shared_ptr<SigmaContext> ctx) {
    std::lock_guard<std::mutex> lk(sigma_mtx_);
    sigma_ = std::move(ctx);
}

bool GpuWorker::async_job_install_enabled() {
    // Single source of truth: pearl::async_job_install_enabled() in env_flags.h.
    // Cached because run() reads it every loop iteration.
    static const int cached = pearl::async_job_install_enabled() ? 1 : 0;
    return cached == 1;
}

void GpuWorker::start_async_install_thread() {
    if (!async_job_install_enabled() || async_install_thread_started_) return;
    async_stop_.store(false);
    async_install_thread_ = std::thread(&GpuWorker::async_install_loop, this);
    async_install_thread_started_ = true;
}

void GpuWorker::stop_async_install_thread() {
    if (!async_install_thread_started_) return;
    {
        std::lock_guard<std::mutex> lk(async_mtx_);
        async_stop_.store(true);
    }
    async_cv_.notify_all();
    if (async_install_thread_.joinable()) async_install_thread_.join();
    async_install_thread_started_ = false;
    {
        std::lock_guard<std::mutex> lk(async_mtx_);
        async_pending_.reset();
        async_ready_.reset();
        async_failed_.reset();
    }
    async_requested_.reset();
    if (install_workspace_) {
        pearl_capi_workspace_free(install_workspace_, install_stream_);
        install_workspace_ = nullptr;
    }
    if (install_stream_) { cuStreamDestroy(install_stream_); install_stream_ = nullptr; }
    if (install_copy_stream_) { cuStreamDestroy(install_copy_stream_); install_copy_stream_ = nullptr; }
}

void GpuWorker::submit_async_install(const std::shared_ptr<SigmaContext>& ctx) {
    // Worker-thread only. Dedup: don't re-request a sigma we already handed off.
    if (!ctx || ctx == async_requested_) return;
    async_requested_ = ctx;
    {
        std::lock_guard<std::mutex> lk(async_mtx_);
        async_pending_ = ctx;
    }
    async_cv_.notify_one();
    share_trace("async-install-request",
                "gpu=" + std::to_string(device_index_) +
                " sigma=" + hex_prefix(ctx->job().sigma.data(),
                                       ctx->job().sigma.size(), 4));
}

std::shared_ptr<SigmaContext> GpuWorker::take_async_ready() {
    std::lock_guard<std::mutex> lk(async_mtx_);
    std::shared_ptr<SigmaContext> ready = std::move(async_ready_);
    async_ready_.reset();
    return ready;
}

bool GpuWorker::async_vram_headroom_ok(bool need_workspace) {
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        return false;  // can't tell -> be conservative, use sync install
    }
    const size_t n = static_cast<size_t>(cfg_.n);
    const size_t k = static_cast<size_t>(cfg_.k);
    const size_t r = static_cast<size_t>(cfg_.r);
    // One ResidentBState set (mirrors ResidentBState::allocate): b + bpeb (n*k
    // each), ebr (n*r), ebr_fp16 + earx_bpeb (2*n*r each), ebl_r + ebl_k (k*r
    // each), b_scales (n*4), leaf_cvs (n*k/1024*32 = n*k/32), plus small fixed.
    const size_t resident_b = 2 * n * k + 5 * n * r + 2 * k * r + 4 * n +
                              (n * k) / 32 + 4096;
    // Staging workspace holds only the noise_B scratch: max(n*k, n*r) * 4 bytes.
    const size_t staging = need_workspace
                               ? n * std::max(k, r) * sizeof(int32_t)
                               : 0;
    const size_t margin = static_cast<size_t>(512) << 20;  // 512 MiB
    const size_t need = resident_b + staging + margin;
    return free_bytes >= need;
}

bool GpuWorker::triple_buffer_enabled() {
    static const int cached = pearl::triple_buffer_enabled() ? 1 : 0;
    return cached == 1;
}

bool GpuWorker::triple_vram_headroom_ok() const {
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        return false;
    }
    const size_t m = static_cast<size_t>(cfg_.m);
    const size_t n = static_cast<size_t>(cfg_.n);
    const size_t k = static_cast<size_t>(cfg_.k);
    const size_t r = static_cast<size_t>(cfg_.r);
    // Per-iteration transcript size (workspace pool).
    const size_t transcript_iter = static_cast<size_t>(m / cfg_.bM) *
                                   static_cast<size_t>(n / cfg_.bN) * 256 * 16 * 4;
    // Batched transcript device buffer (allocated per half).
    const int batch = matmuls_per_poll_.load();
    const size_t transcript_device =
        static_cast<size_t>(batch) * transcript_iter;
    // Per-half device buffers + batched transcript buffer for BLAKE3 offload.
    const size_t half_device = 2 * m * k + m * n * 2 + 6 * m * r + 2 * k * r +
                               m * 4 + (m * k) / 32 + 4096 + transcript_device;
    const size_t noise_b = n * std::max(k, r) * sizeof(int32_t);
    const size_t noise_a = std::max(m * r, m * k) * sizeof(int32_t);
    const size_t workspace_per_half = noise_b + noise_a + transcript_iter + (1ULL << 20);
    const size_t resident_b = 2 * n * k + 5 * n * r + 2 * k * r + 4 * n +
                              (n * k) / 32 + 4096;
    const size_t margin = 512ULL << 20;
    // Steady-state peak: three full halves (device + workspace each) + resident B.
    // ping/pong device buffers are already allocated when this runs in the ctor.
    const size_t peak_total = 3 * (half_device + workspace_per_half) + resident_b +
                              margin;
    const size_t already_device = 2 * half_device;
    const size_t incremental = peak_total > already_device
                                   ? peak_total - already_device
                                   : peak_total;
    // Hard floor at prod N/K: ~11 GiB free required (audit: dual @79% leaves ~6.7).
    if (n >= 262144 && k >= 2048) {
        return free_bytes >= (11ULL << 30) && free_bytes >= incremental;
    }
    return free_bytes >= incremental;
}

const char* GpuWorker::half_tag(const HalfBuffers& half) const {
    if (&half == &ping_) return "ping";
    if (&half == &pong_) return "pong";
    if (&half == &third_) return "third";
    return "half";
}

void GpuWorker::upload_pow_target_all_halves(uint32_t nbits) {
    upload_pow_target(ping_, nbits);
    upload_pow_target(pong_, nbits);
    if (triple_buffer_active_) upload_pow_target(third_, nbits);
}

void GpuWorker::drain_all_halves_for_sigma() {
    if (!defer_share_gpu_enabled()) return;
    wait_until_half_free(ping_);
    wait_until_half_free(pong_);
    if (triple_buffer_active_) wait_until_half_free(third_);
}

void GpuWorker::sync_all_compute_streams() {
    check_cuda(cuStreamSynchronize(ping_.stream), "sync ping stream");
    check_cuda(cuStreamSynchronize(pong_.stream), "sync pong stream");
    if (triple_buffer_active_) {
        check_cuda(cuStreamSynchronize(third_.stream), "sync third stream");
    }
}

void GpuWorker::async_install_loop() {
    // Bind this thread to the same device/primary context as the worker thread.
    CUDA_CHECK(cudaSetDevice(device_index_));
    CUDA_CHECK(cudaFree(0));
    check_cuda(cuStreamCreate(&install_stream_, CU_STREAM_NON_BLOCKING),
               "install stream");
    check_cuda(cuStreamCreate(&install_copy_stream_, CU_STREAM_NON_BLOCKING),
               "install copy stream");

    while (true) {
        std::shared_ptr<SigmaContext> to_install;
        {
            std::unique_lock<std::mutex> lk(async_mtx_);
            async_cv_.wait(lk, [&] {
                return async_stop_.load() || async_pending_ != nullptr;
            });
            // On shutdown, drop any pending install immediately — no point
            // spending seconds on an install we will never mine.
            if (async_stop_.load()) break;
            to_install = std::move(async_pending_);
            async_pending_.reset();
        }
        if (!to_install) continue;

        // Mark this sigma as failed so the mining thread falls back to a
        // synchronous install instead of wedging on the stale job.
        auto mark_failed = [&](const char* reason) {
            std::lock_guard<std::mutex> lk(async_mtx_);
            async_failed_ = to_install;
            share_log("async-install-fallback",
                      "gpu=" + std::to_string(device_index_) +
                      " reason=" + reason);
        };

        // VRAM guard: an async install briefly holds a SECOND resident-B set
        // (plus, the first time, the staging workspace) on top of the live job.
        // If there is not enough free VRAM, skip async and let the worker do a
        // synchronous install (which reuses ping/pong workspaces and has a lower
        // peak). This makes the feature OOM-safe even when enabled at a shape
        // that does not fit.
        if (!async_vram_headroom_ok(install_workspace_ == nullptr)) {
            mark_failed("insufficient_vram");
            continue;
        }

        // Staging workspace for noise_B scratch only (with_noise_A=0 drops the
        // ~1 GiB transcript + noise_A pools the install path never touches).
        // Allocated once, reused. Must NOT be a ping/pong workspace (those are
        // live on the mining thread).
        if (!install_workspace_) {
            void* ws = nullptr;
            int rc = pearl_capi_workspace_alloc(cfg_.m, cfg_.n, cfg_.k, cfg_.r,
                                                /*with_noise_A=*/0,
                                                /*with_noise_B=*/1,
                                                &ws, install_stream_);
            if (rc != 0 || !ws) {
                mark_failed("staging_workspace_alloc");
                continue;  // worker falls back to synchronous install
            }
            install_workspace_ = ws;
        }

        try {
            // Full resident-B install on staging stream/workspace. Idempotent;
            // never reuse an already-installed context (orchestrator always
            // hands us a fresh SigmaContext per job).
            to_install->install(install_stream_, install_workspace_,
                                 device_index_, install_copy_stream_);
            // Publish as ready. If a previous ready sigma was never consumed
            // (worker superseded it), free its resident-B here so we do not
            // accumulate more than one leaked set under fast job churn.
            std::shared_ptr<SigmaContext> superseded;
            {
                std::lock_guard<std::mutex> lk(async_mtx_);
                superseded = std::move(async_ready_);
                async_ready_ = to_install;
            }
            if (superseded && superseded != to_install) {
                superseded->resident().free(install_stream_);
            }
            share_trace("async-install-ready",
                        "gpu=" + std::to_string(device_index_) +
                        " sigma=" + hex_prefix(to_install->job().sigma.data(),
                                               to_install->job().sigma.size(), 4));
        } catch (const std::exception& ex) {
            // Mark failed so the worker thread falls back to a synchronous
            // install for this sigma — mining is never left wedged on a stale job.
            {
                std::lock_guard<std::mutex> lk(async_mtx_);
                async_failed_ = to_install;
            }
            share_log("async-install-fail",
                      "gpu=" + std::to_string(device_index_) +
                      " reason=exception err=" + ex.what());
        } catch (...) {
            {
                std::lock_guard<std::mutex> lk(async_mtx_);
                async_failed_ = to_install;
            }
            share_log("async-install-fail",
                      "gpu=" + std::to_string(device_index_) +
                      " reason=unknown_exception");
        }
    }
}

bool GpuWorker::async_install_failed(const std::shared_ptr<SigmaContext>& ctx) {
    std::lock_guard<std::mutex> lk(async_mtx_);
    if (async_failed_ && async_failed_ == ctx) {
        async_failed_.reset();
        return true;
    }
    return false;
}

void GpuWorker::set_target_nbits(uint32_t nbits) {
    target_nbits_.store(nbits);
    target_dirty_.store(true);
}

void GpuWorker::set_matmuls_per_poll(int mpp) {
    if (mpp > 0) matmuls_per_poll_.store(mpp);
}

void GpuWorker::set_graph_batch(int gb) {
    if (gb > 0) graph_batch_.store(gb);
}

void GpuWorker::upload_pow_target(HalfBuffers& half, uint32_t nbits) {
    if (!half.pow_target || !half.stream) return;
    if (nbits == 0) {
        std::fprintf(stderr, "[gpu] WARN: upload_pow_target skipped (nbits=0)\n");
        return;
    }
    const uint64_t daf = cfg_.difficulty_adjustment_factor();
    auto words = target_le_to_pow_u32(adjusted_pow_target_le(nbits, daf));
    check_cuda(cuMemcpyHtoDAsync(half.pow_target, words.data(), 32, half.stream),
               "pow_target h2d");
    half.pow_target_nbits = nbits;
    if (share_trace_enabled() || std::getenv("PROPMINER_DEBUG_POW_TARGET")) {
        std::fprintf(stderr,
                     "[gpu] pow_target upload nbits=%s daf=%llu "
                     "words[7,6,1,0]=%08x %08x %08x %08x\n",
                     nbits_hex(nbits).c_str(),
                     static_cast<unsigned long long>(daf),
                     words[7], words[6], words[1], words[0]);
    }
}

void GpuWorker::upload_pow_target_both_halves(uint32_t nbits) {
    upload_pow_target(ping_, nbits);
    upload_pow_target(pong_, nbits);
}

double GpuWorker::hashrate() const {
    return hashrate_.load();
}

void GpuWorker::ensure_half_workspace(HalfBuffers& half) {
    // Workspace: allocate once per half, reused across σ rotations.  Keep both
    // noise_A and noise_B scratchpads so σ-install can reuse the same pool.
    if (!half.workspace) {
        void* ws = nullptr;
        int rc = pearl_capi_workspace_alloc(cfg_.m, cfg_.n, cfg_.k, cfg_.r,
                                            1, 1, &ws, half.stream);
        if (rc != 0 || !ws) throw std::runtime_error("workspace_alloc failed");
        half.workspace = ws;
    }
}

void GpuWorker::install_sigma(SigmaContext& ctx, HalfBuffers& half) {
    ensure_half_workspace(half);

    // Ensure resident B state is allocated and computed on device.  Idempotent;
    // on the second half it reuses the already-installed buffers owned by ctx.
    ctx.install(half.stream, half.workspace, device_index_, merkle_copy_stream_);

    bind_sigma_to_half(ctx, half);
}

void GpuWorker::bind_sigma_to_half(SigmaContext& ctx, HalfBuffers& half) {
    // ctx.install() must already have populated ctx.resident(). This is the
    // worker-thread-only tail: it binds device pointers into the half workspace,
    // uploads the PoW target, and captures the CUDA graph. It touches only this
    // half's stream/buffers plus the shared (read-only after install) resident B.
    ensure_half_workspace(half);

    const uint32_t nbits = ctx.job().target_nbits ? ctx.job().target_nbits
                                                  : target_nbits_.load();
    if (nbits != target_nbits_.load()) {
        target_nbits_.store(nbits);
    }
    // PoW target: nbits * DAF -> 32-byte LE uint32 array on device.
    upload_pow_target(half, nbits);
    check_cuda(cuStreamSynchronize(half.stream), "pow_target sync after upload");
    share_trace("gpu-install",
                "sigma=" + hex_prefix(ctx.job().sigma.data(), ctx.job().sigma.size(), 4) +
                " target=" + nbits_hex(nbits) + " gpu=" + std::to_string(device_index_));

    PearlCapiWorkspaceParams p{};
    p.m = cfg_.m; p.n = cfg_.n; p.k = cfg_.k; p.r = cfg_.r;
    p.bM = cfg_.bM; p.bN = cfg_.bN; p.bK = cfg_.bK; p.cM = cfg_.cM; p.cN = cfg_.cN;
    p.th_num_blocks = cfg_.tensor_hash_num_blocks(static_cast<size_t>(cfg_.m) * cfg_.k);
    p.th_threads = cfg_.tensor_hash_threads;
    p.th_stages = cfg_.tensor_hash_stages;
    p.th_leaves = cfg_.tensor_hash_leaves;
    p.sigma_seed = ctx.sigma_seed();

    // A-side buffers.
    p.A = reinterpret_cast<void*>(half.a);
    p.AHash = reinterpret_cast<void*>(half.a_hash);
    p.Roots = reinterpret_cast<void*>(half.roots);
    p.CommitA = reinterpret_cast<void*>(half.commit_a);
    p.CommitB = reinterpret_cast<void*>(half.commit_b);
    p.EAL = reinterpret_cast<void*>(half.eal);
    p.EAL_fp16 = reinterpret_cast<void*>(half.eal_fp16);
    p.EAR_R_major = reinterpret_cast<void*>(half.ear_r);
    p.EAR_K_major = reinterpret_cast<void*>(half.ear_k);
    p.AxEBL_fp16 = reinterpret_cast<void*>(half.ax_ebl_fp16);
    p.ApEA = reinterpret_cast<void*>(half.apea);
    p.A_scales = reinterpret_cast<void*>(half.a_scales);
    // Pure-miner: C=nullptr — transcript stays in registers, C never materialised.
    p.C = nullptr;
    p.host_signal_sync = reinterpret_cast<void*>(half.sync);
    p.pow_target = reinterpret_cast<void*>(half.pow_target);
    p.pow_key = reinterpret_cast<void*>(half.commit_a);

    // B-side resident buffers (shared with the other half via SigmaContext).
    const auto& bs = ctx.resident();
    p.B = reinterpret_cast<void*>(bs.b());
    p.BHash = reinterpret_cast<void*>(bs.b_hash());
    p.Key = reinterpret_cast<void*>(bs.key());
    p.EBR = reinterpret_cast<void*>(bs.ebr());
    p.EBR_fp16 = reinterpret_cast<void*>(bs.ebr_fp16());
    p.EBL_R_major = reinterpret_cast<void*>(bs.ebl_r());
    p.EBL_K_major = reinterpret_cast<void*>(bs.ebl_k());
    p.EARxBpEB_fp16 = reinterpret_cast<void*>(bs.earx_bpeb());
    p.BpEB = reinterpret_cast<void*>(bs.bpeb());
    p.B_scales = reinterpret_cast<void*>(bs.b_scales());

    gemm_.install_params(half.workspace, p);

    // Header buffers: resize/realloc only when batch size changed.
    half.header_size = static_cast<size_t>(gemm_.host_signal_header_size());
    int batch = matmuls_per_poll_.load();
    if (static_cast<int>(half.host_headers.size()) != batch) {
        for (void* h : half.host_headers) {
            if (h) cudaFreeHost(h);
        }
        half.host_header_storage.assign(batch, std::vector<uint8_t>(half.header_size, 0));
        half.host_headers.resize(batch, nullptr);
        for (int i = 0; i < batch; ++i) {
            CUDA_CHECK(cudaHostAlloc(&half.host_headers[i], half.header_size,
                                     cudaHostAllocPortable));
        }
    }

    // Re-capture graph whenever batch size changes.  Capturing on a fresh stream
    // state is safest.
    half.graph_ready = false;
    half.graph_batch_count = 0;
    try {
        prepare_graph(half);
    } catch (const std::exception& ex) {
        std::fprintf(stderr,
            "[gpu] cuda graph unavailable gpu=%d: %s (using iter_batch)\n",
            device_index_, ex.what());
        std::fprintf(stderr,
            "[gpu] debug: run ./scripts/debug_geforce_v2.sh or set "
            "PEARL_GEMM_DEBUG=1 and retry sigma install\n");
        half.graph_ready = false;
        half.graph_batch_count = 0;
        // Drain async errors from failed graph validation so iter_batch can run.
        if (half.stream) {
            cuStreamSynchronize(half.stream);
        }
        cudaGetLastError();
    }
}

void GpuWorker::prepare_graph(HalfBuffers& half) {
    const int batch = graph_batch_.load();
    if (batch <= 0 || half.workspace == nullptr) return;
    if (bench_no_graph_enabled()) {
        half.graph_ready = false;
        half.graph_batch_count = 0;
        std::fprintf(stderr, "[gpu] cuda graph disabled (PROPMINER_BENCH_NO_GRAPH)\n");
        share_trace("graph-ready", "gpu=" + std::to_string(device_index_) +
                                   " enabled=0 reason=PROPMINER_BENCH_NO_GRAPH");
        return;
    }
    // Reset headers to a known state before capture.
    for (int i = 0; i < batch; ++i) {
        std::memset(half.host_headers[i], 0, half.header_size);
    }
    check_cuda(cuMemsetD8Async(half.sync, 0, 256, half.stream), "sync clear");
    // Extended graph path: seed H2D is NOT captured. Each sub-launch uploads
    // seed_lo to half.seed_dev_ptr then calls iter_batch_graph_launch_ex.
    // The legacy captured-H2D path replays a stale seed for every sub-batch
    // when count > graph_batch (duplicate headers at off+3, verify failures).
    upload_seed_for_graph(half, 0);
    check_cuda(cuStreamSynchronize(half.stream), "graph capture seed sync");
    gemm_.iter_batch_graph_prepare_ex(half.workspace, half.stream,
                                      half.host_headers.data(), batch,
                                      half.seed_dev_ptr);
    half.graph_ready = true;
    half.graph_batch_count = batch;
    std::fprintf(stderr,
        "[gpu] cuda graph captured (seed_dev launch path, batch=%d)\n", batch);
    share_trace("graph-ready",
                "gpu=" + std::to_string(device_index_) +
                " enabled=1 batch=" + std::to_string(batch));
}

void GpuWorker::queue_batch(HalfBuffers& half, uint64_t seed_lo_start, int count) {
    half.batch_seed_start = seed_lo_start;
    half.batch_mined_target_nbits = half.pow_target_nbits;
    const char* half_tag = (&half == &ping_) ? "ping" : "pong";
    const int graph_batch = half.graph_batch_count;
    const bool can_graph = half.graph_ready && graph_batch > 0 &&
                           (count % graph_batch) == 0;
    // Clear host headers (ARC INVARIANT 1).
    for (int i = 0; i < count; ++i) {
        std::memset(half.host_headers[i], 0, half.header_size);
    }

    if (half.batch_start_event) {
        CUDA_CHECK(cudaEventRecord(half.batch_start_event,
                                   reinterpret_cast<cudaStream_t>(half.stream)));
    }

    bool launched_graph = false;
    if (can_graph) {
        try {
            // Async seed conveyor (PROPMINER_ASYNC_SEED=1): upload each
            // sub-batch's seed on seed_copy_stream_ so the 8-byte H2D overlaps
            // the CPU header copy, instead of a synchronous H2D on the compute
            // stream. seed_dev is only rewritten after the consuming sub-batch
            // has fully completed (wait_half_stream below), and each launch
            // orders after its seed copy via cudaStreamWaitEvent.
            const bool async_seed = async_seed_enabled() &&
                                    seed_copy_stream_ && seed_copy_done_event_ &&
                                    half.pinned_seed_host && half.seed_dev_ptr;
            bool seed_preloaded = false;
            if (async_seed) {
                upload_next_seed_async(
                    half, seed_lo_start +
                              static_cast<uint64_t>(count - graph_batch));
                seed_preloaded = true;
            }
            // The captured graph writes winners into the first `graph_batch`
            // pinned slots [0..graph_batch). It only SETS status=1 on a hit and
            // never resets it, so a winner from one sub-launch stays flagged and
            // would be copied into every later group (1 real winner smeared into
            // N duplicates → gpu_cpu_jackpot_mismatch on all but one). Clear the
            // scratch slots before each launch so only THIS launch's hits copy.
            // Run off=0 last: its destination overlaps the scratch region.
            for (int off = count - graph_batch; off >= 0; off -= graph_batch) {
                for (int i = 0; i < graph_batch; ++i) {
                    std::memset(half.host_headers[i], 0, half.header_size);
                }
                check_cuda(cuMemsetD8Async(half.sync, 0, 256, half.stream),
                           "graph sub-batch sync clear");
                if (seed_preloaded) {
                    CUDA_CHECK(cudaStreamWaitEvent(
                        reinterpret_cast<cudaStream_t>(half.stream),
                        seed_copy_done_event_, 0));
                } else {
                    upload_seed_for_graph(
                        half, seed_lo_start + static_cast<uint64_t>(off));
                }
                gemm_.iter_batch_graph_launch_ex(half.workspace, half.stream);
                if (half.sub_batch_done_event) {
                    CUDA_CHECK(cudaEventRecord(
                        half.sub_batch_done_event,
                        reinterpret_cast<cudaStream_t>(half.stream)));
                }
                if (!wait_half_stream(half)) {
                    std::fprintf(stderr,
                        "[gpu] graph sub-batch failed on gpu=%d; disabling "
                        "cuda graph and falling back to iter_batch\n",
                        device_index_);
                    std::fprintf(stderr,
                        "[gpu] debug: PEARL_GEMM_DEBUG=1 ./scripts/debug_geforce_v2.sh "
                        "(phase 4 graph harness isolates TMA+graph)\n");
                    half.graph_ready = false;
                    launched_graph = false;
                    cuStreamSynchronize(half.stream);
                    break;
                }
                const int next_off = off - graph_batch;
                if (async_seed && next_off >= 0) {
                    // Sub-batch done — seed_dev is free. Kick off the next
                    // seed copy so it overlaps the header memcpy below.
                    upload_next_seed_async(
                        half, seed_lo_start + static_cast<uint64_t>(next_off));
                    seed_preloaded = true;
                } else {
                    seed_preloaded = false;
                }
                for (int i = 0; i < graph_batch; ++i) {
                    std::memcpy(half.host_headers[off + i],
                                half.host_headers[i],
                                half.header_size);
                }
            }
            launched_graph = true;
        } catch (const std::exception& ex) {
            std::fprintf(stderr,
                "[gpu] graph launch failed, falling back to iter_batch: %s\n",
                ex.what());
            std::fprintf(stderr,
                "[gpu] debug: ./scripts/debug_geforce_v2.sh --phase 4\n");
            half.graph_ready = false;
            launched_graph = false;
            // Drain poisoned graph work before the direct iter_batch path.
            cuStreamSynchronize(half.stream);
        }
    }

    if (!launched_graph) {
        check_cuda(cuMemsetD8Async(half.sync, 0, 256, half.stream), "sync clear");
        gemm_.iter_batch(half.workspace, half.stream, seed_lo_start,
                         half.host_headers.data(), count);
    }
    share_trace("batch-queue",
                "gpu=" + std::to_string(device_index_) +
                " half=" + half_tag +
                " path=" + (launched_graph ? "graph" : "iter_batch") +
                " batch_seed_start=" + u64_hex(seed_lo_start) +
                " count=" + std::to_string(count) +
                " graph_batch=" + std::to_string(graph_batch) +
                " target=" + nbits_hex(half.batch_mined_target_nbits));
    // Record completion so the host can spin-wait without blocking the driver.
    if (half.batch_done_event) {
        CUDA_CHECK(cudaEventRecord(half.batch_done_event,
                                   reinterpret_cast<cudaStream_t>(half.stream)));
    }
}

void GpuWorker::upload_next_seed_async(HalfBuffers& half, uint64_t seed_lo) {
    if (!seed_copy_stream_ || !half.pinned_seed_host) return;
    *half.pinned_seed_host = seed_lo;
    cudaError_t e = cudaMemcpyAsync(half.seed_dev_ptr, half.pinned_seed_host,
                                    sizeof(uint64_t), cudaMemcpyHostToDevice,
                                    seed_copy_stream_);
    if (e != cudaSuccess) check_cuda(CUDA_ERROR_UNKNOWN, "seed h2d");
    if (seed_copy_done_event_) {
        cudaError_t r = cudaEventRecord(seed_copy_done_event_, seed_copy_stream_);
        if (r != cudaSuccess) check_cuda(CUDA_ERROR_UNKNOWN, "seed event record");
    }
    const char* half_tag = (&half == &ping_) ? "ping" : "pong";
    share_trace("seed-upload",
                "gpu=" + std::to_string(device_index_) +
                " half=" + half_tag +
                " seed_lo=" + u64_hex(seed_lo));
}

void GpuWorker::upload_seed_for_graph(HalfBuffers& half, uint64_t seed_lo) {
    if (!half.pinned_seed_host || !half.seed_dev_ptr) {
        throw std::runtime_error("missing graph seed buffers");
    }
    *half.pinned_seed_host = seed_lo;
    cudaError_t e = cudaMemcpyAsync(half.seed_dev_ptr, half.pinned_seed_host,
                                    sizeof(uint64_t), cudaMemcpyHostToDevice,
                                    reinterpret_cast<cudaStream_t>(half.stream));
    if (e != cudaSuccess) {
        check_cuda(CUDA_ERROR_UNKNOWN,
                   (std::string("graph seed h2d: ") + cudaGetErrorString(e)).c_str());
    }
}

int GpuWorker::sync_and_scan(HalfBuffers& half, int batch) {
    auto winners = scan_winners(half, batch);
    return winners.empty() ? -1 : winners[0];
}

std::vector<int> GpuWorker::scan_winners(HalfBuffers& half, int batch) {
    std::vector<int> winners;
    for (int k = 0; k < batch; ++k) {
        HostSignalHeader hdr(static_cast<const uint8_t*>(half.host_headers[k]),
                             half.header_size);
        if (hdr.status() == 1) {
            std::memcpy(half.host_header_storage[k].data(),
                        half.host_headers[k], half.header_size);
            winners.push_back(k);
            const auto header_target = hdr.header_pow_target();
            const auto batch_words =
                target_le_to_pow_u32(adjusted_pow_target_le(
                    half.batch_mined_target_nbits,
                    cfg_.difficulty_adjustment_factor()));
            bool target_match = true;
            for (size_t i = 0; i < header_target.size(); ++i) {
                if (header_target[i] != batch_words[i]) {
                    target_match = false;
                    break;
                }
            }
            share_trace("gpu-hit",
                        "gpu=" + std::to_string(device_index_) +
                        " batch_idx=" + std::to_string(k) +
                        " nonce=" + std::to_string(half.batch_seed_start +
                                                    static_cast<uint64_t>(k)) +
                        " batch_seed_start=" + u64_hex(half.batch_seed_start) +
                        " tile_m=" + std::to_string(hdr.mma_tile_m()) +
                        " tile_n=" + std::to_string(hdr.mma_tile_n()) +
                        " tile_row=" + std::to_string(hdr.tile_row_coord()) +
                        " tile_col=" + std::to_string(hdr.tile_col_coord()) +
                        " regs=" + std::to_string(hdr.num_registers_per_thread()) +
                        " batch_target=" + nbits_hex(half.batch_mined_target_nbits) +
                        " live_target=" + nbits_hex(target_nbits_.load()) +
                        " header_target_match=" + (target_match ? "1" : "0"));
            share_log("gpu-hit",
                      "nonce=" + std::to_string(half.batch_seed_start +
                                                  static_cast<uint64_t>(k)) +
                      " gpu=" + std::to_string(device_index_) +
                      " batch_idx=" + std::to_string(k) +
                      " target_match=" + (target_match ? "1" : "0"));
        }
    }
    return winners;
}

bool GpuWorker::wait_half_stream(HalfBuffers& half) {
    if (!half.sub_batch_done_event) {
        check_cuda(cuStreamSynchronize(half.stream), "half stream sync");
        return true;
    }
    cudaError_t e = cudaEventQuery(half.sub_batch_done_event);
    if (e == cudaSuccess) return true;
    if (e != cudaErrorNotReady) {
        check_cuda(CUDA_ERROR_UNKNOWN, cudaGetErrorString(e));
        return false;
    }
    const int64_t stall_ms = stall_restart_ms();
    if (stall_ms <= 0) {
        CUDA_CHECK(cudaEventSynchronize(half.sub_batch_done_event));
        return true;
    }
    try {
        return spin_wait_batch_event(half.sub_batch_done_event, watchdog_,
                                     stall_ms, &batch_abort_requested_);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[gpu] sub-batch spin-wait failed: %s\n", ex.what());
        return false;
    }
}

bool GpuWorker::wait_for_batch(HalfBuffers& half, int timeout_ms) {
    if (!half.batch_done_event) {
        check_cuda(cuStreamSynchronize(half.stream), "sync batch");
        return true;
    }
    cudaError_t e = cudaEventQuery(half.batch_done_event);
    if (e == cudaSuccess) return true;
    if (e != cudaErrorNotReady) {
        check_cuda(CUDA_ERROR_UNKNOWN, cudaGetErrorString(e));
        return false;
    }
    if (timeout_ms <= 0) {
        const int64_t stall_ms = stall_restart_ms();
        try {
            // Returns false only on a stall (max_wait_ms exceeded); propagate so
            // the run loop can restart the process instead of block-syncing on a
            // wedged stream (which would hang forever).
            return spin_wait_batch_event(half.batch_done_event, watchdog_,
                                         stall_ms, &batch_abort_requested_);
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "[gpu] batch spin-wait failed: %s\n", ex.what());
            if (stall_ms > 0) return false;  // treat CUDA error as a stall too
            CUDA_CHECK(cudaEventSynchronize(half.batch_done_event));
            return true;
        }
    }
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(timeout_ms);
    while (true) {
        if (watchdog_) watchdog_->heartbeat();
        e = cudaEventQuery(half.batch_done_event);
        if (e == cudaSuccess) return true;
        if (e != cudaErrorNotReady) {
            check_cuda(CUDA_ERROR_UNKNOWN, cudaGetErrorString(e));
            return false;
        }
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

bool GpuWorker::process_share_trigger(const ShareTriggerJob& job) {
    try {
        return process_share_trigger_impl(job);
    } catch (const std::exception& ex) {
        share_log("rebuild-fail", "nonce=" + std::to_string(job.nonce) +
                                  " reason=cuda_error err=" + ex.what());
        return false;
    }
}

bool GpuWorker::process_share_trigger_impl(const ShareTriggerJob& job) {
    HalfBuffers& half = *job.half;
    const SigmaContext& ctx = *job.sigma_ctx;

    HostSignalHeader hdr(job.header);
    share_trace("rebuild-start",
                "nonce=" + std::to_string(job.nonce) +
                " target=" + nbits_hex(job.target_nbits) +
                " gpu=" + std::to_string(device_index_) +
                " batch_seed_start=" + u64_hex(half.batch_seed_start) +
                " batch_idx=" + std::to_string(job.nonce - half.batch_seed_start) +
                " tile_coord=" + std::to_string(hdr.tile_row_coord()) + "," +
                std::to_string(hdr.tile_col_coord()) +
                " regs=" + std::to_string(hdr.num_registers_per_thread()));

    std::vector<uint32_t> a_rows, b_cols;
    try {
        hdr.extract_indices(a_rows, b_cols);
    } catch (...) {
        share_log("rebuild-fail", "nonce=" + std::to_string(job.nonce) +
                                  " reason=extract_indices");
        return false;
    }

    // The intermediate iters in this batch overwrote dA. Re-derive dA for the
    // winning nonce before any D2H, or the proof's A bytes won't match the
    // GPU-side hashes and the pool rejects the share with a_merkle_mismatch.
    // First drain everything queued on this half's stream.
    check_cuda(cuStreamSynchronize(half.stream), "drain pre-trigger");

    int64_t a_bytes = static_cast<int64_t>(cfg_.m) * cfg_.k;
    gemm_.lcg_int7_fill(reinterpret_cast<void*>(half.a), a_bytes,
                        job.nonce, ctx.sigma_seed(), half.stream);
    check_cuda(cuStreamSynchronize(half.stream), "sync A regen");

    // Recompute A's leaf-CV table on the GPU so ShareBuilder can use the fast
    // a_proof_from_leaf_cvs path instead of D2H'ing the full A matrix.
    const auto& bs = ctx.resident();
    gemm_.tensor_hash_leaf_cvs(
        reinterpret_cast<const uint8_t*>(half.a),
        static_cast<uint32_t>(a_bytes),
        reinterpret_cast<uint8_t*>(half.a_hash),
        reinterpret_cast<const uint8_t*>(bs.key()),
        cfg_.tensor_hash_num_blocks(a_bytes),
        cfg_.tensor_hash_threads,
        cfg_.tensor_hash_stages,
        cfg_.tensor_hash_leaves,
        reinterpret_cast<uint8_t*>(half.roots),
        reinterpret_cast<uint8_t*>(half.a_leaf_cvs),
        device_index_,
        half.stream);

    // Commitment chain on device only (Key/AHash/BHash are all device ptrs).
    // This matches pearl_capi_iter and is the pow_key the GEMM kernel keyed on.
    gemm_.commitment_hash_from_merkle_roots(
        reinterpret_cast<const uint8_t*>(half.a_hash),
        reinterpret_cast<const uint8_t*>(bs.b_hash()),
        reinterpret_cast<const uint8_t*>(bs.key()),
        reinterpret_cast<uint8_t*>(half.commit_a),
        reinterpret_cast<uint8_t*>(half.commit_b),
        device_index_,
        half.stream);

    if (!half.pinned_hash_a || !half.pinned_commit_a || !half.pinned_commit_b) {
        share_log("rebuild-fail", "nonce=" + std::to_string(job.nonce) +
                                  " reason=missing_pinned_share_buffers");
        return false;
    }
    check_cuda(cuMemcpyDtoHAsync(half.pinned_hash_a, half.a_hash, 32,
                                 half.stream), "a_hash d2h");
    check_cuda(cuMemcpyDtoHAsync(half.pinned_commit_a, half.commit_a, 32,
                                 half.stream), "commit_a d2h");
    check_cuda(cuMemcpyDtoHAsync(half.pinned_commit_b, half.commit_b, 32,
                                 half.stream), "commit_b d2h");
    // D2H compact proof inputs into pinned staging (single PCIe sync below).
    if (!half.pinned_leaf_cvs || half.a_leaf_cv_bytes == 0) {
        share_log("rebuild-fail", "nonce=" + std::to_string(job.nonce) +
                                  " reason=missing_leaf_cvs");
        return false;
    }
    check_cuda(cuMemcpyDtoHAsync(half.pinned_leaf_cvs, half.a_leaf_cvs,
                                 half.a_leaf_cv_bytes, half.stream),
               "a_leaf_cvs d2h");

    const size_t row_bytes = static_cast<size_t>(cfg_.k);
    if (a_rows.size() * row_bytes > half.pinned_a_slice_bytes) {
        share_log("rebuild-fail", "nonce=" + std::to_string(job.nonce) +
                                  " reason=a_slice_overflow rows=" +
                                  std::to_string(a_rows.size()));
        return false;
    }
    for (size_t i = 0; i < a_rows.size(); ++i) {
        const uint32_t row = a_rows[i];
        check_cuda(cuMemcpyDtoHAsync(
            half.pinned_a_slice + i * row_bytes,
            half.a + row * row_bytes,
            row_bytes, half.stream), "a slice d2h");
    }

    const auto a_leaf_indices =
        compute_leaf_indices_from_rows(a_rows, row_bytes);
    const size_t opened_bytes = a_leaf_indices.size() * 1024;
    if (opened_bytes > half.pinned_opened_leaves_bytes) {
        share_log("rebuild-fail", "nonce=" + std::to_string(job.nonce) +
                                  " reason=opened_leaves_overflow leaves=" +
                                  std::to_string(a_leaf_indices.size()));
        return false;
    }
    for (size_t i = 0; i < a_leaf_indices.size(); ++i) {
        const uint64_t byte_off =
            static_cast<uint64_t>(a_leaf_indices[i]) * 1024;
        check_cuda(cuMemcpyDtoHAsync(
            half.pinned_opened_leaves + i * 1024,
            half.a + byte_off,
            1024, half.stream), "a opened leaf d2h");
    }

    check_cuda(cuMemcpyDtoHAsync(half.pinned_hash_b, ctx.resident().b_hash(), 32,
                                 half.stream), "b_hash d2h");
    check_cuda(cuStreamSynchronize(half.stream), "sync trigger d2h");

    std::vector<uint8_t> a_leaf_cvs_host(half.a_leaf_cv_bytes);
    std::memcpy(a_leaf_cvs_host.data(), half.pinned_leaf_cvs, half.a_leaf_cv_bytes);
    std::vector<uint8_t> a_slice(a_rows.size() * row_bytes);
    std::memcpy(a_slice.data(), half.pinned_a_slice, a_slice.size());
    std::vector<uint8_t> a_opened_leaf_data(opened_bytes);
    std::memcpy(a_opened_leaf_data.data(), half.pinned_opened_leaves, opened_bytes);
    std::array<uint8_t, 32> hash_b{};
    std::array<uint8_t, 32> hash_a{};
    std::array<uint8_t, 32> commit_a{};
    std::array<uint8_t, 32> commit_b{};
    std::memcpy(hash_b.data(), half.pinned_hash_b, 32);
    std::memcpy(hash_a.data(), half.pinned_hash_a, 32);
    std::memcpy(commit_a.data(), half.pinned_commit_a, 32);
    std::memcpy(commit_b.data(), half.pinned_commit_b, 32);

    ShareFound share;
    share.job = ctx.job();
    share.installed_target_nbits = job.target_nbits;
    share.job.target_nbits = share.installed_target_nbits;
    share.sigma_ctx = job.sigma_ctx;
    share.nonce = job.nonce;
    share.tile_row = a_rows.empty() ? 0 : a_rows[0];
    share.tile_col = b_cols.empty() ? 0 : b_cols[0];
    share.mma_tile_m = hdr.mma_tile_m();
    share.mma_tile_n = hdr.mma_tile_n();
    share.a_row_indices = std::move(a_rows);
    share.b_col_indices = std::move(b_cols);
    share.hash_b = hash_b;
    share.hash_a = hash_a;
    share.gpu_commit_a = commit_a;
    share.gpu_commit_b = commit_b;
    share.a_slice = std::move(a_slice);
    share.a_opened_leaf_data = std::move(a_opened_leaf_data);
    share.a_leaf_cvs = std::move(a_leaf_cvs_host);

    auto claimed = ShareBuilder::ComputeClaimedHash(
        share, share.job.job_key.data(), hash_a.data(), hash_b.data());
    share.claimed_hash = claimed;

    share_trace("rebuild-done",
                "nonce=" + std::to_string(share.nonce) +
                " rows=" + std::to_string(share.a_row_indices.size()) +
                " cols=" + std::to_string(share.b_col_indices.size()) +
                " hash_a=" + hex_prefix(share.hash_a.data(), 32) +
                " hash_b=" + hex_prefix(share.hash_b.data(), 32) +
                " commit_a=" + hex_prefix(share.gpu_commit_a.data(), 32) +
                " claimed=" + hex_prefix(share.claimed_hash.data(), 32) +
                " batch_target=" + nbits_hex(share.installed_target_nbits));
    share_log("rebuild-ok",
              "nonce=" + std::to_string(share.nonce) +
              " gpu=" + std::to_string(device_index_) +
              " rows=" + std::to_string(share.a_row_indices.size()) +
              " cols=" + std::to_string(share.b_col_indices.size()) +
              " claimed=" + hex_prefix(share.claimed_hash.data(), 32));

    if (sink_) sink_->submit(share);
    return true;
}

bool GpuWorker::handle_trigger(HalfBuffers& half,
                               const std::shared_ptr<SigmaContext>& ctx,
                               const std::vector<uint8_t>& header,
                               uint64_t nonce) {
    ShareTriggerJob job;
    job.half = &half;
    job.header = header;
    job.nonce = nonce;
    job.target_nbits = half.batch_mined_target_nbits
        ? half.batch_mined_target_nbits
        : target_nbits_.load();
    job.sigma_ctx = ctx;
    if (!job.sigma_ctx) return false;
    return process_share_trigger(job);
}

void GpuWorker::run() {
#if defined(__linux__)
    // Pin this thread to a specific CPU core for cache locality.
    // Round-robin across cores: GPU 0 → core 0, GPU 1 → core 1, etc.
    // This avoids cache thrashing in the hot mining loop.
    {
        long ncpus = sysconf(_SC_NPROCESSORS_ONLN);
        if (ncpus > 0) {
            cpu_set_t mask;
            CPU_ZERO(&mask);
            CPU_SET(device_index_ % static_cast<long>(ncpus), &mask);
            pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
        }
    }
#endif

    HalfBuffers* ping = &ping_;
    HalfBuffers* pong = &pong_;

    uint64_t global_iter = 0;
    int batch = matmuls_per_poll_.load();
    bool first = true;
    int launch_i = 0;
    int triple_warmup = 0;
    std::shared_ptr<SigmaContext> current_sigma;
    auto last_mining_log = std::chrono::steady_clock::now();
    uint64_t gpu_hits_since_log = 0;

    while (!stop_flag_) {
        if (pause_flag_.load()) {
            if (watchdog_) watchdog_->heartbeat();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }

        std::shared_ptr<SigmaContext> new_sigma;
        {
            std::lock_guard<std::mutex> lk(sigma_mtx_);
            if (sigma_ && sigma_ != current_sigma) {
                new_sigma = sigma_;
            }
        }

        const bool async_install = async_job_install_enabled();

        if (new_sigma && async_install && current_sigma) {
            if (async_install_failed(new_sigma)) {
                // Background install threw for this sigma — fall through to the
                // synchronous path below so we never wedge on a stale job.
                share_log("async-install-fallback",
                          "gpu=" + std::to_string(device_index_) +
                          " reason=sync_install_after_async_fail");
            } else {
                // Keep mining the current sigma and hand the new one to the
                // background installer. The swap happens below once its resident
                // B is ready — the mining loop never blocks on the install.
                submit_async_install(new_sigma);
                new_sigma.reset();
            }
        }

        if (new_sigma) {
            // Synchronous install: first job (nothing to mine yet) or async
            // disabled. Blocks the loop while resident B is built — unchanged
            // proven path.
            if (current_sigma) {
                drain_all_halves_for_sigma();
                sync_all_compute_streams();
            }
            current_sigma = new_sigma;
            batch = matmuls_per_poll_.load();
            install_sigma(*current_sigma, *ping);
            install_sigma(*current_sigma, *pong);
            if (triple_buffer_active_) {
                install_sigma(*current_sigma, third_);
            }
            target_dirty_.store(false);
            first = true;
        } else if (async_install) {
            // Swap in a background-installed sigma if one is ready AND it is
            // still the latest sigma the pool published (else it is stale and
            // dropped; the newer one is already being installed).
            std::shared_ptr<SigmaContext> ready = take_async_ready();
            if (ready) {
                std::shared_ptr<SigmaContext> want;
                {
                    std::lock_guard<std::mutex> lk(sigma_mtx_);
                    want = sigma_;
                }
                if (ready == want && ready != current_sigma) {
                    // Same barriers as the synchronous path: drain deferred
                    // shares and all compute streams before rebinding the B
                    // pointers, so no in-flight batch/share reads the old B.
                    drain_all_halves_for_sigma();
                    sync_all_compute_streams();
                    current_sigma = ready;
                    batch = matmuls_per_poll_.load();
                    bind_sigma_to_half(*current_sigma, *ping);
                    bind_sigma_to_half(*current_sigma, *pong);
                    if (triple_buffer_active_) {
                        bind_sigma_to_half(*current_sigma, third_);
                    }
                    target_dirty_.store(false);
                    first = true;
                    share_trace("async-install-swap",
                                "gpu=" + std::to_string(device_index_) +
                                " sigma=" + hex_prefix(current_sigma->job().sigma.data(),
                                                       current_sigma->job().sigma.size(), 4));
                } else if (ready != current_sigma) {
                    // Stale installed sigma (pool already superseded it). It was
                    // never mined and no share references it, so free its
                    // resident-B now instead of leaking a full set.
                    ready->resident().free(ping_.stream);
                }
            }
        }

        if (!new_sigma && target_dirty_.exchange(false)) {
            const uint32_t nbits = target_nbits_.load();
            upload_pow_target_all_halves(nbits);
            sync_all_compute_streams();
        }

        if (!current_sigma) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        if (triple_buffer_active_) {
            HalfBuffers* ring[3] = {&ping_, &pong_, &third_};
            const int num_halves = 3;
            const int pipeline_depth = 2;

            if (first) {
                launch_i = 0;
                triple_warmup = pipeline_depth;
                first = false;
            }

            if (triple_warmup > 0) {
                if (!logged_first_queue_) {
                    logged_first_queue_ = true;
                    std::fprintf(stderr,
                        "[gpu] first batch queued (count=%d, graph=%s, triple=on)\n",
                        batch, ring[launch_i]->graph_ready ? "on" : "off");
                }
                wait_until_half_free(*ring[launch_i]);
                queue_batch(*ring[launch_i], seed_base_ + global_iter, batch);
                global_iter += batch;
                launch_i = (launch_i + 1) % num_halves;
                --triple_warmup;
                continue;
            }

            const int complete_i =
                (launch_i + num_halves - pipeline_depth) % num_halves;
            HalfBuffers& complete = *ring[complete_i];

            uint64_t t0 = now_ms();
            wait_until_half_free(*ring[launch_i]);
            if (stop_flag_) break;
            queue_batch(*ring[launch_i], seed_base_ + global_iter, batch);
            global_iter += batch;
            total_iters_ += batch;
            launch_i = (launch_i + 1) % num_halves;

            if (!wait_for_batch(complete, 0)) {
                const long long stall_ms = stall_restart_ms();
                const bool aborted = batch_abort_requested_.exchange(false);
                std::fprintf(stderr,
                    "[gpu] STALL: batch on gpu=%d exceeded %lld ms with no progress "
                    "(wedged CUDA stream%s). Exiting for supervisor restart "
                    "(set PROPMINER_STALL_RESTART_MS=0 to disable).\n",
                    device_index_, stall_ms,
                    aborted ? "; health-monitor abort" : "");
                std::fflush(stderr);
                std::_Exit(42);
            }
            batch_abort_requested_.store(false);

            const auto winners = scan_winners(complete, batch);
            gpu_hits_since_log += winners.size();
            const bool defer_share = defer_share_gpu_enabled();
            for (int winner : winners) {
                const uint64_t nonce =
                    complete.batch_seed_start + static_cast<uint64_t>(winner);
                const auto& header = complete.host_header_storage[winner];
                if (defer_share) {
                    enqueue_share_trigger(complete, nonce, header, current_sigma);
                } else {
                    handle_trigger(complete, current_sigma, header, nonce);
                }
            }
            if (watchdog_) watchdog_->heartbeat();

            if (share_trace_enabled()) {
                const auto now = std::chrono::steady_clock::now();
                if (now - last_mining_log >= std::chrono::seconds(30)) {
                    share_trace("mining-heartbeat",
                                "gpu=" + std::to_string(device_index_) +
                                " iters=" + std::to_string(total_iters_.load()) +
                                " batch=" + std::to_string(batch) +
                                " target=" + nbits_hex(target_nbits_.load()) +
                                " gpu_hits_30s=" + std::to_string(gpu_hits_since_log) +
                                " hashrate_ths=" + std::to_string(hashrate_.load() / 1e12) +
                                " triple=on");
                    gpu_hits_since_log = 0;
                    last_mining_log = now;
                }
            }

            double ms = 0.0;
            bool gpu_timed = false;
            if (complete.batch_start_event && complete.batch_done_event) {
                float gpu_ms = 0.0f;
                cudaError_t te = cudaEventElapsedTime(
                    &gpu_ms, complete.batch_start_event, complete.batch_done_event);
                if (te == cudaSuccess && gpu_ms > 0.0f) {
                    ms = static_cast<double>(gpu_ms);
                    gpu_timed = true;
                }
            }
            if (!gpu_timed) {
                ms = static_cast<double>(now_ms() - t0);
            }
            last_iter_ms_.store(ms);
            if (ms > 0.0) {
                const double sec = ms / 1000.0;
                const double ips = static_cast<double>(batch) / sec;
                const double tmads = mining_mac_volume(cfg_) * ips / 1e12;
                const double tiles_per_iter =
                    static_cast<double>(cfg_.m / cfg_.bM) * (cfg_.n / cfg_.bN);
                const double tiles_per_sec = ips * tiles_per_iter;
                const double hr = tiles_per_sec * cfg_.difficulty_adjustment_factor();
                tmads_per_sec_.store(tmads);
                hashrate_.store(hr);
                if (!logged_first_hashrate_ && hr > 0.0) {
                    logged_first_hashrate_ = true;
                    const HashrateMetrics metrics = hashrate_metrics_from_rates(
                        cfg_, tmads, hr, ms, batch, graph_batch_.load());
                    SystemTelemetry telemetry;
                    SystemSnapshot sys = telemetry.sample(device_index_, 0);
                    const char* stratum_env = std::getenv("PROPMINER_USE_STRATUM");
                    if (!stratum_env || stratum_env[0] != '0') {
                        sys.pool_share_diff = resolve_stratum_share_diff_double();
                    }
                    std::fprintf(stderr,
                        "[gpu] first batch completed in %.0f ms (%s, triple=on)\n",
                        ms, gpu_timed ? "gpu" : "wall");
                    print_hashrate_metrics_line(stderr, "[gpu] ", metrics, &sys);
                    print_hashrate_health(stderr, "[gpu] ", metrics, &sys);
                }
            }
            continue;
        }

        HalfBuffers* cur = ping;
        HalfBuffers* other = pong;

        if (first) {
            if (!logged_first_queue_) {
                logged_first_queue_ = true;
                std::fprintf(stderr,
                    "[gpu] first batch queued (count=%d, graph=%s)\n",
                    batch, cur->graph_ready ? "on" : "off");
            }
            wait_until_half_free(*cur);
            queue_batch(*cur, seed_base_ + global_iter, batch);
            global_iter += batch;
            first = false;
            std::swap(ping, pong);
            continue;
        }

        uint64_t t0 = now_ms();
        wait_until_half_free(*cur);
        if (stop_flag_) break;
        queue_batch(*cur, seed_base_ + global_iter, batch);
        global_iter += batch;
        total_iters_ += batch;

        // Spin-wait for the previously-launched batch on `other`. A false
        // return means the batch exceeded PROPMINER_STALL_RESTART_MS — the GPU
        // stream is wedged (100% util, no progress) and cannot be recovered
        // in-process. Exit so the supervisor relaunches with a clean context.
        if (!wait_for_batch(*other, 0)) {
            const long long stall_ms = stall_restart_ms();
            const bool aborted = batch_abort_requested_.exchange(false);
            std::fprintf(stderr,
                "[gpu] STALL: batch on gpu=%d exceeded %lld ms with no progress "
                "(wedged CUDA stream%s). Exiting for supervisor restart "
                "(set PROPMINER_STALL_RESTART_MS=0 to disable).\n",
                device_index_, stall_ms,
                aborted ? "; health-monitor abort" : "");
            std::fflush(stderr);
            std::_Exit(42);
        }
        batch_abort_requested_.store(false);
        const auto winners = scan_winners(*other, batch);
        gpu_hits_since_log += winners.size();
        const bool defer_share = defer_share_gpu_enabled();
        for (int winner : winners) {
            const uint64_t nonce =
                other->batch_seed_start + static_cast<uint64_t>(winner);
            const auto& header = other->host_header_storage[winner];
            if (defer_share) {
                enqueue_share_trigger(*other, nonce, header, current_sigma);
            } else {
                handle_trigger(*other, current_sigma, header, nonce);
            }
        }
        if (watchdog_) watchdog_->heartbeat();

        if (share_trace_enabled()) {
            const auto now = std::chrono::steady_clock::now();
            if (now - last_mining_log >= std::chrono::seconds(30)) {
                share_trace("mining-heartbeat",
                            "gpu=" + std::to_string(device_index_) +
                            " iters=" + std::to_string(total_iters_.load()) +
                            " batch=" + std::to_string(batch) +
                            " target=" + nbits_hex(target_nbits_.load()) +
                            " gpu_hits_30s=" + std::to_string(gpu_hits_since_log) +
                            " hashrate_ths=" + std::to_string(hashrate_.load() / 1e12));
                gpu_hits_since_log = 0;
                last_mining_log = now;
            }
        }

        // GPU batch time via cudaEventElapsedTime (excludes host share work).
        double ms = 0.0;
        bool gpu_timed = false;
        if (other->batch_start_event && other->batch_done_event) {
            float gpu_ms = 0.0f;
            cudaError_t te = cudaEventElapsedTime(
                &gpu_ms, other->batch_start_event, other->batch_done_event);
            if (te == cudaSuccess && gpu_ms > 0.0f) {
                ms = static_cast<double>(gpu_ms);
                gpu_timed = true;
            }
        }
        if (!gpu_timed) {
            uint64_t t1 = now_ms();
            ms = static_cast<double>(t1 - t0);
        }
        last_iter_ms_.store(ms);
        if (ms > 0.0) {
            const double sec = ms / 1000.0;
            const double ips = static_cast<double>(batch) / sec;
            const double tmads = mining_mac_volume(cfg_) * ips / 1e12;
            const double tiles_per_iter =
                static_cast<double>(cfg_.m / cfg_.bM) * (cfg_.n / cfg_.bN);
            const double tiles_per_sec = ips * tiles_per_iter;
            const double hr = tiles_per_sec * cfg_.difficulty_adjustment_factor();
            tmads_per_sec_.store(tmads);
            hashrate_.store(hr);
            if (!logged_first_hashrate_ && hr > 0.0) {
                logged_first_hashrate_ = true;
                const HashrateMetrics metrics = hashrate_metrics_from_rates(
                    cfg_, tmads, hr, ms, batch, graph_batch_.load());
                SystemTelemetry telemetry;
                SystemSnapshot sys = telemetry.sample(device_index_, 0);
                const char* stratum_env = std::getenv("PROPMINER_USE_STRATUM");
                if (!stratum_env || stratum_env[0] != '0') {
                    sys.pool_share_diff = resolve_stratum_share_diff_double();
                }
                std::fprintf(stderr,
                    "[gpu] first batch completed in %.0f ms (%s)\n",
                    ms, gpu_timed ? "gpu" : "wall");
                print_hashrate_metrics_line(stderr, "[gpu] ", metrics, &sys);
                print_hashrate_health(stderr, "[gpu] ", metrics, &sys);
            }
        }

        std::swap(ping, pong);
    }
}

} // namespace pearl
