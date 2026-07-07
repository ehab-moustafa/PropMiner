#include "gpu_worker.h"
#include "hashrate_metrics.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>

#include <cuda_runtime.h>

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
    bool spin_wait_batch_event(cudaEvent_t event, Watchdog* watchdog) {
        constexpr int kSpinTight = 4096;
        constexpr int kYieldEvery = 64;
        int spins = 0;
        while (true) {
            cudaError_t e = cudaEventQuery(event);
            if (e == cudaSuccess) return true;
            if (e != cudaErrorNotReady) {
                throw std::runtime_error(
                    std::string("cudaEventQuery: ") + cudaGetErrorString(e));
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

void GpuWorker::HalfBuffers::allocate(const MiningConfig& cfg, int device_id, CUstream s) {
    stream = s;
    size_t a_bytes = static_cast<size_t>(cfg.m) * cfg.k;
    // In pure-miner mode the kernel accumulates the transcript in registers and
    // never materialises C in HBM.  We still allocate a small C buffer because
    // the C API install_params expects a non-null pointer; it is not read.
    size_t c_bytes = static_cast<size_t>(cfg.m) * cfg.n * sizeof(uint16_t);
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
    check(cuMemAlloc(&c, c_bytes), "c alloc");
    check(cuMemAlloc(&sync, 256), "sync alloc");
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

    ping_.allocate(cfg_, device_index_, ping_.stream);
    pong_.allocate(cfg_, device_index_, pong_.stream);

    // Per-GPU nonce-space partition: top 16 bits = gpu_index, next 16 bits =
    // time-based entropy.  This lets multiple GPUs mine disjoint ranges without
    // central coordination.
    seed_base_ = (static_cast<uint64_t>(gpu_index_) << 48) |
                 (static_cast<uint64_t>(now_ms() & 0xFFFF) << 32);
}

GpuWorker::~GpuWorker() {
    stop();
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
    running_ = true;
    start_share_gpu_thread();
    thread_ = std::thread(&GpuWorker::run, this);
}

void GpuWorker::stop() {
    stop_flag_ = true;
    if (thread_.joinable()) thread_.join();
    stop_share_gpu_thread();
    running_ = false;
}

bool GpuWorker::defer_share_gpu_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("PROPMINER_DEFER_SHARE_GPU");
        cached = (env && env[0] && env[0] != '0') ? 1 : 0;
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
}

void GpuWorker::wait_until_half_free(HalfBuffers& half) {
    if (half.share_jobs_pending.load(std::memory_order_acquire) == 0) return;
    std::unique_lock<std::mutex> lk(share_mtx_);
    share_done_cv_.wait(lk, [&] {
        return half.share_jobs_pending.load(std::memory_order_acquire) == 0;
    });
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

void GpuWorker::install_sigma(SigmaContext& ctx, HalfBuffers& half) {
    // Workspace: allocate once per half, reused across σ rotations.  Keep both
    // noise_A and noise_B scratchpads so σ-install can reuse the same pool.
    if (!half.workspace) {
        void* ws = nullptr;
        int rc = pearl_capi_workspace_alloc(cfg_.m, cfg_.n, cfg_.k, cfg_.r,
                                            1, 1, &ws, half.stream);
        if (rc != 0 || !ws) throw std::runtime_error("workspace_alloc failed");
        half.workspace = ws;
    }

    // Ensure resident B state is allocated and computed on device.  Idempotent;
    // on the second half it reuses the already-installed buffers owned by ctx.
    ctx.install(half.stream, half.workspace, device_index_, merkle_copy_stream_);

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
    // In pure-miner mode the kernel accumulates the transcript in registers and
    // never materialises C in HBM.  The buffer is still allocated so install_params
    // receives a non-null pointer, but it is not read.
    p.C = reinterpret_cast<void*>(half.c);
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
    } catch (const std::exception&) {
        half.graph_ready = false;
        half.graph_batch_count = 0;
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
                upload_seed_for_graph(
                    half, seed_lo_start + static_cast<uint64_t>(off));
                gemm_.iter_batch_graph_launch_ex(half.workspace, half.stream);
                check_cuda(cuStreamSynchronize(half.stream),
                           "graph sub-batch sync");
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
            half.graph_ready = false;
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
        try {
            return spin_wait_batch_event(half.batch_done_event, watchdog_);
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "[gpu] batch spin-wait failed: %s\n", ex.what());
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
    HalfBuffers* ping = &ping_;
    HalfBuffers* pong = &pong_;

    uint64_t global_iter = 0;
    int batch = matmuls_per_poll_.load();
    bool first = true;
    std::shared_ptr<SigmaContext> current_sigma;
    auto last_mining_log = std::chrono::steady_clock::now();
    uint64_t gpu_hits_since_log = 0;

    while (!stop_flag_) {
        std::shared_ptr<SigmaContext> new_sigma;
        {
            std::lock_guard<std::mutex> lk(sigma_mtx_);
            if (sigma_ && sigma_ != current_sigma) {
                new_sigma = sigma_;
            }
        }

        if (new_sigma) {
            if (defer_share_gpu_enabled()) {
                wait_until_half_free(*ping);
                wait_until_half_free(*pong);
            }
            current_sigma = new_sigma;
            batch = matmuls_per_poll_.load();
            install_sigma(*current_sigma, *ping);
            install_sigma(*current_sigma, *pong);
            target_dirty_.store(false);
            first = true;
        } else if (target_dirty_.exchange(false)) {
            const uint32_t nbits = target_nbits_.load();
            upload_pow_target_both_halves(nbits);
            check_cuda(cuStreamSynchronize(ping_.stream), "vardiff pow_target sync ping");
            check_cuda(cuStreamSynchronize(pong_.stream), "vardiff pow_target sync pong");
        }

        if (!current_sigma) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
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

        // Spin-wait for the previously-launched batch on `other`.
        if (!wait_for_batch(*other, 0)) {
            check_cuda(cuStreamSynchronize(other->stream), "sync batch");
        }
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
            }
        }

        std::swap(ping, pong);
    }
}

} // namespace pearl
