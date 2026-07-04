#include "gpu_worker.h"

#include <chrono>
#include <cstring>
#include <thread>

#include <cuda_runtime.h>

#include "host_signal_header.h"
#include "pearl_blake3.h"

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

    std::array<uint8_t, 32> nbits_to_target_le(uint32_t nbits) {
        int exp = static_cast<int>(nbits >> 24);
        uint32_t mant = nbits & 0xFFFFFFu;
        std::array<uint8_t, 32> t{};
        for (int i = 0; i < 3; ++i) {
            int pos = 32 - exp + i;
            if (pos >= 0 && pos < 32) {
                t[pos] = static_cast<uint8_t>(mant >> (8 * (2 - i)));
            }
        }
        return t;
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

    // Pre-create the timing event so the hot path never allocates.
    if (!batch_done_event) {
        cudaError_t e = cudaEventCreateWithFlags(
            reinterpret_cast<cudaEvent_t*>(&batch_done_event),
            cudaEventDisableTiming);
        if (e != cudaSuccess) batch_done_event = nullptr;
    }

    (void)device_id;
}

void GpuWorker::HalfBuffers::free() {
    auto f = [](CUdeviceptr& p) { if (p) { cuMemFree(p); p = 0; } };
    f(a); f(a_hash); f(roots); f(commit_a); f(commit_b);
    f(eal); f(eal_fp16); f(ear_r); f(ear_k); f(ax_ebl_fp16);
    f(apea); f(a_scales); f(c); f(sync); f(pow_target); f(a_leaf_cvs);
    f(seed_dev);
    seed_dev_ptr = nullptr;
    if (workspace) {
        pearl_capi_workspace_free(workspace, stream);
        workspace = nullptr;
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

    // CPU seed generator: zero hashing math, only nonce/seed pre-generation.
    seed_gen_ = std::make_unique<SeedGenerator>(seed_base_);
    seed_gen_->start(1024);
}

GpuWorker::~GpuWorker() {
    stop();
    seed_gen_.reset();
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
    thread_ = std::thread(&GpuWorker::run, this);
}

void GpuWorker::stop() {
    stop_flag_ = true;
    if (thread_.joinable()) thread_.join();
    running_ = false;
}

void GpuWorker::set_sigma(std::shared_ptr<SigmaContext> ctx) {
    std::lock_guard<std::mutex> lk(sigma_mtx_);
    sigma_ = std::move(ctx);
}

void GpuWorker::set_target_nbits(uint32_t nbits) {
    target_nbits_.store(nbits);
}

void GpuWorker::set_matmuls_per_poll(int mpp) {
    if (mpp > 0) matmuls_per_poll_.store(mpp);
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
    ctx.install(half.stream, half.workspace, device_index_);

    // PoW target: nbits -> 32-byte LE uint32 array on device.
    auto target = nbits_to_target_le(target_nbits_.load());
    std::array<uint32_t, 8> target_u32{};
    for (int i = 0; i < 8; ++i) {
        std::memcpy(&target_u32[i], target.data() + i * 4, 4);
    }
    check_cuda(cuMemcpyHtoDAsync(half.pow_target, target_u32.data(), 32, half.stream),
               "pow_target h2d");

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
    int batch = matmuls_per_poll_.load();
    if (batch <= 0 || half.workspace == nullptr) return;
    // Reset headers to a known state before capture.
    for (int i = 0; i < batch; ++i) {
        std::memset(half.host_headers[i], 0, half.header_size);
    }
    check_cuda(cuMemsetD8Async(half.sync, 0, 256, half.stream), "sync clear");
    // Use the extended graph prepare so the seed copy is not captured; seeds
    // are uploaded asynchronously on seed_copy_stream_ while the other half
    // runs the previous batch.
    gemm_.iter_batch_graph_prepare_ex(half.workspace, half.stream,
                                      half.host_headers.data(), batch,
                                      half.seed_dev_ptr);
    half.graph_ready = true;
    half.graph_batch_count = batch;
}

void GpuWorker::queue_batch(HalfBuffers& half, uint64_t seed_lo_start, int count) {
    // Clear host headers.
    for (int i = 0; i < count; ++i) {
        std::memset(half.host_headers[i], 0, half.header_size);
    }
    // Clear dSync.
    check_cuda(cuMemsetD8Async(half.sync, 0, 256, half.stream), "sync clear");

    if (half.graph_ready && count == half.graph_batch_count) {
        // Extended graph path: seed was uploaded to half.seed_dev on
        // seed_copy_stream_ while the previous batch ran.  Make the compute
        // stream wait for the upload, then launch the graph.
        if (seed_copy_done_event_) {
            cudaError_t w = cudaStreamWaitEvent(
                reinterpret_cast<cudaStream_t>(half.stream),
                seed_copy_done_event_, 0);
            if (w != cudaSuccess) check_cuda(CUDA_ERROR_UNKNOWN, "stream wait event");
        }
        gemm_.iter_batch_graph_launch_ex(half.workspace, half.stream);
    } else {
        // Fallback / dynamic batch size.  The batched C API entry point keeps
        // the per-iter CPU launch cost low even without graphs.
        gemm_.iter_batch(half.workspace, half.stream, seed_lo_start,
                         half.host_headers.data(), count);
    }
    // Record an event so the host can sleep/poll instead of blocking the CPU.
    if (half.batch_done_event) {
        CUDA_CHECK(cudaEventRecord(half.batch_done_event,
                                   reinterpret_cast<cudaStream_t>(half.stream)));
    }
}

void GpuWorker::upload_next_seed_async(HalfBuffers& half, uint64_t seed_lo) {
    if (!seed_copy_stream_) return;
    // Upload the seed for the *next* batch on the dedicated copy stream.
    // This overlaps with GEMM execution on the other half's compute stream.
    cudaError_t e = cudaMemcpyAsync(half.seed_dev_ptr, &seed_lo,
                                    sizeof(uint64_t), cudaMemcpyHostToDevice,
                                    seed_copy_stream_);
    if (e != cudaSuccess) check_cuda(CUDA_ERROR_UNKNOWN, "seed h2d");
    if (seed_copy_done_event_) {
        cudaError_t r = cudaEventRecord(seed_copy_done_event_, seed_copy_stream_);
        if (r != cudaSuccess) check_cuda(CUDA_ERROR_UNKNOWN, "seed event record");
    }
}

int GpuWorker::sync_and_scan(HalfBuffers& half, int batch) {
    check_cuda(cuStreamSynchronize(half.stream), "sync batch");
    for (int k = 0; k < batch; ++k) {
        HostSignalHeader hdr(half.host_header_storage[k]);
        if (hdr.status() == 1) return k;
    }
    return -1;
}

bool GpuWorker::wait_for_batch(HalfBuffers& half, int timeout_ms) {
    if (!half.batch_done_event) {
        check_cuda(cuStreamSynchronize(half.stream), "sync batch");
        return true;
    }
    cudaError_t e = cudaEventQuery(half.batch_done_event);
    if (e == cudaSuccess) return true;
    if (e == cudaErrorNotReady) {
        // Brief nap to let the GPU run without spinning the CPU core.
        // For long batches we wait on the event; for very fast batches we just
        // poll to keep latency low.
        if (timeout_ms > 5) {
            cudaError_t w = cudaEventSynchronize(half.batch_done_event);
            return w == cudaSuccess;
        }
        return false;
    }
    if (e != cudaSuccess) {
        check_cuda(CUDA_ERROR_UNKNOWN, cudaGetErrorString(e));
        return false;
    }
    return false;
}

bool GpuWorker::handle_trigger(HalfBuffers& half, const SigmaContext& ctx,
                               const std::vector<uint8_t>& header,
                               uint64_t nonce) {
    HostSignalHeader hdr(header);
    std::vector<uint32_t> a_rows, b_cols;
    try {
        hdr.extract_indices(a_rows, b_cols);
    } catch (...) {
        return false;
    }

    // The intermediate iters in this batch overwrote dA. Re-derive dA for the
    // winning nonce before any D2H, or the proof's A bytes won't match the
    // GPU-side hashes and the pool rejects the share with a_merkle_mismatch.
    // First drain everything queued on this half's stream.
    check_cuda(cuStreamSynchronize(half.stream), "drain pre-trigger");

    int64_t a_bytes = static_cast<int64_t>(cfg_.m) * cfg_.k;
    gemm_.lcg_int7_fill(reinterpret_cast<void*>(half.a), a_bytes,
                        nonce, ctx.sigma_seed(), half.stream);
    check_cuda(cuStreamSynchronize(half.stream), "sync A regen");

    // Recompute A's leaf-CV table on the GPU so ShareBuilder can use the fast
    // a_proof_from_leaf_cvs path instead of D2H'ing the full A matrix.
    gemm_.tensor_hash_leaf_cvs(
        reinterpret_cast<const uint8_t*>(half.a),
        static_cast<uint32_t>(a_bytes),
        reinterpret_cast<uint8_t*>(half.a_hash),
        ctx.job().job_key.data(),
        cfg_.tensor_hash_num_blocks(a_bytes),
        cfg_.tensor_hash_threads,
        cfg_.tensor_hash_stages,
        cfg_.tensor_hash_leaves,
        reinterpret_cast<uint8_t*>(half.roots),
        reinterpret_cast<uint8_t*>(half.a_leaf_cvs),
        device_index_,
        half.stream);

    // D2H the compact leaf-CV table plus the opened A rows.
    std::vector<uint8_t> a_leaf_cvs_host(half.a_leaf_cv_bytes);
    check_cuda(cuMemcpyDtoHAsync(a_leaf_cvs_host.data(), half.a_leaf_cvs,
                                 half.a_leaf_cv_bytes, half.stream),
               "a_leaf_cvs d2h");

    size_t row_bytes = static_cast<size_t>(cfg_.k);
    std::vector<uint8_t> a_slice(a_rows.size() * row_bytes);
    for (size_t i = 0; i < a_rows.size(); ++i) {
        uint32_t row = a_rows[i];
        check_cuda(cuMemcpyDtoHAsync(
            a_slice.data() + i * row_bytes,
            half.a + row * row_bytes,
            row_bytes, half.stream), "a slice d2h");
    }
    check_cuda(cuStreamSynchronize(half.stream), "sync trigger d2h");

    // BHash from resident state.
    std::array<uint8_t, 32> hash_b{};
    check_cuda(cuMemcpyDtoHAsync(hash_b.data(), ctx.resident().b_hash(), 32,
                                 half.stream), "b_hash d2h");
    check_cuda(cuStreamSynchronize(half.stream), "sync b_hash");

    ShareFound share;
    share.job = ctx.job();
    share.nonce = nonce;
    share.tile_row = a_rows.empty() ? 0 : a_rows[0];
    share.tile_col = b_cols.empty() ? 0 : b_cols[0];
    share.mma_tile_m = hdr.mma_tile_m();
    share.mma_tile_n = hdr.mma_tile_n();
    share.a_row_indices = std::move(a_rows);
    share.b_col_indices = std::move(b_cols);
    share.hash_b = hash_b;
    share.a_slice = std::move(a_slice);
    share.a_leaf_cvs = std::move(a_leaf_cvs_host);

    if (sink_) sink_->submit(share);
    return true;
}

void GpuWorker::run() {
    HalfBuffers* ping = &ping_;
    HalfBuffers* pong = &pong_;

    uint64_t global_iter = 0;
    int batch = matmuls_per_poll_.load();
    bool first = true;
    std::shared_ptr<SigmaContext> current_sigma;

    while (!stop_flag_) {
        std::shared_ptr<SigmaContext> new_sigma;
        {
            std::lock_guard<std::mutex> lk(sigma_mtx_);
            if (sigma_ && sigma_ != current_sigma) {
                new_sigma = sigma_;
            }
        }

        if (new_sigma) {
            current_sigma = new_sigma;
            batch = matmuls_per_poll_.load();
            install_sigma(*current_sigma, *ping);
            install_sigma(*current_sigma, *pong);
            first = true;
        }

        if (!current_sigma) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        HalfBuffers* cur = ping;
        HalfBuffers* other = pong;

        if (first) {
            queue_batch(*cur, seed_base_ + global_iter, batch);
            // Start uploading the seed for the next batch immediately so the
            // conveyor belt is primed.
            upload_next_seed_async(*other, seed_base_ + global_iter + batch);
            global_iter += batch;
            first = false;
            std::swap(ping, pong);
            continue;
        }

        uint64_t t0 = now_ms();
        queue_batch(*cur, seed_base_ + global_iter, batch);
        global_iter += batch;
        total_iters_ += batch;

        // Upload the seed for the *next* batch on the copy stream while the
        // current batch runs on the compute stream.  This is the PCIe Gen5
        // conveyor belt: zero hashing on CPU, only seed movement.
        upload_next_seed_async(*other, seed_base_ + global_iter + batch);

        // Wait for the previously-launched batch on `other` with a short sleep
        // so the host does not burn a core.  We still want low latency in case
        // a share is found.
        wait_for_batch(*other, /*timeout_ms=*/10);

        int winner = sync_and_scan(*other, batch);
        if (winner >= 0) {
            handle_trigger(*other, *current_sigma,
                           other->host_header_storage[winner],
                           seed_base_ + global_iter - 2 * batch + winner);
        }

        uint64_t t1 = now_ms();
        double ms = static_cast<double>(t1 - t0);
        last_iter_ms_.store(ms);
        // Each iter is m*n*k MACs; hashrate in MAC/s.
        double ops_per_iter = static_cast<double>(cfg_.m) * cfg_.n * cfg_.k * 2.0;
        hashrate_.store(ops_per_iter / (ms / 1000.0));

        std::swap(ping, pong);
    }
}

} // namespace pearl
