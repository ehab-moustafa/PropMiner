#include "multiplexer.h"
#include "propminer.h"
#include "cuda_driver.h"
#include "work_queue.h"
#include "result_buffer.h"

#include <cuda_runtime.h>
#include <driver_functions.h>

// Helper to check CUDA Driver API errors
#define CU_CHECK(call)                                            \
    do {                                                          \
        CUresult _err = (call);                                   \
        if (_err != CUDA_SUCCESS) {                               \
            const char* _msg = nullptr;                           \
            cuGetErrorString(_err, &_msg);                        \
            fprintf(stderr, "[cuda] %s:%d %s -> %s (%d)\n",      \
                    __FILE__, __LINE__, #call, _msg, _err);       \
            if (gpu.matrix_data_ptr) cuMemFree(gpu.matrix_data_ptr); \
            cuda_driver_destroy_gpu(&gpu);                        \
            return;                                               \
        }                                                         \
    } while (0)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <thread>
#include <cmath>
#include <condition_variable>

namespace propminer {

Multiplexer::Multiplexer(const MinerConfig& config)
    : config_(config) {
    stratum_ = std::make_unique<StratumClient>();
}

Multiplexer::~Multiplexer() {
    stop();
}

int Multiplexer::init() {
    // CUDA was already initialized by main.cpp via cuda_driver_init()

    // Discover GPUs. Use the CUDA Runtime API because on WSL2 containers the
    // driver API returns error 3 while the runtime API works.
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "[multiplexer] cudaGetDeviceCount failed: %d (%s)\n",
                static_cast<int>(err), cudaGetErrorString(err));
        return -1;
    }

    if (device_count == 0) {
        fprintf(stderr, "[multiplexer] No CUDA devices found\n");
        return -1;
    }

    // Determine which GPUs to use
    if (config_.gpu_indices.empty()) {
        for (int i = 0; i < device_count; i++)
            config_.gpu_indices.push_back(i);
    }

    gpu_count_ = static_cast<int>(config_.gpu_indices.size());
    stats_.resize(gpu_count_);

    // Print available GPUs
    fprintf(stderr, "\n");
    fprintf(stderr, "  %-3s  %-24s  %-10s  %s\n", "ID", "Name", "VRAM(GB)", "SM");
    fprintf(stderr, "  %s\n", "------------------------------------------");
    for (int i = 0; i < device_count; i++) {
        CUdevice dev;
        cuDeviceGet(&dev, i);

        char name[256];
        cuDeviceGetName(name, sizeof(name), dev);

        size_t mem = 0;
        cuDeviceTotalMem(&mem, dev);

        int sm = 0, sm_minor = 0;
        cuDeviceGetAttribute(&sm, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
        cuDeviceGetAttribute(&sm_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);

        char active = config_.gpu_indices.empty() ? '*'
                       : (std::find(config_.gpu_indices.begin(),
                                    config_.gpu_indices.end(), i) != config_.gpu_indices.end()) ? '*' : ' ';

        fprintf(stderr, "  %c%d   %-24s  %6.1f     sm_%d%d\n",
                active, i, name, mem / (1024.0 * 1024.0 * 1024.0), sm, sm_minor);
    }
    fprintf(stderr, "\n");

    // Initialize per-GPU stats
    for (int i = 0; i < gpu_count_; i++) {
        int idx = config_.gpu_indices[i];
        CUdevice dev;
        cuDeviceGet(&dev, idx);
        char name[256];
        cuDeviceGetName(name, sizeof(name), dev);

        stats_[i] = std::make_shared<GpuStats>();
        stats_[i]->index = idx;
        stats_[i]->name = name;
    }

    // Connect to stratum pool
    if (!stratum_->connect(config_.pool_host, config_.pool_port,
                           config_.wallet, config_.worker, config_.password)) {
        fprintf(stderr, "[multiplexer] Failed to connect to pool\n");
        return -1;
    }

    // Set up callbacks
    stratum_->set_callbacks(
        [this](const JobAssignment& job) {
            fprintf(stderr, "[multiplexer] Job dispatched to %d GPUs\n", gpu_count_);
            // Notify GPU workers that a new job is available
            {
                std::lock_guard<std::mutex> lock(job_notify_mutex_);
                new_job_ = true;
            }
            job_notify_cv_.notify_all();
        },
        [this](const ShareResult& result) {
            if (result.accepted) {
                if (result.gpu_index >= 0 && result.gpu_index < gpu_count_)
                    stats_[result.gpu_index]->accepted++;
                else
                    for (auto& s : stats_) s->accepted++;
                fprintf(stderr, "[multiplexer] Share accepted%s (gpu=%d)\n",
                        result.is_block_find ? " *** BLOCK FOUND ***" : "",
                        result.gpu_index);
            } else {
                if (result.gpu_index >= 0 && result.gpu_index < gpu_count_)
                    stats_[result.gpu_index]->rejected++;
                else
                    for (auto& s : stats_) s->rejected++;
            }
        },
        [this](const DifficultyAdjust& adj) {
            fprintf(stderr, "[multiplexer] Difficulty adjusted: %f\n", adj.new_difficulty);
        }
    );

    fprintf(stderr, "[multiplexer] Initialized %d GPU(s)\n", gpu_count_);
    return 0;
}

void Multiplexer::start() {
    running_.store(true);

    // Start stratum mining
    stratum_->start_mining();

    // Start GPU worker threads
    for (int i = 0; i < gpu_count_; i++) {
        gpu_threads_.emplace_back(&Multiplexer::gpu_worker, this, i, stats_[i]);
    }

    // Start monitor thread
    monitor_thread_ = std::make_unique<std::thread>(&Multiplexer::monitor_thread_func, this);
}

void Multiplexer::stop() {
    running_.store(false);

    if (monitor_thread_ && monitor_thread_->joinable()) {
        monitor_thread_->join();
        monitor_thread_.reset();
    }

    for (auto& t : gpu_threads_) {
        if (t.joinable()) t.join();
    }
    gpu_threads_.clear();

    stratum_->stop_mining();
}

void Multiplexer::wait() {
    // Wait for all GPU threads to finish
    for (auto& t : gpu_threads_) {
        if (t.joinable()) t.join();
    }
}

bool Multiplexer::is_connected() const {
    return stratum_->is_connected();
}

bool Multiplexer::reconnect() {
    stop();
    fprintf(stderr, "[multiplexer] Attempting to reconnect...\n");
    int retries = 5;
    for (int i = 0; i < retries; i++) {
        if (stratum_->reconnect()) {
            fprintf(stderr, "[multiplexer] Reconnected successfully\n");
            start();
            return true;
        }
        fprintf(stderr, "[multiplexer] Reconnect attempt %d/%d failed\n", i + 1, retries);
        if (i < retries - 1) {
            int delay = 5000 * (i + 1); // Exponential backoff: 5s, 10s, 15s...
            fprintf(stderr, "[multiplexer] Waiting %dms before next attempt...\n", delay);
            usleep(delay * 1000);
        }
    }
    fprintf(stderr, "[multiplexer] Reconnect failed after %d attempts\n", retries);
    return false;
}

void Multiplexer::gpu_worker(int gpu_index, std::shared_ptr<GpuStats> stats) {
    int idx = config_.gpu_indices[gpu_index];

    // Initialize GPU using cuda_driver
    CudaGpu gpu;
    memset(&gpu, 0, sizeof(gpu));

    size_t work_queue_size = WORK_QUEUE_CAPACITY * sizeof(WorkItem);
    size_t result_buf_size = sizeof(ResultBuffer) + RESULT_BUFFER_SIZE * sizeof(ShareResult);

    if (cuda_driver_init_gpu(&gpu, idx, work_queue_size, result_buf_size) != 0) {
        fprintf(stderr, "[gpu:%d] Failed to initialize\n", idx);
        return;
    }

    fprintf(stderr, "[gpu:%d] Context created on %s\n", idx, stats->name.c_str());

    // Initialize result buffer capacity on device
    uint32_t buf_capacity = RESULT_BUFFER_SIZE;
    cuda_driver_copy_to_device(&gpu, gpu.result_buf_ptr + 4, &buf_capacity, sizeof(uint32_t));
    cuda_driver_sync(&gpu);

    // Set up PoW target (default difficulty target — will be updated when job arrives)
    uint32_t pow_target[8] = {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
                               UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    cuda_driver_copy_to_device(&gpu, gpu.pow_target_ptr, pow_target, 8 * sizeof(uint32_t));
    cuda_driver_sync(&gpu);

    // Set up PoW key (BLAKE3 derived key — use sigma-based seed)
    uint32_t pow_key[8] = {0};
    cuda_driver_copy_to_device(&gpu, gpu.pow_key_ptr, pow_key, 8 * sizeof(uint32_t));
    cuda_driver_sync(&gpu);

    // Launch persistent kernel: 4 blocks, 256 threads each
    uint32_t grid_x = 4;
    uint32_t block_x = 256;

    if (cuda_driver_launch_kernel(&gpu, grid_x, block_x) != 0) {
        fprintf(stderr, "[gpu:%d] Failed to launch kernel\n", idx);
        cuda_driver_destroy_gpu(&gpu);
        return;
    }

    fprintf(stderr, "[gpu:%d] Kernel launched (%d blocks x %d threads)\n",
            idx, grid_x, block_x);

    // Mining loop
    auto last_stats_time = std::chrono::steady_clock::now();
    uint64_t local_shares = 0;
    std::string last_dispatched_job_id;
    std::string current_local_job_id;
    std::string current_local_ntime;

    while (running_.load()) {
        // Check for new job from stratum
        {
            std::lock_guard<std::mutex> lock(stratum_->get_job_mutex());
            auto& current_job = stratum_->get_current_job();

            if (!current_job.job_id.empty() && current_job.job_id != last_dispatched_job_id) {
                // Store current job info for share submission
                current_local_job_id = current_job.job_id;
                current_local_ntime = current_job.ntime;
                last_dispatched_job_id = current_job.job_id;

                // Build work item from job
                WorkItem work;
                memset(&work, 0, sizeof(work));
                work.nonce_base = gpu_index;
                work.nonce_count = 128;
                memcpy(work.sigma, current_job.job_id.c_str(),
                       std::min(current_job.job_id.size(), (size_t)32));
                work.target_nbits = current_job.target_nbits;

                // Push work to GPU
                cuda_driver_push_work(&gpu, &work, sizeof(WorkItem));

                // Update PoW target on device
                uint32_t new_target[8];
                memset(new_target, 0, sizeof(new_target));
                // Decode nbits to target bytes
                uint32_t nbits = current_job.target_nbits;
                uint8_t exponent = (nbits >> 23) & 0xFF;
                uint32_t mantissa = nbits & 0x7FFFFFu;
                if (exponent <= 3) {
                    mantissa >>= (8 * (3 - exponent));
                    for (int i = 0; i < exponent; i++)
                        new_target[0] |= mantissa << (i * 8);
                } else {
                    new_target[0] = mantissa & 0xFF;
                    new_target[1] = (mantissa >> 8) & 0xFF;
                    new_target[2] = (mantissa >> 16) & 0xFF;
                    for (int i = 3; i < (int)exponent - 2; i++)
                        new_target[i] = 0xFF;
                }
                cuda_driver_copy_to_device(&gpu, gpu.pow_target_ptr, new_target, 8 * sizeof(uint32_t));
                cuda_driver_sync(&gpu);
            }
        }

        // Poll result buffer for shares
        uint32_t write_pos = 0;
        cuda_driver_copy_to_host(&gpu, &write_pos, gpu.result_buf_ptr, sizeof(uint32_t));

        if (write_pos > 0 && write_pos <= RESULT_BUFFER_SIZE) {
            // Copy results to host pinned buffer
            cuda_driver_copy_to_host(&gpu, gpu.pinned_result_buf,
                                     gpu.result_buf_ptr + sizeof(ResultBuffer),
                                     write_pos * sizeof(ShareResult));
            cuda_driver_sync(&gpu);

            ShareResult* results = (ShareResult*)((char*)gpu.pinned_result_buf + sizeof(ResultBuffer));

            for (uint32_t i = 0; i < write_pos; i++) {
                auto& r = results[i];
                local_shares++;

                // Submit share via stratum using the actual job_id
                stratum_->submit_share(current_local_job_id, (uint32_t)r.nonce,
                                       current_local_ntime,
                                       gpu_index);
            }

            // Reset write_pos
            uint32_t reset = 0;
            cuda_driver_copy_to_device(&gpu, gpu.result_buf_ptr, &reset, sizeof(uint32_t));
            cuda_driver_sync(&gpu);
        }

        // Update stats every second
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_stats_time).count();

        if (elapsed >= 1000) {
            // Read stats counters from device
            uint64_t dev_stats[3] = {0};
            cuda_driver_copy_to_host(&gpu, dev_stats, gpu.stats_ptr, 3 * sizeof(uint64_t));

            double h = static_cast<double>(dev_stats[2]) * 1e6; // nonces computed
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats->hashrate = h / (elapsed / 1000.0);
                stats->total_computed = dev_stats[2];
                stats->accepted = dev_stats[0];
                stats->rejected = dev_stats[1];
            }

            last_stats_time = now;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // Signal kernel shutdown
    cuda_driver_signal_shutdown(&gpu);
    cuda_driver_sync(&gpu);

    cuda_driver_destroy_gpu(&gpu);

    fprintf(stderr, "[gpu:%d] Stopped\n", idx);
}

void Multiplexer::monitor_thread_func() {
    int frame = 0;

    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        if (!config_.show_hashrate) continue;

        frame++;

        // Collect stats
        double total_hr = 0;
        uint64_t total_acc = 0, total_rej = 0;

        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            for (auto& s : stats_) {
                total_hr += s->hashrate;
                total_acc += s->accepted;
                total_rej += s->rejected;
            }
        }

        // Format total hashrate
        char hr_buf[64];
        if (total_hr >= 1e9)
            snprintf(hr_buf, sizeof(hr_buf), "%.2f GH/s", total_hr / 1e9);
        else if (total_hr >= 1e6)
            snprintf(hr_buf, sizeof(hr_buf), "%.2f MH/s", total_hr / 1e6);
        else if (total_hr >= 1e3)
            snprintf(hr_buf, sizeof(hr_buf), "%.2f KH/s", total_hr / 1e3);
        else
            snprintf(hr_buf, sizeof(hr_buf), "%.0f H/s", total_hr);

        // Print header on first frame
        if (frame == 1) {
            fprintf(stderr, "\n");
            fprintf(stderr, "  %s  %s  %s  %s  %s  %s  %s\n",
                    "GPU", "Name", "Hashrate", "Temp", "Power", "Accepted", "Rejected");
            fprintf(stderr, "  %s\n", "-------------------------------------------------------------");
        }

        std::lock_guard<std::mutex> lock(stats_mutex_);
        for (auto& s : stats_) {
            char gpu_hr[32];
            if (s->hashrate >= 1e9)
                snprintf(gpu_hr, sizeof(gpu_hr), "%.2f GH/s", s->hashrate / 1e9);
            else if (s->hashrate >= 1e6)
                snprintf(gpu_hr, sizeof(gpu_hr), "%.2f MH/s", s->hashrate / 1e6);
            else if (s->hashrate >= 1e3)
                snprintf(gpu_hr, sizeof(gpu_hr), "%.2f KH/s", s->hashrate / 1e3);
            else
                snprintf(gpu_hr, sizeof(gpu_hr), "%.0f H/s", s->hashrate);

            fprintf(stderr, "  %-3d  %-24s  %-10s  %5.1fC  %6.0fW  %-8lu  %lu\n",
                    s->index, s->name.substr(0, 24).c_str(), gpu_hr,
                    s->temperature, s->power,
                    (unsigned long)s->accepted, (unsigned long)s->rejected);
        }

        double uptime = frame * 5.0;
        auto minutes = (int)(uptime / 60);
        auto seconds = (int)(uptime % 60);

        fprintf(stderr, "  %s\n", "---");
        fprintf(stderr, "  Total: %s  |  Shares: %lu accepted, %lu rejected  |  %dm %ds\n",
                hr_buf, (unsigned long)total_acc, (unsigned long)total_rej,
                minutes, seconds);
        fprintf(stderr, "\n");
    }
}

std::vector<GpuStats> Multiplexer::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    std::vector<GpuStats> result;
    for (auto& s : stats_) {
        result.push_back(*s);
    }
    return result;
}

double Multiplexer::get_total_hashrate() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    double total = 0;
    for (auto& s : stats_) {
        total += s->hashrate;
    }
    return total;
}

} // namespace propminer