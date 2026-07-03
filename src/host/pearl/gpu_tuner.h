#pragma once

#include "cuda_compat.h"
#include "pearl_types.h"

#include <vector>

namespace pearl {

// Per-GPU autotuner.  Tunes batch size, matrix shape (M, N), CUDA graph use,
// cluster size, carveout, and compiled tile-shape knobs (BM/BN/BK/stages).
// Each candidate is benchmarked for a configurable duration and the result is
// averaged over repeated measurements with worst-case outlier rejection.
struct TuningResult {
    MiningConfig config;
    int batch_size = 8;
    bool use_graph = true;
    int cluster_m = 1;
    int carveout_percent = -1;
    double hashrate = 0.0;  // MAC/s
    int repeats = 1;
};

class GpuTuner {
public:
    explicit GpuTuner(int device_index);

    // Run the autotune sweep.
    //   seconds_per_candidate: wall-clock time for each measured candidate.
    //   repeats: how many independent measurements to take per candidate; the
    //            slowest repeat is discarded before averaging.
    TuningResult tune(double seconds_per_candidate = 5.0, int repeats = 3);

    // Fine-grained search over matrix shapes.  Called automatically by tune().
    TuningResult tune_shapes(const std::vector<MiningConfig>& candidates,
                             double seconds_per_candidate,
                             int repeats);

    // Pick a sensible base shape for the installed GPU's free VRAM.
    MiningConfig shape_for_vram(size_t free_bytes,
                                int major, int minor) const;

private:
    int device_index_;

    // Evaluate one configuration.  Returns 0 on failure.
    double benchmark_config(const MiningConfig& cfg, int batch, bool use_graph,
                            int cluster_m, int carveout_percent,
                            double seconds) const;

    // Repeatably measure a candidate and return the trimmed-mean hashrate.
    double benchmark_stable(const MiningConfig& cfg, int batch, bool use_graph,
                            int cluster_m, int carveout_percent,
                            double seconds, int repeats) const;
};

} // namespace pearl
