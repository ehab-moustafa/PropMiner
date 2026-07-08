#pragma once

namespace pearl {
namespace grouped {

// Minimum batch for grouped GEMM (below this: serial iter_batch path only).
static constexpr int kGroupedGemmMinBatch = 4;
// Maximum groups per single grouped launch (VRAM / workspace cap).
static constexpr int kGroupedGemmMaxBatch = 16;

}  // namespace grouped
}  // namespace pearl
