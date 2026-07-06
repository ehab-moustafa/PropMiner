# PropMiner Research — Hashrate & RTX 5090

Mission-critical research pack (July 2026). **Reports only** — no production code changes.

## Reports (read `10-synthesis-ph-gap` first)

| # | Folder | Topic |
|---|--------|-------|
| 01 | `01-git-repos-mining/` | 10 cloned repos + mining/GEMM ideas |
| 02 | `02-rtx5090-specs/` | GB202 hardware + 2–3× realistic ceiling |
| 03 | `03-propminer-codebase/` | PropMiner 5090 utilization audit |
| 04 | `04-pearl-protocol/` | Pearl/PRL protocol & economics |
| 05 | `05-noise-hash/` | Noisy GEMM + transcript + BLAKE3 |
| 06 | `06-gpu-mining-fundamentals/` | GPU mining mechanics |
| 07 | `07-community-resources/` | Community articles & folklore vs fact |
| 08 | `08-propminer-headroom/` | Headroom, GPS opportunities |
| 09 | `09-gemm-acceleration/` | GEMM vs Pearl gap, x100 honest answer |
| 10 | `10-synthesis-ph-gap/` | TH/s → PH/s mystery + 90-day plan |

## Cloned repos (`01-git-repos-mining/repos/` — verify with `CLONE_MANIFEST.txt`)

1. `cutlass` — NVIDIA CUTLASS
2. `akoya-miner` — Pearl gRPC reference miner
3. `pearl` — pearl-research-labs protocol
4. `ethminer` — multi-stream CUDA mining patterns
5. `alpha-miner` — Blackwell miner docs/releases
6. `BLAKE3` — hash reference
7. `triton` — GPU kernel autotune
8. `pearl-miner-docker` — vLLM Docker community
9. `cccl` — NVIDIA CUDA C++ libs
10. `blackwell-geforce-nvfp4-gemm` — SM120 ISA truth (replaced cuda-samples due to disk)

## Headline

- **~300 TMAD/s today** (~36% of 838 INT8 TOPS on 5090)
- **Realistic 2–3×** via tune-prod + TMA/geforce kernel + knobs
- **1 PH/s on one GPU: impossible** on Pearl V2 — see synthesis for puzzle pieces
- **Eco:** optimize TH/W (power cap ~500 W), not memory OC folklore
