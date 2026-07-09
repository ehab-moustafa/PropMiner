# GeForce v2 — Post-Implementation Audit Checklist

Run on RTX 5090 host before promoting v2 beyond opt-in.

## Build / binary

- [ ] `cmake -DPEARL_GEMM_BLACKWELL_GEFORCE_V2=ON` builds without error
- [ ] `strings build/libpearl_gemm_capi.so | grep PEARL_GEMM_BUILD_KNOBS` includes `+geforce_v2` when v2 ON
- [ ] v2 build with `PEARL_GEMM_BLACKWELL_GEFORCE_V2=0` omits v2 symbols
- [ ] `sizeof(SharedStorage)` for v2 ≤ 101376 B (compile-time `static_assert` in v2.cu)

## Runtime / dispatch

- [ ] Unset `PEARL_GEMM_KERNEL` → `pearl_capi_active_kernel_name()` returns `geforce_v2` (v2 default)
- [ ] `PEARL_GEMM_KERNEL=geforce_v1` → returns `geforce` (v1)
- [ ] `PEARL_GEMM_KERNEL=geforce` → returns `geforce_v2` (alias for best path)
- [ ] `PEARL_GEMM_KERNEL=consumer` → returns `consumer`
- [ ] `PEARL_GEMM_KERNEL=geforce_v2` on v2-less build → exit -54 (fail-fast)
- [ ] `warmup_kernels_before_graph_capture()` succeeds for active kernel

## Proof correctness (G1, G3)

```bash
./scripts/verify_geforce_transcript.sh
```

- [ ] v1 vs consumer: 100/100 trials both shapes
- [ ] v2 vs consumer: 100/100 trials both shapes
- [ ] v2 vs v1: 100/100 trials both shapes
- [ ] `PEARL_GEMM_KERNEL=geforce_v2 ./build/propminer --self-test --rtx5090 --gpus 0`
- [ ] `PROP_MINER_SELF_TEST_PROD=1 PEARL_GEMM_KERNEL=geforce_v2 ./build/propminer --self-test --rtx5090 --gpus 0`
- [ ] `PROPMINER_GATE_GEFORCE=1 ./scripts/pre_deploy_gate.sh`

## Performance (before default promotion)

- [ ] v2 bench ≥ v1 (no regression)
- [ ] v2 bench ≥ +10% vs consumer (`compare_bench.sh` tolerance 0.10)
- [ ] NCU: tensor-pipe active cycles ≥ v1 + 8 pct points

## Production soak (G7)

- [ ] 24–48 h pool with `PEARL_GEMM_KERNEL=geforce_v2`: rejected shares < 1%
- [ ] No `verify-fail`, `claimed_hash_mismatch`, `gpu_cpu_jackpot_mismatch` spikes
- [ ] σ swap test: shares still accepted after job rotation

## Rollback verified

- [ ] `PEARL_GEMM_KERNEL=geforce_v1` restores v1 instantly (no rebuild)
- [ ] `PEARL_GEMM_KERNEL=consumer` restores consumer instantly (no rebuild)

## Status

| Item | Result | Date | Notes |
|------|--------|------|-------|
| Code landed | Phase 0+1 | — | v2 ON by default (blackwell) |
| G1 memcmp | **PENDING** | — | Requires 5090 run |
| G3 v2≡v1 | **PENDING** | — | Requires 5090 run |
| G2 self-test | **PENDING** | — | Requires 5090 run |
| G6 bench | **PENDING** | — | Requires 5090 run |
| G7 soak | **PENDING** | — | Before prod promotion |

**5090 proof gates (G1–G7) still required after deploy — run `PROD_TO_CONFIRM.MD` #1 section on the rig.**
