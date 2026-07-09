# Production confirmation checklist

Short list of what to verify on the rig after deploying builds from this session (`b3dd10e`+).  
**Script?** = automated helper exists; **No script** = watch logs / pool metrics manually.

---

## #1 — GeForce kernel v2 (default ON on blackwell builds)

Run **first** on the 5090 after any kernel/deploy change. v2 uses `PipelineTmaAsync` + TMA prefetch; v1 remains in the binary for rollback.

| # | Confirm | Script? |
|---|---------|---------|
| 1 | **Full transcript gate:** consumer ≡ v2 ≡ v1 byte-identity (100 trials × 2 shapes) + consumer / v1 / v2 self-tests. | **Yes** — `./scripts/verify_geforce_transcript.sh` |
| 2 | **Prod-shape self-test** with v2 default (unset `PEARL_GEMM_KERNEL`). | **Yes** — `PROP_MINER_SELF_TEST_PROD=1 ./build/propminer --self-test --rtx5090 --gpus 0` |
| 3 | Build includes v2: `strings build/libpearl_gemm_capi.so \| grep PEARL_GEMM_BUILD_KNOBS` shows `+geforce_v2`. | **Manual** |
| 4 | Active kernel is v2 when env unset (`PEARL_GEMM_DEBUG=1` → `Active transcript kernel: geforce_v2`). | **No script** — startup stderr |
| 5 | Pre-deploy gate with GeForce verify (optional but recommended before prod). | **Yes** — `PROPMINER_GATE_GEFORCE=1 ./scripts/pre_deploy_gate.sh` |
| 6 | Pool soak ≥1 h: accepted-share rate unchanged; rejected/invalid <1%. | **No script** — pool dashboard + `shares:` line |

**Rollback (no rebuild):**

| Env | Kernel |
|-----|--------|
| *(unset)* | **geforce_v2** (default) |
| `PEARL_GEMM_KERNEL=geforce_v1` | v1 (manual barriers) |
| `PEARL_GEMM_KERNEL=consumer` | cp.async consumer fallback |

**Build requirement:** blackwell cmake with `-DPEARL_GEMM_BLACKWELL_GEFORCE_V2=ON` (default ON). Rebuild `propminer` after pull — pre-built Docker images won't pick up v2 until rebuilt.

**Audit checklist:** `DISCUSSION_WITH_AI/plans/01-geforce-kernel-v2/AUDIT.md`

---

## #2 — Grouped GEMM (default ON at batch≥4 on blackwell)

Activates only when `pearl_capi_iter_batch` receives **count ≥ 4**. Prod default `PROPMINER_BATCH=1` → **no behavior change** until batch is raised.

| # | Confirm | Script? |
|---|---------|---------|
| 1 | Build knob `+grouped` in `libpearl_gemm_capi.so`. | **Manual** — `strings build/libpearl_gemm_capi.so \| grep grouped` |
| 2 | batch=1 self-test unchanged (grouped off). | **Yes** — `PROPMINER_BATCH=1` self-test |
| 3 | batch=4 serial vs grouped smoke (self-test passes both). | **Yes** — `./scripts/verify_grouped_gemm_transcript.sh` |
| 4 | batch=3 uses serial path only (below threshold). | **Yes** — in verify script step [4/4] |
| 5 | Pool soak with `PROPMINER_BATCH=4` ≥1 h if enabling batch in prod. | **No script** — pool dashboard |

**Rollback (no rebuild):** `PEARL_GEMM_GROUPED=0` forces serial loop even at batch≥4.

**Build requirement:** `-DPEARL_GEMM_GROUPED_GEMM=ON` (default ON for blackwell).

**Audit checklist:** `DISCUSSION_WITH_AI/plans/02-ptr-array-grouped-gemm/AUDIT.md`

---

## Deploy & baseline

| # | Confirm | Script? |
|---|---------|---------|
| 1 | New build SHA in logs (`shares: build=…`) matches the release you intended. | **No script** — check log line |
| 2 | TMAD/s steady-state is in the same ballpark as before (~280 TMAD/s @ N=262144). | **No script** — compare `hashrate:` lines |
| 3 | Pool accepted-share rate unchanged over ≥1 h (not just local TMAD/s). | **No script** — pool dashboard / `shares:` line |
| 4 | Pre-deploy gate passes on 5090 before/after pull. | **Yes** — `scripts/pre_deploy_gate.sh` |
| 5 | GPU self-test passes at default and prod shape. | **Yes** — `scripts/pre_deploy_gate.sh` (or manual `--self-test --rtx5090`) |
| 6 | Prod-shape self-test: `PROP_MINER_SELF_TEST_PROD=1 ./build/propminer --self-test --rtx5090 --gpus 0`. | **Partial** — in `pre_deploy_gate.sh`; manual env |

---

## New binary defaults (no env needed)

| # | Confirm | Script? |
|---|---------|---------|
| 7 | `batch=1`, `graph_batch=1` in `hashrate:` line (unless tune cache/env overrides). | **No script** — read log; unset `PROPMINER_BATCH` / tune cache if old values persist |
| 8 | `N=262144` in `hashrate:` (M=8192, K=4096 stratum). | **No script** — read log |
| 9 | `cluster_m=1` unless tune cache overrides. | **No script** — read log / tune files |
| 10 | GeForce v2 kernel active by default on blackwell (`geforce_v2` when env unset; `geforce_v1` / `consumer` to opt out). | **Yes** — `scripts/verify_geforce_transcript.sh`; `scripts/validate_geforce_kernel_build.sh` |

---

## CPU/GPU overlap (default ON — kill switches only)

| # | Confirm | Script? |
|---|---------|---------|
| 11 | **B-col cache** (`PROPMINER_BCOL_CACHE`, default ON): no rise in `verify-fail` / `claimed_hash_mismatch`. | **No script** — logs; rollback `=0` |
| 12 | **Defer share GPU** (`PROPMINER_DEFER_SHARE_GPU`, default ON): shares still `verify-ok` → `accepted`; no hang at job switch. | **Yes** — `scripts/validate_defer_share.sh` (compile); GPU: both defer modes in self-test |
| 13 | **Async seed** (`PROPMINER_ASYNC_SEED`, default ON): no `gpu_cpu_jackpot_mismatch` or duplicate-nonce rejects in first 30 min. | **Yes** — `scripts/validate_seed_path.sh` (code path); **No script** for pool soak |
| 14 | All three overlap features: watch 30 min after deploy; rollback table in deploy checklist if needed. | **Doc only** — `docs/DEPLOY_CHECKLIST_CPU_GPU_OVERLAP.md` |

---

## Pow target / nbits fix

| # | Confirm | Script? |
|---|---------|---------|
| 15 | Stratum `pool_diff` / share `target=` look sane; no `gpu_hits_30s=0` from wrong target. | **No script** — logs |
| 16 | Host `nbits_to_target_le` roundtrip (fixed UB bug). | **Yes** — `scripts/local_host_tests.sh` (Mac only) |

---

## Async job install (default ON, VRAM-guarded)

| # | Confirm | Script? |
|---|---------|---------|
| 17 | At N=262144 ~79% VRAM: expect `async-install-fallback reason=insufficient_vram` and **sync** job switches (normal). | **No script** — grep logs |
| 18 | If VRAM headroom exists: `async-install-request` / `-ready` / `-swap` traces; no stuck old job / wedged mining. | **No script** — grep logs |
| 19 | Accepted shares still valid after job rotation (`stratum new job` → mine → submit). | **No script** — pool + share trace |
| 20 | Kill switch works: `PROPMINER_ASYNC_JOB_INSTALL=0` restores old synchronous switch. | **No script** — env + restart |

---

## Triple-buffer & pipeline instrumentation

| # | Confirm | Script? |
|---|---------|---------|
| 21 | **`PROPMINER_TRIPLE_BUFFER` is ON by default** — `pipeline: … triple=off` unless you set `=1`. | **No script** — read `pipeline:` line |
| 22 | Pipeline metrics appear every ~5s: `half_wait_count`, `half_wait_ms_max`, `half_wait_ms_total`. | **No script** — read `pipeline:` line |
| 23 | If `half_wait_ms_max` often >5 ms during shares → triple-buffer may be worth trying (if VRAM allows). | **No script** — interpret metrics |
| 24 | With `PROPMINER_TRIPLE_BUFFER=1`: at your shape expect `triple-buffer disabled reason=insufficient_vram` (dual path, safe). | **No script** — startup stderr |
| 25 | If triple actually enables: `triple-buffer enabled` at startup; TMAD/s not regressed; shares still accepted. | **No script** — logs + pool |
| 26 | Dedicated triple-buffer validation script. | **No script yet** — use self-test matrix in `docs/DEPLOY_CHECKLIST_CPU_GPU_OVERLAP.md` |

---

## VRAM & job switch

| # | Confirm | Script? |
|---|---------|---------|
| 27 | VRAM ~25–28 GiB steady; spike on job switch acceptable; no OOM / CUDA alloc errors. | **No script** — `nvidia-smi` / `hashrate:` vram= |
| 28 | Job switch: brief TMAD dip OK; no prolonged 0 H/s after `stratum new job`. | **No script** — logs |
| 29 | Share on old job after switch: pool may `SupersededJob` / `Job not found` — not a local verify failure. | **No script** — `share-reasons:` line |

---

## Self-test matrix (GPU rig — proof gate)

| # | Confirm | Script? |
|---|---------|---------|
| 30 | `PROPMINER_DEFER_SHARE_GPU=0` self-test — all shares verify. | **Manual** — see `validate_defer_share.sh` hints |
| 31 | `PROPMINER_DEFER_SHARE_GPU=1` self-test — all shares verify. | **Manual** |
| 32 | `PROPMINER_ASYNC_SEED=0` soak if async-seed regressions suspected. | **No script** — env + mine |
| 33 | `PROPMINER_TRIPLE_BUFFER=1` self-test only if VRAM guard passed at startup. | **No script** |

---

## Rollback (instant, no rebuild)

| # | Confirm | Script? |
|---|---------|---------|
| 34 | Know kill switches: `PROPMINER_BCOL_CACHE=0`, `DEFER_SHARE_GPU=0`, `ASYNC_SEED=0`, `ASYNC_JOB_INSTALL=0`, `TRIPLE_BUFFER` unset. | **Doc** — `docs/DEPLOY_CHECKLIST_CPU_GPU_OVERLAP.md` |
| 35 | Full rollback to pre-overlap behavior: set all four overlap kills to `0` + no triple flag. | **No script** — env + restart |

---

## Quick reference — scripts that exist

| Script | What it covers |
|--------|----------------|
| `scripts/pre_deploy_gate.sh` | Host tests + GPU self-test (default + prod shape) |
| `scripts/local_host_tests.sh` | Host-only correctness (Mac; includes nbits fix) |
| `scripts/validate_defer_share.sh` | Defer-share compile + self-test commands |
| `scripts/validate_seed_path.sh` | Async seed code-path grep + build smoke |
| `scripts/validate_geforce_kernel_build.sh` | GeForce kernel build flag |
| `scripts/verify_geforce_transcript.sh` | **#1 gate** — v1+v2 transcript memcmp + consumer/v1/v2 self-tests |
| `scripts/run_mining.sh` / `scripts/salad/run.sh` | Production launch (not a validator) |
| `docs/DEPLOY_CHECKLIST_CPU_GPU_OVERLAP.md` | Full deploy/rollback guide (not a script) |

**Not automated today:** pool soak, TMAD/s comparison, `pipeline:` stall analysis, async-job VRAM fallback, triple-buffer GPU validation.
