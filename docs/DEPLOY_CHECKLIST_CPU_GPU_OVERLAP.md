# Deploy checklist — CPU/GPU overlap optimizations

One page for the RTX 5090 rig. Three independent features, each with its own
env-var kill switch. Roll them out in this order; validate between steps.

## What changed

| Feature | Default in binary | Kill switch / opt-in |
|---|---|---|
| 1. Targeted B-column expansion (share path skips full n×k B re-hash) | **ON** | `PROPMINER_BCOL_CACHE=0` |
| 2. Deferred share handling (share rebuild on side thread, GPU keeps mining) | **ON** | `PROPMINER_DEFER_SHARE_GPU=0` |
| 3. Async seed conveyor (next sub-batch seed uploads on copy stream) | **ON** | `PROPMINER_ASYNC_SEED=0` |
| 4. Async job installation (next job's resident B installs on a background thread; GPU keeps mining the old job until a fast swap) | **ON** (VRAM-guarded, self-disables when tight) | `PROPMINER_ASYNC_JOB_INSTALL=0` |

No proof math, share encoding, or transcript logic was touched.
`ShareBuilder::VerifyShare` still gates every submission, so a regression shows
up as dropped/rejected shares in logs — never silent bad submissions.

## Deploy steps (on the rig)

1. **Build**

   ```bash
   cmake --build build --target propminer -j"$(nproc)"
   ```

2. **Self-test, both defer modes**

   ```bash
   PROPMINER_DEFER_SHARE_GPU=0 ./build/propminer --self-test --rtx5090 --gpus 0
   PROPMINER_DEFER_SHARE_GPU=1 ./build/propminer --self-test --rtx5090 --gpus 0
   ```

   Both must pass (all shares verify-ok). Then the production-shape self-test:

   ```bash
   PROP_MINER_SELF_TEST_PROD=1 ./build/propminer --self-test --rtx5090 --gpus 0
   ```

3. **Mine — all three features are on by default**
   - **Seeds are proof-critical — watch accepted vs rejected shares for the
     first 30 minutes.** Any `gpu_cpu_jackpot_mismatch`, `verify-fail`,
     `claimed_hash_mismatch`, or duplicate-nonce rejects → use the rollback
     table below to isolate the culprit (start with `PROPMINER_ASYNC_SEED=0`).
   - If anything looks off, step features off one at a time:
     `PROPMINER_ASYNC_SEED=0` first, then `PROPMINER_DEFER_SHARE_GPU=0`,
     then `PROPMINER_BCOL_CACHE=0`.

4. **Judge success** on pool-side accepted share rate over ≥1 h vs the
   previous deployment, not just local TMAD/s.

5. **Async job installation is ON by default** and removes the GPU idle window
   on every job switch when VRAM allows. It is VRAM-guarded and cannot OOM;
   set `PROPMINER_ASYNC_JOB_INSTALL=0` to force the old synchronous switch.

   ```bash
   PROPMINER_ASYNC_JOB_INSTALL=0 ./build/propminer ... # force synchronous switch
   ```

   - The first job of a session still installs synchronously; only rotations
     (job 2+) overlap. During an overlapping install the GPU holds a **second
     resident-B set** (~2.2 GiB at N=262144/K=4096) plus a one-time **noise-B
     staging workspace** (~4 GiB at that shape) and shares SMs with the install
     kernels, so hashrate dips slightly during the swap window instead of
     dropping to zero.
   - **VRAM guard**: before each background install the miner checks free VRAM
     (`cudaMemGetInfo`); if there isn't room for the staging workspace + a second
     resident-B set + 512 MiB margin, it logs `async-install-fallback
     reason=insufficient_vram` and does a normal synchronous install instead.
     This makes the flag OOM-safe. **Consequence at N=262144/K=4096 on a 32 GiB
     5090 with a ~79% baseline: there is not enough headroom, so async will
     self-disable and you'll see the fallback log every rotation — it only
     actually engages at smaller shapes / with more free VRAM.**
   - Watch `async-install-request` / `-ready` / `-swap` traces and, above all,
     **accepted vs rejected share rate**. Any rise in superseded-job,
     `verify-fail`, or `claimed_hash_mismatch` rejects → `PROPMINER_ASYNC_JOB_INSTALL=0`
     (or just unset it) and restart to return to the synchronous switch.
   - Safety: the swap drains both compute streams and all deferred-share work
     before rebinding B pointers, every share pins its own `sigma_ctx`, and a
     background install failure falls back to a synchronous install — mining is
     never wedged on a stale job.

## Rollback (instant, no rebuild)

| Symptom | Action |
|---|---|
| verify-fail / claimed_hash_mismatch on shares | `PROPMINER_BCOL_CACHE=0` and restart |
| Shares dropped after defer enabled, hangs at job switch | `PROPMINER_DEFER_SHARE_GPU=0` and restart |
| Rejects/mismatches after async seed enabled | `PROPMINER_ASYNC_SEED=0` and restart |
| Superseded/verify rejects after async job install enabled | `PROPMINER_ASYNC_JOB_INSTALL=0` (or unset) and restart |
| Anything unclear | `PROPMINER_BCOL_CACHE=0 PROPMINER_DEFER_SHARE_GPU=0 PROPMINER_ASYNC_SEED=0 PROPMINER_ASYNC_JOB_INSTALL=0` → exact pre-change behavior |

## Validation record (Mac, host-only)

- `scripts/local_host_tests.sh`: no new failures vs baseline; new test
  `test_bseed_targeted_column_expand_matches_full` proves targeted per-column
  expansion is byte-identical to full expansion. (4 pre-existing
  `pow_target_stratum_nbits_roundtrip` failures were present before these
  changes and are unrelated.)
- `share_builder.cpp` and `gpu_worker.cpp` compile clean (`-fsyntax-only`).
- Full CUDA build is validated by CI (`compile-check.yml` docker builder).
