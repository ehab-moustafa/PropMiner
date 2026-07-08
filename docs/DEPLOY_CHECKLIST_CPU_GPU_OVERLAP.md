# Deploy checklist — CPU/GPU overlap optimizations

One page for the RTX 5090 rig. Three independent features, each with its own
env-var kill switch. Roll them out in this order; validate between steps.

## What changed

| Feature | Default in binary | Kill switch |
|---|---|---|
| 1. Targeted B-column expansion (share path skips full n×k B re-hash) | **ON** | `PROPMINER_BCOL_CACHE=0` |
| 2. Deferred share handling (share rebuild on side thread, GPU keeps mining) | **ON** | `PROPMINER_DEFER_SHARE_GPU=0` |
| 3. Async seed conveyor (next sub-batch seed uploads on copy stream) | **ON** | `PROPMINER_ASYNC_SEED=0` |

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

## Rollback (instant, no rebuild)

| Symptom | Action |
|---|---|
| verify-fail / claimed_hash_mismatch on shares | `PROPMINER_BCOL_CACHE=0` and restart |
| Shares dropped after defer enabled, hangs at job switch | `PROPMINER_DEFER_SHARE_GPU=0` and restart |
| Rejects/mismatches after async seed enabled | `PROPMINER_ASYNC_SEED=0` and restart |
| Anything unclear | set all three: `PROPMINER_BCOL_CACHE=0 PROPMINER_DEFER_SHARE_GPU=0 PROPMINER_ASYNC_SEED=0` → exact pre-change behavior |

## Validation record (Mac, host-only)

- `scripts/local_host_tests.sh`: no new failures vs baseline; new test
  `test_bseed_targeted_column_expand_matches_full` proves targeted per-column
  expansion is byte-identical to full expansion. (4 pre-existing
  `pow_target_stratum_nbits_roundtrip` failures were present before these
  changes and are unrelated.)
- `share_builder.cpp` and `gpu_worker.cpp` compile clean (`-fsyntax-only`).
- Full CUDA build is validated by CI (`compile-check.yml` docker builder).
