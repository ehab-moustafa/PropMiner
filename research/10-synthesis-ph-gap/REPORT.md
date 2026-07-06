# The TH/s to PH/s Gap — A Creative Synthesis for PropMiner

**PropMiner Research · Topic 10 · Synthesis**  
**Date:** July 2026  
**Inputs:** Research reports 01–09, performance optimizations 01–08  
**Audience:** Engineers chasing the fantasy number — and the honest ones building the real one

---

## Preface: Why This Document Exists

Somewhere between a pool dashboard showing **~300 TH/s** on a single RTX 5090 and a daydream of **1 PH/s**, there is a thousand-fold chasm. That chasm is not a bug in your miner. It is not a hidden compiler flag. It is physics, protocol design, and unit semantics stacked on top of each other.

This synthesis does not promise 1 PH/s. It maps the territory: what the numbers actually mean, where the silicon ceiling is, which shortcuts are mirages, which puzzle pieces genuinely click together, and what would have to change at the protocol level for the "impossible million" to become even theoretically reachable.

The goal is clarity with ambition — honest enough to guide engineering, hopeful enough to keep tuning on a Sunday night.

**Research inputs:** All reports 01–09, plus performance docs 01–08.

---

## 1. The TH/s vs PH/s Mystery — Units, DAF, Pool vs GPU

### 1.1 Three currencies miners confuse

| Metric | What it counts | RTX 5090 scale |
|--------|----------------|----------------|
| **TMAD/s** | Trillion int8 MACs/s (M×N×K × iters/s) | **~300 TMAD/s** |
| **Pool "TH/s"** | Community dashboards; tracks TMAD/s for dedicated miners | **~280–310 TH/s** |
| **Protocol H/s (DAF-normalized)** | Hash-tile evaluations/s × DAF | **~10⁹ H/s (~1 GH/s)** |

Community **305 TH/s** ≈ PropMiner **~305 TMAD/s**. Protocol H/s is smaller by **DAF ≈ 16,384** (2×64×128):

```
hashrate_H/s = tiles_per_sec × DAF
tiles_per_sec = iters_per_sec × (M/128) × (N/256)
```

**Critical trap:** At fixed TMAD/s, **H/s is N-invariant**. Doubling N doubles tiles per matmul but halves matmuls/s. Production N=262144 maximizes work per job, not headline H/s.

### 1.2 What the pool sees vs what the GPU does

**What the pool sees:** DAF-weighted tile throughput, share acceptance rate, vardiff adjustments, heartbeat hashrate claims. Kryptex gRPC V2 receives `claimed_total_hashrate` at register and adjusts difficulty from measured rates.

**What the GPU actually does:** For each nonce iteration:

1. Regenerate noisy A from LCG + rank-R noise structures
2. Execute a full M×N×K int8 GEMM (8192×262144×128 ≈ **275 trillion MACs**)
3. Maintain a **byte-identical transcript** — XOR-rotate mixing at each K/R boundary
4. Run in-kernel BLAKE3 target checks on periodic hash tiles (headless PoW)
5. On rare hits: Merkle proof reconstruction, A regen, pool submit

The pool never sees TMAD/s directly. It sees **valid shares per hour** at a difficulty that maps back to tile evaluations. A kernel 2× faster at GEMM yields ~2× more shares at fixed difficulty — but only if transcripts remain byte-identical.

### 1.3 PH/s arithmetic

| Start | Target | Multiplier |
|-------|--------|------------|
| 1× 5090 @ ~300 TH/s | 1 PH/s (1000 TH/s) | **~3,300×** |
| 3–4 GPU rig @ ~1 TH/s | 1 PH/s | **~1000×** |
| 1000× 5090s @ 300 TH/s | 300 PH/s aggregate | Linear fleet scale |

**1 PH/s on one card within Pearl V2 is not an engineering target — it is a unit mirage.**

---

## 2. Physics Ceiling on RTX 5090

| Spec | Value |
|------|-------|
| Rated INT8 TOPS | **838** |
| SMs / boost | 170 @ 2.41 GHz |
| PropMiner baseline | **~300 TMAD/s (~36% of peak)** |
| **Realistic sustained max** | **~500–600 TMAD/s (60–75% of peak)** |

For Pearl's production shape, arithmetic intensity is ~7,900 MACs/byte — the kernel needs only ~106 GB/s to saturate 838 TOPS against 1,792 GB/s available (~6% bandwidth at peak). **Pearl on 5090 is compute-bound, not memory-bound.** GDDR7 overclocking is Ethash-era thinking; core clock and tensor-pipe duty cycle are the levers.

The ~64% gap between ~300 TMAD/s observed and 838 TOPS rated:

| Bucket | Impact |
|--------|--------|
| Legacy `mma.sync` vs native UMMA | 15–30% |
| Transcript epilogue | 10–20% |
| Register pressure (~1 block/SM) | 5–15% |
| BLAKE3 PoW + launch gaps | 5–15% |

**Consumer GB202 lacks `tcgen05`, TMEM, UMMA.** B200 at 840+ TMAD/s is a reference, not a port target. sm_120 shared memory cap: **99 KB/block**.

3× (~900 TMAD/s) **exceeds rated dense peak sustained** with proof epilogue — not a planning guarantee.

---

## 3. Puzzle Pieces That DON'T Work — 100× Shortcuts, Drop Transcript, Wrong ISA

These are the mirages. They glitter in Discord threads and old GPU mining guides. They do not survive Pearl's verifier.

| Mirage | Why it fails |
|--------|--------------|
| **Skip transcript** | Consensus-critical; `claimed_hash_mismatch` → 100% rejects. Any "100× faster" kernel that omits transcript bytes is worth *negative* hashrate. |
| **Port B200 tcgen05 verbatim** | `sm_100a` cubins rejected on consumer; `tcgen05.*` fails ptxas for `sm_120a`; no TMEM on GeForce |
| **FP8/FP4/sparse ops** | Breaks int8 proof-canonical path; sparse 1,676 TOPS rating irrelevant |
| **8× N → 8× TH/s** | N-invariant at fixed TMAD/s; larger N improves job economics, not headline H/s |
| **Memory OC +2000 MHz** | <1% on compute-bound GEMM; Igor's LAB confirms diminishing returns on GDDR7 |
| **SeedGenerator / faster seeds** | 8-byte upload overlapped; batch time ~60–120 s; ~0% (doc 08, removed) |
| **SM120 CUTLASS int8 atom rename** | Same `mma.sync` SASS; ~0% (doc 06) |
| **CUDA graphs alone** | 1.1–1.5×; PropMiner already captures extended batch graphs |
| **vLLM long prompts** | Inference-only (`min_m≥1024`); stratum miners ignore token count |
| **100× GPUs = 100× profit** | WTEMA difficulty absorbs gains; network ~24–30 EH/s |
| **"1000× kernel fusion fantasy"** | Amdahl: if GEMM is 90%, infinite-speed epilogue saves at most 10% |

---

## 4. Puzzle Pieces That DO Work — Ranked Roadmap

**Partially multiplicative.** Realistic per-GPU: **1.8–3.5×**. Fleet: linear ×N GPUs.

### Tier 1 — Weeks, low risk (**1.1–1.25×**)

1. Production **N=262144** (largely done)
2. **`tune_prod_5090.sh`** — knobs + cluster + batch (+10–20%)
3. **`PROPMINER_USE_TUNE_CACHE=1`**
4. Batch **4→20** (+3–10%)
5. Power cap **450–510 W** (TH/W win)

### Tier 2 — Months (**1.3–1.5×** cumulative)

6. **TMA gmem→smem** (+10–25%; scaffold exists)
7. Compile knobs: MIN_BLOCKS=2, KBLOCK=64/STAGES=3 (+5–15%)
8. **Cluster + CLC** (+5–15%)
9. **`PROPMINER_DEFER_SHARE_GPU=1`** (+0–2%)
10. **ncu profiling** — tensor pipe >80% target

### Tier 3 — High effort (**2–2.5×** cumulative)

11. **GeForce-native kernel** (NOT blind tcgen05; warp-specialized mma.sync + TMA)
12. CUDA 13 smem spilling for register relief
13. Core +150 MHz (self-test gated)

### Tier 4 — Fleet 5×

14. Multi-GPU linear scaling  
15. H100/B200 (~550–600 TH/s/card)  
16. Efficiency fleet (5080/5070 Ti beat 5090 on TH/W)

| Stage | TMAD/s | × baseline |
|-------|--------|------------|
| Today | ~300 | 1.0× |
| Tier 1+2 | ~400–450 | ~1.4× |
| Tier 1–3 | **~500–600** | **~1.7–2.0×** |
| 5× GPU @ 2× each | ~3000 aggregate | ~5× fleet |

---

## 5. Creative Ideas — Community, Repos, PropMiner Gaps, Eco: J/hash

### 5.1 What the community actually figured out

**Alpha-miner** (closed binary, open benchmarks) proved `sm_120` Blackwell backends at **280–300 TH/s** — parity with PropMiner's open ~300 TMAD/s. The spread is narrow; **kernel correctness is solved**, kernel *efficiency* is the game.

**Static pool difficulty** (`x;d=262144` to `1048576` on AlphaPool) stabilizes share rates. Issue #26 showed **117 TH/s local vs 27 TH/s pool** when misconfigured — always compare local TMAD/s to pool-side acceptance, not just console numbers.

**Undervolt + power cap** beats stock 575 W for 24/7: **305 TH/s @ 500 W** (1.65 J/T) is the community sweet spot. Blackwell thermally throttles; alpha-miner cites **280 vs 300 TH/s** from cooling alone.

**nvoc** (`martinstark/nvoc`) scripts reproducible clocks on Linux cloud hosts — essential when A/B testing kernel changes on vast.ai/RunPod.

### 5.2 Repo ideas worth stealing (conceptually)

| Source | Idea | PropMiner use |
|--------|------|---------------|
| Colfax CUTLASS TMEM tutorial | Warp-specialized TMA + MMA + mbarriers | Inform GeForce path even without TMEM |
| Modular Blackwell matmul | 128×256×16 tiles, K-alignment | Validates consumer tile geometry |
| keryx-miner-supr (Keccak/5090) | Register 229→64 unlocked 28% at power wall | Register pressure is the enemy |
| blackwell-tensorcore-kernels | mma.sync ~63% vs wgmma ~96% of peak | Pearl epilogue absorbs only part of ISA win |
| Cloudrift matmul ladder | Roofline-first | Profile before tuning |
| Pearl #156 | Faster pool jackpot verify | Protocol-side economics |

### 5.3 PropMiner gaps still open

1. **TMA production path** — scaffold compiled, not shipped (doc 02)
2. **Winning knob cache** — tooling exists, bare-metal sweep not fleet-deployed (doc 05)
3. **GeForce-native kernel** — not blind B200 port; warp-specialized consumer path (doc 01)
4. **Published benchmarks** — community reports TH/s + J/T; PropMiner should publish TMAD/s + power + acceptance together
5. **MoE fork readiness** — dense-only today; grouped GEMM is future shape change

### 5.4 Eco-conscious mining: J/hash as the real scoreboard

At ~300 TH/s and 500 W: **~1.67 nJ per community "TH"**. The operator metric that matters is **accepted shares per kilowatt-hour**, not peak TH/s on a leaderboard.

| Strategy | Effect |
|----------|--------|
| Power cap 450–510 W | Often -5% TH/s, -15% W → net J/hash win |
| Undervolt stable curve | ~20% perf/W on tensor workloads (community) |
| Reject stale σ immediately | Zero wasted GEMM on expired jobs |
| Fleet duty-cycling | Pause when spot electricity > revenue |
| Right-size GPU | 5080/5070 Ti beat 5090 on TH/W (MiningBoard) |

**Uncomfortable truth (arXiv:2606.04819):** Dominant Pearl mining is **random int8 matmul**, not useful AI inference. The "2-for-1 compute" narrative is economic framing, not consensus guarantee. Eco-conscious operators optimize **shares per joule**, not narrative alignment.

---

## 6. The "Impossible Million" — Protocol Change Required

**~1000× from ~1 TH/s rig, ~3300× from one GPU.** No register tweak closes this.

| Fantasy lever | Reality |
|---------------|---------|
| Smaller K/M/N | Hard fork; security tradeoffs |
| Drop transcript | Breaks cuPOW thesis |
| Lower DAF | Optics only; pools reject |
| Next-gen silicon | ~500–800 TH/s speculative; still need 1000+ cards for PH/s |
| Difficulty | WTEMA absorbs any local 1000× within ~7 days |

**1 PH/s as sustained advantage is impossible on V2 silicon.** Fleet scale + modest per-GPU gains is the honest path.

---

## 7. 90-Day Action Plan (No Breaking Proofs)

**Gate every change:** `propminer --self-test --rtx5090`

### Days 1–30
- Run `tune_prod_5090.sh`; enable `PROPMINER_USE_TUNE_CACHE=1`
- ncu baseline: tensor pipe %, occupancy
- Verify N=262144 mine logs (65536 CTAs)
- A/B power cap 500 W vs 575 W
- Enable `PROPMINER_DEFER_SHARE_GPU=1` after soak

### Days 31–60
- Ship TMA load path behind compile flag; byte-compare transcripts
- ncu diff TMA vs baseline
- Ship winning compile knobs to Docker
- Cluster sweep cluster_m ∈ {1,2,4,8}

### Days 61–90
- `transcript_gemm_sm120_geforce.cu` — warp-specialized, no tcgen05
- Dual dispatch runtime flag; pool canary 24 h
- Target: **≥450 TMAD/s (1.5×)**; stretch **≥550 (1.8×)**

**Standing rules:** Profile before optimizing. Never port sm_100a. Report TMAD/s + watts + shares/h.

---

## 8. Personal Engineering Opinion — Honest, Hopeful, Eco-Conscious

I think the PH/s dream is **healthy as motivation and poisonous as a schedule**. It forces you to ask the right questions — what is the unit, what is the ceiling, what is consensus-critical. It becomes toxic when it sends you hunting for transcript shortcuts at 2 AM.

**Honest:** PropMiner today is **correct and competitive** at ~300 TH/s on a 5090. That is not a failure; it is **36% of rated INT8 tensor peak with a full proof epilogue on legacy ISA**. The closed-source miners sit in the same band — there is no secret 10× hiding in alpha-miner's binary. The gap to 600 TH/s is **real engineering**: TMA, knobs, occupancy, maybe a GeForce-native kernel redesign. The gap to 1 PH/s on one card is **not engineering** — it is **physics plus protocol**.

**Hopeful:** A **2×** on 5090 is credible within two quarters of disciplined work. **2× per GPU × N GPUs** is how fleets win. PropMiner's architecture — ping-pong graphs, resident B, headless PoW, tune caches — is **already the hard host-side work**. The remaining frontier is the kernel, and the kernel has a known ceiling near **500–600 TMAD/s**, not 838, not 8380.

**Eco-conscious:** The winning operator metric is **accepted shares per kilowatt-hour**, not peak TH/s on a leaderboard. A 5090 capped at 500 W doing 290 TH/s beats a 575 W card doing 305 TH/s on every electricity bill that matters. Pearl's dominant mining mode is **pure matmul**, not AI inference — own that honestly, optimize watts, and let the protocol's economics decide whether the work is "useful" enough.

The fantasy 1 PH/s teaches one durable lesson: **read the units, respect the transcript, multiply fleet not folklore.** The puzzle pieces that click are boring — tune, profile, TMA, ship, measure, repeat. That boredom is how you get to 600 TH/s without breaking a single proof.

---

## Appendix: Input Status

| Report | Status |
|--------|--------|
| 01 git repos, 05 noise-hash, 08 headroom, 09 gemm-accel | Pending |
| 02–04, 06–07 | Read |
| Perf docs 01–08 | Read |

| Quantity | Value |
|----------|-------|
| Baseline TMAD/s | ~300 |
| Realistic max | ~500–600 |
| 90-day target | ~450 (1.5×) |
| 1 GPU → 1 PH/s | ~3300× (impossible V2) |

---

*PropMiner research track 10. Original synthesis. No code executed.*
