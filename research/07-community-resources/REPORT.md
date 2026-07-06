# Community Resources: GPU Mining Hashrate Optimization

**Research date:** 2026-07-06  
**Scope:** Web articles, pool guides, Blackwell/CUDA GEMM literature, Pearl (PRL) community, GitHub discussions  
**Audience:** PropMiner engineering — especially RTX 5090 / `sm_120a`, CUDA 12.8+, tensor-core NoisyGEMM path

---

## Executive synthesis

Community guidance on GPU mining hashrate splits into three layers that matter for PropMiner:

1. **Hardware/OS tuning (proven, transferable with limits)** — Power limits, thermal headroom, static pool difficulty, and Linux tooling (`nvoc`, `nvidia-smi`) reliably affect *effective* earnings (accepted shares per watt), even when raw TMAD/s is unchanged. Blackwell consumer cards (RTX 5090) thermal-throttle aggressively; community benchmarks cluster at **280–310 TH/s @ ~500 W**, not the card’s 575 W TDP.

2. **Kernel/architecture optimization (proven in adjacent domains, PropMiner-specific work in progress)** — NVIDIA/CUTLASS/Colfax/Modular documentation converges on **`tcgen05.mma` + TMEM + TMA warp specialization** as the Blackwell datacenter path. Consumer GB202 (`sm_120`) differs: Colfax explicitly notes **consumer Blackwell lacks TMEM**, which aligns with PropMiner’s current **`mma.sync` fallback** and the planned port documented in `performance optimizations/01-native-tcgen05-tmem-gemm.md`. Community miners (alpha-miner) ship closed Blackwell kernels; official Pearl vLLM builds lagged on `sm_120` through mid-2026.

3. **Pearl-specific folklore vs fact** — Much “AI inference mining” advice (long prompts, vLLM workers, `--enforce-eager`) applies to the **reference vLLM miner**, not PropMiner’s **direct pearlhash / transcript GEMM** stack. The **`min_m ≥ 1024`** threshold and decode-vs-prefill distinction are real for NoisyGEMM-in-inference but irrelevant to dedicated stratum miners. Claims that memory OC dramatically boosts Pearl hashrate are **Ethash-era carryover**; Pearl is **tensor-core / GEMM throughput** bound.

**Economics (July 2026 snapshot):** Network hashrate ~24–30 EH/s; RTX 5090 revenue estimates fell ~49% within weeks of mainnet (MiningBoard). At **$0.10/kWh**, a 5090 @ 500 W costs **~$1.20/day** in power; pool calculators show **~$3–8/day gross** depending on PRL price and difficulty — margins compress quickly. **Efficiency (TH/W)** beats raw TH/s for sustained profit; mid-range Blackwell (5080, 5070 Ti) often wins on $/kWh.

---

## Pearl mining community map

| Resource | URL | Role |
|----------|-----|------|
| Official protocol repo | https://github.com/pearl-research-labs/pearl | Node, wallet, vLLM miner, pearl-gemm source |
| AlphaPool (largest community pool) | https://pearl.alphapool.tech/ | Stratum, static-diff tables, Discord/Telegram (linked in footer), Docker image |
| alpha-miner (closed binary, open docs) | https://github.com/AlphaMine-Tech/alpha-miner | Reference hashrates by arch; `--force-backend blackwell` |
| Kryptex Pearl pool + guide | https://pool.kryptex.com/prl | Multi-miner commands, regional endpoints, Discord support |
| PearlPool | https://pearlpool.cloud/ | lpminer / ARC-Miner setup |
| MiningBoard live stats | https://miningboard.com/mining/pearl | Profitability, efficiency rankings |
| Pearl explorer / analytics | https://explorer.mineaitokens.com/analytics | Network hashrate, difficulty, pool share |
| vLLM Docker optimizations | https://github.com/terrapin88/pearl-miner-docker | Community vLLM path: workers, word-list length, enforce-eager |

**Discord:** Pearl team and AlphaPool both reference Discord for mining guides; links are on pool footers (not stable deep URLs). GitHub issue #83 directs solo miners to Discord for workload synthesis recipes.

---

## Source summaries (≥5 distinct sites)

### 1. MiningBoard — *How to Mine Pearl (PRL) in 2026*

| Field | Detail |
|-------|--------|
| **URL** | https://miningboard.com/guides/how-to-mine-pearl-coin |
| **Date** | Updated July 2026 |
| **Key takeaways** | Pearl uses **NoisyGEMM / PoUW** — tensor throughput dominates, not memory bandwidth like KawPow/Ethash. RTX **5090 ≈ 305 TH/s @ 500 W (0.61 TH/W)**; **5080 / 5070 Ti lead efficiency**. Recommends undervolt + power cap; warns revenue **halved post-launch** (~$33 → ~$17/day per 5090). Suggests **rent-before-buy** on vast.ai-style hosts. Lists AlphaPool, Suprnova, 31 pools total. |
| **PropMiner applicability** | **High** — validates PropMiner’s focus on GEMM TMAD/s rather than memory OC. Efficiency framing supports power-capped headless benchmarks. Confirms 5090 as throughput ceiling but not best $/W. |

---

### 2. Kryptex Pool — *How to mine Pearl (PRL)*

| Field | Detail |
|-------|--------|
| **URL** | https://pool.kryptex.com/articles/how-to-mine-pearl-en |
| **Date** | 2026-06-26 |
| **Key takeaways** | Operational onboarding: wallet, regional stratum (`prl.kryptex.network:7048`), PROP vs SOLO. Lists **SRBMiner, PeakMiner, ForgeMiner, ARC-Miner** — notes gzip compression and OC hooks on some miners. Tensor-core GPUs preferred. HiveOS via custom flight sheets; Docker one-liners for Peak/Forge. |
| **PropMiner applicability** | **Medium** — integration checklist for stratum compatibility, worker naming, SSL. No kernel-level detail; useful for deployment parity testing against PropMiner on Kryptex. |

---

### 3. AlphaPool + alpha-miner v1.3.0 release notes

| Field | Detail |
|-------|--------|
| **URL** | https://github.com/AlphaMine-Tech/alpha-miner/releases/tag/v1.3.0 · pool: https://pearl.alphapool.tech/ |
| **Date** | Release 2026-05-16; pool docs ongoing |
| **Key takeaways** | Community reference performance: **H200 ~600 TH/s, H100 550–580, 5090 280–300 TH/s (thermal dependent), 4090 ~160**. Explicit **`sm_120` Blackwell** backend. **Static difficulty** via `--password 'x;d=N'` — 5090 suggested **262144+** (later releases: up to **1,048,576**). Docker `alphaminetech/pearl-miner` tested on **RunPod / vast.ai**. Per-GPU `--devices`, regional stratum hosts (`us2.alphapool.tech`, not HTTPS dashboard host). |
| **PropMiner applicability** | **High (competitive benchmark)** — PropMiner should target ≥ alpha-miner 5090 numbers with open kernel advantage once **tcgen05 consumer path** lands. Static-diff guidance informs pool-side share rate tuning on AlphaPool. |

---

### 4. Igor’s LAB — *Blackwell RTX 5090/5080/5070 overclocking guide*

| Field | Detail |
|-------|--------|
| **URL** | https://www.igorslab.de/en/geforce-rtx-5090-rtx-5080-rtx-5070-ti-and-rtx-5070-significantly-faster-a-blackwell-overclocking-guide-not-just-for-dummies/ |
| **Date** | Early 2026 (multi-page guide, updated through 2026) |
| **Key takeaways** | Blackwell OC differs from Ada: **OC often beats undervolt** for gaming (~**+10–15%** stable, ~300–400 MHz core). GDDR7 often stable **≥2000 MHz** in Afterburner; **memory OC yields diminishing returns** due to already-high bandwidth. **Dual-tool trick** (MSI Afterburner + EVGA Precision X1) bypasses **2000 MHz mem telemetry cap**. Driver quirks: clock stuck at ~2.3 GHz fix via **`nvidia-smi --reset-gpu-clocks`** or reboot. Power limit + cooling quality matter on 575 W TDP. |
| **PropMiner applicability** | **Low–medium for hashrate, medium for stability** — Pearl GEMM is compute-bound; Igor’s mem OC gains unlikely to transfer 1:1. **Thermal/power stability** advice matters: sustained 24/7 mining needs aggressive cooling; **`nvidia-smi` reset** useful on cloud hosts. Linux miners should use **`nvoc`** (see source 6) instead of Afterburner. |

---

### 5. Colfax Research — *CUTLASS Tutorial: GEMM with Tensor Memory for Blackwell*

| Field | Detail |
|-------|--------|
| **URL** | https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/ |
| **Date** | Part 1 published **2025-04-19**; series updated **2026-01-08** |
| **Key takeaways** | Hopper **`wgmma` deprecated** → Blackwell **`tcgen05.mma` (UMMA)**. Accumulators live in **TMEM (256 KB/SM)**, allocated in **32-column granules**, explicit **`tcgen05.alloc/dealloc`**. **Single-thread MMA issue**; warp specialization with **TMA + mbarriers**. **Critical caveat:** article targets **datacenter SM100**; states **consumer CC 12.0 lacks TMEM** — different programming model than B200. |
| **PropMiner applicability** | **Very high (engineering)** — Directly informs PropMiner’s B200 → GB202 port in `01-native-tcgen05-tmem-gemm.md`. Confirms why current **`mma.sync` SM80 atom on sm_120a** is correct-but-slow. Expect **+30–80% TMAD/s** only after consumer-safe tcgen05 path validated against transcript byte identity. |

---

### 6. Modular — *Matrix Multiplication on Blackwell, Part 2*

| Field | Detail |
|-------|--------|
| **URL** | https://modular.github.io/modular/matmul-on-blackwell-part-2/ |
| **Date** | 2025–2026 (engineering blog series) |
| **Key takeaways** | **`tcgen05.mma` tile up to 128×256×16 (1-SM)** vs Hopper 64×256×16; **2-SM mode up to 256×256×16**. TMEM **128 lanes × 512 columns**, 4-byte cells. Demonstrates **TMA + tcgen05 + stmatrix** pipeline; **~58×** over naive kernel, still below cuBLAS. K must align to instruction K=16 multiples. |
| **PropMiner applicability** | **High** — Tile shape and K-alignment constraints should match PropMiner’s **128×256×128** consumer tile and headless PoW grid (8192 CTAs). Useful for validating whether **2-SM cluster** is viable on GB202 (NVIDIA forums: sm_120 supports clusters but not multicast). |

---

### 7. WhatToMine + Kryptex — RTX 5090 Pearl benchmarks

| Field | Detail |
|-------|--------|
| **URL** | https://www.whattomine.com/coins/469-prl-pearl/gpus/92-nvidia-geforce-rtx-5090 · https://pool.kryptex.com/device/gpu/nvidia/nvidia-geforce-rtx-5090 |
| **Date** | Kryptex updated **2026-06-17**; WhatToMine rolling |
| **Key takeaways** | Consensus community numbers: **305–310 TH/s @ 500–510 W (~1.65 J/T)**. WhatToMine “recommended settings”: **core lock 2400 MHz, mem lock 7001 MHz** — treat as starting point, not proven optimal for GEMM. Kryptex @ $0.05/kWh shows **~$8/day** gross on Pearl (highly price/difficulty dependent). |
| **PropMiner applicability** | **High (target KPI)** — PropMiner baseline ~**300 TMAD/s** on RunPod 5090 aligns with community TH/s reporting. Any optimization should be judged on **TH/s, J/T, and accepted share rate** together. |

---

### 8. MDPI Sustainability — *GPU Mining Overclocking and Undervolting* (peer-reviewed)

| Field | Detail |
|-------|--------|
| **URL** | https://www.mdpi.com/2071-1050/14/14/8708 |
| **Date** | **2022** (8× RTX 3060, Ravencoin/KawPow era) |
| **Key takeaways** | Controlled study: **~20% power reduction** via undervolt + OC raised **mining efficiency ~147%** vs stock (MH/W). Method: MSI Afterburner curve editor. Algorithm-specific — memory-heavy KawPow benefited from mem OC + core downclock. |
| **PropMiner applicability** | **Medium (power economics only)** — Validates **power-limit-first** strategy for 24/7 ops. **Do not assume mem OC transfers** to int8 tensor GEMM; use for **electricity cost modeling** and host billing on vast.ai. |

---

### 9. Cool-mining.org — *Undervolting GPUs for mining*

| Field | Detail |
|-------|--------|
| **URL** | https://cool-mining.org/en/mining-en/undervolting-gpus-reducing-power-consumption-during-mining-%f0%9f%9a%80/ |
| **Date** | Undated; evergreen guide |
| **Key takeaways** | **20–40% power savings** with minimal hashrate loss when voltage + power limit tuned together. Introduces **MH/W** as the operator metric. Table of before/after for GTX/RX cards. |
| **PropMiner applicability** | **Medium** — On cloud rentals where power is bundled, focus on **max TH/s within thermal cap**; on home rigs, **cap at 450–510 W** for 5090 matches community sweet spot. |

---

### 10. arXiv — *The Usefulness Gap in Pearl’s cuPOW* (empirical study)

| Field | Detail |
|-------|--------|
| **URL** | https://arxiv.org/html/2606.04819v1 |
| **Date** | **2026-06** (v1) |
| **Key takeaways** | Measures **~24 EH/s network**, analyzes **8,012 AlphaPool workers**; dominant stratum miners run **seeded matmul without AI inference**. Cloud GPU prices **+38%** after miner release. Separates **protocol narrative (useful AI work)** from **dominant mining software behavior**. |
| **PropMiner applicability** | **Contextual** — PropMiner is in the **dedicated pearlhash camp** (same class as alpha-miner), not vLLM inference. Useful for **external communications** and understanding pool economics; does not change kernel optimization priorities. |

---

### 11. GitHub `martinstark/nvoc` — Linux OC for RTX 50-series

| Field | Detail |
|-------|--------|
| **URL** | https://github.com/martinstark/nvoc/ |
| **Date** | Active 2026 |
| **Key takeaways** | CLI for **locked clocks, offsets, power limit %** on Blackwell Linux. Example 5090: `-c 200,2820 -o 856 -m 2000 -p 105`. Notes **NVML cannot fine-grained undervolt** per voltage point. Requires **nvidia-open 550+**. |
| **PropMiner applicability** | **High (ops)** — PropMiner headless on vast.ai/RunPod should script **`nvoc`/`nvidia-smi` power caps** in entrypoint; reproducible perf across hosts. |

---

### 12. Community vLLM path — `terrapin88/pearl-miner-docker` commits

| Field | Detail |
|-------|--------|
| **URL** | https://github.com/terrapin88/pearl-miner-docker (commits May 2026) |
| **Date** | **2026-05-07 – 2026-05-09** |
| **Key takeaways** | **Not PropMiner architecture**, but influential in Pearl Discord/vast.ai circles: **`WORD_LIST_LENGTH` 700→1400** forces **prefill M≥1024** for NoisyGEMM (~**9×** effective mining vs short prompts). **`PEARL_MAX_TOKENS=1`** minimizes decode. **`PEARL_WORKERS = 32 × GPU_COUNT`** fixes multi-GPU starvation. **`PEARL_ENFORCE_EAGER=1`** required (CUDA graphs crash NoisyGEMM). **`VLLM_USE_DEEP_GEMM=0`**. Documented **+35–50% effective hashrate** when combined. |
| **PropMiner applicability** | **Low for kernel; medium for ecosystem** — Explains vLLM miner behavior and **`min_m`** confusion in GitHub issues. PropMiner bypasses inference batching entirely via direct GEMM. |

---

## Proven techniques vs folklore / myths

| Category | Technique | Verdict | Notes |
|----------|-----------|---------|-------|
| **Proven** | Power-limit / undervolt to **450–510 W** on 5090 | ✅ | Matches pool benchmarks; improves J/T with small TH/s loss |
| **Proven** | **Thermal headroom** (fans, water, datacenter airflow) | ✅ | alpha-miner cites **280 vs 300 TH/s** throttle gap |
| **Proven** | **Static stratum difficulty** (`x;d=N`) | ✅ | AlphaPool + alpha-miner; reduces vardiff warmup, stabilizes share rate |
| **Proven** | **Architecture-specific CUDA kernels** (Hopper vs Blackwell backends) | ✅ | Large hashrate gaps between `--force-backend` paths |
| **Proven** | **CUDA 12.8+ / driver 570+** for sm_120 | ✅ | PyTorch/NVIDIA forums; PTX ISA 8.7 adds `sm_120` |
| **Proven** | **tcgen05/TMEM/TMA** for datacenter Blackwell GEMM | ✅ | CUTLASS, Colfax, Modular — not yet consumer PropMiner default |
| **Proven** | Network **difficulty rise erodes revenue** faster than hardware OC gains | ✅ | MiningBoard, explorer analytics |
| **Folklore** | **Memory OC +2000 MHz** dramatically boosts Pearl TH/s | ❌ / weak | GDDR7 bandwidth already high; Igor’s LAB: diminishing returns; Pearl is tensor-bound |
| **Folklore** | **Core clock max OC** always raises mining income | ❌ | Gaming +10% ≠ GEMM +10%; power/thermals often negate |
| **Folklore** | **“Mine while doing AI inference”** is how most PRL is mined today | ⚠️ Mixed | arXiv study: dominant pool software is **direct matmul**; vLLM path is minority |
| **Folklore** | **Longer prompts alone** help PropMiner/stratum miners | ❌ | Applies only to **vLLM NoisyGEMM** (`min_m`); stratum miners ignore token count |
| **Folklore** | **LN2 / 1000 W BIOS mods** for 24/7 mining ROI | ❌ | TechPowerUp stunt; irrelevant to datacenter economics |
| **Folklore** | **Switching to AI compute rental always beats mining** | ⚠️ Context | MiningLegit-style articles promote io.net/Akash; true at some power prices, not universal |
| **Folklore** | **Official Pearl vLLM miner works on RTX 5090 out of the box** | ❌ (mid-2026) | Issue #104: **sm_90a-only** build; community miners filled gap |

---

## Electricity cost angle: earning vs power draw

### Metrics operators use

- **TH/s** — raw matrix throughput (Pearl pools report as hashrate)
- **J/T or TH/W** — **lower J/T is better**; Kryptex cites **1.65 J/T @ 510 W, 310 TH/s** for 5090
- **Net profit = (TH/s × revenue_per_TH) − (W × 24h × $/kWh)** — MiningBoard uses **~$0.0245/TH/day** revenue (July 2026)

### Example scenarios (RTX 5090 @ 305 TH/s, 500 W)

| Electricity rate | Daily power cost | Gross revenue (~$0.0245/TH) | Approx. net |
|------------------|------------------|----------------------------|-------------|
| $0.05/kWh | $0.60 | $7.47 | **+$6.87** |
| $0.10/kWh | $1.20 | $7.47 | **+$6.27** |
| $0.15/kWh | $1.80 | $7.47 | **+$5.67** |
| $0.25/kWh | $3.00 | $7.47 | **+$4.47** |

*(Revenue illustrative from MiningBoard; PRL price and difficulty move daily. Pool calculators at $0.05/kWh showed higher gross — treat as upper bound during favorable markets.)*

### Efficiency ranking (community, July 2026)

From MiningBoard guide — **5080 (0.652 TH/W) and 5070 Ti (0.583)** beat **5090 (0.610)** on efficiency despite lower absolute TH/s. For **cloud hosts** where $/hour is fixed, maximize **TH/s**; for **owned hardware**, maximize **TH/W**.

### PropMiner implication

Benchmark campaigns should report **TMAD/s, wall or TDP power, J/T, and accepted shares/hour** at pinned difficulty. Optimizations that add +5% TMAD/s but +15% power may **reduce** profit below $0.15/kWh.

---

## GitHub issues and discussions (selected)

| # | Repo | Title | Date | Relevance |
|---|------|-------|------|-----------|
| [#104](https://github.com/pearl-research-labs/pearl/issues/104) | pearl-research-labs/pearl | RTX 50-series (sm_120) not supported — no kernel image | 2026-05-17 | Official vLLM pearl-gemm **sm_90a-only**; alpha-miner already ships Blackwell — mirrors PropMiner’s separate consumer cubin work |
| [#83](https://github.com/pearl-research-labs/pearl/issues/83) | pearl-research-labs/pearl | Solo mining not triggering PoUW — **min_m never reached** | 2026-05-05 | Documents **M≥1024** for NoisyGEMM; decode M=1 useless; community fixed via long prompts / batching — **N/A to PropMiner stratum** but explains protocol |
| [#156](https://github.com/pearl-research-labs/pearl/issues/156) | pearl-research-labs/pearl | Pool-side per-share verify cost / faster jackpot | 2026-06-03 | Protocol-level; may affect future block cadence and share economics |
| [#26](https://github.com/AlphaMine-Tech/alpha-miner/issues/26) | AlphaMine-Tech/alpha-miner | v1.8.0 low hashrate — miner 117 TH/s, **pool 27 TH/s** | 2026-06-19 | **Static diff / pool reporting** mismatch; compare PropMiner pool-side vs local TMAD/s carefully |
| [#28](https://github.com/AlphaMine-Tech/alpha-miner/issues/28) | AlphaMine-Tech/alpha-miner | v1.8.x illegal memory access after 30 min | 2026-06-23 | Stability under OC + pipeline flush; relevant for **long-run vast.ai** reliability testing |

**Related PRs/commits worth watching:** pearl [#118](https://github.com/pearl-research-labs/pearl/pull/118) Blackwell sm_120a support; terrapin88 pearl-miner-docker commits on **WORD_LIST_LENGTH / PEARL_WORKERS** (vLLM path).

---

## CUDA 12.8 / Blackwell / tensor-core — community ↔ PropMiner bridge

| Community knowledge | PropMiner status |
|--------------------|------------------|
| alpha-miner **`blackwell` / `blackwell-native` backends** | PropMiner builds **`PEARL_GEMM_ARCH=blackwell`** → `sm_120a` cubin |
| Colfax/Modular **tcgen05 + TMEM on SM100** | Planned port from `transcript_gemm_sm100.cu`; consumer **may lack TMEM** — verify against GB202 PTX |
| **`mma.sync` SM80 atom** still used on 5090 | Current production path; **byte-identical transcript**, ~300 TMAD/s ceiling |
| **CUDA 12.8, driver 570+, sm_120 PTX** | Required for native Blackwell; avoid PTX JIT fallback penalties |
| **Warp specialization, TMA, mbarriers** | Documented in `01-native-tcgen05-tmem-gemm.md` as target architecture |
| **Headless PoW** (in-kernel BLAKE3) | PropMiner differentiator vs pool miners — community binaries opaque |

**Recommended community benchmarks to reproduce:** alpha-miner v1.3+ on 5090 (**280–300 TH/s**), Kryptex/WhatToMine (**305–310 TH/s @ 500 W**), PropMiner internal **~300 TMAD/s** — close parity suggests kernel correctness; gap to **H100 550+** shows architecture headroom.

---

## Actionable takeaways for PropMiner

1. **Prioritize kernel work (`tcgen05` consumer path) over OC guides** — community TH/s spread on 5090 is narrow; software dominates.
2. **Script power caps in Docker entrypoint** — target **500 W / 2400 MHz core lock** as first A/B baseline (WhatToMine community settings).
3. **Pool integration:** support **`x;d=N`** static difficulty; document recommended **d=262144–1048576** for 5090-class.
4. **Publish TMAD/s + J/T + share acceptance** — distinguish from vLLM “effective hashrate” folklore.
5. **Monitor pearl-research-labs/pearl #104 / PR #118** — official sm_120 may change reference verifier assumptions.
6. **Do not optimize for memory bandwidth** unless profiling shows GEMM memory-bound on GB202 (unlikely for int8 tensor path).

---

## References (quick list)

1. https://miningboard.com/guides/how-to-mine-pearl-coin  
2. https://pool.kryptex.com/articles/how-to-mine-pearl-en  
3. https://github.com/AlphaMine-Tech/alpha-miner/releases/tag/v1.3.0  
4. https://pearl.alphapool.tech/  
5. https://www.igorslab.de/en/geforce-rtx-5090-rtx-5080-rtx-5070-ti-and-rtx-5070-significantly-faster-a-blackwell-overclocking-guide-not-just-for-dummies/  
6. https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/  
7. https://modular.github.io/modular/matmul-on-blackwell-part-2/  
8. https://www.whattomine.com/coins/469-prl-pearl/gpus/92-nvidia-geforce-rtx-5090  
9. https://www.mdpi.com/2071-1050/14/14/8708  
10. https://cool-mining.org/en/mining-en/undervolting-gpus-reducing-power-consumption-during-mining-%f0%9f%9a%80/  
11. https://arxiv.org/html/2606.04819v1  
12. https://github.com/martinstark/nvoc/  
13. https://forums.developer.nvidia.com/t/thread-block-clustering-in-blackwell-gpus/320471  
14. https://docs.nvidia.com/cutlass/4.3.4/media/docs/cpp/blackwell_functionality.html  

---

*Original synthesis from public web sources and PropMiner internal planning docs. Figures are community-reported unless otherwise noted; verify live before financial decisions.*
