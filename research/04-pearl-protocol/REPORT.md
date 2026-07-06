# Pearl (PRL) Protocol & Mining — Research Report

**Date:** 2026-07-06  
**Scope:** Economic and technical overview of Pearl, proof-of-work mechanics, pool wire protocols, job parameters, hashrate normalization, share validation, and planned MoE fork.  
**Primary sources:** Pearl whitepaper, Pearl GitHub monorepo, PropMiner implementation (`proto/mining_v2`, `pearl-gemm`, share builder), ARC-miner protocol docs, Kryptex pool documentation, academic literature on cuPOW.

---

## 1. What is Pearl / PRL?

### 1.1 Economic model

Pearl (ticker **PRL**) is a Layer-1 blockchain launched by **Pearl Research Labs** that replaces Bitcoin-style SHA-256 hashing with **Proof-of-Useful-Work (PoUW)** based on GPU matrix multiplication. The economic pitch is “2-for-1”: the same GPU cycles that run AI training/inference can simultaneously produce block-eligible work, turning energy spent on matmul into a monetary asset ([Pearl Whitepaper](https://pearlresearch.ai/Pearl_Whitepaper.pdf), [Hashrate Index overview](https://hashrateindex.com/blog/pearl-prl-ai-compute-cryptocurrency/)).

Key monetary parameters from the whitepaper:

| Parameter | Value |
|---|---|
| Max supply | 2.1 billion PRL (100× Bitcoin’s 21M) |
| Smallest unit | 10⁻⁸ PRL (“grain”) |
| Target block time | ~194 seconds (3:14) |
| Emission | Smooth 1/t² decay (no halving cliffs); ~50% emitted in first ~4 years |
| Difficulty adjustment | WTEMA (Weighted-Target EMA), ~7-day characteristic time |
| Ledger model | Bitcoin-fork UTXO chain with Taproot addresses, XMSS post-quantum path reserved |

The whitepaper’s “Econ-101” section models two miner populations: GPUs doing real AI work (low marginal cost, ~10% overhead) versus “pure miners” paying full cloud rates (~$1.70/H100-hour in their example). Block rewards flow to whichever population finds valid proofs first; the token’s store-of-value thesis depends on adoption as a payment/settlement layer for AI compute markets.

**Important nuance (empirical):** A 2026 study of live Pearl mining found that miners overwhelmingly use **random int8 matrices**, not actual model weights. The protocol cryptographically guarantees *correct noisy GEMM*, not *useful AI output*. “Usefulness” is framed as an economic property, not a consensus guarantee ([arXiv:2606.04819](https://arxiv.org/html/2606.04819v2)).

### 1.2 Technical stack

Pearl’s reference implementation ([pearl-research-labs/pearl](https://github.com/pearl-research-labs/pearl)) is a Bitcoin-fork full node (`pearld`) plus:

- **Oyster** wallet (JSON-RPC / gRPC)
- **vLLM miner** (Python/CUDA) with `pearl-gateway` bridge
- **zk-pow** — Plonky2/STARKy ZK circuit that compresses plain proofs into ~60KB block certificates
- **pearl-blake3** — keyed Merkle commitments over matrix data
- **py-pearl-mining** — Rust/PyO3 bindings for the mining primitives

Mining at the protocol layer is **NoisyGEMM** (also called **cuPOW** in the academic paper “Proofs of Useful Work from Arbitrary Matrix Multiplication,” Komargodski–Schen–Weinstein). Solo mining connects to `pearld` via UDS/TCP; pool mining uses Stratum or gRPC V2 edge protocols implemented by pool operators.

---

## 2. Proof-of-Work: Noisy GEMM

### 2.1 High-level loop

For each mining attempt the miner:

1. Holds matrices **A** (M×K) and **B** (N×K, stored row-major as Bᵀ rows), with entries in **int7/int8** range (typically −64…+64).
2. **Commits** to A and B via BLAKE3 keyed Merkle trees rooted at `hashA`, `hashB`.
3. Derives **noise seeds** from the job key and commitments, then builds low-rank noise **E**, **F** (rank **R**) and forms **A′ = A + E**, **B′ = B + F**.
4. Runs a **tiled int GEMM** on A′·B′, maintaining a per-tile **transcript** (512-bit state, 16×uint32).
5. After each rank-R chunk along K, XOR-reduces the int32 accumulator tile and **mixes** into the transcript via rotate-left-13 ⊕ XOR.
6. Declares a **hit** when `BLAKE3(transcript, key)` ≤ target, where target encodes difficulty **b** (nbits) scaled by tile geometry.

This binds PoW to the **full execution trace** of the tiled multiply, not merely the final product — preventing shortcuts like multiplying zero matrices or skipping K-dimension accumulation ([Whitepaper §3.1–4.5](https://pearlresearch.ai/Pearl_Whitepaper.pdf)).

### 2.2 Transcript

The transcript is the protocol’s “mining scratchpad.” PropMiner’s `pearl-gemm` kernels implement it identically across architectures:

- At each K-boundary (every **R** steps along K), XOR all int32 values in the proof-canonical tile (e.g. 128×256 for consumer/H100 layout).
- Update slot `ℓ mod 16`: `M[ℓ mod 16] ← (M[ℓ mod 16] ≪ 13) ⊕ X`.
- After K/R steps, keyed BLAKE3 of the 16-word jackpot → 256-bit hash compared to PoW target.

The **headless** mining path fuses GEMM + transcript + target check in-kernel (`launch_transcript_gemm_headless`), avoiding a separate finalize pass. Proof-canonical tile dimensions (**bM=128, bN=256, bK=128**, periodic row/col patterns) must match what the pool verifier expects — PropMiner deliberately keeps the SM80 `mma.sync` atom inside sm_120a builds for byte-identical transcripts ([PropMiner docs](../docs/README.md), [transcript_gemm_sm120_geforce.cu](../../third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm120_geforce.cu)).

### 2.3 BLAKE3 commitments

BLAKE3 serves three roles:

1. **Job key:** `job_key = BLAKE3(σ ‖ config_bytes)` — binds work to blockchain header fragment σ and mining configuration ([job_key.cpp](../../src/host/pearl/job_key.cpp)).
2. **Matrix Merkle trees:** Keyed BLAKE3 over 1024-byte leaf chunks of A rows / B columns → `hashA`, `hashB`.
3. **Jackpot hash:** Keyed BLAKE3 of the 16-word transcript state → `claimed_hash` checked against difficulty.

Noise seeds chain as: `b_noise_seed = BLAKE3(job_key ‖ hashB)`, `a_noise_seed = BLAKE3(b_noise_seed ‖ hashA)` ([share_builder.cpp](../../src/host/pearl/share_builder.cpp)).

### 2.4 Difficulty and DAF (Difficulty Adjustment Factor)

Pool-facing difficulty uses Bitcoin-style **nbits** encoding a 256-bit target. Pearl scales per-tile difficulty by a **DAF** so that larger proof tiles represent proportionally more work:

```
DAF = rows_pattern.size × cols_pattern.size × dot_product_length
```

For PropMiner’s default H100/5090 patterns: **rows = 2**, **cols = 64**, **dot_product_length = K** (quantized to multiples of 128) → DAF = **128 × K** at K=128 → **16,384**.

The effective PoW target is:

```
adjusted_target = nbits_to_target(nbits) × DAF   (256-bit LE multiply, clamped)
```

A share is valid when `claimed_hash ≤ adjusted_target` ([pow_target_utils.h](../../src/host/pearl/pow_target_utils.h), [pearl_types.cpp](../../src/host/pearl/pearl_types.cpp)).

**Protocol-level difficulty** (whitepaper Algorithm 4) uses `BLAKE3(M) ≤ 2^(256−b) · r · tm · tn` — the DAF absorbs tile area and K-depth so pool vardiff can compare miners running different M/N while holding K/R/tile-pattern fixed.

---

## 3. Pool Protocols

Pearl pools expose two families of interfaces. PropMiner v2 implements **gRPC V2 only**; Stratum code exists in-tree but is not linked in the production binary.

### 3.1 gRPC V2 (Kryptex default, Akoya-style edge)

**Wire format:** HTTP/2 + TLS + protobuf over bidirectional stream `MinerService/MiningStream`. Schema: `pearlpool.mining.v2` ([ARC-miner/proto/v2/miner.proto](../../../ARC-miner/proto/v2/miner.proto), mirrored in PropMiner’s hand-rolled [mining_v2.h](../../src/host/pearl/proto/mining_v2.h)).

**Lifecycle:**

1. **`Register`** — wallet, worker, GPU cards, `protocol_version=2`, claimed `k`. Server returns `miner_id`, `session_token`, `initial_difficulty_nbits`, `pool_id`.
2. **`MiningStream`** — first miner message is `AuthEvent` with session token; first pool message is `JobAssignment`.
3. Miner sends `ShareSubmission`, `Heartbeat` (~30s), `Ping`.
4. Pool sends `JobAssignment`, `ShareResult`, `DifficultyAdjust` (vardiff), `Pong`, `Error`, `ReconnectHint`.

**Kryptex specifics:** PropMiner defaults to `prl.kryptex.network:443` (TLS gRPC), not the public Stratum port 7048 documented for third-party miners ([PropMiner README](../../README.md)). Kryptex operates both transports: Stratum for SRBMiner/BzMiner/PeakMiner, gRPC/TLS for PropMiner-class clients.

**Security properties (V2 proto comments):**

- `b_seed` is per-(miner, σ), pool-private; miner echoes it in shares for verification.
- No client-supplied share ID — pool dedupes by recomputed jackpot hash.
- `audit_k` enables random B-tree leaf audits on every share (see §6).

### 3.2 Stratum variants

| Variant | Detection | Transport | Notes |
|---|---|---|---|
| **Classic client-first** | Client sends `mining.subscribe` first | TCP `:7048` (Kryptex global) | JSON-RPC line protocol; `pearlhash` algorithm name |
| **Pearl `pearl/v1` challenge-first** | Pool sends `pearl.challenge` immediately after connect | TLS common (e.g. HeroMiners `:1200`) | BLAKE3 connect PoW (~32 leading zero bits); then `mining.configure` with `pearl/v1` |
| **Positional notify/submit** | `mining.notify` params are arrays, not objects | Same | Share = base64 plain proof; difficulty via password `x;d=N` |

Challenge-first handshake ([ARC-miner PEARL-V1-CHALLENGE.md](../../../ARC-miner/docs/PEARL-V1-CHALLENGE.md)):

```
Pool → pearl.challenge {seed, difficulty}
Client → pearl.challenge_response {nonce, seed}
Client → mining.configure [["pearl/v1"], …]
Client → mining.subscribe / mining.authorize
Pool → pearl.set_mining_params {m,n,k,rank,rows_pattern,cols_pattern,mma_type}
Pool → mining.notify / mining.set_difficulty
```

`pearl.set_mining_params` fixes the **MiningConfiguration** for the session — pools enforce compatible M/N/K/R and tile patterns here.

---

## 4. Job Parameters

### 4.1 σ (sigma)

**σ** is a **76-byte incomplete block header** fragment (`kSigmaHeaderBytes = 76`) representing chain state at job time. It is opaque to miners except for:

- First 8 bytes → `sigma_seed` (little-endian uint64) used as RNG entropy for A regeneration.
- Hashed with config to form `job_key`.

When σ changes, miners **re-install** context: expand B from `b_seed`, rebuild B Merkle tree, upload noise_B to VRAM ([sigma_context.cpp](../../src/host/pearl/sigma_context.cpp)).

### 4.2 b_seed

**32-byte seed** per (miner, σ) assignment. Miners treat it as opaque; pools derive it as `BLAKE3(poolSecret ‖ minerId ‖ σ)` ([miner.proto `JobAssignment`](../../../ARC-miner/proto/v2/miner.proto)).

Deterministic expansion via BLAKE3-XOF produces the full B matrix (N×K bytes). The pool verifies shares by re-expanding and checking Merkle paths — miners cannot invent arbitrary B.

### 4.3 M, N, K, R and pool constraints

**Miner-chosen (reported in share):**

| Field | Role | PropMiner default (5090 prod) |
|---|---|---|
| **M** | A rows | 8192 |
| **N** | B columns (Bᵀ rows) | up to 262144 (VRAM-limited) |
| **K** | Inner dimension | 128 |
| **R** | Noise rank | 128 |

**Protocol-enforced (whitepaper §4.8):**

- m, n ≤ 2²⁴
- 16R ≤ K ≤ 4R², K ≤ 2¹⁶, **64 | K**
- R ∈ {32, 64, …, 1024} (powers of two from 2⁵ to 2¹⁰)
- Tile patterns as 3D arithmetic progressions; partial edge tiles ineligible for PoW
- k(h + w) ≤ 2²² where h, w are committed hash-tile dimensions

**Serialized config (52 bytes)** includes K, R, MMA type (`Int7xInt7ToInt32`), and 6-byte encodings of `rows_pattern` / `cols_pattern` — must match C#/pool reference ([pearl_types.h](../../src/host/pearl/pearl_types.h)).

**Register-time K:** V2 `RegisterRequest.k` declares the miner’s K; pools may reject mismatches. PropMiner sends `cfg.mining_config.k` at register ([worker_orchestrator.cpp](../../src/host/pearl/worker_orchestrator.cpp)).

**audit_k:** Count of random B Merkle leaves the pool audits per share (0–64). Indices derived from `BLAKE3-XOF("akoya-audit-v1" ‖ claimed_hash ‖ b_seed ‖ K)`.

---

## 5. Why Matrix Dimensions Matter for Hashrate

Raw GPU throughput is measured in **tiles/sec** (CTA tiles of size bM×bN). Pools and dashboards expect **DAF-normalized hashrate**:

```
hashrate_H/s ≈ tiles_per_sec × DAF
             = tiles_per_sec × rows × cols × K
```

PropMiner’s orchestrator computes bench hashrate this way ([worker_orchestrator.cpp](../../src/host/pearl/worker_orchestrator.cpp)):

```cpp
tiles_per_iter = (M / bM) × (N / bN)
hashrate = tiles_per_sec × difficulty_adjustment_factor()
```

**Implications:**

1. **N dominates.** Doubling N doubles tile count and roughly doubles normalized hashrate if memory bandwidth allows — this is why RTX 5090 tuning pushes N toward 262144.
2. **M/K/R/patterns are usually fixed** across a pool session; comparing miners requires identical config_bytes.
3. **Misreported M/N** in shares is caught at validation (verifier recomputes geometry from config_bytes and opened indices).
4. **Register `claimed_total_hashrate`** seeds initial vardiff; heartbeats report measured rates. Under-reporting yields easy shares; over-reporting yields rejections until vardiff adjusts ([miner.proto `DifficultyAdjust.measured_hashrate`](../../../ARC-miner/proto/v2/miner.proto)).

Console display may scale to TH/s for readability, but the protocol unit is **DAF-weighted tile evaluations per second** ([PropMiner docs](../../docs/README.md)).

---

## 6. Share Validation — What Cannot Be Cheated or Skipped

When a GPU hits, the host builds a **ShareSubmission** protobuf. The pool (and any full node) re-derives and checks:

### 6.1 Mandatory verification steps

| Step | What is checked | Skip / cheat outcome |
|---|---|---|
| **Merkle proofs for A** | Opened row leaves + siblings → `hashA` under `job_key` | Reject: invalid A commitment |
| **Merkle proofs for B** | Opened column leaves + siblings → `hashB`; `b_seed` matches assignment | Reject: wrong B or wrong miner binding |
| **Noise reconstruction** | From seeds, rebuild E/F strips for opened rows/cols | Reject: transcript mismatch |
| **Jackpot / claimed_hash** | Recompute transcript mixing over K/R steps, BLAKE3 → hash | Reject: `claimed_hash_mismatch` |
| **Target** | `claimed_hash ≤ nbits_target × DAF` | Reject: below target |
| **Tile indices** | `(tile_row, tile_col)` consistent with opened patterns | Reject: wrong tile |
| **Config bytes** | 52-byte serialization matches claimed K, R, patterns | Reject: config drift |
| **Audit proof** (if audit_k > 0) | Random B leaves hash to `hashB` | Reject: skipped B expansion / Merkle build |

PropMiner’s `ShareBuilder::build` recomputes claimed_hash before submit and drops shares that fail the DAF-scaled target locally ([share_builder.cpp](../../src/host/pearl/share_builder.cpp)).

### 6.2 What miners *can* choose (by design)

- **A matrix content** — any int8 values in range; reference impl uses uniform random ([arXiv study](https://arxiv.org/html/2606.04819v2)).
- **Nonce / which tile** — search strategy is free; only hits matter.
- **M/N within protocol bounds** — affects hashrate reporting, not validity (if config is consistent).

### 6.3 What miners *cannot* skip

- **Full K/R transcript steps** for the opened tile — partial GEMM does not produce valid jackpot.
- **BLAKE3 tensor-hash of A** each nonce batch — `hashA` must match opened rows.
- **B expansion from pool `b_seed`** — audit_k > 0 forces random leaf proofs every share.
- **Correct tile periodic patterns** — kernel, config_bytes, and proof must align (misaligned `cluster_m` vs patterns causes silent share failures).

### 6.4 Block finds vs pool shares

Pool shares use **plain proofs** (Merkle + transcript metadata). When `claimed_hash` clears **network_target_nbits**, the pool wraps the share into a **ZK block certificate** (~60KB) for `pearld` — miners do not produce ZK proofs in V2 ([miner.proto comment on ShareSubmission](../../../ARC-miner/proto/v2/miner.proto)).

---

## 7. Future: Mixture-of-Experts (MoE) Fork

Pearl Research Labs has documented a planned **MoE hard fork** (upstream `docs/moe-fork-upgrade-guide.md`, summarized in [ARC-miner MOE-PORT-PLAN.md](../../../ARC-miner/docs/MOE-PORT-PLAN.md)):

### 7.1 What changes

At fork height `MoEForkHeight` (TBD):

- Block certificates upgrade from **V1 (dense) ZK** to **V2 (MoE-capable) ZK**.
- `getblocktemplate` adds `requiredcertversion` (1 = dense-only before fork, 2 = MoE allowed at/after fork).
- **Dense mining remains valid indefinitely** after fork — MoE is additive, not a replacement.

### 7.2 MoE mining mechanics (optional capability)

| Stage | Dense (today) | MoE (future) |
|---|---|---|
| Routing | — | Deterministic `topk_ids` → counting-sort → per-expert segments |
| B matrices | Single B from b_seed | **E expert weight matrices** |
| GEMM | One noisy GEMM | **Grouped GEMM** per expert segment |
| Commitment | hashA, hashB | Includes **routing data** in commitment |
| Plain proof | Fixed layout | Variable-length MoE public data (`MoEProofParams`) |

Pre-fork MoE shares are **invalid** (pools must reject). Miners need explicit gating on `requiredcertversion == 2`.

### 7.3 Ecosystem status (July 2026)

- Reference Rust entry points: `mine` vs `mine_moe` in `py-pearl-mining`.
- `pearl_mining` v0.2.0 adds MoE types and V2 plain-proof serialization.
- No production pool requirement for MoE yet; economics vs dense mining still TBD.
- CUTLASS / pearl-gemm trees contain MoE grouped-GEMM scaffolding (SM100/SM120 tests) but PropMiner production path is **dense only**.

---

## 8. Summary Table

| Topic | Pearl / PRL |
|---|---|
| **Consensus work** | Noisy int GEMM + transcript + BLAKE3 jackpot |
| **Chain** | Bitcoin-fork UTXO, ~194s blocks, 2.1B cap |
| **Solo path** | pearld + pearl-gateway + vLLM miner |
| **Pool (PropMiner)** | gRPC V2 TLS :443 (Kryptex) |
| **Pool (ecosystem)** | Stratum :7048, pearl/v1 challenge TLS, LuckyPool/HeroMiners/etc. |
| **Job inputs** | σ (76 B), b_seed (32 B), target_nbits, audit_k |
| **Hashrate unit** | tiles/s × DAF (rows×cols×K) |
| **Anti-cheat core** | Merkle commitments + full transcript recompute + optional B audits |
| **Next fork** | MoE grouped GEMM (optional); dense survives |

---

## References

1. Pearl Research Labs, *Pearl Whitepaper* — https://pearlresearch.ai/Pearl_Whitepaper.pdf  
2. Pearl Research Labs GitHub — https://github.com/pearl-research-labs/pearl  
3. Komargodski, Schen, Weinstein, *Proofs of Useful Work from Arbitrary Matrix Multiplication* (cuPOW foundation)  
4. arXiv:2606.04819v2, *The Usefulness Gap in Proof-of-Useful-Work: An Empirical Study of Pearl’s cuPOW Protocol*  
5. Hashrate Index, *Pearl (PRL): The AI-Compute Cryptocurrency, Explained* — https://hashrateindex.com/blog/pearl-prl-ai-compute-cryptocurrency/  
6. Kryptex Pool Pearl docs — https://pool.kryptex.com/articles/how-to-mine-pearl-en  
7. PropMiner source: `src/host/pearl/proto/mining_v2.h`, `pearl_types.h`, `share_builder.cpp`, `sigma_context.cpp`  
8. pearl-gemm: `third_party/pearl-gemm/csrc/portable/transcript_kernel.cuh`, Blackwell/consumer transcript kernels  
9. ARC-miner: `proto/v2/miner.proto`, `docs/PEARL-V1-CHALLENGE.md`, `docs/MOE-PORT-PLAN.md`, `docs/POOLS.md  

---

*Report produced for PropMiner research track 04. No code changes.*
