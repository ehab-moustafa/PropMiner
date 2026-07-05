# PCIe Gen5 x16 + PSU Headroom for RTX 5090 Pearl Mining

**PropMiner performance optimization series — document 07**  
**Scope:** Hardware and host configuration (not miner code)  
**Target GPU:** NVIDIA GeForce RTX 5090 (GB202, Compute Capability 12.0)  
**Expected uplift for Pearl specifically:** **0–10%**, and only if you are *already* limited by PCIe width, power delivery, or thermal throttling — not from “faster bus” alone.

---

## 1. Executive Summary

The RTX 5090 is a **575 W-class** card on a **PCIe Gen5 x16** link. For most workloads, Gen5 bandwidth is headroom; for **PropMiner Pearl mining**, it is almost always **not the bottleneck**.

PropMiner’s RTX 5090 path is deliberately **GPU-isolated**:

- Matrix **B** and noise state stay **VRAM-resident** across iterations (`SigmaContext` resident buffers).
- The host uploads an **8-byte seed** per batch on a dedicated copy stream (`seed_copy_stream_`), overlapped with Tensor Core GEMM on ping/pong compute streams.
- **Device-to-host (D2H)** traffic happens only on a **rare share hit** (compact proof inputs: leaf CVs, A row slices, opened leaves, 32-byte B hash).

That design means Pearl spends its time in **GDDR7 bandwidth and Tensor Core math**, not on the PCIe bus. Upgrading from Gen4 x16 to Gen5 x16, or fixing a marginal PSU, will **not** magically add 10% hashrate on a healthy build — but fixing **x8 lane downgrade**, **12VHPWR undervoltage**, or **power-limit throttling** can recover lost performance that *looks* like a software regression.

**Bottom line:** Treat this guide as **infrastructure hygiene** for a 5090 mining rig. Verify Gen5 x16 and PSU headroom once at build time; then spend optimization effort on kernel tuning, occupancy (`Rtx5090Profile`), and thermals — not PCIe tuning knobs.

---

## 2. Why It Matters for PropMiner (Honest Assessment)

### What PropMiner actually moves over PCIe

| Traffic type | Direction | Size (typical) | Frequency |
|--------------|-----------|----------------|-----------|
| Batch seed | H2D | **8 bytes** (`uint64_t`) | Every batch (~16–24 GEMM iters) |
| PoW target / sigma install | H2D | Small (KB scale) | Per job / sigma rotation |
| Share proof reconstruction | D2H | KB–low MB (pinned staging) | **Rare** (only on PoW hit) |
| Matrix B / noise tensors | — | **None per iter** | Resident in VRAM |

The `GpuWorker` comments describe a “PCIe Gen5 conveyor belt” for seeds — that is **architectural overlap** (upload next seed while GEMM runs), not a high-bandwidth data pipeline. Even at 20 batches per second, seed traffic is **160 bytes/s**.

### PCIe Gen5 x16 vs Pearl demand

| Link | Approx. effective bandwidth (one direction) | Pearl seed demand |
|------|---------------------------------------------|-------------------|
| PCIe 4.0 x16 | ~32 GB/s | ~0.0000002 GB/s |
| PCIe 5.0 x16 | ~64 GB/s | ~0.0000002 GB/s |

Pearl uses **far less than 0.001%** of Gen4 x16 capacity for normal mining. **Gen5 vs Gen4 is irrelevant to steady-state hashrate** unless something else is wrong (lane bifurcation, broken riser, ASPM power states causing latency spikes — all uncommon).

### Where hardware *can* still matter (+0–10%)

| Failure mode | Symptom | Mechanism |
|--------------|---------|-----------|
| GPU in **x8** or **x4** slot | `nvidia-smi` / BIOS shows reduced lanes | Still enough for seeds, but may correlate with bad slot/riser/power delivery issues |
| **12VHPWR** high resistance / adapter | Crashes, errors, **power throttling** | GPU cannot sustain 575 W peaks → lower clocks |
| **Undersized PSU** or shared 12V rail | Transient shutdowns, PL cap | Sustained hashrate drop |
| **Thermal + power** limit interaction | Clock oscillation under load | Looks like “unstable” H/s |
| **Wrong PCIe slot** on multi-GPU board | Secondary slot wired x8 | Irrelevant for Pearl bandwidth; relevant for power/thermals placement |

**Realistic expectation:** A correctly configured Gen4 x16 + adequate PSU performs **the same** as Gen5 x16 for Pearl. Gains appear when you **fix an existing defect**, not when you “optimize” an already-healthy link.

---

## 3. PCIe Gen5 x16 Requirements

### 3.1 Motherboard slot

- Use the **primary CPU-direct PCIe x16 slot** (usually top slot, labeled PCIEX16_1 or similar).
- Confirm in the manual: **PCIe 5.0 x16** from the CPU (not chipset-only Gen4).
- Avoid sharing lanes with multiple NVMe drives in configurations that drop the GPU to **x8** — check the board’s lane allocation table.

**RTX 5090 reference:** Single card should negotiate **x16** at **Gen 5** (reported as `16` link width, `5` or `5.0` link speed).

### 3.2 Risers and extenders

Mining frames often use risers. Risers are a common source of **instability**, not bandwidth starvation:

| Riser type | Guidance |
|------------|----------|
| **PCIe 4.0 x16** riser | Sufficient bandwidth for Pearl; prefer quality shielded units |
| **PCIe 5.0** riser | Only needed if chasing marginal latency stability at extreme power; not required for Pearl throughput |
| **x1 USB risers** | **Avoid for RTX 5090** — power and signaling inadequate for 575 W class cards |
| **Dual / angled 12VHPWR** adapters | High risk; use native PSU cable or certified adapter |

For Pearl, a **stable x16 Gen4** riser beats a **flaky Gen5** riser every time.

### 3.3 Lane splitting and multi-GPU

- **Bifurcation** (e.g., x16 → x8/x8) is normal on workstation boards. Pearl does not need x16 electrical bandwidth, but each 5090 still needs **adequate 12VHPWR** and cooling.
- Running **two 5090s** on one consumer board: verify the second slot’s **physical x8 vs x16** and **distance from CPU** — thermal and power matter more than lane count.

### 3.4 BIOS settings (physical PC)

- **Above 4G Decoding:** Enabled (required for large BAR / VRAM).
- **Resizable BAR (ReBAR):** Enabled — helps some CUDA allocations; Pearl benefits indirectly.
- **PCIe Link Speed:** Auto or Gen5 — do not force Gen1/Gen2 for “stability” unless diagnosing a fault.
- **ASPM:** If you see mysterious latency under light load, test with ASPM disabled — rare for mining (GPU always loaded).

---

## 4. Verification Commands

Run these after install, after any hardware change, and when diagnosing underperformance.

### 4.1 `nvidia-smi` (Windows & cross-platform)

```text
nvidia-smi
```

Check: GPU name `NVIDIA GeForce RTX 5090`, no Xid errors in driver logs, temperature and power within limits.

**Link status (critical):**

```text
nvidia-smi -q -d PCIE
```

Look for:

| Field | Healthy value |
|-------|----------------|
| `PCIe Generation (max)` | 5 |
| `PCIe Generation (current)` | 5 |
| `Link Width (max)` | 16x |
| `Link Width (current)` | 16x |

**Power and throttle state:**

```text
nvidia-smi -q -d PERFORMANCE,CLOCK,POWER
```

Look for `SW Power Cap`, `HW Slowdown`, `HW Thermal Slowdown`, `SW Thermal Slowdown` — any **Active** flag under load explains hashrate loss better than PCIe.

**Continuous monitor during a benchmark:**

```text
nvidia-smi dmon -s pucvmet -d 1
```

Watch `pwr` (W), `gtemp`, `clocks` — sustained throttling shows here before PropMiner logs complain.

### 4.2 `lspci` (when available on the host OS)

On hosts that expose PCI config (some bare-metal Linux installs, diagnostic live USBs):

```text
lspci -nn | grep -i nvidia
lspci -vvv -s <bus:device.function> | grep -E "LnkCap|LnkSta|Width|Speed"
```

Interpret:

| `LnkSta` | Meaning |
|----------|---------|
| `Speed 32GT/s, Width x16` | Gen5 x16 — good |
| `Speed 16GT/s, Width x16` | Gen4 x16 — fine for Pearl |
| `Width x8` or `x4` | Investigate slot, riser, or bifurcation |

**Windows alternative** (no `lspci`): GPU-Z → *Bus Interface* field; or NVIDIA Control Panel → System Information.

### 4.3 PropMiner-specific sanity check

After hardware verification, run the built-in RTX 5090 benchmark on the target machine:

```text
propminer --bench 60 --rtx5090 --gpus 0
```

Compare H/s before and after hardware changes. Use the same driver version and power limit for A/B tests.

---

## 5. PSU Requirements

### 5.1 RTX 5090 power budget

| Parameter | Typical value |
|-----------|----------------|
| TDP / default power limit | **575 W** |
| Transient spikes | Briefly above PL (depends on VRM and driver) |
| 12VHPWR | **12V-2×6** (formerly 12VHPWR) — **450 W rated connector**, card can draw more from combined rails per NVIDIA spec |
| Auxiliary recommendation | **850 W PSU minimum** for 5090-only builds; **1000–1200 W** for high-end CPU + 5090; **1200 W+** for dual 5090 |

Pearl mining holds **continuous Tensor Core load** — closer to **Furmark-like duty cycle** than gaming. Budget PSU for **sustained** draw, not short bursts.

### 5.2 12VHPWR connection

- Prefer **native 12V-2×6 cable from the PSU** (ATX 3.1 / PCIe 5.1 compliant).
- **Fully insert** until latch clicks — partial seating caused widespread 4090 issues; 5090 is the same class risk.
- Avoid **two 8-pin to 12VHPWR** pigtail adapters unless the adapter and PSU are explicitly rated for **600 W** on that rail.
- Do not mix modular cables between PSU brands.

### 5.3 Rail allocation

- Single-rail 12 V designs: ensure **combined GPU + CPU** does not exceed rail OCP limits.
- Multi-rail: connect GPU to the **dedicated PCIe / 12VHPWR** ports documented for VGA in the manual — not peripheral SATA/Molex chains.

### 5.4 Symptoms of insufficient PSU / cable

- Random **black screens** or driver resets under load.
- `nvidia-smi` shows power stuck **below** configured limit while GPU is “busy”.
- Xid **79** or **119** errors (power / PCIe related) in Event Viewer or `dmesg` on applicable hosts.
- PropMiner **watchdog** context resets correlated with power spikes.

---

## 6. Thermal / Power Limit Tuning

Pearl on RTX 5090 is **compute-bound** on Tensor Cores and GDDR7. Thermals and power limits interact directly with sustained clocks.

### 6.1 Recommended starting policy

Aligns with `docs/RTX5090_BLUEPRINT.md` clock guidance:

| Setting | Recommendation |
|---------|----------------|
| Power limit | **95–100%** (550–575 W) for max H/s; reduce only if thermally constrained |
| Core clock offset | **+0 to +200 MHz** max; validate with `--self-test` after each step |
| Memory clock | **Stock** or mild OC; do not downclock VRAM for “stability” |
| Fan curve | Aggressive — GDDR7 and VRM hotspots matter under 24/7 mining |

### 6.2 Useful commands

```text
nvidia-smi -i 0 -pl 575
nvidia-smi -i 0 -q -d POWER
```

Inspect `Max Power Limit` vs `Current Power Limit`. Some OEM cards cap below 575 W unless vendor tool is used.

**Vendor tools** (MSI Afterburner, ASUS GPU Tweak, etc.) may expose higher PL or fan controls than `nvidia-smi`.

### 6.3 Thermal throttling vs PCIe

| Observation | Likely cause |
|-------------|--------------|
| Hot spot \> 90°C, clocks drop | Thermal limit — fix airflow, repaste only if warranted |
| Power at PL cap, clocks oscillate | Power limit — PSU or PL setting |
| PCIe at Gen5 x16, still low H/s | **Not PCIe** — profile kernel, occupancy, or VRAM clock |

### 6.4 PropMiner self-test gate

After any power or clock change:

```text
propminer --self-test --rtx5090 --gpus 0
```

Invalid proofs from aggressive OC are worse than a few percent hashrate loss.

---

## 7. Salad vs Physical PC Differences

**Salad** is a distributed compute marketplace: you deploy PropMiner as a containerized workload on a host you do not own. A **physical PC** is hardware you control end-to-end.

| Aspect | Salad (hosted GPU) | Physical PC |
|--------|-------------------|-------------|
| PCIe slot / riser choice | **No control** — provider’s hardware | Full control |
| PSU / 12VHPWR quality | **No control** | Choose ATX 3.1 PSU, native cable |
| BIOS (Gen5, ReBAR, lanes) | **No control** | Configure explicitly |
| Driver version | Host-managed | You pin driver |
| Multi-tenant neighbors | Possible power / thermal interference | Isolated |
| Diagnostic access | Limited to logs, `nvidia-smi` in container | Full hardware monitoring |
| PropMiner PCIe sensitivity | **Low** — if container starts and `--bench` is stable, bus is sufficient | Same — but you can fix underlying hardware |

**Practical guidance:**

- On **Salad**, verify **`nvidia-smi -q -d PCIE`** and run **`--bench 60 --rtx5090`** after deploy. If results match other hosts, **do not chase PCIe tuning** — you cannot change the hardware anyway.
- On a **physical PC**, this document’s checklist is worth doing **once** at build time.
- Salad hosts may run PropMiner in GPU-passthrough environments; that does not change Pearl’s minimal PCIe footprint — it only affects what you can measure and fix.

---

## 8. When This Actually Helps PropMiner vs Other Workloads

### High impact for Pearl (do these first)

1. **Power / thermal headroom** — sustained 575 W class load without throttle.
2. **Driver + CUDA 12.8+** — correct `sm_120` path.
3. **`--rtx5090` profile** — `M=8192`, large `N` from `pick_n_for_vram()`, batch 16–24, CUDA graphs.
4. **Occupancy** — 8192 CTAs on 170 SMs; avoid shrinking `N` unnecessarily for a tiny tail wave.
5. **Kernel selection** — native Blackwell MMA vs SM80 atom (see `RTX5090_BLUEPRINT.md`).

### Low impact for Pearl (this document)

1. PCIe Gen5 vs Gen4 at x16 width.
2. Faster CPU PCIe root complex for seed uploads.
3. Pinned host memory tuning beyond what PropMiner already allocates.
4. Riser “bandwidth upgrade” when current link is x16 Gen4+.

### Other workloads where PCIe Gen5 matters more

| Workload | Why PCIe matters |
|----------|------------------|
| LLM inference (large batch, CPU offload) | Constant weight streaming |
| GPU ↔ CPU pipeline (pre/post on CPU) | Large H2D/D2H per frame |
| Multiple GPUs P2P without NVLink | Peer copies across PCIe |
| Dataset loaders feeding GPU every step | Bandwidth-bound host pipeline |
| Video encode/decode bounce buffers | Real-time D2H/H2D |

Pearl is none of these: **GEMM + on-device hash** with **8-byte seeds**.

---

## 9. Checklist for User's Build

Use this when assembling or buying an RTX 5090 PropMiner rig.

### Motherboard & case

- [ ] CPU-direct **PCIe 5.0 x16** slot identified in manual  
- [ ] **Above 4G Decoding** + **ReBAR** enabled in BIOS  
- [ ] Case airflow: **intake + exhaust** sized for 575 W continuous  
- [ ] GPU clearance for **12V-2×6** cable bend radius (no sharp fold at connector)

### Power

- [ ] PSU **≥ 1000 W** (quality tier A/B from reputable review sites)  
- [ ] **ATX 3.1 / PCIe 5.1** with native **12V-2×6** cable  
- [ ] GPU power cable fully seated; **no** SATA/Molex adapters  
- [ ] UPS optional but recommended for farm stability  

### Link verification

- [ ] `nvidia-smi -q -d PCIE` → **Gen 5, x16** (Gen 4 x16 acceptable for Pearl)  
- [ ] No corrective action needed if width is x16 and bench is stable  
- [ ] If x8/x4 → reseat card, try direct slot (no riser), update BIOS  

### PropMiner validation

- [ ] `propminer --self-test --rtx5090 --gpus 0` passes  
- [ ] `propminer --bench 60 --rtx5090 --gpus 0` recorded as baseline H/s  
- [ ] `nvidia-smi dmon` during bench shows no thermal/power slowdown flags  
- [ ] Orchestrator log shows: `GPU-isolated path (VRAM-resident B, CUDA graphs, pinned PCIe, no CPU mining)`

### Ongoing monitoring

- [ ] Weekly: compare H/s to baseline (pool-side if available)  
- [ ] Monthly: dust filters, fan health, connector inspection  
- [ ] After driver update: repeat `--self-test` and short bench  

---

## 10. Diagnostic if Underperforming

Work through this tree **in order**. Stop when the anomaly is found.

```
Underperforming H/s
│
├─ 1. Software baseline
│     ├─ Same driver as last known-good?
│     ├─ Using --rtx5090 and production config (not capped bench N)?
│     └─ --self-test passes?
│
├─ 2. Power & thermals (MOST COMMON hardware cause)
│     ├─ nvidia-smi dmon: pwr, gtemp, clocks under load
│     ├─ HW/SW slowdown flags active?
│     ├─ PL below 575 W or OEM-capped?
│     └─ Fix: airflow, fan curve, PSU upgrade, reseat 12VHPWR
│
├─ 3. PCIe width (not speed)
│     ├─ Link Width current == 16x (or stable 8x on validated board)?
│     ├─ If downgraded: direct slot test, replace riser
│     └─ Note: x16 Gen4 is NOT a Pearl bottleneck
│
├─ 4. PCIe Gen speed
│     ├─ Stuck at Gen1/Gen2? → reseat, BIOS Auto, check riser
│     └─ Gen4 vs Gen5 at x16 → expect NO Pearl hashrate delta
│
├─ 5. GPU health
│     ├─ Xid errors in logs?
│     ├─ ECC / row remapping (if exposed)?
│     └─ Thermal paste / VRM issue if hot-spot throttles early
│
└─ 6. PropMiner-specific
      ├─ VRAM pressure → N reduced by pick_n_for_vram?
      ├─ CUDA graph disabled → higher launch overhead
      ├─ Watchdog resets → correlate with power / driver
      └─ Compare against SRBMiner on same pool for external reference
```

### Quick reference: symptom → action

| Symptom | First action |
|---------|--------------|
| Hashrate dropped after moving card to riser | Direct-slot test; replace riser; check x16 width |
| Crashes only under mining load | PSU / 12VHPWR / power limit |
| `Link Width x8` | Manual lane allocation; move NVMe; different slot |
| Gen5 capable but runs Gen4 | Acceptable for Pearl; check BIOS if chasing marginal stability |
| Power well below 575 W at full load | PL setting, PSU rail limit, or thermal throttle |
| Salad host underperforms vs bare metal | Compare `nvidia-smi` PL and thermals; host may throttle tenants — not fixable in software |

### When to stop investigating PCIe

If all are true:

1. `Link Width` = **x16** (or stable **x8** on known-good board)  
2. `Link Speed` ≥ **Gen 4**  
3. No power or thermal throttling under `dmon`  
4. `--self-test` passes  

…then **PCIe and PSU are not your levers**. Return to kernel profile, `N` selection, batch size, and autotune (`PROPMINER_AUTOTUNE=1`) per `docs/RTX5090_BLUEPRINT.md`.

---

## Related PropMiner Documentation

| Document | Topic |
|----------|-------|
| `docs/RTX5090_BLUEPRINT.md` | Kernel, occupancy, clocks, VRAM architecture |
| `docs/RTX5090_LINUX_TASKS.md` | Build, benchmark, and profiling workflow |
| `src/host/pearl/rtx5090_profile.h` | Default M/N/K, batch, SM count |
| `src/host/pearl/gpu_worker.h` | Seed copy stream, pinned share staging |

---

## Revision notes

- **Audience:** RTX 5090 Pearl miners on physical Windows PCs and Salad-hosted GPUs.  
- **Explicitly out of scope:** Linux/WSL2 setup, vCPU container sizing, Docker internals.  
- **Honest ceiling:** PCIe Gen5 is **insurance**, not a Pearl hashrate multiplier; **PSU and thermals** are the hardware knobs that can recover real lost performance.
