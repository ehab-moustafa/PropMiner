# CPU-Side BLAKE3 Offload Test Plan

> **Goal:** Validate byte-identical BLAKE3 computation when moved from the CUDA GEMM kernel to a separate CPU-side kernel, running entirely on macOS without GPU.

---

## 1. Architecture & Data Flow Summary

### 1.1 The CUDA Path (Current — Ground Truth)

```
WorkItem (σ[32], b_seed[32], target_nbits)
    │
    ├─ LCG noise → A_tile[64×64] int8  (shared mem)
    ├─ LCG noise → B_tile[64×65] int8  (shared mem, padded)
    │
    └─ INT8 GEMM (mma.sync m16n8k32) → C[8] int32 accumulators
         │
         ├─ Initialize transcript[16] from σ (byte-sliced, LE u32)
         ├─ XOR-reduce 8 MMA outputs into transcript[0..7] via pow_rotl_xor (rotate-left-13 ^ y)
         ├─ Copy pow_key[8] → chaining[8]
         ├─ blake3::compress_msg_block_u32(transcript, chaining, B3_PARAMS_KEYED_SINGLE)
         │    counter=0, block_len=64, flags=KEYED|CHUNK_START|CHUNK_END|ROOT = 0x1F
         │    6 rounds + permutation, 1 final round (no perm)
         │    Output: chaining[i] = s[i] ^ s[i+8]
         └─ Compare chaining[7..0] (big-endian) vs pow_target[7..0]
              │
              └─ If found → write ShareResult{σ, nonce, hash[32], tile coords}
```

### 1.2 The CPU Path (Target — Must Match)

```
Same WorkItem (σ[32], b_seed[32], target_nbits)
    │
    ├─ LCG noise → A_tile[64×64] int8  (CPU reference)
    ├─ LCG noise → B_tile[64×65] int8  (CPU reference)
    │
    └─ INT8 GEMM (CPU matmul) → C[8] int32 accumulators
         │
         ├─ Initialize transcript[16] from σ (byte-sliced, LE u32)
         ├─ XOR-reduce 8 MMA outputs via rotl-13 ^ y
         ├─ CPU BLAKE3 compress (reference impl or portable path)
         │    Same params: counter=0, block_len=64, flags=0x1F
         └─ Same target comparison
```

### 1.3 Key Constants (from `propminer_config.h`)

| Constant | Value | Meaning |
|---|---|---|
| `B3_MSG_BLOCK_BYTES` | 64 | Message block size (16 × u32) |
| `B3_MSG_BLOCK_U32` | 16 | Message block as u32 words |
| `B3_CHAINING_VALUE_BYTES` | 32 | Chaining value size |
| `B3_CHAINING_VALUE_U32` | 8 | Chaining value as u32 words |
| `B3_ROUNDS` | 7 | 6 with perm + 1 final |
| `B3_FLAGS_KEYED_SINGLE` | `0x1F` | `KEYED_HASH \| CHUNK_START \| CHUNK_END \| ROOT` |
| `HASH_ACCUMULATE_ROTATION` | 13 | Rotate-left amount for XOR tree |

### 1.4 BLAKE3 Flags Breakdown

```
CHUNK_START  = 1 << 0  = 0x01
CHUNK_END    = 1 << 1  = 0x02
PARENT       = 1 << 2  = 0x04
ROOT         = 1 << 3  = 0x08
KEYED_HASH   = 1 << 4  = 0x10
────────────────────────────────
KEYED_SINGLE = 0x1F     (all five for PoW)
```

---

## 2. Existing Test Infrastructure

### 2.1 Reference BLAKE3 Implementation (Already Exists)

**File:** `src/host/tests/blake3_reference.c` + `blake3_reference.h`

This is the **official BLAKE3 reference implementation** (Rust port to C). It provides:
- `blake3_hasher_init()` — standard (unkeyed) hash
- `blake3_hasher_init_keyed()` — keyed hash
- `blake3_hasher_update()` — streaming input
- `blake3_hasher_finalize()` — produce output

**Usage in tests:** `src/host/tests/ref_blake3.cpp` wraps it as `pearl::ref::Blake3Ref`.

### 2.2 CUDA BLAKE3 Implementation (Device Code)

**File:** `src/cuda/include/blake3.cuh`

Key function: `compress_msg_block_u32(msg[16], chaining[8], params)`
- Uses PTX primitives: `b3_rotr` (rotate via `shf.l.wrap.b32`), `b3_xor3` (XOR via `lop3.b32`)
- 16 named state registers, not arrays
- 6 rounds with `BLAKE3_ROUND` + `BLAKE3_PERMUTE`, 1 final round without permute
- Output: `chaining[i] = s[i] ^ s[i+8]`

### 2.3 Alternative GPU BLAKE3 (pearl-gemm)

**File:** `third_party/pearl-gemm/csrc/blake3/blake3.cuh`

Same algorithm, different implementation style:
- Uses `rState(i)` / `rBlock(i)` accessor macros
- Includes `blake3_rounds.inc` for round/permutation logic
- Has `compress_msg_block_u32` template with CUTLASS tensors
- Also has `compress_full` (returns 16 words: both directions of XOR)

### 2.4 HIP/SYCL BLAKE3 (Cross-Platform GPU)

**File:** `third_party/pearl-gemm/csrc/rocm/blake3_device.cuh`

Pure C++ inline functions (no PTX). **This is the best portable reference** for GPU BLAKE3 because:
- Uses standard C++ arithmetic: `rightrotate32(x, n) = (x << (32-n)) | (x >> n)`
- Same `BLAKE3_ROUND` / `BLAKE3_PERMUTE` macros via `blake3_rounds.inc`
- Provides `compress()`, `compress_full()`, `parent_cv()`, `chunk_cv()`
- Has `init_cv()`, `cv_to_bytes()` helpers

### 2.5 Existing Unit Tests

**File:** `src/host/tests.cpp`

Currently tests:
- `test_reference_blake3()` — empty hash, "hello" hash, keyed hash
- `test_reference_bseed_expand()` — XOF expansion
- `test_reference_merkle_tree()` — Merkle tree with keyed BLAKE3
- `test_reference_claimed_hash_deterministic()` — full Pearl hash pipeline
- `test_share_claimed_hash_roundtrip()` — end-to-end share verification

**Missing:** No tests for the specific `compress_msg_block_u32` function with the exact transcript format used in the PoW kernel.

### 2.6 Pearl Challenge Solver (CPU Unrolled BLAKE3)

**File:** `src/host/pearl/pearl_challenge.cpp`

This is a **hand-unrolled BLAKE3 solver** for nonce brute-force:
- Uses `try_nonce()` with 7 manually unrolled rounds
- Message: `seed[32] || nonce_le[8] || zero_pad[24]` (40 bytes total → padded to 64)
- Flags: `CHUNK_START | CHUNK_END | ROOT = 0x0D` (NOT keyed)
- This is a **different mode** than the PoW kernel (unkeyed, different message layout)

---

## 3. Test Suite Design

### TEST GROUP 1: BLAKE3 Compress Correctness

#### Test 1.1: Basic Compress with Zero Input

**Purpose:** Verify the CPU compress function produces correct output for the simplest possible input.

**Input:**
- `msg[16]` = all zeros (64 zero bytes)
- `chaining[8]` = IV (0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19)
- `params` = `{counter: 0, block_len: 64, flags: 0x1F}`

**Reference:** Use `blake3_reference.c` — initialize keyed hasher, feed 64 zero bytes, finalize.

**Expected Output:** Compute via both the CUDA-style compress and the reference hasher. The results must match byte-for-byte.

**Verification:**
```cpp
// CPU compress function
uint32_t cv[8] = {IV0, IV1, IV2, IV3, IV4, IV5, IV6, IV7};
uint32_t msg[16] = {0};
compress(cv, msg, 0, 64, 0x1F);

// Reference
blake3_hasher hasher;
blake3_hasher_init_keyed(&hasher, iv_bytes); // IV as key for keyed mode
uint8_t zero_block[64] = {0};
blake3_hasher_update(&hasher, zero_block, 64);
uint32_t ref_cv[8];
blake3_hasher_finalize(&hasher, ref_cv, 32);
// ... compare cv vs ref_cv
```

#### Test 1.2: Compress with Random Input

**Purpose:** Test with non-trivial message data.

**Input:** Generate 16 random u32 values (64 bytes):
```cpp
uint32_t msg[16] = {
    0xDEADBEEF, 0xCAFEBABC, 0x12345678, 0xFEDCBA98,
    0xCAFEBABE, 0xBEEFCAFE, 0x42424242, 0x7F7F7F7F,
    0x80808080, 0xFFFFFFFF, 0x00000000, 0x01010101,
    0x80000000, 0x7FFFFFFF, 0x11111111, 0x22222222
};
uint32_t chaining[8] = {1, 2, 3, 4, 5, 6, 7, 8};
```

**Reference:** Same as Test 1.1 — compare CUDA-style compress output against `blake3_reference.c`.

#### Test 1.3: Cross-Implementation Compress Match

**Purpose:** Verify that the three implementations produce identical output:
1. `src/cuda/include/blake3.cuh` — PTX version (can only verify algorithmically)
2. `third_party/pearl-gemm/csrc/rocm/blake3_device.cuh` — C++ inline version
3. `src/host/tests/blake3_reference.c` — official reference

**Method:** Port the PTX version's logic to C++ (replacing PTX intrinsics with C++ arithmetic) and compare against the reference. The rocm version is already in portable C++.

**Verification:** Run all three on the same 100 random inputs. All must produce identical 8-word outputs.

#### Test 1.4: Flags Variation Test

**Purpose:** Verify that different flag combinations produce different (but correct) outputs.

**Input:** Same `msg[16]` and `chaining[8]` for all variants.

**Test variants:**
| Flags | Value | Meaning |
|---|---|---|
| `0x1F` | `KEYED|START|END|ROOT` | PoW mode (single-block keyed) |
| `0x1D` | `KEYED|START|END` | Without ROOT |
| `0x17` | `KEYED|START|ROOT` | Without CHUNK_END |
| `0x0D` | `START|END|ROOT` | Unkeyed |
| `0x1F \| PARENT` | `KEYED|START|END|ROOT|PARENT` | With PARENT flag |

**Reference:** Use `blake3_reference.c` with appropriate flag manipulation.

**Expected:** All five produce different outputs. The reference implementation's flag handling must match.

---

### TEST GROUP 2: Transcript Assembly & XOR Reduction

#### Test 2.1: Transcript Initialization from σ

**Purpose:** Verify the transcript is correctly initialized from sigma bytes.

**CUDA Code (ground truth):**
```cpp
uint32_t transcript[B3_MSG_BLOCK_U32] = {};
for (int i = 0; i < B3_MSG_BLOCK_U32; i++) {
    int idx = (i * 4) % 32;
    transcript[i] = ((uint32_t)work_sigma[idx]       ) |
                    ((uint32_t)work_sigma[idx + 1] << 8)  |
                    ((uint32_t)work_sigma[idx + 2] << 16) |
                    ((uint32_t)work_sigma[idx + 3] << 24);
}
```

**Input:** `sigma[32]` with known values, e.g.:
```cpp
uint8_t sigma[32] = {
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10,
    0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
    0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20
};
```

**Expected transcript[16]:**
```
transcript[0] = LE32(sigma[0..3])   = 0x04030201
transcript[1] = LE32(sigma[4..7])   = 0x08070605
transcript[2] = LE32(sigma[8..11])  = 0x0C0B0A09
transcript[3] = LE32(sigma[12..15]) = 0x100F0E0D
transcript[4] = LE32(sigma[16..19]) = 0x14131211
transcript[5] = LE32(sigma[20..23]) = 0x18171615
transcript[6] = LE32(sigma[24..27]) = 0x1C1B1A19
transcript[7] = LE32(sigma[28..31]) = 0x201F1E1D
transcript[8]  = LE32(sigma[0..3])   = 0x04030201  (wraps: (8*4)%32 = 0)
transcript[9]  = LE32(sigma[4..7])   = 0x08070605
...
transcript[15] = LE32(sigma[28..31]) = 0x201F1E1D
```

**Verification:** CPU code must produce the exact same 16 u32 values.

#### Test 2.2: XOR Reduction with MMA Outputs

**Purpose:** Verify the `pow_rotl_xor` operation that folds GEMM accumulator into transcript.

**CUDA Code:**
```cpp
uint32_t c_arr[] = {c0, c1, c2, c3, c4, c5, c6, c7};
for (int i = 0; i < 8; i++) {
    transcript[i] = pow_rotl_xor(transcript[i], c_arr[i]);
}
```

Where `pow_rotl_xor(x, y) = rotl(x, 13) ^ y`.

**Input:**
```cpp
// Transcript after sigma init (from Test 2.1)
uint32_t transcript[16] = {0x04030201, 0x08070605, 0x0C0B0A09, 0x100F0E0D,
                           0x14131211, 0x18171615, 0x1C1B1A19, 0x201F1E1D,
                           0x04030201, 0x08070605, 0x0C0B0A09, 0x100F0E0D,
                           0x14131211, 0x18171615, 0x1C1B1A19, 0x201F1E1D};

// Simulated GEMM accumulator outputs (8 values)
uint32_t c_arr[8] = {0xDEADBEEF, 0xCAFEBABC, 0x12345678, 0xFEDCBA98,
                     0xCAFEBABE, 0xBEEFCAFE, 0x42424242, 0x7F7F7F7F};
```

**Expected (first 4 values):**
```
transcript[0] = rotl(0x04030201, 13) ^ 0xDEADBEEF = 0x00806040 ^ 0xDEADBEEF = 0xDE4BDEBF
transcript[1] = rotl(0x08070605, 13) ^ 0xCAFEBABC = ...
```

**Verification:** Compute on CPU, compare bit-by-bit.

#### Test 2.3: Full Transcript Pipeline (σ → XOR → BLAKE3)

**Purpose:** End-to-end test: sigma + GEMM output → BLAKE3 compress → hash.

**Input:**
- `sigma[32]` = known pattern (e.g., 0xab repeated)
- `pow_key[8]` = known pattern (e.g., 0x42 repeated)
- `c_arr[8]` = known GEMM accumulator values

**Pipeline:**
1. Initialize transcript from sigma
2. XOR-reduce c_arr into transcript[0..7]
3. Copy pow_key into chaining
4. Call BLAKE3 compress
5. Compare chaining vs target

**Reference:** Implement the same pipeline using `blake3_reference.c`:
```cpp
// Step 1-2: Build transcript
uint32_t transcript[16];
// ... init from sigma, XOR with c_arr ...

// Step 3-4: Use reference keyed hash
blake3_hasher hasher;
blake3_hasher_init_keyed(&hasher, pow_key_bytes);
// Feed transcript as 64 bytes (LE u32 → bytes)
uint8_t msg[64];
for (int i = 0; i < 16; i++) {
    msg[i*4]     = transcript[i] & 0xFF;
    msg[i*4 + 1] = (transcript[i] >> 8) & 0xFF;
    msg[i*4 + 2] = (transcript[i] >> 16) & 0xFF;
    msg[i*4 + 3] = (transcript[i] >> 24) & 0xFF;
}
blake3_hasher_update(&hasher, msg, 64);
uint8_t hash[32];
blake3_hasher_finalize(&hasher, hash, 32);
```

**⚠️ Critical Note:** The `blake3_reference.c` streaming API does NOT directly expose the single-block compress with custom flags. You must verify that:
- The reference's keyed hash with 64 bytes of input and flags `CHUNK_START|CHUNK_END|ROOT` (without KEYED_HASH) produces the same output as the keyed mode with flags `KEYED_HASH|CHUNK_START|CHUNK_END|ROOT`.

Actually, looking at the reference code more carefully:
- `blake3_hasher_init_keyed()` sets `flags = KEYED_HASH`
- The reference's `blake3_hasher_update()` + `finalize()` handles chunking internally

**The CUDA kernel does NOT use the streaming API.** It uses a **single-block compress** with `counter=0, block_len=64, flags=0x1F`. This is a domain-separated keyed hash.

**Verification approach:** Port the `compress()` function from `blake3_device.cuh` to plain C++ and verify it against the reference's output for the same input.

---

### TEST GROUP 3: Edge Cases

#### Test 3.1: All-Zero Transcript

**Input:**
- `sigma[32]` = all zeros
- `c_arr[8]` = all zeros
- `pow_key[8]` = all zeros

**Expected:** Deterministic output. Compute with both CUDA-style compress and reference. Must match.

#### Test 3.2: Max Nonce / Max Values

**Input:**
- `sigma[32]` = all `0xFF`
- `c_arr[8]` = all `0xFFFFFFFF`
- `pow_key[8]` = all `0xFFFFFFFF`

**Expected:** Deterministic output. Must match between implementations.

#### Test 3.3: Single-Bit Transitions

**Input:** For each i in 0..127, set bit i of the 64-byte message to 1, all else zero.

**Purpose:** Verify that changing a single input bit changes the output significantly (avalanche effect) and that the change is consistent across implementations.

**Expected:** Each single-bit change produces a different 32-byte hash. The hashes from CPU must match CUDA.

#### Test 3.4: Boundary Values

**Test cases:**
| Test | msg[0] | chaining[0] | block_len | flags |
|---|---|---|---|---|
| Normal | 0x12345678 | 0x6A09E667 | 64 | 0x1F |
| block_len=0 | 0x12345678 | 0x6A09E667 | 0 | 0x1F |
| block_len=1 | 0x12345678 | 0x6A09E667 | 1 | 0x1F |
| counter=1 | 0x12345678 | 0x6A09E667 | 64 | 0x1F |
| counter=UINT32_MAX | 0x12345678 | 0x6A09E667 | 64 | 0x1F |

**Expected:** All produce valid hashes. Counter value affects output (different counter → different output).

#### Test 3.5: σ Trigger Pattern

**Purpose:** Test the transcript initialization with the exact σ pattern used when a share trigger fires.

**Input:** Use actual σ from a real (or synthetic) work item. The σ is 32 bytes from the stratum protocol.

**Method:** Extract σ from `test_reference_blake3()` or `test_share_claimed_hash_roundtrip()` in `tests.cpp`.

**Verification:** The transcript built from σ must match between CPU and CUDA initialization code.

---

### TEST GROUP 4: Hash Comparison Against Target

#### Test 4.1: Target Comparison Logic

**Purpose:** Verify the big-endian comparison of the 256-bit hash against the target.

**CUDA Code:**
```cpp
bool found = true;
for (int i = B3_CHAINING_VALUE_U32 - 1; i >= 0; i--) {
    if (chaining[i] > pow_target[i]) { found = false; break; }
    if (chaining[i] < pow_target[i])  break;
}
```

This is a standard big-endian unsigned comparison of two 256-bit integers stored as 8 little-endian u32 words.

**Test cases:**

| Hash (u32[8], big-endian view) | Target (u32[8]) | Expected |
|---|---|---|
| All zeros | Any | found=true |
| 0x00000001...00 | 0x00000001...00 | found=true (equal) |
| 0x00000001...00 | 0x00000000...FF | found=false |
| 0xFFFFFFFF...FF | 0x00000000...00 | found=false |
| 0x00000000...01 | 0x00000000...01 | found=true |

**Verification:** Implement the same comparison on CPU. For each test case, verify the boolean result matches.

#### Test 4.2: Difficulty nbits → Target Conversion

**Purpose:** Verify that `target_nbits` from the WorkItem correctly maps to the 32-byte target array.

**Reference:** `src/host/pearl/pow_target_utils.cpp` — `nbits_to_target_le()`.

**Test:** For `target_nbits = 0x1b01fffe` (difficulty ~32768):
```cpp
auto target = nbits_to_target_le(0x1b01fffe);
// Expected LE layout: 0x01fffe << 192 → bytes 24..26
EXPECT(target[24] == 0xfe);
EXPECT(target[25] == 0xff);
EXPECT(target[26] == 0x01);
```

**Verification:** Already tested in `test_pow_target_stratum_nbits_roundtrip()`. Add explicit comparison with CUDA's target handling.

---

### TEST GROUP 5: Merkle Leaf CV Computation

#### Test 5.1: Leaf CV (Chunk CV) Computation

**Purpose:** Verify that the leaf CV computation matches between CUDA and CPU.

**CUDA Code (from `blake3_device.cuh`):**
```cpp
__device__ inline void chunk_cv(const uint8_t* chunk, u64 idx, const u32* key,
                                bool is_root, u32 out_cv[8]){
  u32 cv[8]; init_cv(cv, key);
  u32 base = (key?KEYED_HASH:0);
  for(int b=0;b<16;++b){
    u32 block[16];
    for(int i=0;i<16;++i) block[i]=load_le32(chunk+b*64+i*4);
    u32 f=base; if(b==0)f|=CHUNK_START; if(b==15){ f|=CHUNK_END; if(is_root)f|=ROOT; }
    compress(cv, block, idx, MSG_BLOCK_SIZE, f);
  }
  for(int i=0;i<8;++i) out_cv[i]=cv[i];
}
```

**Input:**
- `chunk` = 1024 bytes (16 × 64-byte blocks), all zeros
- `idx` = 0 (first chunk)
- `key` = all zeros (or a known 32-byte key)
- `is_root` = true

**Reference:** Use `pearl::ref::Blake3Ref::chunk_cv()`.

**Verification:** CPU and CUDA must produce identical 8-word CV.

#### Test 5.2: Parent CV Computation

**Purpose:** Verify parent node computation.

**CUDA Code:**
```cpp
__device__ inline void parent_cv(const u32 l[8], const u32 r[8], const u32* key,
                                 bool is_root, u32 out_cv[8]){
  u32 cv[8]; init_cv(cv, key);
  u32 block[16];
  for(int i=0;i<8;++i){ block[i]=l[i]; block[i+8]=r[i]; }
  u32 f=(key?KEYED_HASH:0)|PARENT; if(is_root)f|=ROOT;
  compress(cv, block, 0, MSG_BLOCK_SIZE, f);
  for(int i=0;i<8;++i) out_cv[i]=cv[i];
}
```

**Input:**
- `left[8]` = known chaining value (e.g., from Test 5.1)
- `right[8]` = known chaining value
- `key` = known 32-byte key

**Reference:** `pearl::ref::Blake3Ref::parent_cv()`.

**Verification:** Byte-identical output.

#### Test 5.3: Full Merkle Tree Root

**Purpose:** Build a small Merkle tree on CPU and verify against the CUDA tensor_hash logic.

**Input:** 4 leaves of 1024 bytes each, keyed with a known 32-byte key.

**Method:**
1. Compute leaf CVs (Test 5.1)
2. Compute parent CVs (Test 5.2)
3. Compute root CV

**Reference:** `pearl::ref::RefMerkleTree` — already tested in `test_reference_merkle_tree()`.

**Verification:** The root must match between CPU `RefMerkleTree` and CUDA's `ComputeBlakeMTKernel`.

---

### TEST GROUP 6: Full Pipeline Integration

#### Test 6.1: Complete PoW Pipeline (σ → GEMM → XOR → BLAKE3 → Target Check)

**Purpose:** End-to-end test matching the exact CUDA kernel flow.

**Input:** Synthetic WorkItem:
```cpp
uint8_t sigma[32] = {0xab};  // all 0xab
uint8_t b_seed[32] = {0xcd};
uint32_t target_nbits = 24;  // easy target for testing
uint32_t pow_key[8] = {0x42};  // all 0x42
uint32_t pow_target[8] = {0};  // very easy (all zeros = always found)
```

**Steps:**
1. Generate A_tile, B_tile using splitmix64 (same as CUDA)
2. Compute GEMM C = A × B^T on CPU (64×64 × 64×64 → 64×64)
3. Extract 8 accumulator values (c0..c7 from first warp)
4. Initialize transcript from sigma
5. XOR-reduce c_arr into transcript
6. BLAKE3 compress with pow_key
7. Compare against pow_target

**Reference:** Implement each step in plain C++ using the same algorithms as CUDA.

**Verification:** Every intermediate value (transcript, chaining, hash, found flag) must match.

#### Test 6.2: Determinism Test

**Purpose:** Run the same input 1000 times and verify identical output.

**Method:**
```cpp
for (int i = 0; i < 1000; i++) {
    auto result = blake3_offload_test(sigma, pow_key, c_arr);
    EXPECT(result == first_result);
}
```

**Verification:** All 1000 runs produce byte-identical hash.

#### Test 6.3: Cross-Platform Consistency

**Purpose:** Verify the test produces the same results on different CPU architectures.

**Method:** Run the same test binary on:
1. macOS (Apple Silicon — ARM64)
2. Linux (x86_64)
3. Any other available platform

**Verification:** All platforms produce identical 32-byte hash for the same input.

---

## 4. Implementation Plan

### 4.1 Files to Create

```
src/host/tests/
├── blake3_compress_test.cpp      # Tests 1.1-1.4, 3.1-3.5, 4.1
├── transcript_pipeline_test.cpp  # Tests 2.1-2.3, 6.1
├── merkle_cv_test.cpp            # Tests 5.1-5.3
├── edge_cases_test.cpp           # Tests 3.1-3.5, 4.2
└── integration_test.cpp          # Tests 6.1-6.3
```

### 4.2 Reference Implementation to Port

The most portable reference is `third_party/pearl-gemm/csrc/rocm/blake3_device.cuh` — it's already pure C++ with no GPU dependencies. Copy the `compress()` function and adapt it for host testing.

Alternatively, use the existing `blake3_reference.c` and construct the equivalent keyed single-block hash.

### 4.3 Build Configuration

Add to the existing test binary (`src/host/tests.cpp`):
```cpp
#define PROP_MINER_HOST_ONLY_TESTS 1  // No CUDA runtime needed
```

This ensures tests compile and run without libcuda.dylib.

### 4.4 Test Runner

```bash
# Build host-only tests
CXX=clang++ CXXFLAGS="-DPROP_MINER_HOST_ONLY_TESTS=1 -O2" \
    g++ -o blake3_test \
    src/host/tests.cpp \
    src/host/tests/blake3_reference.c \
    src/host/tests/ref_blake3.cpp \
    src/host/tests/ref_pearl.cpp \
    src/host/tests/blake3_compress_test.cpp \
    src/host/tests/transcript_pipeline_test.cpp \
    src/host/tests/merkle_cv_test.cpp \
    src/host/tests/edge_cases_test.cpp \
    src/host/tests/integration_test.cpp

# Run
./blake3_test
```

---

## 5. Verification Matrix

| Test | CUDA Reference | CPU Reference | Match Criteria |
|---|---|---|---|
| 1.1 Basic Compress | PTX compress | blake3_reference.c | 8 × u32 identical |
| 1.2 Random Compress | PTX compress | blake3_reference.c | 8 × u32 identical |
| 1.3 Cross-Impl | PTX compress | rocm compress + ref | All 3 identical |
| 1.4 Flags | PTX compress | blake3_reference.c | Different outputs, consistent |
| 2.1 σ → Transcript | CUDA init | CPU init | 16 × u32 identical |
| 2.2 XOR Reduction | CUDA rotl_xor | CPU rotl_xor | 8 × u32 identical |
| 2.3 Full Pipeline | CUDA kernel | CPU pipeline | 32-byte hash identical |
| 3.1 All-Zero | CUDA kernel | CPU pipeline | 32-byte hash identical |
| 3.2 Max Values | CUDA kernel | CPU pipeline | 32-byte hash identical |
| 3.3 Single-Bit | CUDA kernel | CPU pipeline | 128 hashes match |
| 3.4 Boundary | CUDA kernel | CPU pipeline | All variants match |
| 3.5 σ Trigger | CUDA kernel | CPU pipeline | 32-byte hash identical |
| 4.1 Target Compare | CUDA kernel | CPU compare | Boolean identical |
| 4.2 nbits → Target | pow_target_utils | pow_target_utils | LE bytes identical |
| 5.1 Leaf CV | chunk_cv() | ref::chunk_cv() | 8 × u32 identical |
| 5.2 Parent CV | parent_cv() | ref::parent_cv() | 8 × u32 identical |
| 5.3 Merkle Root | RefMerkleTree | RefMerkleTree | 32-byte root identical |
| 6.1 Full Pipeline | CUDA kernel | CPU pipeline | Everything matches |
| 6.2 Determinism | N/A | CPU pipeline | 1000× identical |
| 6.3 Cross-Platform | N/A | CPU pipeline | macOS = Linux |

---

## 6. Extracting Exact BLAKE3 Input Format from CUDA Kernel

### 6.1 Message Block Layout

The 64-byte message block passed to `compress_msg_block_u32` is the **transcript** — 16 little-endian u32 values:

```
Message = transcript[0] || transcript[1] || ... || transcript[15]  (LE bytes)
        = 64 bytes total

transcript[i] = LE32(sigma[(i*4)%32 .. (i*4+3)%32])  for i < 8
transcript[i] = rotl_13(transcript[i]) ^ c_arr[i]    for i < 8 (after XOR)
transcript[i] = 0                                     for i >= 8 (unused)
```

### 6.2 Chaining Value Input

```
chaining[8] = pow_key[8]  (copied directly from work item)
```

### 6.3 Compress Parameters

```
CompressParams {
    counter   = 0,
    block_len = 64 (B3_MSG_BLOCK_BYTES),
    flags     = 0x1F (B3_FLAGS_KEYED_SINGLE)
}
```

### 6.4 Output Format

```
chaining[8] = s[i] ^ s[i+8]  for i = 0..7
hash[32] = LE bytes of chaining[8]
```

---

## 7. Known-Good Inputs/Outputs from Existing Tests

### 7.1 Empty BLAKE3 Hash (from `test_reference_blake3`)

```
Input:   (empty)
Output:  af1349b9 f5f9a1a6 a0404dea 36dcdc49 9bcbb25c9ad1c112 b7cc9a93 cae41f32 62
```

### 7.2 "hello" BLAKE3 Hash

```
Input:   "hello" (5 bytes)
Output:  (computed by test, verify against known BLAKE3 vector)
```

### 7.3 BLAKE3 IV (Hardcoded)

```
IV = {0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
      0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19}
```

These are the BLAKE3 initialization constants from the official spec.

---

## 8. BLAKE3 Test Vectors in the Codebase

### 8.1 Official BLAKE3 Reference Vectors

The `blake3_reference.c` file implements the official reference. To get test vectors:

```c
// Empty input vector (known-good, from test_reference_blake3)
blake3_hasher_init(&hasher);
blake3_hasher_finalize(&hasher, out, 32);
// out = [af,13,49,b9,f5,f9,a1,a6,a0,40,4d,ea,36,dc,dc,49,
//        9b,cb,25,c9,ad,c1,12,b7,cc,9a,93,ca,e4,1f,32,62]
```

### 8.2 Custom Vectors for Pearl-Specific Modes

The Pearl miner uses **keyed single-block** mode, which is NOT the standard BLAKE3 input hashing mode. Standard BLAKE3 test vectors won't apply directly. You need to generate custom vectors:

```cpp
// Generate vector for testing
uint32_t msg[16] = {0};  // all zeros
uint32_t key[8] = {0};   // all zeros
compress(key, msg, 0, 64, 0x1F);
// key[8] now contains the chaining value — this is your test vector
```

---

## 9. Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| PTX intrinsics behavior differs from C++ arithmetic | Low | Use rocm `blake3_device.cuh` as intermediate reference (already C++) |
| Endianness mismatch between CUDA and CPU | Medium | Explicit LE byte assembly in both paths |
| Floating-point vs integer differences | None | All arithmetic is integer-only (int8, int32, uint32) |
| Splitmix64 differences | Low | Standard algorithm, no platform dependencies |
| GEMM accumulator differences | High | CPU GEMM must use exact same int8×int8→int32 accumulation |
| XOR reduction ordering | Low | Deterministic loop, no parallelism |

---

## 10. Priority Order

1. **Test 1.1** — Basic compress (foundation for all other tests)
2. **Test 2.1** — Transcript init (verifies σ → u32 conversion)
3. **Test 2.2** — XOR reduction (verifies rotl-13 ^ y)
4. **Test 2.3** — Full pipeline (integrates 1.1 + 2.1 + 2.2)
5. **Test 3.1** — All-zero edge case (sanity check)
6. **Test 4.1** — Target comparison (verifies the "found" logic)
7. **Test 6.1** — Complete PoW pipeline (end-to-end validation)
8. **Tests 5.1-5.3** — Merkle CV (separate but related)
9. **Tests 3.2-3.5, 4.2, 6.2-6.3** — Remaining edge cases and integration
