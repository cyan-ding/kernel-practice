You’ve got the right foundation already — a course gives you the vocabulary (threads, blocks, memory hierarchy, coalescing). What comes next is a **progression of kernels** where each one teaches a technique you’ll reuse in the next. Your `kernel-practice` repo is empty, which is fine: treat it as a sequence of small, benchmarked exercises.

## The core idea

Most “deep learning kernels” are really three things:

1. **GEMM** (matrix multiply) — the workhorse
2. **Reductions** (sum, max, softmax) — normalization and attention
3. **Memory layout transforms** (im2col, transpose, reshape) — conv and attention

If you can write a decent matmul and a decent softmax, you’re ~70% of the way to understanding everything else.

---

## Phase 0: Setup (1–2 days)

Pick one stack and stick with it:


| Stack        | Good if…                                                |
| ------------ | ------------------------------------------------------- |
| **CUDA C++** | You want to understand hardware deeply                  |
| **Triton**   | You want faster iteration, still learn the concepts     |
| **Both**     | Best long-term: CUDA for fundamentals, Triton for speed |


For each kernel you write:

- A **naive reference** (CPU or simple GPU)
- A **correctness test** (compare against NumPy / PyTorch)
- A **benchmark** (time vs reference implementation)
- One **roofline-style question**: “Am I memory-bound or compute-bound?”

Repo structure that works well:

```
kernels/
  matmul/
    naive.cu
    tiled.cu
    test.py
    bench.py
  softmax/
  conv/
  ...
```

---

## Phase 1: Matmul — the foundation (2–3 weeks)

This is the most important kernel. Don’t rush past it.

### Step 1 — Naive parallel matmul

- One thread computes one output element: `C[i,j] = sum_k A[i,k]*B[k,j]`
- Learn: indexing, launch config, correctness
- You’ll be slow — that’s the point

### Step 2 — Tiled matmul with shared memory

- Load tiles of A and B into `__shared__` memory
- Reuse data across threads in a block
- Learn: **shared memory**, **bank conflicts**, **tiling**

### Step 3 — Register blocking / thread tiling

- Each thread computes a small `TM×TN` tile of C
- Learn: **register pressure**, **ILP**, amortizing loads

### Step 4 — Vectorized loads

- Use `float4` / `half2` loads where aligned
- Learn: **coalescing**, alignment requirements

### Step 5 — Compare against cuBLAS

- Your goal isn’t to beat cuBLAS — it’s to understand the gap
- Typical student tiled kernel: ~10–30% of cuBLAS
- Read why: tensor cores, autotuning, sophisticated scheduling (CUTLASS)

**Milestone:** `C = A @ B` for `(M,K) @ (K,N)` with FP32, within ~5× of cuBLAS on a mid-size problem (e.g. 4096³).

**Techniques you’ll reuse everywhere:** tiling, shared memory, register blocking, occupancy tradeoffs.

---

## Phase 2: Batched & strided ops (3–5 days)

DL rarely does one matmul — it does thousands in parallel.

Implement:

- **Batched GEMM**: `(B, M, K) @ (B, K, N)`
- **Strided batched GEMM**: tensors aren’t contiguous in batch dim

This mirrors how PyTorch’s `torch.bmm` and cuBLAS `cublasGemmStridedBatched` work.

**Milestone:** match `torch.bmm` within ~2× for batch=32, 1024×1024.

---

## Phase 3: Reductions & softmax (1 week)

These show up in loss, attention, and normalization.

### Reduction kernel

- Parallel sum / max with shared-memory tree reduction
- Learn: **warp shuffle** (`__shfl_down_sync`) — much faster than shared mem for 32-lane reductions

### Softmax

- `softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))`
- Implement in **two passes** first (max, then sum+exp), then **one-pass fused** per row
- Learn: **numerical stability**, **row-wise parallelism**

**Milestone:** match `torch.softmax` on `(batch, seq, dim)` within ~2×.

---

## Phase 4: LayerNorm / RMSNorm (3–5 days)

Combines matmul-like patterns with reductions:

```
y = (x - mean) / sqrt(var + eps) * gamma + beta   # LayerNorm
y = x / sqrt(mean(x²) + eps) * gamma              # RMSNorm (Llama-style)
```

- One block (or warp) per row/token
- Reduction for mean/variance, then elementwise scale/shift
- Learn: **fusing** reduction + elementwise in one kernel

**Milestone:** match `torch.nn.functional.layer_norm` / RMSNorm.

---

## Phase 5: Conv2D via im2col + GEMM (1–2 weeks)

Don’t start with a direct conv kernel — im2col teaches the pattern used everywhere.

1. **im2col**: unfold input patches into columns → `(C_out, C_in·K_h·K_w) @ (C_in·K_h·K_w, H_out·W_out)`
2. Reuse your batched GEMM
3. Optional: **direct conv** with shared-memory tiling (harder, teaches spatial locality)

**Milestone:** match `F.conv2d` for a simple case (3×3, stride 1, padding 1).

---

## Phase 6: Attention (2–3 weeks)

This is where it all comes together.

### Naive attention

```
S = Q @ K^T / sqrt(d)     # (seq, seq) — GEMM
P = softmax(S, dim=-1)    # reduction
O = P @ V                 # GEMM
```

Implement each piece with your existing kernels. You’ll hit memory limits fast on long sequences.

### Optimized attention (concepts, then code)

- **Tiled attention**: don’t materialize full `(seq, seq)` matrix
- **.Online softmax**: update running max/sum as you stream K/V blocks
- Study **FlashAttention** paper + a minimal implementation

**Milestone:** correct attention for `seq=512, d=64`; then optimize to handle `seq=2048` without OOM.

---

## Phase 7: Fusion & elementwise (ongoing)

Real systems fuse ops to cut memory traffic:


| Fused kernel           | Why                        |
| ---------------------- | -------------------------- |
| `matmul + bias + relu` | One write instead of three |
| `softmax + dropout`    | Attention training         |
| `rmsnorm + linear`     | Llama block                |


Pick one fusion after you have the parts working separately.

---

## What to study alongside coding

1. **Roofline model** — compute vs memory bound for each kernel
2. **Nsight Compute** — inspect your matmul (memory throughput, occupancy)
3. **CUTLASS docs** — see how pros structure GEMM (even if you don’t use it yet)
4. **Triton tutorials** — matmul tutorial is excellent for building intuition fast

---

## Suggested timeline (part-time, ~5–10 hrs/week)


| Weeks | Focus                              |
| ----- | ---------------------------------- |
| 1–3   | Matmul (naive → tiled → optimized) |
| 4     | Batched GEMM                       |
| 5     | Softmax + reductions               |
| 6     | LayerNorm / RMSNorm                |
| 7–8   | Conv2D (im2col)                    |
| 9–11  | Attention (naive → tiled)          |
| 12+   | Fusion, FP16/BF16, tensor cores    |


---

## Common mistakes to avoid

- **Skipping correctness** — always test against PyTorch first
- **Optimizing too early** — get naive working, profile, then optimize the hot spot
- **Ignoring memory layout** — row-major vs col-major matters enormously for GEMM
- **Trying to beat cuBLAS on day one** — compare for learning, not ego
- **Jumping to FlashAttention before matmul** — you’ll be debugging memory bugs without understanding why tiling exists

---

## A good first week

1. Day 1–2: naive CUDA matmul + Python test vs NumPy
2. Day 3–4: tiled matmul with shared memory
3. Day 5: benchmark both; run Nsight Compute on tiled version
4. Day 6–7: read one CUTLASS or Triton matmul tutorial; note 3 things they do that you don’t

If you want, I can scaffold the `kernel-practice` repo with that first matmul exercise (naive CUDA + PyTorch test harness + benchmark script) so you have a concrete starting point.