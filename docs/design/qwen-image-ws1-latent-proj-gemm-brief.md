# Pre-implementation brief: `latent_proj_gemm` (Qwen-Image WS1)

**Status:** research only. No kernels in this document.
**Issue:** [RL-Align/RL-Kernel#386](https://github.com/RL-Align/RL-Kernel/issues/386) row `latent_proj_gemm`, claimed by `@haoruilee`.
**Context pin:** `upstream/main` @ `ccb70e3041f36fd272c0ec8cbbce53d952b65936` (2026-09-08). This fork's `origin/main` is **596 commits behind** that tip. Implement against a branch cut from `upstream/main`, not from this fork's current `HEAD`.

```bash
git fetch https://github.com/RL-Align/RL-Kernel.git main
git checkout -b feat/latent-proj-gemm FETCH_HEAD
```

---

## 0. What the issue actually asks for

One named single-GPU operator that covers **both** Qwen-Image boundary linears:

| Direction | Model module | Math | fwd / bwd | WS2 |
|---|---|---|---|---|
| `img_in` | `QwenImageTransformer2DModel.img_in` | packed latent `64 → 3072` | both | **no** |
| `proj_out` | `QwenImageTransformer2DModel.proj_out` | hidden `3072 → 64` | both | **no** |

Pinned numerics from #386 (repo-wide contribution rules, plus this row's extra "FP32 throughout"):

1. Fixed reduction tree, frozen and documented. Addition order never changes with batch size, token count, SM count, or launch geometry.
2. FP32 accumulators everywhere.
3. No Split-K, Stream-K, or atomic partial accumulation.
4. No fast-math, no `__expf` / `__logf`, no TF32, no compiler reassociation.
5. Generic WS1 rule says "one FP32→BF16 cast at final output only". **This row overrides that:** "FP32 throughout". Treat that as **no BF16 cast at all** unless maintainers say otherwise (open question Q1).
6. Batch invariance: a row's output bytes do not depend on who it is batched with or how it is tiled.
7. CUDA is the bit-level reference. Triton must match CUDA byte-for-byte, or register an explicit tolerance profile. No silent fallback.

Contribution ceremony (issue text: "refer to #204"): python + triton + cuda + binding + registry gating + clean fallback + tests + benchmark + validation-environment table.

Official model facts (not in the issue table, recovered from Diffusers + the published checkpoint):

| Fact | Value | Source |
|---|---|---|
| `in_channels` | 64 | `Qwen/Qwen-Image` `transformer/config.json` |
| `out_channels` | 16 | same |
| `patch_size` | 2 | same → packed channels = `2*2*16 = 64` |
| hidden | `24 * 128 = 3072` | `num_attention_heads * attention_head_dim` |
| `img_in` | `nn.Linear(64, 3072)` (default `bias=True`) | Diffusers `transformer_qwenimage.py` |
| `proj_out` | `nn.Linear(3072, 64, bias=True)` | same, `patch_size * patch_size * out_channels` |
| checkpoint tensors | `img_in.weight`, `img_in.bias`, `proj_out.weight`, `proj_out.bias` | `diffusion_pytorch_model.safetensors.index.json` |
| VAE `z_dim` | 16 (sibling op `latent_normalize`) | `vae/config.json` |
| spatial compression | 8× | `AutoencoderKLQwenImage` `dim_mult=[1,2,4,4]` |
| pack | `latent_pack_unpack` (sibling, still open) consumes `[B,16,H,W]` → `[B, (H/2)*(W/2), 64]` | #386 + Diffusers `_pack_latents` |

`latent_proj_gemm` must assume **already-packed** tokens. Do not fold pack/unpack into this kernel.

---

## 1. Closest existing GEMM / linear ops and file layout to copy

### 1.1 Do not treat this as "just call `det_gemm`"

`det_gemm` (WS1 #146 / PR #180) is the closest **kernel family**, but it is the wrong public contract for this row.

| Existing op | Why it is close | Why it is not enough |
|---|---|---|
| `det_gemm` | Fixed mid-split K-tree, no Split-K, FP32 accum, CUDA+Triton+tests+bench+docs, registry key | `check_in` is **BF16-only**. Naive tree uses **BF16 internal adds** (`bf16_add`) between K-tree children. Public API is `C = A[M,K] @ B[K,N]`, not HF `Linear`. `DetGemmOp.__call__` asserts BF16. |
| `DetGemmOp.linear` | HF `[N,K]` weight, `dX = dY @ W`, `dW = dY.T @ X` | Still BF16. Hopper auto path can select `cublaslt_nosplitk` (env-contract, not a documented FP32 tree). |
| `canonical_linear_fp32` | FP32 GEMM used by WS1 full-model backward session | Requires `active_session()`, logical keys, SM90 `det_gemm_rowwise_fwd_fp32`. Not a standalone op. |
| `NativeMatmulOp` / `matmul` | PyTorch gold `forward_fp32 = torch.matmul(a.float(), b.float())` | Gold for Qwen3 projections; **not** batch-invariant on GPU (cuBLAS). `det_gemm` docs explicitly exclude this from registry dispatch. |
| `NativeLMHeadOp` / `SM90LMHeadOp` | HF `[out,in]` linear + optional bias; per-row GEMV gold; `forward` / `forward_fp32` dual path | Vocab-scale, BF16 training path, backward routes through BF16 `det_gemm_*`. Best **Python API** template, not the CUDA body. |
| `Qwen3FFNOp` | Specialized deterministic GEMM orchestration | SwiGLU + TP collectives. Pattern for a named wrapper, not for a 64↔3072 boundary linear. |

**Copy the file layout from PR #180 (`det_gemm`) and the PR ceremony from PR #204 (`batch_invariant_logp` CUDA).** Copy the **callable contract** from `lm_head` (HF weight, optional bias, `forward` + `forward_fp32`).

### 1.2 Upstream paths to read before writing anything

GEMM family (copy structure, not BF16 tree internals):

```
csrc/cuda/gemm/det_gemm_kernel.cu          # mid-split tree, BM=128 BN=64 BK=32, naive fallback
csrc/cuda/gemm/det_gemm_tma.cuh            # Hopper TMA helper (BF16)
rl_engine/kernels/ops/cuda/matmul/det_gemm.py
rl_engine/kernels/ops/triton/matmul/det_gemm.py   # BLOCK 64/64/32, allow_tf32=False, no autotune
rl_engine/kernels/ops/pytorch/matmul/det_gemm.py  # NativeGemmOp — benchmark only
rl_engine/kernels/ops/pytorch/linear/matmul.py    # NativeMatmulOp gold
docs/operators/det-gemm.md
docs/operators/matmul.md
tests/test_det_gemm.py
benchmarks/benchmark_det_gemm.py
```

HF-linear family (copy API + bias + gold style):

```
rl_engine/kernels/ops/pytorch/linear/lm_head.py   # _fixed_k_projection + _strict_fp32_matmul
rl_engine/kernels/ops/cuda/linear/lm_head.py
docs/operators/lm_head.md
tests/test_lm_head.py
csrc/cuda/embedding_lm_head_sm90.cu               # det_gemm_rowwise_fwd_fp32 lives here
```

Ceremony / harness (copy registration, not logp math) — PR #204 file list:

```
csrc/ops.cpp
setup.py
rl_engine/_C.pyi
rl_engine/kernels/registry.py
rl_engine/kernels/gtest/operator_specs.py
rl_engine/kernels/gtest/operator_inputs.py
scripts/check_operator.py
docs/operators/<op>.md
docs/.nav.yml
docs/operators/README.md
docs/contributing/operator-doc-template.md
docs/contributing/gtest-usage.md
```

PR #180 is the better **GEMM** file-add list (19 files: cuda/triton/pytorch + kernel + tma + binding + registry + gtest + tests + bench + docs + nav). PR #199 is the Native+Triton-first split if CUDA lands in a follow-up.

### 1.3 Reduction tree already frozen on `det_gemm` — and why this op must re-document its own

From `csrc/cuda/gemm/det_gemm_kernel.cu` (upstream):

- Leaf width `K_TREE_LEAF = BK = 32`.
- Recurrence: `mid = lo + (hi-lo)//2`; child sums combined with **`bf16_add`** (cast to FP32, add, cast back to BF16).
- SM90 path: one CTA per `[BM=128, BN=64]` tile, walks K in steps of 32, no Split-K. `N` and `K` must be tile-aligned or it falls back to naive. `M` is zero-padded up to `BM` so kernel choice does not depend on batch.
- Simulated TP=2 (contiguous half-K) matches TP=1; TP=8 left-fold does not. **WS2 is "no" for this row**, so do not inherit the TP=2 half-K story as an acceptance requirement.

For `latent_proj_gemm` the issue wants **FP32 throughout**. Reusing `bf16_add` internals would violate the row spec even if inputs were secretly BF16. Freeze a **new** tree in the operator doc:

> Mid-split K-tree, leaf width 32, **FP32 leaves and FP32 internal adds**, single final store (FP32). One thread or one output element owns the full K range. No atomics.

That tree happens to be what the Triton `det_gemm` path already does when `PROMOTE_INPUTS=True` (`tl.float32` acc, `allow_tf32=False`, one program per output tile, no split-K). The CUDA naive path does **not**.

### 1.4 Tile alignment vs the two pinned shapes

Existing SM90 `det_gemm` tiles: `BM=128`, `BN=64`, `BK=32`.

| Direction | logical `K` | logical `N` (out) | SM90-aligned? |
|---|---|---|---|
| `img_in` | 64 = 2×BK | 3072 = 48×BN | yes |
| `proj_out` | 3072 = 96×BK | 64 = 1×BN | yes |

So a future BF16/SM90 specialization would align. **v1 should not depend on that path**: it is BF16-load, optional `KERNEL_ALIGN_DET_GEMM_SM90`, and `det_gemm_fwd_fp32` is currently **unbound** in `csrc/ops.cpp` / `rl_engine/_C.pyi` (symbol exists in the `.cu` but is not `m.def`'d; the function also calls `gemm_dispatch` without `output_fp32=true`). `det_gemm_rowwise_fwd_fp32` lives in the SM90-only `embedding_lm_head_sm90.cu` and is also unbound in `ops.cpp`.

Recommendation: **new always-compiled CUDA source** (like `det_gemm_kernel.cu`, not `sm90_srcs`), naive/tiled FP32, SM80+ or even "any CUDA". Hopper TMA is optional later and must not change the documented tree.

---

## 2. Exact API contract for `latent_proj_gemm`

### 2.1 Recommended public Python surface

One registry name, two pinned directions, HF `nn.Linear` layout. Match `lm_head` so gtest / autograd / docs stay uniform.

```python
from rl_engine.kernels.registry import kernel_registry

op = kernel_registry.get_op("latent_proj_gemm")

# img_in:  x[..., 64] @ W[3072, 64].T + b[3072]  -> y[..., 3072]
y = op(x, weight, bias=bias)                 # execution dtype (FP32)
y32 = op.forward_fp32(x, weight, bias=bias)  # gold: FP32 in, acc, out

# proj_out: x[..., 3072] @ W[64, 3072].T + b[64] -> y[..., 64]
```

| Argument | Shape | Dtype | Layout / rules |
|---|---|---|---|
| `x` | `[..., in_features]` | **FP32** | Packed image tokens. Leading dims flatten to `M`. Contiguous last dim. |
| `weight` | `[out_features, in_features]` | **FP32** | HF `nn.Linear` `[out, in]`. Same as `lm_head`, **not** `det_gemm`'s `[K,N]`. |
| `bias` | `[out_features]` or `None` | **FP32** | Official Qwen-Image has bias on **both** modules. Support both `None` and present; default tests use bias. |
| output | `x.shape[:-1] + (out_features,)` | **FP32** | No dtype cast. |

Pinned `(in_features, out_features)` only:

| `direction` | `in_features` | `out_features` | `weight` | `bias` |
|---|---|---|---|---|
| `img_in` | 64 | 3072 | `[3072, 64]` | `[3072]` |
| `proj_out` | 3072 | 64 | `[64, 3072]` | `[64]` |

Infer `direction` from shapes; raise on any other pair. Do not accept Qwen3 4096-wide projections through this op.

Suggested helpers (keep them thin):

- `forward` / `__call__` — FP32→FP32, same as `forward_fp32` for this op (the dual path still exists so gtest can call `gold_method="forward_fp32"`).
- `forward_fp32` — gold: disable autocast + `torch.backends.cuda.matmul.allow_tf32 = False`, then **per-row** `torch.mv(weight, row) + bias` (copy `NativeLMHeadOp._fixed_k_projection`). Do **not** gold-path `torch.matmul` on CUDA.
- Autograd: `dX = dY @ W` (FP32 tree over `out_features`), `dW = dY.T @ X` (FP32 tree over `M`, tokens in ascending order), `dB = sum_M dY` (left-to-right / ascending-row fold, FP32). Multi-path sums use that fixed order.

### 2.2 Semantics

```
Y[m, n] = sum_{k=0..K-1}^{mid-split tree} X[m, k] * W[n, k]  +  B[n]
```

- `K = in_features`, `N = out_features`, `M = prod(x.shape[:-1])`.
- Weight is consumed as `[out, in]` (no caller-facing transpose).
- Bias add is after the GEMM, in FP32, and must not be fused in a way that changes the GEMM tree.
- Empty `M=0` returns an empty `[0, out]` of the right dtype (lm_head already has this guard).

### 2.3 Backends and fallback (no silent fallback)

| Platform | Priority | Notes |
|---|---|---|
| CUDA | `CudaLatentProjGemmOp` → `TritonLatentProjGemmOp` | CUDA is bit-reference. If the CUDA extension is missing **and** the caller asked for CUDA, **raise** (copy `DetGemmOp`: "refusing non-strict fallback"). Auto `get_op` may degrade CUDA→Triton only when the CUDA class fails to import, and must record `fallback=true` + reason. |
| ROCm | Triton only | Must match CUDA bytes on the same dataset **or** ship an explicit hardware tolerance profile. Default plan: bitwise vs CUDA on NVIDIA; on ROCm document "Triton is the local reference, cross-checked on NVIDIA CI". |
| CPU | `NativeLatentProjGemmOp` | Gold / gtest plumbing only. |
| MUSA / NPU | inherit CPU gold unless a later backend exists | Do not invent a silent CUDA stub. |

Registry pattern to copy (`rl_engine/kernels/registry.py`):

- Add `OpBackend.CUDA_LATENT_PROJ_GEMM`, `TRITON_LATENT_PROJ_GEMM`, `PYTORCH_LATENT_PROJ_GEMM`.
- `cuda["latent_proj_gemm"] = [CUDA, TRITON]`
- `rocm["latent_proj_gemm"] = [TRITON]`
- `cpu` / `musa` = `[PYTORCH]`
- **Do not** put `Native*` on the CUDA auto list (same reason as `det_gemm`: cuBLAS is not batch-invariant).

Gating:

- CUDA class `__init__` checks `_EXT_AVAILABLE` and required `_C` symbols; missing → `RuntimeError`.
- Triton class `__init__` checks `import triton`.
- Dtype/shape mismatch → `TypeError` / `ValueError`, not a silent cast to BF16.
- `setup.py` already has `KERNEL_ALIGN_USE_FAST_MATH` → `--use_fast_math`. New `.cu` must compile with IEEE FP32 and must not opt into that flag. Do not pass `-use_fast_math` / `--ftz` / `--prec-div=false`. Disable TF32 at the op boundary (`allow_tf32 = False`) like `NativeLMHeadOp._strict_fp32_matmul`.

### 2.4 Trace / fingerprint (issue acceptance item)

Minimum record, one dict per launch (forward and backward). Reuse `record_backward` (`rl_engine/kernels/ops/backward_runtime.py`) plus a small forward provenance helper:

| Field | Pinned value |
|---|---|
| `reduction_order` | `mid_split_k_tree` |
| `k_tree_leaf` | `32` |
| `accumulator_dtype` | `float32` |
| `output_dtype` | `float32` |
| `split_k` | `disabled` |
| `stream_k` | `disabled` |
| `tf32` | `disabled` |
| `fast_math` | `disabled` |
| `kernel_fingerprint` | `rlkernel.latent_proj_gemm.<cuda\|triton\|pytorch>.v1` |
| `direction` | `img_in` \| `proj_out` |
| `backend` | actual, not requested |

`BackendProvenance` for WS1 gtest (`docs/contributing/gtest-usage.md` §6.3) should set `execution_dtype=float32`, `accumulation_dtype=float32`, `output_dtype=float32`, `reference_dtype=float32`, both TF32 flags false. The stock contract policy says WS1 EXIT execution is BF16 — this op is an **explicit FP32 exception**; do not fake `cuda_bf16` provenance.

### 2.5 WS2

Issue column is **no**. Do not add `latent_proj_gemm_parallel`, TP row/col splits, or collective staging. Do not accept `cublaslt_nosplitk` as a backend (it is a Hopper GEMM contract for Qwen3 FFN, not this op).

---

## 3. Checklist of files to add / change

All paths below are on **upstream `main`**. Several do not exist on this fork yet.

### 3.1 New files

| Path | Role |
|---|---|
| `rl_engine/kernels/ops/pytorch/linear/latent_proj_gemm.py` | Gold: per-row FP32 GEMV + bias, TF32/autocast off. |
| `rl_engine/kernels/ops/triton/linear/latent_proj_gemm.py` | Pinned-block Triton GEMM, FP32 acc, no autotune. Clone `ops/triton/matmul/det_gemm.py` but HF weight + bias + FP32 I/O. |
| `rl_engine/kernels/ops/cuda/linear/latent_proj_gemm.py` | Autograd wrapper around `_C` symbols; `record_backward`; strict raise if extension missing. |
| `csrc/cuda/gemm/latent_proj_gemm_kernel.cu` | Always-compiled CUDA. New tree (FP32 internal adds). Fwd + `dX` + `dW` + `dB`. |
| `tests/test_latent_proj_gemm.py` | Correctness, bitwise CUDA↔Triton, batch/chunk/pad invariance, both directions, bias, pinned image token counts. |
| `benchmarks/benchmark_latent_proj_gemm.py` | Native / Triton / CUDA on the three image Ms × both directions. Overhead vs cuBLAS FP32, not a speedup claim. |
| `docs/operators/latent-proj-gemm.md` | Copy `docs/contributing/operator-doc-template.md`. Document the frozen tree. |

Optional later (not required for the first PR if v1 is naive CUDA): `csrc/cuda/gemm/latent_proj_gemm_tma.cuh`.

### 3.2 Existing files to edit

| Path | Change |
|---|---|
| `rl_engine/kernels/ops/pytorch/linear/__init__.py` | Export `NativeLatentProjGemmOp`. |
| `rl_engine/kernels/ops/triton/linear/__init__.py` | Export Triton op. |
| `rl_engine/kernels/ops/cuda/linear/__init__.py` | Export CUDA op (file currently only exports embedding/lm_head). |
| `rl_engine/kernels/ops/cuda/__init__.py` | Already imports `matmul`; linear is imported via package. Confirm no extra hook needed. |
| `csrc/ops.cpp` | Declare + `m.def` fwd / dx / dw / dbias (and a compiled-marker if you follow `det_gemm_sm90_compiled`). |
| `rl_engine/_C.pyi` | Stub the new symbols. |
| `setup.py` | Append the new `.cu` to the always-on `cuda_sources` list next to `csrc/cuda/gemm/det_gemm_kernel.cu` (around the block that starts ~line 134). Do **not** hide it behind `KERNEL_ALIGN_FORCE_SM90` / `KERNEL_ALIGN_DET_GEMM_SM90`. |
| `rl_engine/kernels/registry.py` | `OpBackend` strings + priority map for `cuda` / `rocm` / `musa` / `cpu` / `npu`. |
| `rl_engine/kernels/gtest/operator_specs.py` | `OP_SPECS["latent_proj_gemm"]`: `op_class="reduction"`, gold = native `forward_fp32`, candidates `pytorch` / `triton` / `cuda`. `grad_input_names=("x", "weight", "bias")`. |
| `rl_engine/kernels/gtest/operator_inputs.py` | Builder + `operator_shape_name`. Add CLI-friendly `--in-features` / `--out-features` **or** reuse `--k-dim`/`--n-dim` with defaults `64`/`3072` and a `--direction` flag. Wire `scripts/check_operator.py` if new flags are added. |
| `tests/test_operator_inputs.py` | Append `"latent_proj_gemm"` to the parametrize list. |
| `tests/test_kernel_registry.py` | Dispatch smoke if you add hardware-gated prepend (only needed if SM90-only). |
| `docs/operators/README.md` | Link the new page. |
| `docs/.nav.yml` | Add `operators/latent-proj-gemm.md` under Operators (upstream nav already lists `det-gemm.md`; this fork's nav does not — edit whichever tree the PR targets). |
| `rl_engine/kernels/gtest/gradient_adapters.py` | New adapter if C4 bitwise grad invariance is in-scope for the first PR. |
| Issue #386 table | After the PR exists: fill `PR` column and set status `👀`. Claim comment already landed 2026-09-08. |

### 3.3 Do not touch (for v1)

- `csrc/cuda/gemm/det_gemm_kernel.cu` internals (Qwen3 BF16 contract).
- FFN / `cublaslt_nosplitk` routing.
- WS2 collective / TP files.
- `tolerance_contract.json` BF16 reduction rows, unless you add an explicit `float32` reduction row for this op's gtest (likely needed so `check_operator.py --dtype fp32` does not inherit BF16 tols). Check `rl_engine/kernels/gtest/tolerance.py` resolver before inventing private `atol`.

### 3.4 Suggested first-PR split (same as #199 → #204)

1. Native gold + Triton + tests + docs + registry + gtest inputs (CPU + CUDA Triton).
2. CUDA kernel + binding + bitwise CUDA↔Triton + benchmark + validation-env table.

#386 asks for all of it in "a new ops" PR; one PR is fine if it stays reviewable. If split, land Native+Triton first so CUDA has a bitwise target.

---

## 4. Test shapes and acceptance harness usage

### 4.1 Image → token counts (must appear in tests)

VAE spatial `/8`, then `latent_pack_unpack` 2×2 pack. `M = (H/8/2) * (W/8/2)`.

| Issue shape | Pixel `H×W` | Latent `h×w` | Packed `ph×pw` | `M` tokens | `img_in` GEMM | `proj_out` GEMM |
|---|---|---|---|---|---|---|
| `1024²` | 1024×1024 | 128×128 | 64×64 | **4096** | `[4096,64]×[64,3072]` | `[4096,3072]×[3072,64]` |
| `1328²` | 1328×1328 | 166×166 | 83×83 | **6889** | `[6889,64]×[64,3072]` | `[6889,3072]×[3072,64]` |
| `1664×928` | 1664×928 | 208×116 | 104×58 | **6032** | `[6032,64]×[64,3072]` | `[6032,3072]×[3072,64]` |

Also run `M ∈ {1, 7, 31, 64, 128, 256}` so tile edges (`BM=128` if anyone later reuses SM90 padding) and ragged Triton masks are covered. `1328²` / `1664×928` are **not** multiples of 128 — that is the point.

CI can keep `M≤256` plus one full `M=4096` smoke if GPU memory allows (~50–90 MiB FP32 activations; cheap). Mark `6889` / `6032` as GPU nightly if CI is tight; #386 still requires those shapes in the PR's recorded run.

### 4.2 pytest matrix (`tests/test_latent_proj_gemm.py`)

Mirror `tests/test_det_gemm.py` + `tests/test_lm_head.py`:

- Forward correctness vs native `forward_fp32` (bitwise on CUDA gold if gold is the per-row GEMV; Triton/CUDA must be bitwise to **CUDA**, and within contract vs native if native uses a different tree — see Q5).
- Backward `dX`, `dW`, `dB` vs a fixed-order FP32 reference.
- Batch invariance: row in `M=1` vs same row in `M=N` (`torch.equal`).
- Chunked prefill: `cat(op(x[:t]), op(x[t:])) == op(x)`.
- Padding invariance: extra dummy rows must not change valid rows.
- Both directions + bias present / `bias=None`.
- Reject wrong shapes (`in=4096`, `out=12288`, 3-D weight, BF16 input).
- Registry: `get_op("latent_proj_gemm")` returns CUDA on SM80+ with extension, else Triton, else raise/native by platform.
- Cross-backend: `torch.equal(cuda(x,W,b), triton(x,W,b))` for every pinned shape that fits CI.
- Trace dict keys present after one forward+backward.

Do **not** assert bitwise equality against `torch.matmul` / cuBLAS.

### 4.3 gtest / `check_operator.py`

After `OP_SPECS` + inputs land:

```bash
# plumbing
python scripts/check_operator.py --op latent_proj_gemm --candidate pytorch --device cpu --dtype fp32 \
  --batch 2 --seq 16 --k-dim 64 --n-dim 3072

# Triton candidate (CUDA device)
python scripts/check_operator.py --op latent_proj_gemm --candidate triton --device cuda --dtype fp32 \
  --batch 1 --seq 64 --k-dim 64 --n-dim 3072 --check-grad --grad-mode random

python scripts/check_operator.py --op latent_proj_gemm --candidate triton --device cuda --dtype fp32 \
  --batch 1 --seq 64 --k-dim 3072 --n-dim 64 --check-grad

# CUDA candidate
python scripts/check_operator.py --op latent_proj_gemm --candidate cuda --device cuda --dtype fp32 \
  --batch 1 --seq 4096 --k-dim 64 --n-dim 3072 --check-grad
```

`--batch * --seq` is how `operator_inputs.py` builds `M` for `det_gemm` today (`M = batch * seq`, 2-D `[M,K]`). Either keep that and treat `seq` as packed tokens, or add an explicit `--tokens` / `--direction`. Prefer `--direction {img_in,proj_out}` so reviewers do not swap K/N.

Issue #386 says "Forward matches the FP32 CPU reference **byte for byte**". That is stricter than the WS1 four-judgment `forward_accuracy` tols for BF16 reduction. For this FP32 op:

- Gold method = `forward_fp32`.
- Candidate vs gold: **bitwise** if gold implements the same mid-split tree; otherwise gold must be rewritten to that tree (a Python recursive mid-split, like `_k_tree_gemm` in `tests/test_det_gemm.py` but **without** the BF16 leaf cast).
- Do not use `contract["accuracy"]["default"]["reduction"]["bfloat16"]` as the gate.

C3/C4 (`scripts/check_forward_invariance.py`, `scripts/check_gradient_invariance.py`) are the WS1 Qwen3 chain harness. Nice-to-have for this multimodal row; **not** a substitute for the issue's own batch-invariance tests. If added, register a `gradient_adapters.py` entry.

### 4.4 Benchmark

`python benchmarks/benchmark_latent_proj_gemm.py`

Columns: direction, `M`, backend ms, peak extra MB. Shapes = the three image `M`s × `{img_in, proj_out}`. Compare CUDA vs Triton vs `torch.matmul` FP32 (TF32 off). Report overhead, same tone as `benchmarks/benchmark_det_gemm.py`.

### 4.5 Validation environment table (copy #204)

Every kernel PR in this repo is expected to paste a table like:

| Item | Value (fill at run time) |
|---|---|
| GPU | e.g. H200 / A100 |
| Driver / CUDA | |
| `nvcc` | |
| PyTorch | |
| Python | |
| Host compiler | |
| Extension symbols | `_C.latent_proj_gemm_fwd` (name TBD) present |
| Build vars | `TORCH_CUDA_ARCH_LIST=...`; **no** `KERNEL_ALIGN_USE_FAST_MATH` |
| Commands | pytest + the `check_operator.py` lines above + benchmark |

### 4.6 Issue #386 acceptance checklist (this row)

- [ ] Forward matches the FP32 CPU **tree** reference byte for byte (CUDA and Triton).
- [ ] Backward matches the fixed FP32 reference; `dW` / `dB` reductions use a frozen token/row order.
- [ ] Batch invariance for `M` and launch geometry.
- [ ] Shapes cover `{1024², 1328², 1664×928}` token counts above.
- [ ] Trace records reduction order, acc precision, Split-K/Stream-K, TF32, fingerprint.
- [ ] FP32 CPU reference + bit-equality harness shipped.
- [ ] Pinned items have explicit tests: both directions, K∈{64,3072}, bias, FP32-only rejection of BF16.

---

## 5. Pitfalls and open questions

### 5.1 Pitfalls (will break review if ignored)

1. **Wrong base branch.** Implementing on this fork's `main` means fighting a 596-commit gap (no `det_gemm`, no gtest contract, no current registry). Cut from `upstream/main`.
2. **Reusing `det_gemm` unchanged.** BF16 `check_in`, BF16 K-tree internals, unbound `det_gemm_fwd_fp32`, Hopper `cublaslt_nosplitk` auto-route. A wrapper that casts FP32→BF16, calls `det_gemm`, casts back **fails** "FP32 throughout" and the bitwise CPU tree.
3. **Gold = `torch.matmul`.** That is `NativeGemmOp` / cuBLAS. It will fail batch-invariance on GPU and will not match a mid-split tree bitwise. Gold must be the documented tree (per-row left-to-right leaf of width 32, mid-split combine, all FP32).
4. **HF layout vs `A @ B`.** Callers and the checkpoint store `[out, in]`. `det_gemm` tests use `[K, N]`. Mixing these is the most likely silent bug. Prefer `DetGemmOp.linear` / `lm_head` (`Y = X @ W.T`).
5. **Bias omitted.** The issue table does not say "+ bias", but Diffusers and the published `img_in.bias` / `proj_out.bias` tensors do. Shipping a bias-free kernel makes the assembled MMDiT block (#386 WS1 exit) wrong.
6. **Generic rule #5 (BF16 store) vs this row.** If the CUDA kernel "helpfully" stores BF16 to look like other WS1 GEMMs, `img_in` would inject a cast into every DiT block input and `proj_out` would quantize the VAE-facing residual. Keep FP32 stores. A later `dtype_cast` inventory op already exists for explicit casts.
7. **Silent fallback.** Registry and CUDA wrapper must raise. #386: "No silent fallback."
8. **Autotune / shape-dependent kernels.** Triton `autotune` is forbidden (`det_gemm` comment: autotune picks per-shape configs → breaks invariance). CUDA must not pick naive vs tiled based on `M` in a way that changes a valid row's bytes. If two algorithms exist, `M`-padding (as SM90 `det_gemm` does) or a single algorithm for all `M` is required.
9. **`KERNEL_ALIGN_USE_FAST_MATH` and TF32.** `setup.py` can inject `--use_fast_math`. Document that validation builds leave it unset. Pin `allow_tf32 = False` inside the op, do not trust the caller.
10. **`dW` token reduction.** When `M` is small, `det_gemm`'s `det_gemm_db_small_k` visits tokens `0..M-1` and rounds once. Replicate that **order** (ascending tokens, one FP32 acc, one store) so `M=1` vs `M=4096` `dW` for a shared prefix stays well-defined. No atomics across CTAs for partial `dW`.
11. **Claim protocol.** #386 says: put your handle in the table **and** open a PR against that issue. The handle is already there; the implementation PR must reference `#386` and update the `PR` / status cells (on `upstream`, not only this fork).
12. **Sibling ops.** `latent_pack_unpack` is still `🙋`. Tests should construct packed `[M,64]` tensors directly; do not block on pack. `txt_in` is a **different** linear (`3584→3072`) and is a separate row.

### 5.2 Open questions (ask `@zhangj1an` on #386 before locking the CUDA tree)

| ID | Question | Recommendation if no reply |
|---|---|---|
| **Q1** | Does "FP32 throughout" mean tensors stay FP32 (no BF16 store), or only "accum FP32" with a final BF16 cast as in generic rule 5? | **Stay FP32.** The table singles this row out the same way as `flow_sde_step_logp`. Put the exception in `docs/operators/latent-proj-gemm.md`. |
| **Q2** | Bias yes/no? Table omits it; checkpoint has it. | **Optional `bias` argument, default tests with bias.** |
| **Q3** | One registry op vs two (`latent_img_in_gemm`, `latent_proj_out_gemm`)? | **One op**, shape-inferred direction. Matches the kernel table name. |
| **Q4** | Must CUDA v1 be Hopper TMA, or is naive FP32 scalar acceptable (as `det_gemm`'s first milestone)? | **Naive/tiled FP32 first.** #146 explicitly accepted a slow deterministic baseline. Skinny `K=64` / `N=64` does not need WGMMA for correctness. |
| **Q5** | Is the CPU gold the mid-split tree (bitwise with CUDA) or a left-to-right `torch.mv`? #386 says "byte for byte" vs "FP32 CPU reference". | **Implement the CPU gold as the same mid-split tree** so bitwise is possible. Also keep a `torch.mv` / sequential leaf check as a loose sanity, not the gate. |
| **Q6** | Should gtest grow a `float32` reduction contract row, or is bitwise-only enough? | Bitwise tests are the issue gate. Add a `float32` contract row only if `check_operator.py` otherwise mis-resolves tols. |
| **Q7** | May Triton on ROCm ship a documented non-bitwise profile? | Yes, but only with a written hardware delta. NVIDIA CUDA remains the reference. |
| **Q8** | Is `M` allowed to include a batch dim of images (`B * tokens`) or must tests be `B=1`? | Flatten leading dims; invariance is **per packed token row**, same as `det_gemm` vs `M`. |

### 5.3 Suggested implementation order (still no kernel code here)

1. Confirm Q1–Q3 on #386 in one comment.
2. Branch from `upstream/main`.
3. Land Native gold + gtest inputs + docs contract (tree written down **before** CUDA).
4. Land Triton matching that tree; pytest invariance on `M∈{1,7,31,64,128,4096}` and both directions.
5. Land CUDA matching Triton bitwise; bind; registry; benchmark; paste validation env on the PR (the #204 section).
6. Tick the #386 row to `👀` with the PR link.

### 5.4 Non-goals

- No kernel listings, tile diagrams that prescribe PTX, or PoC implementations in this brief.
- No WS2 / TP / sequence-parallel variant.
- No fusion with `latent_normalize`, `norm_out` / AdaLN, or pack/unpack.
- No Qwen3 FFN / lm_head shape support on this registry key.
