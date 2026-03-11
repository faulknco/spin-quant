# Experiment 17: Per-Row All-Layer Quantization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Quantize all 24 MLP layers of GPT-2 with per-row codebooks and measure whether the single-layer PPL improvement carries through to all-layer stacking.

**Architecture:** New script modelled on experiments/all_layer_quant.py and experiments/per_row_codebook.py. Replaces all 24 c_fc and c_proj layers with per-row quantized versions.

**Tech Stack:** PyTorch, HuggingFace transformers, WikiText-2, src/codebook.py (kmeans), experiments/eval_perplexity.py (conv1d_to_linear, eval_perplexity)

---

## Task 1: Create `experiments/per_row_all_layer.py`

### 1a. Write the file

Create `/Users/faulknco/projects/spin-quant/experiments/per_row_all_layer.py` with the exact content below.

```python
"""
Per-row codebook all-layer quantization.

Experiment 17. Exp 14 showed per-row k-means dramatically outperforms flat at
single-layer level (PPL=71 vs 154 at bpw=0.5). Exp 9 showed flat all-layer
(all 24 MLP layers) gives PPL=3,042 — 8× amplification from single-layer.

Question: does the per-row improvement (2.2× at bpw=0.5) carry through to
all-layer stacking? Expected if proportional: 3,042 × (71/154) ≈ 1,400.

Per-row doesn't alter activation distributions (precomputed W_q, no scale
migration), so no sequential calibration is needed — unlike SmoothQuant.

Design:
  - Quantize all 24 MLP layers (h0-h11 × c_fc, c_proj) with per-row codebooks
  - Three configs: bpw=0.5, 0.313, 0.750
  - Compare to: flat all-layer PPL=3,042 (Exp 9), per-row single-layer results

Usage:
    .venv/bin/python experiments/per_row_all_layer.py
"""

import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.codebook import kmeans
from experiments.eval_perplexity import conv1d_to_linear, eval_perplexity


MODEL_NAME = "gpt2"
N_EVAL     = 100
MAX_LEN    = 128
DEVICE     = "cpu"

# Reference numbers from prior experiments
EXP9_PPL = 3042.435       # flat all-layer bd=16 K=256 bpw=0.5
SINGLE_LAYER = {
    "per-row bd=8  K=16  bpw=0.500": 71.014,
    "per-row bd=16 K=32  bpw=0.313": 58.423,
    "per-row bd=8  K=64  bpw=0.750": 56.397,
}

# Configs: (label, desc, block_dim, K, bpw)
CONFIGS = [
    ("A", "per-row bd=8  K=16  bpw=0.500",  8,  16, 0.500),
    ("B", "per-row bd=16 K=32  bpw=0.313", 16,  32, 0.313),
    ("C", "per-row bd=8  K=64  bpw=0.750",  8,  64, 0.750),
]


class _ReconLinear(nn.Module):
    """Linear layer with precomputed reconstructed weight matrix."""
    def __init__(self, W_q, bias=None):
        super().__init__()
        self.register_buffer("W_q", W_q.float())
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None

    def forward(self, x):
        return F.linear(x, self.W_q, self.bias)


def quantize_per_row(W, b, block_dim, K):
    """
    Per-row k-means quantization: one codebook per output row.

    Each output row W[i] of shape [in_features] is split into
    n_blocks = in_features // block_dim blocks of shape [n_blocks, block_dim].
    K-means is run independently on each row's blocks. The reconstructed row
    W_q[i] = centroids_i[labels_i].reshape(1, in_features).

    Args:
        W:         [out_features, in_features] weight tensor
        b:         bias tensor or None
        block_dim: block size d (must divide in_features)
        K:         codebook size per row

    Returns:
        _ReconLinear with precomputed W_q of shape [out_features, in_features]
    """
    W = W.float().cpu()
    out_f, in_f = W.shape
    assert in_f % block_dim == 0, (
        f"in_features={in_f} must be divisible by block_dim={block_dim}"
    )
    n_blocks = in_f // block_dim

    # Pre-allocate reconstructed weight matrix
    W_q = torch.zeros_like(W)

    for i in range(out_f):
        if i > 0 and i % 500 == 0:
            print(f"  row {i}/{out_f}...", flush=True)

        # Shape: [n_blocks, block_dim]
        row_blocks = W[i].reshape(n_blocks, block_dim)

        # Run k-means on this row's blocks only
        centroids_i, labels_i = kmeans(row_blocks, K, n_iter=50, seed=42)

        # Reconstruct and store
        W_q[i] = centroids_i[labels_i].reshape(in_f)

    return _ReconLinear(W_q, b)


def build_all_layer_model(base_model, block_dim, K):
    """
    Deep-copy the base model and replace all 24 MLP layers (h0-h11 × c_fc, c_proj)
    with per-row quantized versions.

    Per-row quantization uses only the weight matrix itself (no H_diag calibration),
    so all 24 layers can be quantized from the base model directly — no sequential
    calibration required.

    Args:
        base_model: original FP32 GPT-2 model
        block_dim:  block size d for per-row k-means
        K:          codebook size per row

    Returns:
        Modified deep-copy of base_model with all 24 MLP layers quantized.
    """
    model = copy.deepcopy(base_model)
    for bi in range(12):
        block = model.transformer.h[bi]
        for attr in ["c_fc", "c_proj"]:
            raw = getattr(block.mlp, attr)
            lin = conv1d_to_linear(raw)
            W = lin.weight.data.clone()
            b = lin.bias.data.clone() if lin.bias is not None else None
            layer = quantize_per_row(W, b, block_dim, K)
            setattr(block.mlp, attr, layer)
            print(f"  h{bi}.{attr} done", flush=True)
    return model


def main():
    print(f"Loading {MODEL_NAME} ...")
    model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading WikiText-2 ...")
    test_data  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_texts = [t for t in test_data["text"] if len(t.strip()) > 50][:N_EVAL]

    print(f"\nEvaluating baseline ({N_EVAL} texts, max_len={MAX_LEN}) ...")
    ppl_base = eval_perplexity(model, tokenizer, eval_texts, MAX_LEN, DEVICE)
    print(f"  Baseline PPL = {ppl_base:.3f}")

    results = {}
    for (label, desc, bd, K, bpw) in CONFIGS:
        print(f"\n[{label}] {desc}  bd={bd}  K={K}  bpw={bpw:.3f}")
        print(f"  Quantizing all 24 MLP layers with per-row k-means ...")
        q_model = build_all_layer_model(model, bd, K)
        print(f"  Evaluating fully-quantized model ...")
        ppl = eval_perplexity(q_model, tokenizer, eval_texts, MAX_LEN, DEVICE)
        delta = ppl - ppl_base
        print(f"  PPL = {ppl:.3f}  (delta = {delta:+.3f})")
        results[label] = (desc, bd, K, bpw, ppl, delta)
        del q_model

    # Summary table
    print(f"\n{'='*80}")
    print(f"Experiment 17 Summary — Per-Row All-Layer Quantization (all 24 MLP layers)")
    print(f"{'='*80}")
    print(f"  Baseline FP32:                           PPL = {ppl_base:.3f}")
    print(f"  Flat all-layer Exp 9 (bd=16 K=256 bpw=0.5): PPL = {EXP9_PPL:.3f}")
    print()

    header = (
        f"  {'Config':<32}  {'bd':>3}  {'K':>4}  {'bpw':>5}  "
        f"{'PPL':>9}  {'delta':>9}  {'vs single-layer':>16}  {'vs Exp9':>9}"
    )
    print(header)
    print(f"  {'-' * (len(header.strip()))}")

    for label, (desc, bd, K, bpw, ppl, delta) in results.items():
        sl_ppl   = SINGLE_LAYER.get(desc)
        if sl_ppl is not None:
            sl_str = f"{ppl / sl_ppl:>+.2f}×"
        else:
            sl_str = "       n/a"

        # Only compare vs Exp9 for config A (same bpw=0.5)
        if label == "A":
            exp9_str = f"{ppl / EXP9_PPL:>+.3f}×"
        else:
            exp9_str = "      n/a"

        print(
            f"  ({label}) {desc:<29}  {bd:>3}  {K:>4}  {bpw:>5.3f}  "
            f"{ppl:>9.3f}  {delta:>+9.3f}  {sl_str:>16}  {exp9_str:>9}"
        )

    print()
    print(f"  Amplification factors (all-layer delta / single-layer delta vs FP32):")
    for label, (desc, bd, K, bpw, ppl, delta) in results.items():
        sl_ppl = SINGLE_LAYER.get(desc)
        if sl_ppl is not None:
            sl_delta   = sl_ppl - ppl_base
            all_delta  = delta
            factor     = all_delta / sl_delta if sl_delta > 0 else float("nan")
            print(f"    [{label}] {desc}: {factor:.2f}× amplification")

    print()
    # Check if per-row all-layer beats flat all-layer at bpw=0.5
    if "A" in results:
        ppl_A = results["A"][4]
        if ppl_A < EXP9_PPL:
            improvement = (EXP9_PPL - ppl_A) / EXP9_PPL * 100
            print(f"  Config A beats flat Exp 9 by {improvement:.1f}% PPL reduction.")
        else:
            regression = (ppl_A - EXP9_PPL) / EXP9_PPL * 100
            print(f"  Config A is {regression:.1f}% WORSE than flat Exp 9 — unexpected.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
```

### 1b. Verify syntax

```bash
cd /Users/faulknco/projects/spin-quant && .venv/bin/python -c "import experiments.per_row_all_layer"
```

Expected output: no errors, no tracebacks. If an ImportError appears, check that the `sys.path.insert` line correctly points `..` to the repo root (it does — same pattern as all other experiment files).

### 1c. Commit

```bash
cd /Users/faulknco/projects/spin-quant
git add experiments/per_row_all_layer.py
git commit -m "Add Experiment 17: per-row all-layer quantization script"
```

---

## Task 2: Run the experiment

### 2a. Launch in background

```bash
cd /Users/faulknco/projects/spin-quant
.venv/bin/python experiments/per_row_all_layer.py
```

Run this in the background. The script prints per-layer progress as it goes (`h0.c_fc done`, `h0.c_proj done`, ..., `h11.c_proj done`) so you can confirm it is running.

**Expected runtime:** 30–45 minutes on CPU.
- Each config quantizes 24 layers × ~3,072 output rows (c_fc: 768→3072, c_proj: 3072→768) = ~90,000 per-row k-means calls total across all 3 configs.
- Eval is ~100 texts × 3 configs = manageable.

**Expected output shape:**

```
Loading gpt2 ...
Loading WikiText-2 ...

Evaluating baseline (100 texts, max_len=128) ...
  Baseline PPL = 59.640

[A] per-row bd=8  K=16  bpw=0.500  bd=8  K=16  bpw=0.500
  Quantizing all 24 MLP layers with per-row k-means ...
  h0.c_fc done
  h0.c_proj done
  ...
  h11.c_proj done
  Evaluating fully-quantized model ...
  PPL = XXXX.XXX  (delta = +XXXX.XXX)

[B] ...
[C] ...

================================================================================
Experiment 17 Summary — Per-Row All-Layer Quantization (all 24 MLP layers)
================================================================================
  ...
```

### 2b. Collect results

Once the run completes, read the terminal output or redirect stdout to a log file if needed. Record the PPL values for configs A, B, C.

---

## Task 3: Record findings in FINDINGS_LOG.md

### 3a. Insert Experiment 17 section

Open `/Users/faulknco/projects/spin-quant/FINDINGS_LOG.md` and insert the following section **before** `## Experiment 15` (Exp 15 is currently the highest-numbered experiment in the log; insert Exp 17 at the top of the experiment section list, before Exp 15).

The exact insertion anchor is the line:

```
## Experiment 15 — Block-local H-weighted k-means
```

Insert the new section immediately before that line. Fill in `RESULT_A`, `RESULT_B`, `RESULT_C`, `DELTA_A`, `DELTA_B`, `DELTA_C`, `AMP_A`, `AMP_B`, `AMP_C` with the actual numbers from the run output.

```markdown
## Experiment 17 — Per-row all-layer quantization

**Question:** Does the 2.2× single-layer PPL improvement from per-row codebooks
(Exp 14, PPL=71 vs flat 154 at bpw=0.5) carry through when all 24 MLP layers are
quantized simultaneously?

**Design:** `experiments/per_row_all_layer.py`. Three configs applied to all 24
c_fc and c_proj layers in h0–h11. No calibration data needed — per-row quantization
is fully determined by the weight matrix, with no activation distribution migration.
Deep-copy base model, replace all 24 layers, evaluate.

**Results:**

```
Config                              bd    K    bpw       PPL      delta   vs single-layer   vs Exp9
(A) per-row bd=8  K=16  bpw=0.500   8   16  0.500  RESULT_A  +DELTA_A          AMP_A×       X.XXX×
(B) per-row bd=16 K=32  bpw=0.313  16   32  0.313  RESULT_B  +DELTA_B          AMP_B×          n/a
(C) per-row bd=8  K=64  bpw=0.750   8   64  0.750  RESULT_C  +DELTA_C          AMP_C×          n/a
```

Reference:
- Baseline FP32:          PPL =    59.640
- Flat all-layer Exp 9:   PPL = 3,042.435  (bd=16 K=256 bpw=0.5)
- Per-row single-layer A: PPL =    71.014  (Exp 14, bd=8 K=16 bpw=0.5)
- Per-row single-layer B: PPL =    58.423  (Exp 14, bd=16 K=32 bpw=0.313)
- Per-row single-layer C: PPL =    56.397  (Exp 14, bd=8 K=64 bpw=0.75)
```

### Key findings

**1. Amplification factor: per-row vs flat.**
Flat all-layer at bpw=0.5 (Exp 9) showed ~8× amplification from single-layer PPL delta
(single-layer delta ≈ +321, all-layer delta ≈ +2,982). Fill in whether per-row all-layer
shows a smaller, equal, or larger amplification factor.

If AMP_A < 8×: per-row error is better-conditioned for stacking — each layer's
reconstruction is tighter-tailed, so residual errors interact less destructively downstream.

If AMP_A ≈ 8×: amplification is a property of GPT-2 depth/width, not of the per-layer
error distribution. Per-row improves the base error but not the stacking multiplier.

If AMP_A > 8×: per-row introduces some new failure mode at all-layer scale (unlikely given
no activation migration, but worth noting if observed).

**2. Physics interpretation.**
Per-row codebooks address cross-row magnitude heterogeneity (different output neurons have
different weight scales). Flat k-means mixes hot and cold rows in a single codebook,
wasting centroids on the dynamic range. Per-row gives each row its own K centroids,
recovering ~2.2× in PPL at single-layer.

At all-layer scale, the question is whether the improved per-row reconstruction (smaller
||W - W_q||_F per row) translates linearly to smaller accumulated activation error.
The activation perturbation at layer l is δh_l = (W_q - W)x_l. For subsequent layers,
x_{l+1} = f(W_q x_l + b) vs x*_{l+1} = f(W x*_l + b), so errors compound through
the nonlinear residual stream. Tighter W_q (per-row) → smaller δh_l → proportionally
smaller compounding if the residual stream is approximately linear in the perturbation.

**3. Implications for next experiments.**
- If per-row all-layer PPL is proportional to single-layer improvement: per-row is the
  correct foundation for further compression. Next step: per-row bpw sweep across configs
  to find the PPL cliff (Exp 16 / per_row_bpw_sweep.py).
- If per-row all-layer still gives PPL > 1,000: multi-layer stacking requires a different
  approach (e.g., layer-wise distillation, activation-aware quantization order).
- Config B (bpw=0.313, near-lossless single-layer) is the most informative: if even
  near-lossless single-layer quantization degrades badly at all-layer scale, that points to
  inter-layer sensitivity rather than per-layer reconstruction error as the bottleneck.

---

```

### 3b. Commit

```bash
cd /Users/faulknco/projects/spin-quant
git add FINDINGS_LOG.md
git commit -m "Record Experiment 17 findings: per-row all-layer quantization results"
```

---

## Reference: complete file listing

- Script to create:  `/Users/faulknco/projects/spin-quant/experiments/per_row_all_layer.py`
- Log to update:     `/Users/faulknco/projects/spin-quant/FINDINGS_LOG.md`
- This plan:         `/Users/faulknco/projects/spin-quant/docs/plans/2026-03-11-per-row-all-layer.md`
- Design doc:        `/Users/faulknco/projects/spin-quant/docs/plans/2026-03-11-per-row-all-layer-design.md`
- Reference scripts: `/Users/faulknco/projects/spin-quant/experiments/per_row_codebook.py`
                     `/Users/faulknco/projects/spin-quant/experiments/all_layer_quant.py`

## Checklist

- [ ] Task 1: `experiments/per_row_all_layer.py` created and syntax-verified
- [ ] Task 1: committed
- [ ] Task 2: experiment run to completion
- [ ] Task 2: PPL values for configs A, B, C recorded
- [ ] Task 3: `## Experiment 17` section inserted in FINDINGS_LOG.md with actual numbers
- [ ] Task 3: committed
