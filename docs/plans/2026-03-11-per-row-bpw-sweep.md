# Experiment 16: Per-Row BPW Sweep Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Sweep K values for per-row k-means on h0.c_fc to find where the per-row phase transition occurs, comparing to flat k-means results from Exp 6/8.

**Architecture:** New script modelled on experiments/per_row_codebook.py. Reuses quantize_per_row logic. Sweeps two block_dim values (8 and 16) across all valid K values.

**Tech Stack:** PyTorch, HuggingFace transformers, WikiText-2, src/codebook.py (kmeans), experiments/eval_perplexity.py (conv1d_to_linear, eval_perplexity)

---

## Task 1: Create `experiments/per_row_bpw_sweep.py`

Create the file `/Users/faulknco/projects/spin-quant/experiments/per_row_bpw_sweep.py` with the following complete content:

```python
"""
Per-row codebook BPW sweep — finding the phase transition.

Experiment 16. Exp 14 showed per-row codebooks dramatically outperform flat
k-means at bpw=0.5 (PPL=71 vs 154) and achieve near-lossless at bpw=0.313.

Flat k-means has a sharp phase transition at bpw≈0.5 (K=256, bd=16).
Per-row eliminates cross-row centroid competition — the per-row critical bpw
should be much lower. This experiment sweeps K for bd=8 and bd=16 to find
where (if anywhere) per-row breaks down.

Target: h0.c_fc. Compare to flat sweep (Exp 6/8).

Usage:
    .venv/bin/python experiments/per_row_bpw_sweep.py
"""

import sys, os, math, copy
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

# Flat k-means reference PPLs from prior experiments (hardcoded).
# Keys are (block_dim, K).
FLAT_REF = {
    (8,  16):  154.340,   # Exp 8
    (8,  32):  16423.0,   # Exp 8 non-monotone
    (16, 256): 380.618,   # Exp 6
}


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


def run_sweep(W, b, block_dim, K_list, model, tokenizer, eval_texts, ppl_base):
    """
    Run per-row k-means for each K in K_list and return a list of result dicts.

    Each result dict has keys: K, bpw, ppl, delta, note.
    """
    out_f, in_f = W.shape
    n_blocks_per_row = in_f // block_dim
    results = []

    for K in K_list:
        bpw = math.log2(K) / block_dim
        print(f"\n  bd={block_dim}  K={K}  bpw={bpw:.3f}  (n_blocks_per_row={n_blocks_per_row})")
        assert K <= n_blocks_per_row, (
            f"K={K} exceeds n_blocks_per_row={n_blocks_per_row} for bd={block_dim}"
        )

        print(f"  Running per-row k-means over {out_f} rows ...")
        layer = quantize_per_row(W, b, block_dim, K)

        m = copy.deepcopy(model)
        m.transformer.h[0].mlp.c_fc = layer
        ppl = eval_perplexity(m, tokenizer, eval_texts, MAX_LEN, DEVICE)
        del m

        delta = ppl - ppl_base
        note = "BROKEN" if ppl > 10 * ppl_base else ""
        print(f"  PPL = {ppl:.3f}  (delta = {delta:+.3f})  {note}")

        results.append(dict(K=K, bpw=bpw, ppl=ppl, delta=delta, note=note))

    return results


def print_table(block_dim, results, ppl_base):
    """Print summary table for one block_dim sweep."""
    n_blocks_per_row = 768 // block_dim   # GPT-2 h0.c_fc in_features=768
    print()
    print(f"  bd={block_dim}  (n_blocks_per_row={n_blocks_per_row})")
    print(f"  Baseline (FP32): {ppl_base:.3f}")
    print()
    print(f"  {'K':>4}  {'bpw':>6}  {'PPL':>10}  {'delta':>10}  {'note':<8}  flat_ref")
    print(f"  {'-'*60}")
    for r in results:
        flat_ref_ppl = FLAT_REF.get((block_dim, r["K"]))
        flat_str = f"{flat_ref_ppl:.3f}" if flat_ref_ppl is not None else "      —"
        print(
            f"  {r['K']:>4}  {r['bpw']:>6.3f}  {r['ppl']:>10.3f}  "
            f"{r['delta']:>+10.3f}  {r['note']:<8}  {flat_str}"
        )


def main():
    print(f"Loading {MODEL_NAME} ...")
    model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading WikiText-2 ...")
    test_data  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_texts = [t for t in test_data["text"] if len(t.strip()) > 50][:N_EVAL]

    raw_layer     = model.transformer.h[0].mlp.c_fc
    target_linear = conv1d_to_linear(raw_layer)
    W = target_linear.weight.data.clone()
    b = target_linear.bias.data.clone() if target_linear.bias is not None else None

    out_f, in_f = W.shape
    print(f"Target: h0.c_fc  {tuple(W.shape)}  (out_features={out_f}, in_features={in_f})")

    # Baseline
    print("\nBaseline (full precision) ...")
    ppl_base = eval_perplexity(model, tokenizer, eval_texts, MAX_LEN, DEVICE)
    print(f"  PPL = {ppl_base:.3f}")

    # bd=8 sweep — n_blocks_per_row = 768/8 = 96, so K <= 96
    # bpw = log2(K) / 8
    #   K=2  → 0.125
    #   K=4  → 0.250
    #   K=8  → 0.375
    #   K=16 → 0.500  (matches Exp 14C)
    #   K=32 → 0.625
    #   K=64 → 0.750  (matches Exp 14D)
    #   K=96 → 0.896  (log2(96)/8 ≈ 0.833; note 96 is not a power of 2)
    K_list_bd8 = [2, 4, 8, 16, 32, 64, 96]

    # bd=16 sweep — n_blocks_per_row = 768/16 = 48, so K <= 48
    # bpw = log2(K) / 16
    #   K=2  → 0.063
    #   K=4  → 0.125
    #   K=8  → 0.250
    #   K=16 → 0.250  (log2(16)/16 = 0.250)
    #   K=32 → 0.313  (matches Exp 14E)
    #   K=48 → 0.363  (log2(48)/16 ≈ 0.363; max valid K)
    K_list_bd16 = [2, 4, 8, 16, 32, 48]

    print("\n" + "="*70)
    print("Sweep 1: bd=8")
    print("="*70)
    results_bd8 = run_sweep(W, b, 8, K_list_bd8, model, tokenizer, eval_texts, ppl_base)

    print("\n" + "="*70)
    print("Sweep 2: bd=16")
    print("="*70)
    results_bd16 = run_sweep(W, b, 16, K_list_bd16, model, tokenizer, eval_texts, ppl_base)

    # Final summary
    print("\n" + "="*70)
    print("Experiment 16 Summary — Per-Row BPW Sweep (h0.c_fc)")
    print("="*70)
    print_table(8,  results_bd8,  ppl_base)
    print()
    print_table(16, results_bd16, ppl_base)

    # Identify phase transition (first K where PPL > 2x baseline)
    print()
    for bd, results in [(8, results_bd8), (16, results_bd16)]:
        broken = [r for r in results if r["note"] == "BROKEN"]
        good   = [r for r in results if r["note"] != "BROKEN"]
        if broken:
            min_broken_K = min(r["K"] for r in broken)
            print(f"  bd={bd}: phase transition at K={min_broken_K} "
                  f"(bpw={math.log2(min_broken_K)/bd:.3f})")
        else:
            best = min(results, key=lambda r: r["ppl"])
            print(f"  bd={bd}: no breakdown found — best PPL={best['ppl']:.3f} "
                  f"at K={best['K']} (bpw={best['bpw']:.3f})")


if __name__ == "__main__":
    main()
```

After writing the file, verify syntax with:

```bash
cd /Users/faulknco/projects/spin-quant && .venv/bin/python -c "import experiments.per_row_bpw_sweep"
```

Expected output: no errors, no output (the module-level code is all inside `main()`).

Then commit:

```bash
cd /Users/faulknco/projects/spin-quant && git add experiments/per_row_bpw_sweep.py && git commit -m "Add Experiment 16: per-row BPW sweep script"
```

---

## Task 2: Run experiment

Run the experiment in the background (it will take ~20–40 minutes on CPU, 13 configs × ~3 min each):

```bash
cd /Users/faulknco/projects/spin-quant && .venv/bin/python experiments/per_row_bpw_sweep.py 2>&1 | tee /tmp/exp16_output.txt
```

Wait for completion. The script prints per-row progress every 500 rows and the full summary tables at the end. When done, read the output:

```bash
cat /tmp/exp16_output.txt
```

The final output will look like (example values — actual results will differ):

```
======================================================================
Experiment 16 Summary — Per-Row BPW Sweep (h0.c_fc)
======================================================================

  bd=8  (n_blocks_per_row=96)
  Baseline (FP32): 59.XXX

     K     bpw         PPL       delta  note      flat_ref
  ------------------------------------------------------------
     2   0.125      XXX.XXX   +XXX.XXX            —
     4   0.250      XXX.XXX   +XXX.XXX            —
     8   0.375      XXX.XXX   +XXX.XXX            —
    16   0.500      XXX.XXX   +XXX.XXX            154.340
    32   0.625      XXX.XXX   +XXX.XXX            16423.000
    64   0.750      XXX.XXX   +XXX.XXX            —
    96   0.896      XXX.XXX   +XXX.XXX            —

  bd=16  (n_blocks_per_row=48)
  Baseline (FP32): 59.XXX

     K     bpw         PPL       delta  note      flat_ref
  ------------------------------------------------------------
     2   0.063      XXX.XXX   +XXX.XXX            —
     4   0.125      XXX.XXX   +XXX.XXX            —
     8   0.250      XXX.XXX   +XXX.XXX            —
    16   0.250      XXX.XXX   +XXX.XXX            —
    32   0.313      XXX.XXX   +XXX.XXX            —
    48   0.363      XXX.XXX   +XXX.XXX            —
```

Record the actual output values for use in Task 3.

---

## Task 3: Record findings in FINDINGS_LOG.md

Open `/Users/faulknco/projects/spin-quant/FINDINGS_LOG.md`. Find the line `## Experiment 15` and insert the following block **immediately before it** (leaving one blank line between the new section and `## Experiment 15`):

```markdown
## Experiment 16 — Per-row BPW sweep (phase transition search)

**Script:** `experiments/per_row_bpw_sweep.py`
**Date:** 2026-03-11
**Target:** h0.c_fc (GPT-2 h[0].mlp.c_fc, shape [3072, 768])
**Eval:** 100 WikiText-2 test texts, MAX_LEN=128

### Setup

Sweep K for per-row k-means on two block_dim values:
- bd=8:  K ∈ {2, 4, 8, 16, 32, 64, 96}  — n_blocks_per_row=96
- bd=16: K ∈ {2, 4, 8, 16, 32, 48}      — n_blocks_per_row=48

bpw = log2(K) / block_dim. Prior work:
- Flat bd=8  K=16  (bpw=0.500): PPL=154 (Exp 8)
- Flat bd=8  K=32  (bpw=0.625): PPL=16,423 — chaos (Exp 8)
- Flat bd=16 K=256 (bpw=0.500): PPL=381 (Exp 6)

### Results

**Baseline (FP32):** PPL = [FILL FROM OUTPUT]

**bd=8 sweep:**

| K  | bpw   | PPL      | delta    | note   | flat_ref  |
|----|-------|----------|----------|--------|-----------|
| 2  | 0.125 | [FILL]   | [FILL]   |        | —         |
| 4  | 0.250 | [FILL]   | [FILL]   |        | —         |
| 8  | 0.375 | [FILL]   | [FILL]   |        | —         |
| 16 | 0.500 | [FILL]   | [FILL]   |        | 154.340   |
| 32 | 0.625 | [FILL]   | [FILL]   |        | 16423.0   |
| 64 | 0.750 | [FILL]   | [FILL]   |        | —         |
| 96 | 0.896 | [FILL]   | [FILL]   |        | —         |

**bd=16 sweep:**

| K  | bpw   | PPL      | delta    | note   | flat_ref  |
|----|-------|----------|----------|--------|-----------|
| 2  | 0.063 | [FILL]   | [FILL]   |        | —         |
| 4  | 0.125 | [FILL]   | [FILL]   |        | —         |
| 8  | 0.250 | [FILL]   | [FILL]   |        | —         |
| 16 | 0.250 | [FILL]   | [FILL]   |        | —         |
| 32 | 0.313 | [FILL]   | [FILL]   |        | —         |
| 48 | 0.363 | [FILL]   | [FILL]   |        | —         |

### Findings

[Fill in after experiment completes. Address the following:]

**Phase transition location:** [Where does per-row break down — first K with PPL > 10× baseline, or "none found in range"?]

**Transition character:** [Is the degradation sharp (first-order, like flat k-means) or gradual?]

**Comparison to flat k-means:**
- At matching bpw values (bd=8 K=16: bpw=0.5), per-row PPL vs flat 154.340?
- At bd=8 K=32 (bpw=0.625), flat collapsed to 16,423 — does per-row remain stable?
- What is the minimum bpw where per-row still gives near-lossless (delta < 10)?

**Physics interpretation:** [Fill in. Suggested framing: flat k-means transition is a capacity crisis — below K=256 the single codebook can't cover the full range of row norms. Per-row eliminates the norm competition, so the critical K for per-row is set by within-row block diversity alone. How many distinct block shapes does one row of h0.c_fc need to represent? The answer appears to be ≈ [FILL] (the K where per-row first degrades significantly).]

**Next experiment candidates:**
- If per-row stays stable all the way to K=2: the bottleneck is not codebook size but the shared-codebook norm competition. The correct next move is multi-layer per-row sweep (Exp 17 candidate).
- If per-row breaks at moderate K: there IS a within-row phase transition. Could try K-means++ init, more iterations, or hierarchical codebooks.

---
```

After inserting the section and filling in all `[FILL]` placeholders from the actual experiment output, commit:

```bash
cd /Users/faulknco/projects/spin-quant && git add FINDINGS_LOG.md && git commit -m "Record Experiment 16 findings: per-row BPW sweep"
```
