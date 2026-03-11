"""
Per-row codebook quantization.

Experiment 14. All prior k-means experiments (Exp 6-13) used a SINGLE shared
codebook for all blocks in the weight matrix. The root problem: hot output rows
(large weights) and cold output rows (small weights) compete for the same K
centroids. The codebook covers the full range, giving poor resolution to either.

Fix: per-row codebooks. Each of the out_features output rows gets its own K
centroids, run on just the blocks of that row. Centroids are tailored to each
row's weight distribution. No cross-row contamination.

This is the k-means analog of "per-channel quantization" in INT quant.

Hypothesis: per-row codebooks should outperform flat k-means at the same bpw,
especially for small K (where the shared codebook is most starved).

All configs at bpw=0.5 (K=256 bd=16 or K=16 bd=8). Also test K=64 bd=16
(bpw=0.375) to see if per-row enables more aggressive compression.

Target: h0.c_fc. Compare to:
  flat bd=16 K=256: PPL=381 (Exp 6)
  flat bd=8  K=16:  PPL=154 (Exp 8, current best)

Usage:
    .venv/bin/python experiments/per_row_codebook.py
"""

import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.codebook import kmeans, quantize_blocks
from experiments.eval_perplexity import conv1d_to_linear, eval_perplexity


MODEL_NAME = "gpt2"
N_CALIB    = 0      # no calibration needed — no H_diag
N_EVAL     = 150
MAX_LEN    = 128
DEVICE     = "cpu"


class _ReconLinear(nn.Module):
    """Linear layer with precomputed reconstructed weight matrix."""
    def __init__(self, W_q, bias=None):
        super().__init__()
        self.register_buffer("W_q", W_q.float())
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None

    def forward(self, x):
        return F.linear(x, self.W_q, self.bias)


def quantize_flat(W, b, block_dim, K):
    """
    Standard flat k-means quantization: one shared codebook for all blocks.

    Args:
        W:         [out_features, in_features] weight tensor
        b:         bias tensor or None
        block_dim: block size d (must divide in_features)
        K:         codebook size

    Returns:
        _ReconLinear with precomputed W_q
    """
    centroids, labels, W_shape = quantize_blocks(W, block_dim, K, n_iter=50)
    W_q = centroids[labels].reshape(W_shape)
    return _ReconLinear(W_q, b)


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

    # Configs: (label, description, mode, block_dim, K)
    # Note: per-row k-means requires K <= n_blocks_per_row = in_features / block_dim
    #   bd=16: n_blocks = 768/16 = 48  → K must be ≤ 48
    #   bd=8:  n_blocks = 768/8  = 96  → K must be ≤ 96
    # bpw = log2(K) / block_dim
    configs = [
        ("A", "flat    bd=16 K=256 bpw=0.500", "flat",     16, 256),  # Exp 6 reference
        ("B", "flat    bd=8  K=16  bpw=0.500", "flat",      8,  16),  # Exp 8 best
        ("C", "per-row bd=8  K=16  bpw=0.500", "per-row",   8,  16),  # per-row at same bpw as B
        ("D", "per-row bd=8  K=64  bpw=0.750", "per-row",   8,  64),  # per-row at higher bpw
        ("E", "per-row bd=16 K=32  bpw=0.313", "per-row",  16,  32),  # per-row bd=16, K≤48
        ("F", "per-row bd=16 K=16  bpw=0.250", "per-row",  16,  16),  # per-row bd=16, lower bpw
    ]

    results = {}
    for (label, desc, mode, bd, K) in configs:
        bpw = (K.bit_length() - 1) / bd
        print(f"\n[{label}] {desc}  mode={mode}  bd={bd}  K={K}  bpw={bpw:.3f}")

        if mode == "flat":
            layer = quantize_flat(W, b, bd, K)
        else:
            print(f"  Running per-row k-means over {out_f} rows ...")
            layer = quantize_per_row(W, b, bd, K)

        m = copy.deepcopy(model)
        m.transformer.h[0].mlp.c_fc = layer
        ppl = eval_perplexity(m, tokenizer, eval_texts, MAX_LEN, DEVICE)
        del m

        delta = ppl - ppl_base
        print(f"  PPL = {ppl:.3f}  (delta = +{delta:.3f})")
        results[label] = (desc, mode, bd, K, bpw, ppl, delta)

    # Summary table
    ppl_A = results["A"][5]
    print(f"\n{'='*80}")
    print(f"Experiment 14 Summary  (h0.c_fc)  — per-row codebooks")
    print(f"{'='*80}")
    print(f"  Baseline (FP32):  {ppl_base:>9.3f}")
    print()
    header = f"  {'Config':<28}  {'mode':<9}  {'bd':>3}  {'K':>5}  {'bpw':>5}  {'PPL':>9}  {'delta':>9}  {'vs flat-A':>9}"
    print(header)
    print(f"  {'-'*len(header.strip())}")

    for label, (desc, mode, bd, K, bpw, ppl, delta) in results.items():
        vs_A = ppl - ppl_A
        if label == "A":
            vs_A_str = "     ref"
        else:
            vs_A_str = f"{vs_A:>+9.3f}"
        best_tag = " *" if ppl == min(r[5] for r in results.values()) else "  "
        print(
            f"  ({label}) {desc:<26}  {mode:<9}  {bd:>3}  {K:>5}  {bpw:>5.3f}  "
            f"{ppl:>9.3f}  {delta:>+9.3f}  {vs_A_str}{best_tag}"
        )

    best_label = min(results, key=lambda k: results[k][5])
    best_ppl   = results[best_label][5]
    print()
    print(f"  Best config: ({best_label}) — PPL={best_ppl:.3f}")
    if ppl_A > ppl_base:
        gap_recovered = (ppl_A - best_ppl) / (ppl_A - ppl_base) * 100
        print(f"  Gap recovered vs flat bd=16 K=256: {gap_recovered:.1f}%")
    print(f"  Remaining gap to FP32: {best_ppl - ppl_base:.3f}")


if __name__ == "__main__":
    main()
