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
