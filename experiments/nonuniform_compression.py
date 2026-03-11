"""
Data-driven non-uniform compression.

Experiment 19A. Exp 18 measured per-layer ΔPPL for per-row K=64 across all
24 MLP sublayers. The distribution is highly uneven (h6.c_proj +79.4 vs
h0.c_fc +0.14 — 550× range). This experiment reallocates bits budget-neutrally:
give the top-N most-sensitive layers more K, give the rest less K.

ΔPPL rankings (Exp 18, per-row bd=8 K=64, sorted descending):
  h6.c_proj +79.4, h5.c_proj +38.2, h7.c_proj +33.6, h6.c_fc +33.2,
  h10.c_proj +22.3, h4.c_proj +21.2, h8.c_proj +20.2, h11.c_fc +19.8,
  h7.c_fc +16.3, h5.c_fc +15.4, h11.c_proj +15.2, h3.c_proj +12.5, ...

Config B (top-8, K_high=256, K_low=32) is exactly budget-neutral at 0.75 bpw:
  avg_bpw = (8*log2(256) + 16*log2(32)) / (24*8) = (64+80)/192 = 0.75.

Baseline: uniform K=64 all-layer (Exp 17C) PPL = 421.9.

Usage:
    .venv/bin/python experiments/nonuniform_compression.py
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
N_EVAL     = 50
MAX_LEN    = 128
DEVICE     = "cpu"

# Exp 18 per-layer ΔPPL for per-row bd=8 K=64 (delta from previous step)
DELTA_PPL_EXP18 = {
    "h0.c_fc":    0.144,
    "h0.c_proj":  3.875,
    "h1.c_fc":    3.146,
    "h1.c_proj":  0.799,
    "h2.c_fc":    4.574,
    "h2.c_proj":  8.619,
    "h3.c_fc":    6.509,
    "h3.c_proj":  12.515,
    "h4.c_fc":    11.420,
    "h4.c_proj":  21.201,
    "h5.c_fc":    15.420,
    "h5.c_proj":  38.180,
    "h6.c_fc":    33.156,
    "h6.c_proj":  79.425,
    "h7.c_fc":    16.348,
    "h7.c_proj":  33.582,
    "h8.c_fc":    -6.740,  # negative: adding this layer actually reduced PPL
    "h8.c_proj":  20.214,
    "h9.c_fc":    3.363,
    "h9.c_proj":  6.803,
    "h10.c_fc":   6.667,
    "h10.c_proj": 22.258,
    "h11.c_fc":   19.765,
    "h11.c_proj": 15.218,
}

# Sorted by ΔPPL descending (most sensitive first)
RANKED = sorted(DELTA_PPL_EXP18.items(), key=lambda x: x[1], reverse=True)

# Non-uniform configs: (label, N_sensitive, K_high, K_low)
# N=None means uniform (K_high applied to all 24 layers)
CONFIGS = [
    ("Uniform K=64",       None,  64,  64),   # Exp 17C baseline
    ("Uniform K=128",      None, 128, 128),   # upper bound
    ("top-4  K256/K32",       4, 256,  32),
    ("top-8  K256/K32 (=0.75bpw)",  8, 256,  32),  # budget-neutral
    ("top-8  K128/K32",       8, 128,  32),
    ("top-12 K128/K32",      12, 128,  32),
    ("top-4  K128/K64",       4, 128,  64),
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
    """Per-row k-means: one codebook per output row.

    K is capped at n_blocks_per_row (= in_features // block_dim) since k-means
    requires K <= N. For c_fc layers (in_features=768, bd=8): max K=96.
    For c_proj layers (in_features=3072, bd=8): max K=384.
    """
    W = W.float().cpu()
    out_f, in_f = W.shape
    n_blocks = in_f // block_dim
    K_eff = min(K, n_blocks)  # cap at n_blocks_per_row
    W_q = torch.zeros_like(W)
    for i in range(out_f):
        row_blocks = W[i].reshape(n_blocks, block_dim)
        centroids_i, labels_i = kmeans(row_blocks, K_eff, n_iter=50, seed=42)
        W_q[i] = centroids_i[labels_i].reshape(in_f)
    return _ReconLinear(W_q, b)


def make_k_map(N_sensitive, K_high, K_low):
    """
    Build a dict mapping each of the 24 layer names to its assigned K.

    Top-N layers by Exp 18 ΔPPL get K_high; the remaining 24-N get K_low.
    If N_sensitive is None, all layers get K_high (uniform).
    """
    if N_sensitive is None:
        return {name: K_high for name, _ in RANKED}
    sensitive = set(name for name, _ in RANKED[:N_sensitive])
    return {name: (K_high if name in sensitive else K_low) for name, _ in RANKED}


def build_nonuniform_model(base_model, block_dim, k_map):
    """
    Deep-copy base_model and quantize all 24 MLP sublayers with per-row k-means,
    using the layer-specific K values from k_map.

    Args:
        base_model: FP32 GPT-2 (never modified)
        block_dim:  block size (8 for all configs here)
        k_map:      dict mapping "h{bi}.{attr}" -> K

    Returns:
        Quantized deep-copy of base_model.
    """
    model = copy.deepcopy(base_model)
    for bi in range(12):
        for attr in ["c_fc", "c_proj"]:
            name = f"h{bi}.{attr}"
            K = k_map[name]
            raw = getattr(model.transformer.h[bi].mlp, attr)
            lin = conv1d_to_linear(raw)
            W = lin.weight.data.clone()
            b = lin.bias.data.clone() if lin.bias is not None else None
            layer = quantize_per_row(W, b, block_dim, K)
            setattr(model.transformer.h[bi].mlp, attr, layer)
    return model


def avg_bpw(k_map, block_dim):
    """Compute average bits-per-weight across all 24 layers."""
    import math
    total_bits_per_block = sum(math.log2(K) for K in k_map.values())
    return total_bits_per_block / (len(k_map) * block_dim)


def main():
    print(f"Loading {MODEL_NAME} ...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading WikiText-2 ...")
    test_data  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_texts = [t for t in test_data["text"] if len(t.strip()) > 50][:N_EVAL]

    ppl_base = eval_perplexity(base_model, tokenizer, eval_texts, MAX_LEN, DEVICE)
    print(f"FP32 baseline PPL = {ppl_base:.3f}")

    BLOCK_DIM = 8  # bd=8 for all configs

    print(f"\n{'='*75}")
    print("Experiment 19A — Data-Driven Non-Uniform Compression")
    print(f"{'='*75}")
    print(f"{'Config':<30}  {'N':>3}  {'K_h':>4}  {'K_l':>4}  {'bpw':>5}  {'PPL':>9}  {'vs K=64':>9}")
    print(f"{'-'*75}")

    ppl_uniform_k64 = None
    results = []

    for (label, N_sensitive, K_high, K_low) in CONFIGS:
        k_map = make_k_map(N_sensitive, K_high, K_low)
        bpw   = avg_bpw(k_map, BLOCK_DIM)

        print(f"Running: {label} ...", flush=True)
        q_model = build_nonuniform_model(base_model, BLOCK_DIM, k_map)
        ppl     = eval_perplexity(q_model, tokenizer, eval_texts, MAX_LEN, DEVICE)
        del q_model

        if label == "Uniform K=64":
            ppl_uniform_k64 = ppl

        vs_k64 = f"{ppl / ppl_uniform_k64:.3f}×" if ppl_uniform_k64 is not None else "—"
        N_str  = str(N_sensitive) if N_sensitive is not None else "all"
        print(
            f"{label:<30}  {N_str:>3}  {K_high:>4}  {K_low:>4}  {bpw:>5.3f}  "
            f"{ppl:>9.3f}  {vs_k64:>9}"
        )
        results.append((label, N_sensitive, K_high, K_low, bpw, ppl))

    print(f"\n{'='*75}")
    print("Summary")
    print(f"{'='*75}")
    print(f"  FP32 baseline:    {ppl_base:.3f}")
    print(f"  Uniform K=64:     {ppl_uniform_k64:.3f}")
    print()
    print("  Top-N sensitive layers (Exp 18 ΔPPL ranking):")
    for rank, (name, delta) in enumerate(RANKED[:12], 1):
        print(f"    {rank:>2}. {name:<12}  ΔPPL={delta:>+7.3f}")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
