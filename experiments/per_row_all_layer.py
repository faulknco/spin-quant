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
