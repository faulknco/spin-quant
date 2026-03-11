"""
SmoothQuant + optimal block_dim combination.

Experiment 11. Combines the two best single-layer improvements found so far:
  - Experiment 8:  bd=8, K=16 at bpw=0.5 gives PPL=165 (vs 360 for bd=16)
  - Experiment 10: SmoothQuant α=0.5 gives PPL=170 (vs 381 for bd=16, no scaling)

Tests whether combining both improvements gives further PPL reduction:
  (A) Flat, bd=16, K=256:   PPL=381 (baseline from Exp 6)
  (B) Flat, bd=8,  K=16:    PPL=165 (best from Exp 8)
  (C) Smooth α=0.5, bd=16, K=256: PPL=170 (best from Exp 10)
  (D) Smooth α=0.5, bd=8,  K=16:  PPL=?   ← new combination

All at bpw=0.5. Target: h0.c_fc.

Usage:
    python experiments/smooth_combined.py
"""

import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.codebook import quantize_blocks, kmeans
from src.hessian import estimate_h_diag
from experiments.eval_perplexity import conv1d_to_linear, eval_perplexity


MODEL_NAME = "gpt2"
N_CALIB    = 100
N_EVAL     = 150
MAX_LEN    = 128
DEVICE     = "cpu"
ALPHA      = 0.5    # confirmed optimal from Exp 10b


class _FlatLinear(nn.Module):
    def __init__(self, centroids, labels, W_shape, bias=None):
        super().__init__()
        self.register_buffer("centroids", centroids.float())
        self.register_buffer("labels",    labels)
        self._W_shape = W_shape
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None
    def forward(self, x):
        return F.linear(x, self.centroids[self.labels].reshape(self._W_shape), self.bias)


class _SmoothLinear(nn.Module):
    def __init__(self, centroids, labels, W_shape, scale, bias=None):
        super().__init__()
        self.register_buffer("centroids", centroids.float())
        self.register_buffer("labels",    labels)
        self.register_buffer("scale",     scale.float())
        self._W_shape = W_shape
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None
    def forward(self, x):
        W_smooth = self.centroids[self.labels].reshape(self._W_shape)
        return F.linear(x * self.scale, W_smooth, self.bias)


def eval_config(model, tokenizer, eval_texts, raw_layer, W, b, h_diag,
                block_dim, K, smooth=False, alpha=0.5, label=""):
    bpw = (K.bit_length() - 1) / block_dim
    print(f"\n[{label}] bd={block_dim}, K={K}, bpw={bpw:.3f}, smooth={smooth}")

    if smooth:
        col_std = W.float().std(dim=0).clamp(min=1e-8)
        h_sqrt  = h_diag.float().clamp(min=1e-8).sqrt()
        col_scale = (col_std ** alpha) * (h_sqrt ** (1 - alpha))
        col_scale = col_scale.clamp(min=1e-8)
        ratio = col_scale.max() / col_scale.min()
        print(f"  col_scale ratio: {ratio:.2f}×  (α={alpha})")
        W_smooth = W.float() / col_scale.unsqueeze(0)
        centroids, labels, W_shape = quantize_blocks(W_smooth, block_dim, K, n_iter=50)
        layer = _SmoothLinear(centroids, labels, W_shape, col_scale, b)
    else:
        centroids, labels, W_shape = quantize_blocks(W, block_dim, K, n_iter=50)
        layer = _FlatLinear(centroids, labels, W_shape, b)

    m = copy.deepcopy(model)
    m.transformer.h[0].mlp.c_fc = layer
    ppl = eval_perplexity(m, tokenizer, eval_texts, MAX_LEN, DEVICE)
    del m
    return ppl


def main():
    print(f"Loading {MODEL_NAME} ...")
    model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading WikiText-2 ...")
    train_data  = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test_data   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    calib_texts = [t for t in train_data["text"] if len(t.strip()) > 50][:N_CALIB]
    eval_texts  = [t for t in test_data["text"]  if len(t.strip()) > 50][:N_EVAL]

    raw_layer     = model.transformer.h[0].mlp.c_fc
    target_linear = conv1d_to_linear(raw_layer)
    W = target_linear.weight.data.clone()
    b = target_linear.bias.data.clone() if target_linear.bias is not None else None

    print(f"Target: h0.c_fc  {tuple(W.shape)}")

    # Baseline
    print("\nBaseline (full precision) ...")
    ppl_base = eval_perplexity(model, tokenizer, eval_texts, MAX_LEN, DEVICE)
    print(f"  PPL = {ppl_base:.3f}")

    # Calibration
    print(f"\nCalibrating H_diag ({N_CALIB} texts) ...")
    h_diag = estimate_h_diag(model, tokenizer, raw_layer, calib_texts, MAX_LEN, DEVICE)
    print(f"  H_diag range: [{h_diag.min():.4f}, {h_diag.max():.4f}]")

    configs = [
        # (label, block_dim, K, smooth)
        ("A", 16, 256, False),   # Flat, bd=16 — baseline
        ("B",  8,  16, False),   # Flat, bd=8  — best from Exp 8
        ("C", 16, 256, True),    # Smooth α=0.5, bd=16 — best from Exp 10
        ("D",  8,  16, True),    # Smooth α=0.5, bd=8  — new combination
    ]

    results = {}
    for (label, bd, K, smooth) in configs:
        ppl = eval_config(model, tokenizer, eval_texts, raw_layer, W, b, h_diag,
                          bd, K, smooth=smooth, alpha=ALPHA, label=label)
        delta = ppl - ppl_base
        bpw = (K.bit_length() - 1) / bd
        print(f"  PPL = {ppl:.3f}  (delta = +{delta:.3f})")
        results[label] = (bd, K, bpw, smooth, ppl, delta)

    # Summary
    print(f"\n{'='*65}")
    print(f"Experiment 11 Summary  (h0.c_fc, bpw=0.500)")
    print(f"{'='*65}")
    print(f"  Baseline (FP32):   {ppl_base:>9.3f}")
    print(f"\n  {'Config':<30}  {'bd':>3}  {'K':>5}  {'PPL':>9}  {'delta':>9}  {'vs A':>9}")
    ppl_A = results['A'][4]
    for label, (bd, K, bpw, smooth, ppl, delta) in results.items():
        mode = "smooth α=0.5" if smooth else "flat     "
        vs_A = ppl_A - ppl
        vs_A_pct = vs_A / (ppl_A - ppl_base) * 100 if ppl_A > ppl_base else 0
        tag = " ← best" if ppl == min(r[4] for r in results.values()) else ""
        print(f"  ({label}) {mode}  bd={bd:>2}  K={K:>5}  {ppl:>9.3f}  {delta:>+9.3f}  {vs_A_pct:>+8.1f}%{tag}")

    best_label = min(results, key=lambda k: results[k][4])
    best_ppl = results[best_label][4]
    gap_recovered = (ppl_A - best_ppl) / (ppl_A - ppl_base) * 100
    print(f"\n  Best config: ({best_label}) — PPL={best_ppl:.3f}")
    print(f"  Gap recovered vs flat bd=16: {gap_recovered:.1f}%")
    print(f"  Remaining gap to FP32: {best_ppl - ppl_base:.3f} ({(best_ppl/ppl_base - 1)*100:.1f}× above baseline)")


if __name__ == "__main__":
    main()
