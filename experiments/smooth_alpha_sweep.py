"""
SmoothQuant alpha sweep: find optimal scale balance.

Experiment 10b. Experiment 10 showed that combined scaling
s_j = std(W[:,j])^α * sqrt(H_diag[j])^(1-α) at α=0.5 gives PPL=170 (vs 381 for flat).

This sweep tests α in {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}:
  α=0.0 → pure H-diag:  s_j = sqrt(H_diag[j])
  α=0.5 → balanced:     s_j = std_j^0.5 * H_j^0.25
  α=1.0 → pure W-std:   s_j = std(W[:,j])

Goal: find the α that minimises PPL and understand the sensitivity of the sweep.

Usage:
    python experiments/smooth_alpha_sweep.py
"""

import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.codebook import quantize_blocks
from src.hessian import estimate_h_diag
from experiments.eval_perplexity import conv1d_to_linear, eval_perplexity


MODEL_NAME = "gpt2"
K          = 256
BLOCK_DIM  = 16
N_CALIB    = 100
N_EVAL     = 150
MAX_LEN    = 128
DEVICE     = "cpu"

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


class _SmoothLinear(nn.Module):
    """Drop-in layer: applies scale to x, uses flat-block quantized W_smooth."""
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


def smooth_quantize(W, col_scale, block_dim, K, n_iter=50):
    """Scale W columns by 1/col_scale, run flat k-means, return quantized layer state."""
    W_smooth = W.float() / col_scale.unsqueeze(0)   # [out_f, in_f] / [1, in_f]
    centroids, labels, W_shape = quantize_blocks(W_smooth, block_dim, K, n_iter=n_iter)
    return centroids, labels, W_shape


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
    W  = target_linear.weight.data.clone()
    b  = target_linear.bias.data.clone() if target_linear.bias is not None else None

    bpw = (K.bit_length() - 1) / BLOCK_DIM
    print(f"Target: h0.c_fc  {tuple(W.shape)}  K={K}  bd={BLOCK_DIM}  bpw={bpw:.3f}")

    # Baseline
    print("\nBaseline (full precision) ...")
    ppl_base = eval_perplexity(model, tokenizer, eval_texts, MAX_LEN, DEVICE)
    print(f"  PPL = {ppl_base:.3f}")

    # Calibration
    print(f"\nCalibrating H_diag ({N_CALIB} texts) ...")
    h_diag  = estimate_h_diag(model, tokenizer, raw_layer, calib_texts, MAX_LEN, DEVICE)
    col_std = W.float().std(dim=0).clamp(min=1e-8)   # [in_f]
    h_sqrt  = h_diag.float().clamp(min=1e-8).sqrt()  # [in_f]
    print(f"  col_std  range: [{col_std.min():.4f}, {col_std.max():.4f}]  ratio={col_std.max()/col_std.min():.2f}×")
    print(f"  H_diag   range: [{h_diag.min():.4f}, {h_diag.max():.4f}]  ratio={h_sqrt.max()/h_sqrt.min():.2f}×")

    # Flat k-means reference
    print("\nFlat k-means (no scaling) ...")
    centroids_flat, labels_flat, W_shape_flat = quantize_blocks(W, BLOCK_DIM, K, n_iter=50)

    class _FlatLinear(nn.Module):
        def __init__(self, c, l, s, bias=None):
            super().__init__()
            self.register_buffer("centroids", c.float())
            self.register_buffer("labels", l)
            self._W_shape = s
            self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None
        def forward(self, x):
            return F.linear(x, self.centroids[self.labels].reshape(self._W_shape), self.bias)

    m_flat = copy.deepcopy(model)
    m_flat.transformer.h[0].mlp.c_fc = _FlatLinear(centroids_flat, labels_flat, W_shape_flat, b)
    ppl_flat = eval_perplexity(m_flat, tokenizer, eval_texts, MAX_LEN, DEVICE)
    del m_flat
    print(f"  PPL = {ppl_flat:.3f}  (delta = +{ppl_flat - ppl_base:.3f})")

    # Alpha sweep
    print(f"\n{'α':>5}  {'col_ratio':>10}  {'PPL':>10}  {'delta':>10}  {'vs flat':>10}")
    print("-" * 55)

    results = []
    for alpha in ALPHAS:
        # s_j = std_j^alpha * H_j^(0.5*(1-alpha))
        col_scale = (col_std ** alpha) * (h_sqrt ** (1 - alpha))
        col_scale = col_scale.clamp(min=1e-8)
        ratio = (col_scale.max() / col_scale.min()).item()

        centroids, labels, W_shape = smooth_quantize(W, col_scale, BLOCK_DIM, K)
        layer = _SmoothLinear(centroids, labels, W_shape, col_scale, b)

        m = copy.deepcopy(model)
        m.transformer.h[0].mlp.c_fc = layer
        ppl = eval_perplexity(m, tokenizer, eval_texts, MAX_LEN, DEVICE)
        del m

        vs_flat = (ppl_flat - ppl) / (ppl_flat - ppl_base) * 100
        note = " ← best" if ppl < min([r[1] for r in results], default=float('inf')) else ""
        print(f"  {alpha:>3.1f}  {ratio:>10.2f}×  {ppl:>10.3f}  {ppl-ppl_base:>+10.3f}  {vs_flat:>+9.1f}%{note}")
        results.append((alpha, ppl, ratio, vs_flat))

    best_alpha, best_ppl, best_ratio, best_vs_flat = min(results, key=lambda r: r[1])

    print(f"\n{'='*55}")
    print(f"Alpha sweep summary  (h0.c_fc, K={K}, bd={BLOCK_DIM}, bpw={bpw:.3f})")
    print(f"{'='*55}")
    print(f"  Baseline (FP32):    {ppl_base:>9.3f}")
    print(f"  Flat k-means:       {ppl_flat:>9.3f}  (delta={ppl_flat-ppl_base:>+8.3f})")
    print(f"  Best SmoothQuant:   {best_ppl:>9.3f}  (α={best_alpha:.1f}, ratio={best_ratio:.2f}×, "
          f"{best_vs_flat:+.1f}% of gap recovered)")


if __name__ == "__main__":
    main()
