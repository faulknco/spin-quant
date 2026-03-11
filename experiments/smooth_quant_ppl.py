"""
Experiment 10: SmoothQuant-style column scaling as a fix for scale heterogeneity.

Background / motivation:
  All H-weighted quantization experiments failed because column scale heterogeneity
  (14× ratio) causes reconstruction errors. The key insight: instead of absorbing the
  per-column scale into the weight reconstruction path (which amplifies errors), migrate
  the scale to the ACTIVATION path.

  SmoothQuant formula:
      y = W @ x = W_smooth @ (x * s)
      where W_smooth[:, j] = W[:, j] / s_j

  W_smooth has uniform column scale → flat k-means on W_smooth works well (no
  cross-column contamination). At inference:
      F.linear(x * s, W_smooth_quant, bias)   [equivalent to F.linear(x, W, bias)]

  Critical difference from previous attempts: quantize W_smooth with FLAT ROW-MAJOR
  blocks (preserves within-row weight correlation), apply s to x in forward pass.

Tests 4 scaling strategies at bpw=0.5 (K=256, bd=16) on h0.c_fc:
  1. Baseline:  flat k-means on W (no scaling) — PPL reference
  2. W-std:     s_j = std(W[:, j])  — removes weight column magnitude heterogeneity
  3. H-diag:    s_j = sqrt(H_diag[j])  — removes activation sensitivity heterogeneity
  4. Combined:  s_j = std(W[:, j])^0.5 * sqrt(H_diag[j])^0.5  — balance both
"""

import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.codebook import quantize_blocks, reconstruct
from src.hessian import estimate_h_diag
from experiments.eval_perplexity import conv1d_to_linear, eval_perplexity


MODEL_NAME = "gpt2"
DEVICE     = "cpu"
MAX_LEN    = 128
K          = 256
BLOCK_DIM  = 16
N_CALIB    = 100
N_EVAL     = 150


# ---------------------------------------------------------------------------
# SmoothQuant-style inline layer
# ---------------------------------------------------------------------------

class _SmoothLinear(nn.Module):
    """
    Drop-in linear layer applying SmoothQuant-style activation scaling.

    Forward: F.linear(x * s, W_smooth_quant, bias)
    which is equivalent to F.linear(x, W, bias) in full precision.

    W_smooth_quant is stored as flat row-major codebook blocks so that the
    quantized weight has uniform column scale (no cross-column contamination).
    """
    def __init__(self, centroids, labels, W_shape, scale, bias=None):
        super().__init__()
        self.register_buffer("centroids", centroids.float())
        self.register_buffer("labels",    labels)
        self.register_buffer("scale",     scale.float())   # [in_f]
        self._W_shape = W_shape
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None

    def forward(self, x):
        W_smooth = self.centroids[self.labels].reshape(self._W_shape)
        return F.linear(x * self.scale, W_smooth, self.bias)


class _FlatLinear(nn.Module):
    """Flat k-means baseline layer (no scaling)."""
    def __init__(self, centroids, labels, W_shape, bias=None):
        super().__init__()
        self.register_buffer("centroids", centroids.float())
        self.register_buffer("labels",    labels)
        self._W_shape = W_shape
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None

    def forward(self, x):
        W = self.centroids[self.labels].reshape(self._W_shape)
        return F.linear(x, W, self.bias)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def smooth_quantize(W: torch.Tensor, scale: torch.Tensor, block_dim: int, K: int):
    """
    Compute W_smooth = W / s (column-wise), run flat k-means on W_smooth,
    return (centroids, labels, W_shape).
    """
    # W_smooth[:, j] = W[:, j] / s_j
    W_smooth = W.float() / scale.unsqueeze(0).clamp(min=1e-8)
    centroids, labels, W_shape = quantize_blocks(W_smooth, block_dim, K, n_iter=50)
    return centroids, labels, W_shape


def h_rmse(W: torch.Tensor, Wq: torch.Tensor, h_diag: torch.Tensor) -> float:
    """H-weighted RMSE: sqrt(mean((W - Wq)^2 * H_diag_j))."""
    err = (W.float() - Wq.float()) ** 2
    return (err * h_diag.unsqueeze(0)).mean().sqrt().item()


def col_scale_stats(scale: torch.Tensor) -> tuple[float, float, float]:
    """Return (min, max, max/min ratio) of a per-column scale vector."""
    s = scale.float()
    s_min = s.min().item()
    s_max = s.max().item()
    ratio = s_max / max(s_min, 1e-12)
    return s_min, s_max, ratio


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {MODEL_NAME} ...")
    model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading WikiText-2 ...")
    train_data  = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test_data   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    calib_texts = [t for t in train_data["text"] if len(t.strip()) > 50][:N_CALIB]
    eval_texts  = [t for t in test_data["text"]  if len(t.strip()) > 50][:N_EVAL]

    # Target layer: h0.c_fc
    raw_layer     = model.transformer.h[0].mlp.c_fc
    target_linear = conv1d_to_linear(raw_layer)
    W = target_linear.weight.data.clone()   # [out_f, in_f] = [3072, 768]
    b = target_linear.bias.data.clone() if target_linear.bias is not None else None

    out_f, in_f = W.shape
    bpw = (K.bit_length() - 1) / BLOCK_DIM
    print(f"Target: h0.c_fc  {tuple(W.shape)}  K={K}  block_dim={BLOCK_DIM}  bpw={bpw:.3f}")

    # --- H_diag calibration
    print(f"\nCalibrating H_diag ({N_CALIB} texts) ...")
    h_diag = estimate_h_diag(model, tokenizer, raw_layer, calib_texts, MAX_LEN, DEVICE)
    print(f"  H_diag range: [{h_diag.min():.4f}, {h_diag.max():.4f}]  "
          f"ratio={h_diag.max()/h_diag.min().clamp(min=1e-12):.1f}x  "
          f"CV={h_diag.std()/h_diag.mean():.3f}")

    # -----------------------------------------------------------------------
    # [1/5] Baseline: flat k-means on W (no scaling)
    # -----------------------------------------------------------------------
    print(f"\n[1/5] Baseline: flat k-means (no scaling)")
    centroids_base, labels_base, _ = quantize_blocks(W, BLOCK_DIM, K, n_iter=50)
    Wq_base = reconstruct(centroids_base, labels_base, W.shape)
    hr_base = h_rmse(W, Wq_base, h_diag)

    base_layer = _FlatLinear(centroids_base, labels_base, W.shape, b)
    model_base = copy.deepcopy(model)
    model_base.transformer.h[0].mlp.c_fc = base_layer
    ppl_base = eval_perplexity(model_base, tokenizer, eval_texts, MAX_LEN, DEVICE)
    del model_base

    # col_scale ratio = 1.00x for baseline (no scale applied)
    print(f"  PPL = {ppl_base:.3f}  H-RMSE = {hr_base:.6f}  col_scale ratio = 1.00x")

    # -----------------------------------------------------------------------
    # [2/5] W-std scaling: s_j = std(W[:, j])
    # -----------------------------------------------------------------------
    print(f"\n[2/5] W-std scaling: s_j = std(W[:,j])")
    scale_wstd = W.float().std(dim=0).clamp(min=1e-8)   # [in_f]
    s_min, s_max, s_ratio = col_scale_stats(scale_wstd)
    print(f"  col_std ratio: max/min = {s_ratio:.2f}x")

    centroids_ws, labels_ws, _ = smooth_quantize(W, scale_wstd, BLOCK_DIM, K)
    # Reconstruct W_smooth_quant, then multiply columns back by s_j for H-RMSE
    W_smooth_ws = reconstruct(centroids_ws, labels_ws, W.shape)
    Wq_ws = W_smooth_ws.float() * scale_wstd.unsqueeze(0)
    hr_ws = h_rmse(W, Wq_ws, h_diag)

    ws_layer = _SmoothLinear(centroids_ws, labels_ws, W.shape, scale_wstd, b)
    model_ws = copy.deepcopy(model)
    model_ws.transformer.h[0].mlp.c_fc = ws_layer
    ppl_ws = eval_perplexity(model_ws, tokenizer, eval_texts, MAX_LEN, DEVICE)
    del model_ws

    gain_ws = (ppl_ws - ppl_base) / ppl_base * 100
    print(f"  PPL = {ppl_ws:.3f}  H-RMSE = {hr_ws:.6f}  "
          f"(vs baseline: {gain_ws:+.1f}% PPL gain)")

    # -----------------------------------------------------------------------
    # [3/5] H-diag scaling: s_j = sqrt(H_diag[j])
    # -----------------------------------------------------------------------
    print(f"\n[3/5] H-diag scaling: s_j = sqrt(H_diag[j])")
    scale_hd = h_diag.float().clamp(min=1e-8).sqrt()    # [in_f]
    s_min, s_max, s_ratio = col_scale_stats(scale_hd)
    print(f"  col_hdiag ratio: max/min = {s_ratio:.2f}x")

    centroids_hd, labels_hd, _ = smooth_quantize(W, scale_hd, BLOCK_DIM, K)
    W_smooth_hd = reconstruct(centroids_hd, labels_hd, W.shape)
    Wq_hd = W_smooth_hd.float() * scale_hd.unsqueeze(0)
    hr_hd = h_rmse(W, Wq_hd, h_diag)

    hd_layer = _SmoothLinear(centroids_hd, labels_hd, W.shape, scale_hd, b)
    model_hd = copy.deepcopy(model)
    model_hd.transformer.h[0].mlp.c_fc = hd_layer
    ppl_hd = eval_perplexity(model_hd, tokenizer, eval_texts, MAX_LEN, DEVICE)
    del model_hd

    gain_hd = (ppl_hd - ppl_base) / ppl_base * 100
    print(f"  PPL = {ppl_hd:.3f}  H-RMSE = {hr_hd:.6f}  "
          f"(vs baseline: {gain_hd:+.1f}% PPL gain)")

    # -----------------------------------------------------------------------
    # [4/5] Combined: s_j = std(W[:,j])^0.5 * sqrt(H_diag[j])^0.5
    # -----------------------------------------------------------------------
    print(f"\n[4/5] Combined scaling: s_j = std(W[:,j])^0.5 * sqrt(H_diag[j])^0.5")
    scale_comb = (scale_wstd.clamp(min=1e-8) ** 0.5) * (scale_hd.clamp(min=1e-8) ** 0.5)
    s_min, s_max, s_ratio = col_scale_stats(scale_comb)
    print(f"  col_combined ratio: max/min = {s_ratio:.2f}x")

    centroids_cb, labels_cb, _ = smooth_quantize(W, scale_comb, BLOCK_DIM, K)
    W_smooth_cb = reconstruct(centroids_cb, labels_cb, W.shape)
    Wq_cb = W_smooth_cb.float() * scale_comb.unsqueeze(0)
    hr_cb = h_rmse(W, Wq_cb, h_diag)

    cb_layer = _SmoothLinear(centroids_cb, labels_cb, W.shape, scale_comb, b)
    model_cb = copy.deepcopy(model)
    model_cb.transformer.h[0].mlp.c_fc = cb_layer
    ppl_cb = eval_perplexity(model_cb, tokenizer, eval_texts, MAX_LEN, DEVICE)
    del model_cb

    gain_cb = (ppl_cb - ppl_base) / ppl_base * 100
    print(f"  PPL = {ppl_cb:.3f}  H-RMSE = {hr_cb:.6f}  "
          f"(vs baseline: {gain_cb:+.1f}% PPL gain)")

    # -----------------------------------------------------------------------
    # [5/5] Full-precision reference
    # -----------------------------------------------------------------------
    print(f"\n[5/5] Full-precision reference")
    ppl_fp = eval_perplexity(model, tokenizer, eval_texts, MAX_LEN, DEVICE)
    print(f"  PPL = {ppl_fp:.3f}  (full precision)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Experiment 10 Summary  (h0.c_fc, K={K}, block_dim={BLOCK_DIM}, bpw={bpw:.3f})")
    print(f"{'='*70}")
    print(f"  {'Variant':<30}  {'PPL':>9}  {'H-RMSE':>10}  {'vs baseline':>12}")
    print(f"  {'-'*30}  {'-'*9}  {'-'*10}  {'-'*12}")
    print(f"  {'Full precision':<30}  {ppl_fp:>9.3f}")
    print(f"  {'Flat k-means (baseline)':<30}  {ppl_base:>9.3f}  {hr_base:>10.6f}  {'ref':>12}")

    for name, ppl, hr in [
        ("W-std scaling",    ppl_ws, hr_ws),
        ("H-diag scaling",   ppl_hd, hr_hd),
        ("Combined scaling", ppl_cb, hr_cb),
    ]:
        delta = ppl - ppl_base
        gain_pct = -delta / ppl_base * 100
        print(f"  {name:<30}  {ppl:>9.3f}  {hr:>10.6f}  {gain_pct:>+11.1f}%")

    print(f"\n  H-RMSE shows metric-PPL relationship:")
    print(f"    baseline={hr_base:.6f}  W-std={hr_ws:.6f}  H-diag={hr_hd:.6f}  combined={hr_cb:.6f}")


if __name__ == "__main__":
    main()
