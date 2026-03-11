"""
Hessian-weighted codebook: perplexity evaluation.

Lightweight single-layer experiment. Swaps only h0.c_fc to avoid memory pressure.
Compares three conditions at K=256 (bpw=0.5):
  1. Baseline    — full precision
  2. Flat        — standard k-means codebook (CodebookLinear)
  3. H-weighted  — Hessian-weighted k-means (HessianLinear)

Usage:
    python experiments/hessian_ppl.py
"""

import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.codebook import quantize_blocks, reconstruct
from src.hessian import estimate_h_diag, hessian_quantize, hessian_reconstruct
from experiments.eval_perplexity import conv1d_to_linear, eval_perplexity


MODEL_NAME  = "gpt2"
K           = 256
BLOCK_DIM   = 16
N_CALIB     = 100
N_EVAL      = 150
MAX_LEN     = 128
DEVICE      = "cpu"

# Clip percentiles to sweep — controls max scale amplification
# 100 = no clipping (original, catastrophic), 50 = very gentle
CLIP_PERCENTILES = [50, 75, 90, 95, 99, 100]


# ---------------------------------------------------------------------------
# Minimal drop-in layers (self-contained, no import chain issues)
# ---------------------------------------------------------------------------

class _FlatLinear(nn.Module):
    """Flat k-means codebook linear layer."""
    def __init__(self, centroids, labels, W_shape, bias=None):
        super().__init__()
        self.register_buffer("centroids", centroids.float())
        self.register_buffer("labels", labels)
        self._W_shape = W_shape
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None

    def forward(self, x):
        W = self.centroids[self.labels].reshape(self._W_shape)
        return F.linear(x, W, self.bias)


class _HessianLinear(nn.Module):
    """H-weighted k-means codebook linear layer."""
    def __init__(self, centroids_scaled, labels, scale, W_shape, bias=None):
        super().__init__()
        self.register_buffer("centroids_scaled", centroids_scaled.float())
        self.register_buffer("labels", labels)
        self.register_buffer("scale", scale.float())
        self._W_shape = W_shape
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None

    def forward(self, x):
        out_f, in_f = self._W_shape
        W_scaled = self.centroids_scaled[self.labels].reshape(out_f, in_f)
        W = W_scaled / self.scale.clamp(min=1e-8).unsqueeze(0)
        return F.linear(x, W, self.bias)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {MODEL_NAME} ...")
    model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Data
    print("Loading WikiText-2 ...")
    train_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test_data  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    calib_texts = [t for t in train_data["text"] if len(t.strip()) > 50][:N_CALIB]
    eval_texts  = [t for t in test_data["text"]  if len(t.strip()) > 50][:N_EVAL]

    # Target layer
    raw_layer    = model.transformer.h[0].mlp.c_fc
    target_linear = conv1d_to_linear(raw_layer)
    W  = target_linear.weight.data.clone()
    b  = target_linear.bias.data.clone() if target_linear.bias is not None else None

    print(f"Target: h0.c_fc  {tuple(W.shape)}  K={K}  bpw={K.bit_length()-1}/{BLOCK_DIM:.3f}")

    # --- Baseline perplexity
    print("\n[1/3] Baseline (full precision) ...")
    ppl_base = eval_perplexity(model, tokenizer, eval_texts, MAX_LEN, DEVICE)
    print(f"  PPL = {ppl_base:.3f}")

    # --- Calibration: H_diag
    print(f"\nCalibrating H_diag ({N_CALIB} texts) ...")
    h_diag = estimate_h_diag(model, tokenizer, raw_layer, calib_texts, MAX_LEN, DEVICE)
    print(f"  H_diag range: [{h_diag.min():.4f}, {h_diag.max():.4f}]  CV={h_diag.std()/h_diag.mean():.3f}")

    # --- Flat k-means
    print(f"\n[2/3] Flat k-means (K={K}) ...")
    centroids_flat, labels_flat, _ = quantize_blocks(W, BLOCK_DIM, K, n_iter=50)
    flat_layer = _FlatLinear(centroids_flat, labels_flat, W.shape, b)

    model_flat = copy.deepcopy(model)
    model_flat.transformer.h[0].mlp.c_fc = flat_layer
    ppl_flat = eval_perplexity(model_flat, tokenizer, eval_texts, MAX_LEN, DEVICE)
    del model_flat
    print(f"  PPL = {ppl_flat:.3f}  (delta = +{ppl_flat - ppl_base:.3f})")

    # --- H-weighted k-means: sweep clip percentiles
    print(f"\n[3/3] H-weighted k-means sweep over clip percentiles ...")
    print(f"      (clip_pct=100 → no clipping; lower → gentler scaling)")

    hq_results = []
    for clip_pct in CLIP_PERCENTILES:
        state_hq = hessian_quantize(W, h_diag, BLOCK_DIM, K, n_iter=50,
                                    scale_clip_percentile=clip_pct)
        # effective max/min scale ratio after clipping
        scale = state_hq["scale"]
        scale_ratio = (scale.max() / scale.clamp(min=1e-8).min()).item()

        hq_layer = _HessianLinear(
            state_hq["centroids_scaled"],
            state_hq["labels"],
            state_hq["scale"],
            W.shape, b,
        )
        model_hq = copy.deepcopy(model)
        model_hq.transformer.h[0].mlp.c_fc = hq_layer
        ppl_hq = eval_perplexity(model_hq, tokenizer, eval_texts, MAX_LEN, DEVICE)
        del model_hq

        hq_results.append((clip_pct, scale_ratio, ppl_hq))
        print(f"  clip={clip_pct:>3}%  scale_ratio={scale_ratio:>5.1f}×  PPL={ppl_hq:>9.3f}  "
              f"(delta={ppl_hq-ppl_base:>+9.3f})")

    # --- Summary
    best_clip, best_ratio, best_ppl = min(hq_results, key=lambda x: x[2])
    print(f"\n{'='*65}")
    print(f"PPL Summary  (single layer: h0.c_fc, K={K}, bpw=0.5)")
    print(f"{'='*65}")
    print(f"  Baseline (FP32):             {ppl_base:>9.3f}")
    print(f"  Flat k-means:                {ppl_flat:>9.3f}  (delta={ppl_flat-ppl_base:>+8.3f})")
    for clip_pct, ratio, ppl_hq in hq_results:
        marker = " ← best" if clip_pct == best_clip else ""
        print(f"  H-weighted clip={clip_pct:>3}%:        {ppl_hq:>9.3f}  (delta={ppl_hq-ppl_base:>+8.3f})  "
              f"ratio={ratio:.1f}×{marker}")

    ppl_gain = ppl_flat - best_ppl
    ppl_gain_pct = ppl_gain / (ppl_flat - ppl_base) * 100 if ppl_flat > ppl_base else 0
    print(f"\n  Best H-weighted (clip={best_clip}%) vs flat: {ppl_gain:+.3f}  "
          f"({ppl_gain_pct:.1f}% of quantization gap recovered)")


if __name__ == "__main__":
    main()
