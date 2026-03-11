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
from src.hessian import (estimate_h_diag, hessian_quantize, hessian_reconstruct,
                         hessian_quantize_sorted, hessian_reconstruct_sorted)
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

    # --- H-weighted sorted: sort dims by H_diag before blocking
    print(f"\n[3/3] H-weighted sorted k-means (K={K}) ...")
    print(f"      (dims sorted by H_diag so each block has uniform sensitivity)")
    state_sorted = hessian_quantize_sorted(W, h_diag, BLOCK_DIM, K, n_iter=50)

    # Measure within-block scale ratio to confirm uniformity
    sort_idx = state_sorted["sort_idx"]
    h_sorted = h_diag[sort_idx]
    n_blocks_per_row = W.shape[1] // BLOCK_DIM
    h_blocks = h_sorted.reshape(n_blocks_per_row, BLOCK_DIM)
    within_block_ratios = (h_blocks.max(dim=1).values / h_blocks.min(dim=1).values.clamp(min=1e-8))
    print(f"  Within-block H_diag ratio: mean={within_block_ratios.mean():.2f}×  "
          f"max={within_block_ratios.max():.2f}×  (vs 214× global before sorting)")

    class _SortedHLinear(nn.Module):
        def __init__(self, state, bias=None):
            super().__init__()
            self.register_buffer("centroids", state["centroids"].float())
            self.register_buffer("labels",    state["labels"])
            self.register_buffer("unsort_idx", state["unsort_idx"])
            self._W_shape = state["W_shape"]
            self._bd = state["block_dim"]
            self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None
        def forward(self, x):
            out_f, in_f = self._W_shape
            W_sorted = self.centroids[self.labels].reshape(out_f, in_f)
            W = W_sorted[:, self.unsort_idx]
            return F.linear(x, W, self.bias)

    sorted_layer = _SortedHLinear(state_sorted, b)
    model_sorted = copy.deepcopy(model)
    model_sorted.transformer.h[0].mlp.c_fc = sorted_layer
    ppl_sorted = eval_perplexity(model_sorted, tokenizer, eval_texts, MAX_LEN, DEVICE)
    del model_sorted
    print(f"  PPL = {ppl_sorted:.3f}  (delta = +{ppl_sorted - ppl_base:.3f})")

    # --- Summary
    print(f"\n{'='*65}")
    print(f"PPL Summary  (single layer: h0.c_fc, K={K}, bpw=0.5)")
    print(f"{'='*65}")
    print(f"  Baseline (FP32):             {ppl_base:>9.3f}")
    print(f"  Flat k-means:                {ppl_flat:>9.3f}  (delta={ppl_flat-ppl_base:>+8.3f})")
    print(f"  H-weighted sorted:           {ppl_sorted:>9.3f}  (delta={ppl_sorted-ppl_base:>+8.3f})")

    ppl_gain = ppl_flat - ppl_sorted
    ppl_gain_pct = ppl_gain / (ppl_flat - ppl_base) * 100 if ppl_flat > ppl_base else 0
    print(f"\n  H-weighted sorted vs flat:   {ppl_gain:>+9.3f}  "
          f"({ppl_gain_pct:.1f}% of quantization gap {'recovered' if ppl_gain > 0 else 'added'})")


if __name__ == "__main__":
    main()
