"""
Per-column H-weighted quantization: perplexity evaluation.

Experiment 5. Tests whether the Experiment 4 failures were due to block structure
(mixing hot and cold H_diag dimensions within a block) by using a column-grouped layout.

Key idea: form groups of `group_size` consecutive output-dimension weights from the SAME
input column j. Every element in the group shares H_diag[j], so H-weighting has no
within-block distortion. The scale cancels uniformly on reconstruction.

Conditions at bpw=0.5 (group_size=16, K=256):
  1. Baseline     — full precision
  2. Flat block   — standard k-means on consecutive weight blocks (control)
  3. Col-grouped  — k-means on column-grouped blocks, scaled by sqrt(H_diag[j])

Usage:
    python experiments/hessian_col_ppl.py
"""

import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.codebook import kmeans
from src.hessian import estimate_h_diag
from experiments.eval_perplexity import conv1d_to_linear, eval_perplexity


MODEL_NAME = "gpt2"
K          = 256
GROUP_SIZE = 16       # output-dim rows per group (all share same column's H_diag)
N_CALIB    = 100
N_EVAL     = 150
MAX_LEN    = 128
DEVICE     = "cpu"


# ---------------------------------------------------------------------------
# Per-column grouped quantization
# ---------------------------------------------------------------------------

def col_grouped_quantize(W: torch.Tensor, h_diag: torch.Tensor,
                         group_size: int, K: int, n_iter: int = 50) -> dict:
    """
    Column-grouped H-weighted k-means quantization.

    Groups `group_size` consecutive output-dim weights from the same input column j.
    All group elements share H_diag[j] — no within-group sensitivity mixing.

    Layout:
      W: [out_f, in_f]
      W.T: [in_f, out_f]  — each row is one input column
      After grouping: [in_f * (out_f // group_size), group_size]
      Each group (j, g) contains W.T[j, g*group_size : (g+1)*group_size]
      All scaled uniformly by sqrt(H_diag[j]).

    bpw = log2(K) / group_size  (same formula as flat block k-means)
    """
    out_f, in_f = W.shape
    assert out_f % group_size == 0, f"out_f={out_f} must be divisible by group_size={group_size}"

    scale = (h_diag.float() + 1e-8).sqrt()   # [in_f]

    # Column-major layout: [in_f, out_f], each row = one input column
    W_T = W.float().T                          # [in_f, out_f]

    # Scale each column uniformly by sqrt(H_diag[j])
    W_T_scaled = W_T * scale.unsqueeze(1)      # [in_f, out_f]

    # Form groups: [in_f * n_groups_per_col, group_size]
    n_groups_per_col = out_f // group_size
    W_groups = W_T_scaled.reshape(in_f * n_groups_per_col, group_size).cpu()

    # K-means in scaled space
    centroids, labels = kmeans(W_groups, K, n_iter=n_iter)

    return {
        "centroids":       centroids,     # [K, group_size]  in scaled space
        "labels":          labels,        # [in_f * n_groups_per_col]
        "scale":           scale.cpu(),   # [in_f]
        "group_size":      group_size,
        "n_groups_per_col": n_groups_per_col,
        "W_shape":         W.shape,
    }


class _ColGroupedLinear(nn.Module):
    """Drop-in layer using per-column H-weighted codebook."""
    def __init__(self, state: dict, bias=None):
        super().__init__()
        self.register_buffer("centroids", state["centroids"].float())
        self.register_buffer("labels",    state["labels"])
        self.register_buffer("scale",     state["scale"].float())
        self._W_shape        = state["W_shape"]
        self._group_size     = state["group_size"]
        self._n_groups_per_col = state["n_groups_per_col"]
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None

    def forward(self, x):
        out_f, in_f = self._W_shape

        # Reconstruct scaled column-major matrix
        W_T_scaled = self.centroids[self.labels].reshape(in_f, out_f)  # [in_f, out_f]

        # Unscale each column uniformly by 1/sqrt(H_diag[j])
        W_T = W_T_scaled / self.scale.clamp(min=1e-8).unsqueeze(1)    # [in_f, out_f]

        # Back to [out_f, in_f]
        W = W_T.T.contiguous()
        return F.linear(x, W, self.bias)


# ---------------------------------------------------------------------------
# Flat block k-means (control — same bpw, standard block layout)
# ---------------------------------------------------------------------------

class _FlatBlockLinear(nn.Module):
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

    # Target layer
    raw_layer     = model.transformer.h[0].mlp.c_fc
    target_linear = conv1d_to_linear(raw_layer)
    W  = target_linear.weight.data.clone()    # [out_f, in_f] = [3072, 768]
    b  = target_linear.bias.data.clone() if target_linear.bias is not None else None

    out_f, in_f = W.shape
    bpw = (K.bit_length() - 1) / GROUP_SIZE
    print(f"Target: h0.c_fc  {tuple(W.shape)}  K={K}  group_size={GROUP_SIZE}  bpw={bpw:.3f}")

    # --- Baseline
    print("\n[1/3] Baseline (full precision) ...")
    ppl_base = eval_perplexity(model, tokenizer, eval_texts, MAX_LEN, DEVICE)
    print(f"  PPL = {ppl_base:.3f}")

    # --- H_diag calibration
    print(f"\nCalibrating H_diag ({N_CALIB} texts) ...")
    h_diag = estimate_h_diag(model, tokenizer, raw_layer, calib_texts, MAX_LEN, DEVICE)
    print(f"  H_diag range: [{h_diag.min():.4f}, {h_diag.max():.4f}]  "
          f"CV={h_diag.std()/h_diag.mean():.3f}")

    # --- Flat block k-means (control)
    print(f"\n[2/3] Flat block k-means (K={K}, group_size={GROUP_SIZE}, bpw={bpw:.3f}) ...")
    from src.codebook import quantize_blocks
    centroids_flat, labels_flat, _ = quantize_blocks(W, GROUP_SIZE, K, n_iter=50)
    flat_layer = _FlatBlockLinear(centroids_flat, labels_flat, W.shape, b)

    model_flat = copy.deepcopy(model)
    model_flat.transformer.h[0].mlp.c_fc = flat_layer
    ppl_flat = eval_perplexity(model_flat, tokenizer, eval_texts, MAX_LEN, DEVICE)
    del model_flat
    print(f"  PPL = {ppl_flat:.3f}  (delta = +{ppl_flat - ppl_base:.3f})")

    # --- Per-column H-weighted k-means
    print(f"\n[3/3] Per-column H-weighted k-means (K={K}, group_size={GROUP_SIZE}) ...")
    print(f"      (each group contains {GROUP_SIZE} weights from the same input column j)")
    print(f"      (all share H_diag[j] — no within-group sensitivity mixing)")

    # Verify within-group scale uniformity (should be 1.0× by construction)
    scale = (h_diag.float() + 1e-8).sqrt()
    n_groups_per_col = out_f // GROUP_SIZE
    # In col-grouped layout, within each group ALL elements have the same scale[j]
    # So within-group scale ratio = 1.0 for all groups by construction
    print(f"  Within-group H_diag ratio: 1.00× (by construction — same column per group)")
    print(f"  Column-wise H_diag ratio (max/min across columns): {scale.max()/scale.min():.2f}×")
    print(f"  (This affects k-means centroid placement but not reconstruction fidelity per group)")

    state_col = col_grouped_quantize(W, h_diag, GROUP_SIZE, K, n_iter=50)
    col_layer  = _ColGroupedLinear(state_col, b)

    model_col = copy.deepcopy(model)
    model_col.transformer.h[0].mlp.c_fc = col_layer
    ppl_col = eval_perplexity(model_col, tokenizer, eval_texts, MAX_LEN, DEVICE)
    del model_col
    print(f"  PPL = {ppl_col:.3f}  (delta = +{ppl_col - ppl_base:.3f})")

    # --- H-RMSE comparison
    from src.codebook import reconstruct
    Wq_flat = reconstruct(centroids_flat, labels_flat, W.shape)
    Wq_col  = state_col["centroids"][state_col["labels"]].reshape(in_f, out_f).T.float()
    scale_v = state_col["scale"].clamp(min=1e-8)
    Wq_col  = (Wq_col.T / scale_v.unsqueeze(1)).T   # unscale

    h_diag_v = h_diag.to(W.device)
    err_flat = (W - Wq_flat.to(W.device)) ** 2
    err_col  = (W - Wq_col.to(W.device)) ** 2
    h_rmse_flat = (err_flat * h_diag_v.unsqueeze(0)).mean().sqrt().item()
    h_rmse_col  = (err_col  * h_diag_v.unsqueeze(0)).mean().sqrt().item()
    gain_pct = (h_rmse_flat - h_rmse_col) / h_rmse_flat * 100

    # --- Summary
    print(f"\n{'='*65}")
    print(f"PPL Summary  (single layer: h0.c_fc, K={K}, bpw={bpw:.3f})")
    print(f"{'='*65}")
    print(f"  Baseline (FP32):             {ppl_base:>9.3f}")
    print(f"  Flat block k-means:          {ppl_flat:>9.3f}  (delta={ppl_flat-ppl_base:>+8.3f})")
    print(f"  Col-grouped H-weighted:      {ppl_col:>9.3f}  (delta={ppl_col-ppl_base:>+8.3f})")

    ppl_gain = ppl_flat - ppl_col
    ppl_gap  = ppl_flat - ppl_base
    ppl_gain_pct = ppl_gain / ppl_gap * 100 if ppl_gap > 0 else 0
    print(f"\n  H-col-grouped vs flat:       {ppl_gain:>+9.3f}  "
          f"({ppl_gain_pct:.1f}% of quantization gap {'recovered' if ppl_gain > 0 else 'added'})")
    print(f"\n  H-RMSE flat:    {h_rmse_flat:.6f}")
    print(f"  H-RMSE col:     {h_rmse_col:.6f}  (gain={gain_pct:+.1f}%)")


if __name__ == "__main__":
    main()
