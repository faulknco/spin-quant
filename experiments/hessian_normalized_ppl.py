"""
Per-column normalized k-means: perplexity evaluation.

Experiment 5b. Tests per-column normalization as the fix for cross-column scale
contamination in shared codebooks.

Root cause of Experiment 4/5 failures: shared codebook at bpw=0.5 with 14× column
scale heterogeneity. Hot-column centroids get amplified 14× when used for cold columns.

Fix: normalize each input column j to (mean=0, std=1) before k-means. Store per-column
μ_j, σ_j as side information (768 × 2 floats = 6KB, ~0 extra bpw). The shared codebook
operates on unit-scale data with no cross-column contamination.

Reconstruction: W_approx[:, j] = centroids[labels_j] * σ_j + μ_j  (no amplification)

Conditions at bpw≈0.5 (group_size=16, K=256):
  1. Baseline        — full precision
  2. Flat block      — standard k-means (rows-then-cols layout, no normalization) [control]
  3. Col-normalized  — per-column normalized k-means, shared codebook
  4. Col-norm + H    — same but k-means weighted by H_diag (hot cols matter more)

Usage:
    python experiments/hessian_normalized_ppl.py
"""

import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.codebook import kmeans, quantize_blocks, reconstruct
from src.hessian import estimate_h_diag
from experiments.eval_perplexity import conv1d_to_linear, eval_perplexity


MODEL_NAME = "gpt2"
K          = 256
GROUP_SIZE = 16
N_CALIB    = 100
N_EVAL     = 150
MAX_LEN    = 128
DEVICE     = "cpu"


# ---------------------------------------------------------------------------
# Per-column normalized quantization
# ---------------------------------------------------------------------------

def col_normalized_quantize(W: torch.Tensor, group_size: int, K: int,
                             h_diag=None, n_iter: int = 50) -> dict:
    """
    Per-column normalized k-means quantization with shared codebook.

    Steps:
      1. For each input column j: compute μ_j = mean(W[:, j]), σ_j = std(W[:, j])
      2. Normalize: W_norm[:, j] = (W[:, j] - μ_j) / σ_j  (unit variance per column)
      3. Transpose to column-major: W_norm.T [in_f, out_f]
      4. Form groups of group_size consecutive output-dim weights per column
      5. If h_diag given: weight each group's contribution by H_diag[j] in k-means
         (hot columns get more influence on centroid placement)
      6. Run k-means on all groups (all unit-scale — no cross-column contamination)
      7. Store: centroids + labels + μ + σ (side info, negligible bits)

    Reconstruction: W_approx[:, j] = centroids[labels_j].reshape(...) * σ_j + μ_j

    bpw = log2(K) / group_size  (plus ~0.01 bpw for per-column μ, σ)
    """
    out_f, in_f = W.shape
    assert out_f % group_size == 0

    W_f = W.float()

    # Per-column statistics
    col_mean = W_f.mean(dim=0)           # [in_f]
    col_std  = W_f.std(dim=0).clamp(min=1e-8)  # [in_f]

    # Normalize: W_norm[:, j] has mean=0, std=1
    W_norm = (W_f - col_mean.unsqueeze(0)) / col_std.unsqueeze(0)  # [out_f, in_f]

    # Column-major layout: [in_f, out_f]
    W_norm_T = W_norm.T  # [in_f, out_f]

    # Form groups: [in_f * n_groups_per_col, group_size]
    n_groups_per_col = out_f // group_size
    W_groups = W_norm_T.reshape(in_f * n_groups_per_col, group_size).cpu()
    # W_groups[j * n_groups + g] = W_norm[g*group_size:(g+1)*group_size, j]
    # All elements share column j → all unit-scale after normalization

    # Optional H-weighting: duplicate groups proportional to sqrt(H_diag[j])
    # This biases centroid placement toward hot columns without changing reconstruction
    if h_diag is not None:
        # Weight each column's groups by H_diag[j] in the k-means objective.
        # We implement this via sqrt-weighting of the groups fed to k-means:
        # rescale each group by sqrt(H_diag[j]) so that hot-column distances count more.
        # After k-means: centroids are in H-rescaled normalized space.
        # Reconstruction: unrescale before applying σ_j + μ_j.
        h_scale = (h_diag.float().clamp(min=1e-8)).sqrt()  # [in_f]
        # [in_f * n_groups_per_col, group_size]: each group scaled by h_scale[j]
        h_scale_rep = h_scale.unsqueeze(1).expand(in_f, n_groups_per_col)  # [in_f, n_groups_per_col]
        h_scale_rep = h_scale_rep.reshape(in_f * n_groups_per_col, 1).cpu()
        W_groups_h = W_groups * h_scale_rep
        centroids_h, labels = kmeans(W_groups_h, K, n_iter=n_iter)
        # Unscale centroids back to normalized space — we can't simply unscale because
        # different groups share centroids with different h_scale values.
        # Instead, store labels and centroid in h-scaled space, unscale at reconstruction time.
        # But at reconstruction, we don't know which column a centroid came from...
        #
        # Alternative: store centroids in normalized (unit) space by dividing back.
        # This doesn't work cleanly with shared centroids.
        #
        # Pragmatic approach: store centroids_h (in H-scaled space) and at reconstruction
        # unscale per-group using h_scale[j]. This IS well-defined because each group
        # belongs to exactly one column j, and h_scale[j] is stored.
        #
        return {
            "centroids":   centroids_h,       # [K, group_size]  H-scaled normalized space
            "labels":      labels,            # [in_f * n_groups_per_col]
            "col_mean":    col_mean.cpu(),    # [in_f]
            "col_std":     col_std.cpu(),     # [in_f]
            "h_scale":     h_scale.cpu(),     # [in_f]  for unscaling centroids
            "group_size":  group_size,
            "n_groups_per_col": n_groups_per_col,
            "W_shape":     W.shape,
            "h_weighted":  True,
        }
    else:
        centroids, labels = kmeans(W_groups, K, n_iter=n_iter)
        return {
            "centroids":   centroids,         # [K, group_size]  normalized space
            "labels":      labels,
            "col_mean":    col_mean.cpu(),
            "col_std":     col_std.cpu(),
            "h_scale":     None,
            "group_size":  group_size,
            "n_groups_per_col": n_groups_per_col,
            "W_shape":     W.shape,
            "h_weighted":  False,
        }


def col_normalized_reconstruct(state: dict) -> torch.Tensor:
    """Reconstruct weight matrix from per-column normalized codebook."""
    out_f, in_f = state["W_shape"]
    group_size  = state["group_size"]
    n_groups_per_col = state["n_groups_per_col"]

    # [in_f * n_groups_per_col, group_size]
    W_groups_recon = state["centroids"][state["labels"]]

    if state["h_weighted"] and state["h_scale"] is not None:
        # Centroids are in H-scaled normalized space; unscale per column
        h_scale = state["h_scale"]  # [in_f]
        h_scale_rep = h_scale.unsqueeze(1).expand(in_f, n_groups_per_col)
        h_scale_rep = h_scale_rep.reshape(in_f * n_groups_per_col, 1)
        W_groups_recon = W_groups_recon / h_scale_rep.clamp(min=1e-8)

    # [in_f, out_f] — reshape back to column-major
    W_norm_T = W_groups_recon.reshape(in_f, out_f)

    # Denormalize each column: W[:, j] = W_norm_T[j, :] * σ_j + μ_j
    col_mean = state["col_mean"]  # [in_f]
    col_std  = state["col_std"]   # [in_f]

    W_T = W_norm_T * col_std.unsqueeze(1) + col_mean.unsqueeze(1)  # [in_f, out_f]
    return W_T.T.contiguous()   # [out_f, in_f]


class _ColNormLinear(nn.Module):
    """Drop-in layer using per-column normalized shared codebook."""
    def __init__(self, state: dict, bias=None):
        super().__init__()
        self.register_buffer("centroids", state["centroids"].float())
        self.register_buffer("labels",    state["labels"])
        self.register_buffer("col_mean",  state["col_mean"].float())
        self.register_buffer("col_std",   state["col_std"].float())
        if state["h_scale"] is not None:
            self.register_buffer("h_scale", state["h_scale"].float())
        else:
            self.h_scale = None
        self._W_shape        = state["W_shape"]
        self._group_size     = state["group_size"]
        self._n_groups       = state["n_groups_per_col"]
        self._h_weighted     = state["h_weighted"]
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None

    def forward(self, x):
        out_f, in_f = self._W_shape

        W_groups = self.centroids[self.labels]   # [in_f * n_groups, group_size]

        if self._h_weighted and self.h_scale is not None:
            h_scale_rep = self.h_scale.unsqueeze(1).expand(in_f, self._n_groups)
            h_scale_rep = h_scale_rep.reshape(in_f * self._n_groups, 1)
            W_groups = W_groups / h_scale_rep.clamp(min=1e-8)

        W_norm_T = W_groups.reshape(in_f, out_f)
        W_T = W_norm_T * self.col_std.unsqueeze(1) + self.col_mean.unsqueeze(1)
        W = W_T.T.contiguous()
        return F.linear(x, W, self.bias)


class _FlatLinear(nn.Module):
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
    print("\n[1/4] Baseline (full precision) ...")
    ppl_base = eval_perplexity(model, tokenizer, eval_texts, MAX_LEN, DEVICE)
    print(f"  PPL = {ppl_base:.3f}")

    # --- H_diag calibration
    print(f"\nCalibrating H_diag ({N_CALIB} texts) ...")
    h_diag = estimate_h_diag(model, tokenizer, raw_layer, calib_texts, MAX_LEN, DEVICE)
    print(f"  H_diag range: [{h_diag.min():.4f}, {h_diag.max():.4f}]  "
          f"CV={h_diag.std()/h_diag.mean():.3f}")

    # --- Flat block k-means (control)
    print(f"\n[2/4] Flat block k-means (K={K}, group_size={GROUP_SIZE}, bpw={bpw:.3f}) ...")
    centroids_flat, labels_flat, _ = quantize_blocks(W, GROUP_SIZE, K, n_iter=50)
    flat_layer = _FlatLinear(centroids_flat, labels_flat, W.shape, b)

    model_flat = copy.deepcopy(model)
    model_flat.transformer.h[0].mlp.c_fc = flat_layer
    ppl_flat = eval_perplexity(model_flat, tokenizer, eval_texts, MAX_LEN, DEVICE)
    del model_flat
    print(f"  PPL = {ppl_flat:.3f}  (delta = +{ppl_flat - ppl_base:.3f})")

    # --- Per-column normalized (no H-weighting)
    print(f"\n[3/4] Per-column normalized k-means (no H-weighting) ...")
    print(f"      (each column normalized to unit variance before k-means)")
    print(f"      (μ, σ stored per column as side info — negligible bpw overhead)")
    state_norm = col_normalized_quantize(W, GROUP_SIZE, K, h_diag=None, n_iter=50)
    norm_layer  = _ColNormLinear(state_norm, b)

    model_norm = copy.deepcopy(model)
    model_norm.transformer.h[0].mlp.c_fc = norm_layer
    ppl_norm = eval_perplexity(model_norm, tokenizer, eval_texts, MAX_LEN, DEVICE)
    del model_norm
    print(f"  PPL = {ppl_norm:.3f}  (delta = +{ppl_norm - ppl_base:.3f})")

    # --- Per-column normalized + H-weighted k-means
    print(f"\n[4/4] Per-column normalized + H-weighted k-means ...")
    print(f"      (unit-variance columns; H_diag[j] weights centroid placement toward hot cols)")
    state_nh = col_normalized_quantize(W, GROUP_SIZE, K, h_diag=h_diag, n_iter=50)
    nh_layer  = _ColNormLinear(state_nh, b)

    model_nh = copy.deepcopy(model)
    model_nh.transformer.h[0].mlp.c_fc = nh_layer
    ppl_nh = eval_perplexity(model_nh, tokenizer, eval_texts, MAX_LEN, DEVICE)
    del model_nh
    print(f"  PPL = {ppl_nh:.3f}  (delta = +{ppl_nh - ppl_base:.3f})")

    # --- H-RMSE comparison
    Wq_flat = reconstruct(centroids_flat, labels_flat, W.shape).to(W.device)
    Wq_norm = col_normalized_reconstruct(state_norm).to(W.device)
    Wq_nh   = col_normalized_reconstruct(state_nh).to(W.device)

    h_diag_v = h_diag.to(W.device)
    def h_rmse(Wq):
        err = (W - Wq) ** 2
        return (err * h_diag_v.unsqueeze(0)).mean().sqrt().item()

    hr_flat = h_rmse(Wq_flat)
    hr_norm = h_rmse(Wq_norm)
    hr_nh   = h_rmse(Wq_nh)

    # --- Summary
    print(f"\n{'='*65}")
    print(f"PPL Summary  (single layer: h0.c_fc, K={K}, bpw={bpw:.3f})")
    print(f"{'='*65}")
    print(f"  Baseline (FP32):             {ppl_base:>9.3f}")
    print(f"  Flat block k-means:          {ppl_flat:>9.3f}  (delta={ppl_flat-ppl_base:>+8.3f})")
    print(f"  Col-normalized:              {ppl_norm:>9.3f}  (delta={ppl_norm-ppl_base:>+8.3f})")
    print(f"  Col-norm + H-weighted:       {ppl_nh:>9.3f}  (delta={ppl_nh-ppl_base:>+8.3f})")

    ppl_gap = ppl_flat - ppl_base
    for name, ppl in [("col-norm", ppl_norm), ("col-norm+H", ppl_nh)]:
        gain = ppl_flat - ppl
        pct  = gain / ppl_gap * 100 if ppl_gap > 0 else 0
        verb = "recovered" if gain > 0 else "added"
        print(f"\n  {name} vs flat:  {gain:>+9.3f}  ({pct:.1f}% of quantization gap {verb})")

    print(f"\n  H-RMSE flat:    {hr_flat:.6f}")
    print(f"  H-RMSE norm:    {hr_norm:.6f}  (gain={100*(hr_flat-hr_norm)/hr_flat:+.1f}%)")
    print(f"  H-RMSE norm+H:  {hr_nh:.6f}  (gain={100*(hr_flat-hr_nh)/hr_flat:+.1f}%)")


if __name__ == "__main__":
    main()
