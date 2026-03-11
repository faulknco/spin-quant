"""
Physics-inspired quantization schemes.

Three regimes explored here, ordered by increasing physics-alignment:

1. Scalar codebook  (standard, baseline)
   Each block -> nearest centroid in R^d.

2. O(N)-style factorization  (Heisenberg-inspired)
   Each block = norm * unit_direction.
   Quantize direction with a spherical codebook, norm with a discrete ladder.
   Effective bits = (log2(K_dir) + log2(n_levels)) / block_dim

3. Hierarchical / RG-style  (renormalization group inspired)
   Coarse codebook captures global structure (low-k modes).
   Residual codebook corrects per-block deviations (high-k modes).
   Quantize as coarse_centroid + residual_centroid.
"""

import torch
from .codebook import kmeans


# ---------------------------------------------------------------------------
# O(N)-style: direction + norm
# ---------------------------------------------------------------------------

def on_factorize(W_blocks: torch.Tensor, eps: float = 1e-8):
    """
    Factor each block as norm * unit_direction.

    Args:
        W_blocks: [N, d]
    Returns:
        dirs:  [N, d]  unit vectors (spin orientations)
        norms: [N]     scalar magnitudes
    """
    norms = W_blocks.norm(dim=1, keepdim=True).clamp(min=eps)
    dirs = W_blocks / norms
    return dirs, norms.squeeze(1)


def quantize_scalar_ladder(values: torch.Tensor, n_levels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Uniform scalar quantization over [min, max] with n_levels bins.

    Returns:
        bins:   [n_levels] the discrete level values
        idx:    [N] long index of nearest bin per value
    """
    vmin, vmax = values.min(), values.max()
    bins = torch.linspace(vmin.item(), vmax.item(), steps=n_levels, device=values.device)
    idx = (values.unsqueeze(1) - bins.unsqueeze(0)).abs().argmin(dim=1)
    return bins, idx


def on_quantize(
    W: torch.Tensor,
    block_dim: int,
    K_dir: int,
    n_levels: int = 16,
    n_iter: int = 50,
):
    """
    O(N)-style quantization of weight matrix W.

    Returns a dict with everything needed for reconstruction and analysis.
    """
    out_f, in_f = W.shape
    W_blocks = W.reshape(-1, block_dim).float().cpu()

    dirs, norms = on_factorize(W_blocks)

    dir_centroids, dir_labels = kmeans(dirs, K_dir, n_iter=n_iter)
    norm_bins, norm_idx = quantize_scalar_ladder(norms, n_levels)

    return {
        "dir_centroids": dir_centroids,   # [K_dir, block_dim]
        "dir_labels":    dir_labels,       # [N_blocks]
        "norm_bins":     norm_bins,        # [n_levels]
        "norm_idx":      norm_idx,         # [N_blocks]
        "W_shape":       W.shape,
        "block_dim":     block_dim,
        "bits_per_weight": (dir_labels.float().log2().mean() + torch.tensor(n_levels).float().log2()) / block_dim,
    }


def on_reconstruct(state: dict) -> torch.Tensor:
    dirs_q = state["dir_centroids"][state["dir_labels"]]
    norms_q = state["norm_bins"][state["norm_idx"]]
    W_blocks = dirs_q * norms_q.unsqueeze(1)
    return W_blocks.view(state["W_shape"])


# ---------------------------------------------------------------------------
# RG / hierarchical: coarse + residual
# ---------------------------------------------------------------------------

def rg_quantize(
    W: torch.Tensor,
    block_dim: int,
    K_coarse: int,
    K_residual: int,
    n_iter: int = 50,
):
    """
    Two-level (RG-style) codebook quantization.

    Level 1 (coarse): global codebook, captures large-scale weight structure.
    Level 2 (residual): per-block correction from a finer local codebook.

    Analogy to RG: coarse = integrated-out long-wavelength modes,
                   residual = short-wavelength corrections.

    Returns a state dict for reconstruction and analysis.
    """
    out_f, in_f = W.shape
    W_blocks = W.reshape(-1, block_dim).float().cpu()

    # --- Level 1: coarse
    coarse_centroids, coarse_labels = kmeans(W_blocks, K_coarse, n_iter=n_iter)
    coarse_approx = coarse_centroids[coarse_labels]  # [N, d]

    # --- Level 2: residual
    residuals = W_blocks - coarse_approx
    res_centroids, res_labels = kmeans(residuals, K_residual, n_iter=n_iter)

    return {
        "coarse_centroids":   coarse_centroids,   # [K_coarse, d]
        "coarse_labels":      coarse_labels,       # [N_blocks]
        "res_centroids":      res_centroids,       # [K_residual, d]
        "res_labels":         res_labels,          # [N_blocks]
        "W_shape":            W.shape,
        "block_dim":          block_dim,
    }


def rg_reconstruct(state: dict) -> torch.Tensor:
    coarse = state["coarse_centroids"][state["coarse_labels"]]
    residual = state["res_centroids"][state["res_labels"]]
    return (coarse + residual).view(state["W_shape"])
