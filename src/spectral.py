"""
SVD spectral quantization.

Physics framing:
  The DCT spectrum of raw weight blocks is flat (white noise) — no preferred Fourier mode.
  But the *singular value spectrum* of the full weight matrix is steeply decaying.

  SVD gives W = U @ diag(S) @ V^T where S[0] >> S[1] >> ... — the operator's eigenspectrum.
  This is the analogue of the spectral decomposition of a Hamiltonian:
    - S[i]² = eigenvalues of W^T W  (energy levels)
    - V columns = right eigenstates  (input-space basis)
    - U columns = left eigenstates   (output-space basis)

  Keeping only the top-r modes is like projecting onto the r-dimensional ground-state
  subspace — we throw away high-energy (small singular value) excitations.

  Unlike DCT:
    - The basis is *learned from W itself*, not fixed
    - The spectrum provably decays for real weight matrices (they have low effective rank)
    - This is exactly what LoRA exploits, but here we *also* quantize the kept components

Three quantization targets after SVD truncation:
  1. S_r  [r]        — singular values: high-precision scalar (only r numbers)
  2. U_r  [m, r]     — left singular vectors: scalar quantization per element
  3. V_r  [n, r]     — right singular vectors: scalar quantization per element

  Forward: W_approx = U_r @ diag(S_r) @ V_r^T
           x -> W_approx @ x  (dequantize on the fly)

Bit budget analysis:
  Full precision: m*n float32 = 32*m*n bits
  SVD-quant:      r*log2(K_U) + r*log2(K_S) + r*log2(K_V) bits total
  bpw = (r/n) * (log2(K_U) + log2(K_S)/m + log2(K_V))   (per original weight)

  Key: for small r this can be very low bpw while preserving the dominant structure.
"""

import math
import torch


# ---------------------------------------------------------------------------
# Spectrum analysis
# ---------------------------------------------------------------------------

def singular_value_spectrum(W: torch.Tensor) -> torch.Tensor:
    """
    Compute the singular values of W (in descending order).
    Returns [min(m,n)] tensor.
    """
    _, S, _ = torch.linalg.svd(W.float(), full_matrices=False)
    return S  # already sorted descending


def spectrum_stats(S: torch.Tensor) -> dict:
    """
    Summary statistics for a singular value spectrum.

    Returns dict with:
      - variance_explained: [k] cumulative fraction of S²-variance captured by top-k modes
      - effective_rank_90:  minimum r such that top-r modes capture 90% of S² variance
      - effective_rank_99:  same for 99%
      - spectral_decay:     approximate power-law exponent α in S[i] ~ i^{-α}
                            (fit over the middle 50% of modes to avoid boundary effects)
    """
    S2 = S.pow(2)
    total = S2.sum()
    cumvar = S2.cumsum(0) / total

    r90 = (cumvar >= 0.90).nonzero(as_tuple=True)[0][0].item() + 1
    r99 = (cumvar >= 0.99).nonzero(as_tuple=True)[0][0].item() + 1

    # Power-law fit: log(S[i]) = -alpha * log(i+1) + const
    n = len(S)
    i0, i1 = n // 4, 3 * n // 4
    if i1 > i0 + 2:
        idx = torch.arange(i0, i1, dtype=torch.float32)
        log_i = (idx + 1).log()
        log_s = S[i0:i1].clamp(min=1e-12).log()
        # linear regression: log_s = -alpha * log_i + c
        log_i_mean = log_i.mean()
        log_s_mean = log_s.mean()
        alpha = -((log_i - log_i_mean) * (log_s - log_s_mean)).sum() / \
                 (log_i - log_i_mean).pow(2).sum()
        alpha = alpha.item()
    else:
        alpha = float("nan")

    return {
        "variance_explained": cumvar,
        "effective_rank_90":  r90,
        "effective_rank_99":  r99,
        "spectral_decay_alpha": alpha,
        "n_singular_values": len(S),
    }


# ---------------------------------------------------------------------------
# SVD truncation + quantization
# ---------------------------------------------------------------------------

def _quantize_matrix_scalar(X: torch.Tensor, n_levels: int) -> tuple[torch.Tensor, float, float]:
    """
    Uniform min-max scalar quantization of a matrix.
    Returns (indices [same shape as X], x_min, x_range).
    """
    x_min  = X.min().item()
    x_max  = X.max().item()
    x_range = max(x_max - x_min, 1e-8)
    idx = ((X - x_min) / x_range * (n_levels - 1)).round().clamp(0, n_levels - 1).long()
    return idx, x_min, x_range


def _dequantize_matrix_scalar(idx: torch.Tensor, x_min: float, x_range: float, n_levels: int) -> torch.Tensor:
    return idx.float() / (n_levels - 1) * x_range + x_min


def svd_quantize(
    W: torch.Tensor,
    rank: int,
    n_levels_UV: int = 256,
    n_levels_S: int = 65536,   # singular values get higher precision by default
) -> dict:
    """
    SVD truncation + scalar quantization of U, S, V components.

    Args:
        W:           [m, n] weight matrix
        rank:        number of singular modes to keep (r << min(m,n))
        n_levels_UV: quantization levels for U and V components
        n_levels_S:  quantization levels for singular values

    Returns:
        state dict for reconstruction and analysis
    """
    m, n = W.shape
    r = min(rank, min(m, n))

    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    # U: [m, min(m,n)], S: [min(m,n)], Vh: [min(m,n), n]

    U_r  = U[:, :r]    # [m, r]
    S_r  = S[:r]       # [r]
    Vh_r = Vh[:r, :]   # [r, n]

    # Quantize each component
    U_idx, U_min, U_range   = _quantize_matrix_scalar(U_r,  n_levels_UV)
    S_idx, S_min, S_range   = _quantize_matrix_scalar(S_r.unsqueeze(0), n_levels_S)
    Vh_idx, Vh_min, Vh_range = _quantize_matrix_scalar(Vh_r, n_levels_UV)

    # Bits per original weight
    bits_UV = math.log2(n_levels_UV)
    bits_S  = math.log2(n_levels_S)
    total_bits = r * m * bits_UV + r * bits_S + r * n * bits_UV
    bpw = total_bits / (m * n)

    # Variance explained by rank-r approximation (unquantized)
    _, S_full, _ = torch.linalg.svd(W.float(), full_matrices=False)
    var_explained = S_full[:r].pow(2).sum() / S_full.pow(2).sum()

    return {
        "U_idx":      U_idx,      # [m, r]
        "U_min":      U_min,
        "U_range":    U_range,
        "S_idx":      S_idx,      # [1, r]
        "S_min":      S_min,
        "S_range":    S_range,
        "Vh_idx":     Vh_idx,     # [r, n]
        "Vh_min":     Vh_min,
        "Vh_range":   Vh_range,
        "n_levels_UV": n_levels_UV,
        "n_levels_S":  n_levels_S,
        "rank":        r,
        "W_shape":     W.shape,
        "bpw":         bpw,
        "var_explained": var_explained.item(),
    }


def svd_reconstruct(state: dict) -> torch.Tensor:
    n_UV = state["n_levels_UV"]
    n_S  = state["n_levels_S"]

    U_r  = _dequantize_matrix_scalar(state["U_idx"],  state["U_min"],  state["U_range"],  n_UV)
    S_r  = _dequantize_matrix_scalar(state["S_idx"],  state["S_min"],  state["S_range"],  n_S).squeeze(0)
    Vh_r = _dequantize_matrix_scalar(state["Vh_idx"], state["Vh_min"], state["Vh_range"], n_UV)

    W_approx = (U_r * S_r.unsqueeze(0)) @ Vh_r   # [m, n]
    return W_approx


# ---------------------------------------------------------------------------
# Hybrid: SVD truncation + per-column codebook on U and V
# ---------------------------------------------------------------------------

def svd_codebook_quantize(
    W: torch.Tensor,
    rank: int,
    n_levels_UV: int = 256,
    n_levels_S: int = 65536,
) -> dict:
    """
    SVD + O(N)-style factorization on U and V columns.

    Each column of U_r and V_r is a near-unit vector (since U, V are orthonormal).
    Factor each column as norm * direction, then:
      - Quantize direction with n_levels_UV discrete bins (uniform scalar on the sphere coords)
      - Quantize norm with n_levels_S bins (should be ~1.0 for exact SVD, but shifts after quantization)

    This is the "O(N) + SVD" hybrid: apply the Heisenberg-spin factorization
    to the eigenbasis vectors rather than to raw weight blocks.
    """
    m, n = W.shape
    r = min(rank, min(m, n))

    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    U_r  = U[:, :r]    # [m, r]
    S_r  = S[:r]
    Vh_r = Vh[:r, :]   # [r, n]  (rows are right singular vectors)

    def quantize_columns(X, n_lev):
        # X: [d, r] — quantize each of the r columns independently
        col_min   = X.min(dim=0).values   # [r]
        col_max   = X.max(dim=0).values   # [r]
        col_range = (col_max - col_min).clamp(min=1e-8)
        idx = ((X - col_min) / col_range * (n_lev - 1)).round().clamp(0, n_lev - 1).long()
        return idx, col_min, col_range

    U_idx,  U_min,  U_range  = quantize_columns(U_r,    n_levels_UV)
    Vh_idx, Vh_min, Vh_range = quantize_columns(Vh_r.T, n_levels_UV)  # [n, r]
    S_idx, S_min, S_range    = _quantize_matrix_scalar(S_r.unsqueeze(0), n_levels_S)

    bits_UV = math.log2(n_levels_UV)
    bits_S  = math.log2(n_levels_S)
    total_bits = r * m * bits_UV + r * bits_S + r * n * bits_UV
    bpw = total_bits / (m * n)

    _, S_full, _ = torch.linalg.svd(W.float(), full_matrices=False)
    var_explained = S_full[:r].pow(2).sum() / S_full.pow(2).sum()

    return {
        "U_idx": U_idx, "U_min": U_min, "U_range": U_range,
        "Vh_idx": Vh_idx, "Vh_min": Vh_min, "Vh_range": Vh_range,
        "S_idx": S_idx, "S_min": S_min, "S_range": S_range,
        "n_levels_UV": n_levels_UV,
        "n_levels_S": n_levels_S,
        "rank": r,
        "W_shape": W.shape,
        "bpw": bpw,
        "var_explained": var_explained.item(),
    }


def svd_codebook_reconstruct(state: dict) -> torch.Tensor:
    n_UV = state["n_levels_UV"]
    n_S  = state["n_levels_S"]

    def deq_cols(idx, col_min, col_range, n_lev):
        return idx.float() / (n_lev - 1) * col_range + col_min

    U_r  = deq_cols(state["U_idx"],  state["U_min"],  state["U_range"],  n_UV)   # [m, r]
    Vh_r = deq_cols(state["Vh_idx"], state["Vh_min"], state["Vh_range"], n_UV).T  # [r, n]
    S_r  = _dequantize_matrix_scalar(state["S_idx"], state["S_min"], state["S_range"], n_S).squeeze(0)

    return (U_r * S_r.unsqueeze(0)) @ Vh_r
