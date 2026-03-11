"""
Frequency-domain quantization schemes.

Physics framing:
  Field theory / condensed matter always decomposes fields into momentum modes.
  Low-k (low-frequency) modes carry most of the "energy" (variance) in weight matrices.
  High-k modes are small corrections that can be aggressively truncated or quantized.

Two schemes here:

1. DCT quantization
   Each weight block -> DCT -> keep top-k coefficients -> scalar quantize coefficients.
   Analogy: JPEG compression / momentum-space truncation in lattice field theory.
   The energy compaction property of the DCT means most variance lives in a few modes.

2. Walsh-Hadamard (WHT) expansion
   Same idea but with the Hadamard basis {-1/sqrt(d), +1/sqrt(d)}^d.
   Hardware-efficient: no multiplications in the transform.
   Used in QuIP# for LLM quantization (they call it "incoherence processing").
   Analogy: spin-wave basis on a discrete lattice.
"""

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DCT utilities (1D type-II, scipy-compatible, pure PyTorch)
# ---------------------------------------------------------------------------

def dct1d(x: torch.Tensor) -> torch.Tensor:
    """
    1-D DCT-II (orthonormal) along the last dimension.
    Implemented via FFT for efficiency.

    x: [..., d]  -> X: [..., d]
    """
    d = x.shape[-1]
    # Reorder: [x0, x1, ..., x_{d-1}, x_{d-1}, ..., x1]  (length 2d)
    v = torch.cat([x, x.flip(-1)], dim=-1)
    V = torch.fft.rfft(v, dim=-1)[..., :d]
    # DCT-II from FFT
    k = torch.arange(d, device=x.device, dtype=x.dtype)
    phase = torch.exp(-1j * math.pi * k / (2 * d))
    X = (V * phase).real
    # Orthonormal scaling
    X[..., 0] /= math.sqrt(4 * d)
    X[..., 1:] /= math.sqrt(2 * d)
    return X


def idct1d(X: torch.Tensor) -> torch.Tensor:
    """
    1-D DCT-III (inverse of DCT-II, orthonormal) along last dimension.
    """
    d = X.shape[-1]
    k = torch.arange(d, device=X.device, dtype=X.dtype)
    # Undo orthonormal scaling
    X = X.clone()
    X[..., 0] *= math.sqrt(4 * d)
    X[..., 1:] *= math.sqrt(2 * d)
    # Phase correction then iFFT
    phase = torch.exp(1j * math.pi * k / (2 * d))
    V = torch.zeros(*X.shape[:-1], 2 * d, dtype=torch.complex64, device=X.device)
    V[..., :d] = X.to(torch.complex64) * phase
    v = torch.fft.irfft(V, n=2 * d, dim=-1)
    return v[..., :d]


# ---------------------------------------------------------------------------
# DCT quantization
# ---------------------------------------------------------------------------

def dct_energy_spectrum(W: torch.Tensor, block_dim: int) -> torch.Tensor:
    """
    Compute the mean squared DCT coefficient at each frequency bin across all blocks.
    Returns [block_dim] tensor — the "power spectrum" of weight blocks.

    This is the key diagnostic: if most energy is in low-k bins, DCT quantization wins.
    """
    W_blocks = W.reshape(-1, block_dim).float()
    coeffs = dct1d(W_blocks)         # [N, d]
    power = coeffs.pow(2).mean(0)    # [d]  mean over blocks
    return power


def dct_quantize(
    W: torch.Tensor,
    block_dim: int,
    keep_k: int,
    n_levels: int = 256,
):
    """
    DCT-domain quantization.

    For each block:
      1. Apply DCT-II
      2. Keep only the top-keep_k lowest-frequency coefficients (k=0..keep_k-1)
      3. Scalar-quantize each retained coefficient to n_levels bins
      4. Store: (coefficient bin indices [keep_k], scalar min/scale per block)

    Args:
        W:         [out_features, in_features]
        block_dim: block size d (must divide in_features)
        keep_k:    number of DCT coefficients to retain (1 <= keep_k <= block_dim)
        n_levels:  quantization levels per coefficient

    Returns:
        state dict for reconstruction and analysis
    """
    assert keep_k <= block_dim
    out_f, in_f = W.shape
    assert in_f % block_dim == 0

    W_blocks = W.reshape(-1, block_dim).float()  # [N, d]
    N = W_blocks.shape[0]

    coeffs = dct1d(W_blocks)                     # [N, d]
    kept = coeffs[:, :keep_k]                    # [N, keep_k]  low-frequency modes

    # Per-block min-max scalar quantization of each retained coefficient
    c_min = kept.min(dim=0).values   # [keep_k]
    c_max = kept.max(dim=0).values   # [keep_k]

    # Avoid zero range
    c_range = (c_max - c_min).clamp(min=1e-8)
    # Quantize to [0, n_levels-1]
    c_norm = (kept - c_min) / c_range            # [N, keep_k] in [0,1]
    idx = (c_norm * (n_levels - 1)).round().long().clamp(0, n_levels - 1)  # [N, keep_k]

    # Effective bits per original weight:
    # We store keep_k indices of log2(n_levels) bits each, spread over block_dim weights
    bpw = (keep_k * math.log2(n_levels)) / block_dim

    return {
        "idx":       idx,       # [N, keep_k]  quantized DCT coefficients
        "c_min":     c_min,     # [keep_k]
        "c_range":   c_range,   # [keep_k]
        "n_levels":  n_levels,
        "keep_k":    keep_k,
        "block_dim": block_dim,
        "W_shape":   W.shape,
        "bpw":       bpw,
    }


def dct_reconstruct(state: dict) -> torch.Tensor:
    idx     = state["idx"].float()
    c_min   = state["c_min"]
    c_range = state["c_range"]
    n_lev   = state["n_levels"]
    keep_k  = state["keep_k"]
    d       = state["block_dim"]

    # Dequantize retained coefficients
    kept_q = idx / (n_lev - 1) * c_range + c_min   # [N, keep_k]

    # Pad high-frequency bins with zero (truncated modes = discarded)
    N = kept_q.shape[0]
    coeffs_q = torch.zeros(N, d, device=kept_q.device, dtype=kept_q.dtype)
    coeffs_q[:, :keep_k] = kept_q

    W_blocks = idct1d(coeffs_q)
    return W_blocks.reshape(state["W_shape"])


# ---------------------------------------------------------------------------
# Walsh-Hadamard quantization
# ---------------------------------------------------------------------------

def hadamard_matrix(d: int, device=None) -> torch.Tensor:
    """
    Construct the d x d normalised Hadamard matrix (d must be a power of 2).
    H @ H.T = I
    """
    assert d > 0 and (d & (d - 1)) == 0, f"d={d} must be a power of 2"
    H = torch.tensor([[1.0]], device=device)
    while H.shape[0] < d:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0) / math.sqrt(2)
    return H  # [d, d]


def wht_quantize(
    W: torch.Tensor,
    block_dim: int,
    keep_k: int,
    n_levels: int = 256,
):
    """
    Walsh-Hadamard transform quantization.

    Same pipeline as DCT but using the Hadamard basis:
      block -> WHT -> keep first keep_k rows -> quantize -> store

    The Hadamard transform has only +/-1 entries (scaled) so it can be
    implemented with additions/subtractions — no floating point multiply
    in the transform step, making it hardware-friendly.

    After the transform the rows are ordered by "sequency" (number of sign
    changes), analogous to frequency ordering in the DCT.
    """
    assert block_dim & (block_dim - 1) == 0, "block_dim must be power of 2 for WHT"
    assert keep_k <= block_dim
    out_f, in_f = W.shape
    assert in_f % block_dim == 0

    H = hadamard_matrix(block_dim, device=W.device)   # [d, d]
    W_blocks = W.reshape(-1, block_dim).float()        # [N, d]

    coeffs = W_blocks @ H.T                            # [N, d]  WHT coefficients
    kept = coeffs[:, :keep_k]                          # [N, keep_k]

    c_min   = kept.min(dim=0).values
    c_max   = kept.max(dim=0).values
    c_range = (c_max - c_min).clamp(min=1e-8)
    c_norm  = (kept - c_min) / c_range
    idx     = (c_norm * (n_levels - 1)).round().long().clamp(0, n_levels - 1)

    bpw = (keep_k * math.log2(n_levels)) / block_dim

    return {
        "idx":       idx,
        "c_min":     c_min,
        "c_range":   c_range,
        "n_levels":  n_levels,
        "keep_k":    keep_k,
        "block_dim": block_dim,
        "W_shape":   W.shape,
        "H":         H,
        "bpw":       bpw,
    }


def wht_reconstruct(state: dict) -> torch.Tensor:
    idx     = state["idx"].float()
    c_min   = state["c_min"]
    c_range = state["c_range"]
    n_lev   = state["n_levels"]
    keep_k  = state["keep_k"]
    d       = state["block_dim"]
    H       = state["H"]

    kept_q = idx / (n_lev - 1) * c_range + c_min

    N = kept_q.shape[0]
    coeffs_q = torch.zeros(N, d, device=kept_q.device, dtype=kept_q.dtype)
    coeffs_q[:, :keep_k] = kept_q

    W_blocks = coeffs_q @ H   # inverse WHT = H itself (H is orthonormal)
    return W_blocks.reshape(state["W_shape"])
