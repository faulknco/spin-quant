"""
Hessian-weighted codebook quantization.

Physics framing (spin-glass):
  The spin-glass Hamiltonian is H({σ}) = Σᵢⱼ Jᵢⱼ σᵢ σⱼ where Jᵢⱼ encodes the
  coupling strength between spin i and spin j.

  For LLM quantization, the analogous coupling is the *loss Hessian* H_W:
    Jᵢⱼ ∝ ∂²L / ∂Wᵢ ∂Wⱼ

  A weight in a direction d with high Hessian eigenvalue λ costs λ·ε² to quantize
  with error ε. A direction with low λ can absorb a large error cheaply.

  Flat k-means minimises Σᵢ (Wᵢ - Wqᵢ)² — it treats all directions equally.
  Hessian-weighted k-means minimises Σᵢⱼ Hᵢⱼ (Wᵢ - Wqᵢ)(Wⱼ - Wqⱼ) — it protects
  sensitive directions and is aggressive on insensitive ones.

Diagonal approximation:
  Full Hessian for a layer W ∈ ℝ^{m×n} has n² parameters — intractable.
  We use the diagonal approximation: H_diag[j] = E_x[x_j²] for the j-th input dimension.
  This is exact for the output-space MSE loss: ∂²(||Wx - Wqx||²) / ∂W_ij² = E[x_j²].
  (The same approximation used by GPTQ.)

  Intuition: input dimension j is "important" if it is frequently activated (high E[x_j²]).
  Weights that multiply important inputs should be quantized more carefully.

Implementation:
  1. Forward calibration pass: run n_calib tokens, record input activations X to target layer
  2. Compute H_diag[j] = mean(X[:, j]²) for j = 0..in_features-1
  3. Scale weights:   W_scaled[:, j] = W[:, j] * sqrt(H_diag[j])
  4. Run k-means on W_scaled blocks (now each block has equal "loss importance")
  5. Unscale centroids: C[:, block_positions] /= sqrt(H_diag[block_positions])
  6. Reconstruct: Wq = C[labels].view(W.shape)

  Result: centroids are in unscaled weight space, labels assign each block
  to the centroid that minimises loss-weighted error.
"""

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from .codebook import kmeans, reconstruct


# ---------------------------------------------------------------------------
# Calibration hook: capture layer input activations
# ---------------------------------------------------------------------------

class _ActivationCapture:
    """Context manager that captures the input to a nn.Module."""
    def __init__(self, module: nn.Module):
        self.activations = []
        self._hook = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        # input is a tuple; first element is the activation tensor
        x = input[0].detach().float()
        # Flatten all token positions: [batch, seq, features] -> [batch*seq, features]
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.dim() == 2:
            pass
        self.activations.append(x.cpu())

    def get_h_diag(self) -> torch.Tensor:
        """H_diag[j] = E[x_j²] — mean squared activation per input dimension."""
        all_acts = torch.cat(self.activations, dim=0)   # [N_tokens, in_features]
        return all_acts.pow(2).mean(dim=0)               # [in_features]

    def close(self):
        self._hook.remove()


def estimate_h_diag(
    model: nn.Module,
    tokenizer,
    layer_module: nn.Module,
    texts: list[str],
    max_length: int = 128,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Run calibration forward passes and return the diagonal Hessian H_diag[j] = E[x_j²]
    for the inputs to `layer_module`.

    Args:
        model:        full GPT-2 (or similar) model
        tokenizer:    matching tokenizer
        layer_module: the specific layer to capture inputs for (e.g. block.mlp.c_fc)
        texts:        calibration texts
        max_length:   token length per text

    Returns:
        h_diag: [in_features] tensor
    """
    capture = _ActivationCapture(layer_module)
    model.eval()
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            model(enc["input_ids"].to(device))
    capture.close()
    return capture.get_h_diag()


# ---------------------------------------------------------------------------
# Hessian-weighted k-means quantization
# ---------------------------------------------------------------------------

def hessian_quantize(
    W: torch.Tensor,
    h_diag: torch.Tensor,
    block_dim: int,
    K: int,
    n_iter: int = 50,
    scale_clip_percentile: float = 95.0,
) -> dict:
    """
    Hessian-weighted codebook quantization.

    Args:
        W:         [out_features, in_features]  weight matrix
        h_diag:    [in_features]               diagonal Hessian H_diag[j] = E[x_j²]
        block_dim: block size (must divide in_features)
        K:         codebook size

    Returns:
        state dict with centroids, labels and metadata for reconstruction + analysis
    """
    out_f, in_f = W.shape
    assert in_f % block_dim == 0
    assert h_diag.shape[0] == in_f

    # --- Scale factor per input dimension: sqrt(H_diag[j])
    # Clip H_diag at the given percentile before sqrt to prevent explosive
    # amplification of cold-dimension errors on reconstruction.
    # Without clipping: max/min scale ratio can be 14×, causing 14× amplified
    # reconstruction errors in cold dims which catastrophically hurt PPL.
    h_clipped = h_diag.clone()
    cap = torch.quantile(h_diag.float(), scale_clip_percentile / 100.0)
    h_clipped = h_clipped.clamp(max=cap.item())
    scale = (h_clipped + 1e-8).sqrt()       # [in_f]

    # --- Scale weight columns by sqrt(H_diag): makes all directions equally costly
    W_scaled = W.float() * scale.unsqueeze(0)  # [out_f, in_f]

    # --- Reshape into blocks and run k-means in scaled space
    W_scaled_blocks = W_scaled.reshape(-1, block_dim).cpu()  # [N_blocks, block_dim]

    # The block scale is the H-scale for the block's input dimensions
    # We need to tile the scale to match block layout
    # scale: [in_f] → repeated for each output row → [N_blocks, block_dim]
    scale_blocks = scale.reshape(-1, block_dim)               # [in_f//block_dim, block_dim]
    scale_blocks = scale_blocks.unsqueeze(0).expand(out_f, -1, -1)  # [out_f, n_blocks_per_row, block_dim]
    scale_blocks = scale_blocks.reshape(-1, block_dim).cpu()  # [N_blocks, block_dim]

    # k-means on scaled blocks
    centroids_scaled, labels = kmeans(W_scaled_blocks, K, n_iter=n_iter)

    # --- Unscale centroids back to original weight space
    # Each centroid represents a block in scaled space.
    # The centroid covers a specific set of input-dimension positions.
    # Since all blocks at the same input-position range share the same scale pattern,
    # we need a per-unique-block-position unscaling.
    #
    # Simpler approach: the centroids are in scaled space, and at reconstruction time
    # we just reconstruct in scaled space and unscale the full matrix at once.
    # This is equivalent and cleaner.

    return {
        "centroids_scaled": centroids_scaled,   # [K, block_dim]  in H-scaled space
        "labels":           labels,             # [N_blocks]
        "scale":            scale.cpu(),        # [in_f]
        "block_dim":        block_dim,
        "W_shape":          W.shape,
        "K":                K,
    }


def hessian_reconstruct(state: dict) -> torch.Tensor:
    """Reconstruct weight matrix from Hessian-weighted codebook."""
    out_f, in_f = state["W_shape"]
    block_dim = state["block_dim"]
    scale = state["scale"]  # [in_f]

    # Reconstruct in scaled space
    centroids_scaled = state["centroids_scaled"]
    labels = state["labels"]
    W_scaled_blocks = centroids_scaled[labels]           # [N_blocks, block_dim]
    W_scaled = W_scaled_blocks.reshape(out_f, in_f)

    # Unscale: W_approx[:, j] = W_scaled[:, j] / scale[j]
    inv_scale = 1.0 / scale.clamp(min=1e-8)
    W_approx = W_scaled * inv_scale.unsqueeze(0)
    return W_approx


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def h_diag_stats(h_diag: torch.Tensor) -> dict:
    """
    Analyse the structure of the diagonal Hessian.

    Key questions:
      - Is H_diag uniform (flat) → no gain from Hessian weighting
      - Is H_diag highly non-uniform → big gain expected
      - What is the dynamic range max/min?
    """
    h = h_diag.float()
    return {
        "mean":        h.mean().item(),
        "std":         h.std().item(),
        "min":         h.min().item(),
        "max":         h.max().item(),
        "dynamic_range_db": 10 * (h.max() / h.clamp(min=1e-12).min()).log10().item(),
        "cv":          (h.std() / h.mean()).item(),   # coefficient of variation
        # fraction of dimensions that carry >10× mean activation
        "hot_fraction": (h > 10 * h.mean()).float().mean().item(),
    }
