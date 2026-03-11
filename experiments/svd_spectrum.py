"""
SVD singular value spectrum diagnostic + RMSE sweep.

Key question: does the singular value spectrum of GPT-2 weight matrices
actually decay fast? Contrast with the flat DCT spectrum.

Physics framing:
  σᵢ  = singular values = square-root eigenvalues of W^T W
  σᵢ² = variance explained by mode i

  If σᵢ ~ i^{-α} with large α → fast spectral decay → low effective rank
  → SVD truncation is well-motivated (most of W lives in a small subspace)

Also sweeps rank r and compares:
  - SVD truncation (unquantized) — theoretical ceiling
  - SVD + scalar quantization of U, S, V
  - Scalar codebook baseline (from sweep.py results)

Usage:
    python experiments/svd_spectrum.py
"""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM

from src.spectral import (
    singular_value_spectrum, spectrum_stats,
    svd_quantize, svd_reconstruct,
    svd_codebook_quantize, svd_codebook_reconstruct,
)


MODEL_NAME  = "gpt2"
N_LEVELS_UV = 256    # 8-bit for U and V
N_LEVELS_S  = 65536  # 16-bit for singular values (only r of them)


def load_all_weights(model):
    """Return dict of {layer_name: weight_tensor} for all MLP layers."""
    from transformers.pytorch_utils import Conv1D
    weights = {}
    for i, block in enumerate(model.transformer.h):
        for attr in ["c_fc", "c_proj"]:
            layer = getattr(block.mlp, attr)
            W = layer.weight.data.float()
            if isinstance(layer, Conv1D):
                W = W.T   # Conv1D: [in, out] -> [out, in]
            weights[f"h{i}.{attr}"] = W
    return weights


def print_spectrum_summary(name: str, S: torch.Tensor):
    stats = spectrum_stats(S)
    m = len(S)
    print(f"\n{name}  (rank={m})")
    print(f"  effective rank for 90% variance: {stats['effective_rank_90']:>4}  "
          f"({stats['effective_rank_90']/m*100:.1f}% of full rank)")
    print(f"  effective rank for 99% variance: {stats['effective_rank_99']:>4}  "
          f"({stats['effective_rank_99']/m*100:.1f}% of full rank)")
    print(f"  spectral decay exponent α:       {stats['spectral_decay_alpha']:.3f}")
    # Print first 20 singular values normalised
    S_norm = S / S[0]
    vals = "  ".join(f"{v:.3f}" for v in S_norm[:20].tolist())
    print(f"  σ[0..19]/σ[0]: {vals}")


def rmse_sweep_svd(W: torch.Tensor, name: str):
    """
    For a single weight matrix, sweep rank r and compare:
      (a) SVD truncation only (no quantization) — oracle ceiling
      (b) SVD + scalar quantization
      (c) For reference: what bpw does each rank correspond to?
    """
    m, n = W.shape
    min_dim = min(m, n)

    ranks = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256]
    ranks = [r for r in ranks if r <= min_dim]

    print(f"\n{'='*72}")
    print(f"RMSE sweep: {name}  [{m} x {n}]")
    print(f"{'='*72}")
    print(f"{'rank':>6}  {'bpw':>6}  {'var%':>6}  {'RMSE_trunc':>12}  {'RMSE_svdq':>12}")
    print(f"{'-'*72}")

    # Precompute full SVD once
    U_full, S_full, Vh_full = torch.linalg.svd(W.float(), full_matrices=False)

    results = []
    for r in ranks:
        # (a) truncation only (no quantization)
        W_trunc = (U_full[:, :r] * S_full[:r].unsqueeze(0)) @ Vh_full[:r, :]
        rmse_trunc = (W - W_trunc).pow(2).mean().sqrt().item()

        # (b) SVD + scalar quantization
        state = svd_quantize(W, rank=r, n_levels_UV=N_LEVELS_UV, n_levels_S=N_LEVELS_S)
        W_svdq = svd_reconstruct(state)
        rmse_svdq = (W - W_svdq).pow(2).mean().sqrt().item()
        bpw = state["bpw"]
        var_pct = state["var_explained"] * 100

        print(f"{r:>6}  {bpw:>6.3f}  {var_pct:>5.1f}%  {rmse_trunc:>12.6f}  {rmse_svdq:>12.6f}")
        results.append((r, bpw, var_pct, rmse_trunc, rmse_svdq))

    return results


def compare_at_same_bpw(W: torch.Tensor):
    """
    At ~0.5 bpw, compare SVD-quant RMSE vs scalar codebook RMSE.
    The scalar codebook result comes from K=256, block_dim=16.
    """
    from src.codebook import quantize_blocks, reconstruct

    print(f"\n{'='*55}")
    print(f"Head-to-head at ~0.5 bpw")
    print(f"{'='*55}")

    # SVD: find rank that gives ~0.5 bpw
    m, n = W.shape
    # bpw(r) = r*(m+n)*log2(n_levels_UV) / (m*n)  (ignoring tiny S term)
    # → r ≈ 0.5 * m*n / ((m+n)*log2(n_levels_UV))
    bits_per_rank = (m + n) * math.log2(N_LEVELS_UV)
    r_target = max(1, round(0.5 * m * n / bits_per_rank))

    state = svd_quantize(W, rank=r_target, n_levels_UV=N_LEVELS_UV, n_levels_S=N_LEVELS_S)
    W_svdq = svd_reconstruct(state)
    rmse_svdq = (W - W_svdq).pow(2).mean().sqrt().item()
    print(f"  SVD-quant (r={r_target}):        bpw={state['bpw']:.3f}  RMSE={rmse_svdq:.6f}  var={state['var_explained']*100:.1f}%")

    # Scalar codebook K=256, block_dim=16 → bpw=0.5
    centroids, labels, _ = quantize_blocks(W, block_dim=16, K=256, n_iter=30)
    W_scalar = reconstruct(centroids, labels, W.shape)
    rmse_scalar = (W.cpu() - W_scalar).pow(2).mean().sqrt().item()
    print(f"  Scalar codebook (K=256,d=16):    bpw=0.500  RMSE={rmse_scalar:.6f}")

    winner = "SVD-quant" if rmse_svdq < rmse_scalar else "Scalar codebook"
    print(f"  Winner: {winner}")


def main():
    print(f"Loading {MODEL_NAME} ...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    weights = load_all_weights(model)

    # --- 1. Spectrum summary across all layers
    print("\n\n" + "="*65)
    print("SINGULAR VALUE SPECTRUM SUMMARY  (all MLP layers)")
    print("="*65)
    print(f"{'layer':>14}  {'shape':>12}  {'r90':>5}  {'r90%':>6}  {'α':>7}")
    print("-"*65)

    for name, W in weights.items():
        S = singular_value_spectrum(W)
        stats = spectrum_stats(S)
        m, n = W.shape
        r90 = stats["effective_rank_90"]
        alpha = stats["spectral_decay_alpha"]
        print(f"{name:>14}  {str(tuple(W.shape)):>12}  {r90:>5}  {r90/min(m,n)*100:>5.1f}%  {alpha:>7.3f}")

    # --- 2. Detailed spectrum for one representative layer
    W0 = weights["h0.c_fc"]
    S0 = singular_value_spectrum(W0)
    print_spectrum_summary("h0.c_fc", S0)

    # Side-by-side: SVD spectrum vs DCT spectrum (the contrast)
    from src.frequency import dct_energy_spectrum
    dct_power = dct_energy_spectrum(W0, block_dim=16)
    dct_norm  = (dct_power / dct_power.sum()).tolist()
    S0_norm   = (S0[:16] / S0[:16].sum()).tolist()

    print(f"\n{'k':>4}  {'SVD σₖ²/Σ':>12}  {'DCT pₖ/Σ':>12}  (first 16 modes)")
    print("-"*42)
    for k, (sv, dct) in enumerate(zip(S0_norm, dct_norm)):
        sv_bar  = "█" * int(sv * 60)
        dct_bar = "░" * int(dct * 60)
        print(f"{k:>4}  {sv:>11.4f}  {dct:>11.4f}  {sv_bar}{dct_bar}")

    # --- 3. RMSE sweep on layer 0
    results = rmse_sweep_svd(W0, "h0.c_fc")

    # --- 4. Head-to-head vs scalar codebook at same bpw
    from src.codebook import quantize_blocks, reconstruct
    compare_at_same_bpw(W0)


if __name__ == "__main__":
    main()
