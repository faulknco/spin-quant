"""
DCT power spectrum diagnostic.

The core question before running full DCT quantization:
  Do GPT-2 weight blocks actually have low-frequency structure?
  i.e. does the DCT power spectrum decay with mode index k?

If yes → DCT quantization is well-motivated (low-k modes carry the weight).
If no  → weight blocks look like white noise in DCT space → no gain over scalar codebook.

Also runs a sweep over keep_k (number of retained DCT coefficients) to show
RMSE vs bits-per-weight for the DCT scheme vs the scalar codebook baseline.

Usage:
    python experiments/dct_spectrum.py
"""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM
from src.frequency import dct_energy_spectrum, dct_quantize, dct_reconstruct, wht_quantize, wht_reconstruct
from src.codebook import quantize_blocks, reconstruct


MODEL_NAME = "gpt2"
BLOCK_DIM  = 16   # must divide in_features=768 for GPT-2 c_fc
N_LEVELS   = 256  # 8-bit coefficient quantization


def load_weight(layer_idx: int = 0, attr: str = "c_fc") -> torch.Tensor:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    layer = getattr(model.transformer.h[layer_idx].mlp, attr)
    # GPT-2 uses Conv1D: weight is [in, out] → transpose to [out, in]
    W = layer.weight.data
    if W.shape[0] < W.shape[1]:   # Conv1D layout
        W = W.T
    return W.clone()


def print_spectrum(W: torch.Tensor, block_dim: int = BLOCK_DIM, label: str = ""):
    power = dct_energy_spectrum(W, block_dim)   # [d]
    total = power.sum().item()
    cumulative = power.cumsum(0) / total

    print(f"\n{'='*55}")
    print(f"DCT Power Spectrum  {label}")
    print(f"{'='*55}")
    print(f"{'k':>4}  {'power':>10}  {'% total':>8}  {'cumul%':>8}")
    print(f"{'-'*55}")
    for k, (p, c) in enumerate(zip(power.tolist(), cumulative.tolist())):
        bar = "#" * int(p / total * 40)
        print(f"{k:>4}  {p:>10.5f}  {p/total*100:>7.2f}%  {c*100:>7.2f}%  {bar}")

    k50 = (cumulative >= 0.50).nonzero(as_tuple=True)[0][0].item()
    k90 = (cumulative >= 0.90).nonzero(as_tuple=True)[0][0].item()
    k99 = (cumulative >= 0.99).nonzero(as_tuple=True)[0][0].item()
    print(f"\n50% energy in k=0..{k50}  ({k50+1}/{block_dim} modes, {(k50+1)/block_dim*100:.0f}%)")
    print(f"90% energy in k=0..{k90}  ({k90+1}/{block_dim} modes, {(k90+1)/block_dim*100:.0f}%)")
    print(f"99% energy in k=0..{k99}  ({k99+1}/{block_dim} modes, {(k99+1)/block_dim*100:.0f}%)")


def rmse_sweep(W: torch.Tensor, block_dim: int = BLOCK_DIM):
    """Compare DCT, WHT, and scalar codebook across a range of bits-per-weight."""
    print(f"\n{'='*65}")
    print(f"RMSE vs bits-per-weight  (block_dim={block_dim})")
    print(f"{'='*65}")
    print(f"{'scheme':>18}  {'keep_k / K':>12}  {'bpw':>6}  {'RMSE':>10}")
    print(f"{'-'*65}")

    results = []

    # DCT: vary keep_k from 1 to block_dim
    for keep_k in [1, 2, 4, 6, 8, 10, 12, 14, 16]:
        if keep_k > block_dim:
            continue
        state = dct_quantize(W, block_dim, keep_k, N_LEVELS)
        Wq    = dct_reconstruct(state)
        rmse  = (W.float() - Wq).pow(2).mean().sqrt().item()
        bpw   = state["bpw"]
        print(f"{'DCT':>18}  {f'keep_k={keep_k}':>12}  {bpw:>6.3f}  {rmse:>10.6f}")
        results.append(("DCT", keep_k, bpw, rmse))

    # WHT: same sweep (block_dim must be power of 2)
    if block_dim & (block_dim - 1) == 0:
        for keep_k in [1, 2, 4, 6, 8, 10, 12, 14, 16]:
            if keep_k > block_dim:
                continue
            state = wht_quantize(W, block_dim, keep_k, N_LEVELS)
            Wq    = wht_reconstruct(state)
            rmse  = (W.float() - Wq).pow(2).mean().sqrt().item()
            bpw   = state["bpw"]
            print(f"{'WHT':>18}  {f'keep_k={keep_k}':>12}  {bpw:>6.3f}  {rmse:>10.6f}")
            results.append(("WHT", keep_k, bpw, rmse))

    # Scalar codebook baseline: vary K for comparable bpw
    # bpw = log2(K) / block_dim  →  K = 2^(bpw * block_dim)
    for K in [4, 16, 64, 256, 1024, 4096, 16384, 65536]:
        bpw = math.log2(K) / block_dim
        if bpw > 10:
            break
        centroids, labels, _ = quantize_blocks(W, block_dim, K, n_iter=30)
        Wq = reconstruct(centroids, labels, W.shape)
        rmse = (W.float().cpu() - Wq).pow(2).mean().sqrt().item()
        print(f"{'Scalar codebook':>18}  {f'K={K}':>12}  {bpw:>6.3f}  {rmse:>10.6f}")
        results.append(("Scalar", K, bpw, rmse))

    return results


def main():
    print(f"Loading {MODEL_NAME} layer 0 c_fc ...")
    W = load_weight(layer_idx=0, attr="c_fc")
    print(f"Weight shape: {W.shape}")

    # --- Spectrum diagnostic
    print_spectrum(W, BLOCK_DIM, label=f"GPT-2 h[0].mlp.c_fc  (block_dim={BLOCK_DIM})")

    # --- Spectrum across layers (first 6)
    print("\n\nSpectrum summary across layers (% energy in first 4 DCT modes):")
    print(f"{'layer':>8}  {'attr':>8}  {'top-4 energy%':>14}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    for i in range(min(6, len(model.transformer.h))):
        for attr in ["c_fc", "c_proj"]:
            layer = getattr(model.transformer.h[i].mlp, attr)
            W_ = layer.weight.data
            if W_.shape[0] < W_.shape[1]:
                W_ = W_.T
            power = dct_energy_spectrum(W_, BLOCK_DIM)
            top4_pct = (power[:4].sum() / power.sum() * 100).item()
            print(f"{i:>8}  {attr:>8}  {top4_pct:>13.1f}%")

    # --- RMSE sweep
    W = load_weight(0, "c_fc")
    results = rmse_sweep(W, BLOCK_DIM)

    # --- Summary: at 0.5 bpw, which scheme wins?
    print("\n\nAt ~0.5 bpw, RMSE comparison:")
    target_bpw = 0.5
    for scheme, _, bpw, rmse in results:
        if abs(bpw - target_bpw) < 0.1:
            print(f"  {scheme:>18}  bpw={bpw:.3f}  RMSE={rmse:.6f}")


if __name__ == "__main__":
    main()
