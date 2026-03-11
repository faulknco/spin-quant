"""
Hessian-weighted codebook experiment.

Single-layer, lightweight. Runs in under a minute.

Compares on h0.c_fc only:
  1. Flat k-means (baseline)
  2. Hessian-weighted k-means

Reports both flat RMSE and H-weighted RMSE for a complete picture.

Usage:
    python experiments/hessian_codebook.py
"""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.codebook import quantize_blocks, reconstruct, quantization_rmse
from src.hessian import estimate_h_diag, hessian_quantize, hessian_reconstruct, h_diag_stats
from experiments.eval_perplexity import conv1d_to_linear


MODEL_NAME = "gpt2"
BLOCK_DIM  = 16
K_LIST     = [16, 64, 256, 1024]
N_CALIB    = 100    # calibration texts
MAX_LEN    = 128
DEVICE     = "cpu"


def h_weighted_rmse(W: torch.Tensor, Wq: torch.Tensor, h_diag: torch.Tensor) -> float:
    """
    Hessian-weighted RMSE: sqrt(mean over j of H_diag[j] * mean_i((W-Wq)[:,j]²))
    This is the metric we actually care about — it weights errors by input importance.
    """
    h = h_diag.float().to(W.device)
    diff = (W.float() - Wq.float())           # [out, in]
    col_mse = diff.pow(2).mean(dim=0)         # [in]   mean over output rows
    h_rmse = (h * col_mse).mean().sqrt()
    return h_rmse.item()


def main():
    print(f"Loading {MODEL_NAME} ...")
    model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Target layer
    target_layer_raw = model.transformer.h[0].mlp.c_fc
    target_linear    = conv1d_to_linear(target_layer_raw)
    W = target_linear.weight.data.clone()
    print(f"Target layer: h0.c_fc  shape={tuple(W.shape)}")

    # --- Calibration: estimate H_diag
    print(f"\nRunning calibration ({N_CALIB} texts, max_len={MAX_LEN}) ...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_texts = [t for t in dataset["text"] if len(t.strip()) > 50][:N_CALIB]

    h_diag = estimate_h_diag(
        model, tokenizer,
        layer_module=target_layer_raw,   # hook on the original Conv1D
        texts=calib_texts,
        max_length=MAX_LEN,
        device=DEVICE,
    )
    print(f"H_diag shape: {h_diag.shape}")

    stats = h_diag_stats(h_diag)
    print(f"\n--- H_diag statistics ---")
    print(f"  mean:             {stats['mean']:.4f}")
    print(f"  std:              {stats['std']:.4f}")
    print(f"  min:              {stats['min']:.6f}")
    print(f"  max:              {stats['max']:.4f}")
    print(f"  dynamic range:    {stats['dynamic_range_db']:.1f} dB")
    print(f"  coeff variation:  {stats['cv']:.3f}  (0=uniform, >>1=highly non-uniform)")
    print(f"  'hot' dims (>10x mean): {stats['hot_fraction']*100:.1f}%")

    print("\n" + "="*72)
    print(f"Quantization comparison  (block_dim={BLOCK_DIM})")
    print("="*72)
    print(f"{'K':>6}  {'bpw':>5}  "
          f"{'flat_RMSE':>12}  {'h_RMSE_flat':>12}  "
          f"{'h_RMSE_hq':>12}  {'gain%':>7}")
    print("-"*72)

    for K in K_LIST:
        bpw = math.log2(K) / BLOCK_DIM

        # --- (a) Flat k-means
        centroids_flat, labels_flat, _ = quantize_blocks(W, BLOCK_DIM, K, n_iter=40)
        Wq_flat = reconstruct(centroids_flat, labels_flat, W.shape)

        flat_rmse   = (W.float() - Wq_flat).pow(2).mean().sqrt().item()
        h_rmse_flat = h_weighted_rmse(W, Wq_flat, h_diag)

        # --- (b) Hessian-weighted k-means
        state_hq = hessian_quantize(W, h_diag, BLOCK_DIM, K, n_iter=40)
        Wq_hq    = hessian_reconstruct(state_hq)

        h_rmse_hq = h_weighted_rmse(W, Wq_hq, h_diag)

        # gain: how much better is H-weighted on the H-weighted metric?
        gain_pct = (h_rmse_flat - h_rmse_hq) / h_rmse_flat * 100

        print(f"{K:>6}  {bpw:>5.3f}  "
              f"{flat_rmse:>12.6f}  {h_rmse_flat:>12.6f}  "
              f"{h_rmse_hq:>12.6f}  {gain_pct:>6.1f}%")

    print("\nColumns:")
    print("  flat_RMSE   = ||W - Wq||² (uniform, what k-means minimises)")
    print("  h_RMSE_flat = ||H^½(W - Wq)||² for flat k-means (what we care about)")
    print("  h_RMSE_hq   = ||H^½(W - Wq)||² for Hessian-weighted k-means")
    print("  gain%       = reduction in h_RMSE from using Hessian weighting")


if __name__ == "__main__":
    main()
