"""
Block-local H-weighted k-means quantization.

Experiment 15. Global H-weighted k-means (Exp 3-5b) catastrophically failed:
PPL=300,000+ despite better H-RMSE. Root cause: H_diag[j] range is [0.005, 0.991]
(14× max/min ratio). Hot dimensions dominate centroid placement; cold dimensions
get near-zero centroid coverage; reconstruction errors in cold dims devastate PPL.

Fix: block-local H normalization. For each block of block_dim dimensions:
  1. Normalize H_diag within the block to unit mean
  2. Apply sqrt(H_norm) scale within the block
  3. Run k-means on scaled blocks (global codebook, but locally-normalized scale)
  4. Unscale at reconstruction

Within each block, hot dims still get more centroid attention than cold dims
(the relative ordering is preserved). But the cross-block scale ratio is bounded
to the within-block variation of H_diag — much smaller than the global 14×.

Comparison includes global H-weighted (config E) to confirm this experiment
replicates the known failure mode, validating the block-local fix is genuine.

Target: h0.c_fc. Compare to:
  flat bd=16 K=256: PPL=381 (Exp 6)
  flat bd=8  K=16:  PPL=154 (Exp 8, current best)
  global H (Exp 3):  PPL=300,000+ (known failure)

Usage:
    .venv/bin/python experiments/block_local_hessian.py
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "gpt2"
N_CALIB    = 100
N_EVAL     = 150
MAX_LEN    = 128
DEVICE     = "cpu"


# ---------------------------------------------------------------------------
# Helper: reconstructed weight linear layer
# ---------------------------------------------------------------------------

class _ReconLinear(nn.Module):
    """Linear layer with precomputed reconstructed weight matrix."""
    def __init__(self, W_q, bias=None):
        super().__init__()
        self.register_buffer("W_q", W_q.float())
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None

    def forward(self, x):
        return F.linear(x, self.W_q, self.bias)


# ---------------------------------------------------------------------------
# Quantization functions
# ---------------------------------------------------------------------------

def quantize_flat(W, b, block_dim, K, n_iter=50):
    """
    Standard flat k-means block quantization.

    Baseline identical to Exp 6 (bd=16, K=256) and Exp 8 (bd=8, K=16).

    Args:
        W:         [out_f, in_f] weight matrix
        b:         bias tensor or None
        block_dim: block size (must divide in_f)
        K:         codebook size
        n_iter:    Lloyd iterations

    Returns:
        _ReconLinear with quantized weights
    """
    centroids, labels, _ = quantize_blocks(W, block_dim, K, n_iter=n_iter)
    W_q = reconstruct(centroids, labels, W.shape)
    return _ReconLinear(W_q, b)


def quantize_block_local_h(W, b, h_diag, block_dim, K, n_iter=50):
    """
    Block-local H-weighted k-means quantization.

    Key idea: instead of globally scaling by sqrt(H_diag[j]), normalize H_diag
    within each block of block_dim consecutive input dimensions, then scale.

    Within each block, the relative importance ordering of dimensions is preserved
    (hot dims within the block still attract more centroid mass than cold dims in
    the same block). But the cross-block scale ratio is eliminated: all blocks
    enter k-means at the same overall magnitude, so no block can starve another
    of centroid coverage.

    Within-block H_diag variation is much smaller than the global 14× ratio.
    Consecutive dims in GPT-2 tend to have similar H_diag magnitudes; the worst
    case within-block ratio is governed by local H_diag structure, not the global
    hot/cold split.

    Algorithm:
      For each block position b covering dims [b*bd : (b+1)*bd]:
        1. h_block = H_diag[b*bd : (b+1)*bd]             shape [bd]
        2. h_norm  = h_block / h_block.mean().clamp(1e-8) mean=1.0 within block
        3. s       = h_norm.sqrt()                         shape [bd]
        4. scale all weight rows in that block: w_scaled = w_block * s

      Run global k-means on all scaled blocks (single shared codebook).
      Reconstruct: w_block_q = centroids[label] / s

    Args:
        W:         [out_f, in_f] weight matrix
        b:         bias tensor or None
        h_diag:    [in_f] diagonal Hessian H_diag[j] = E[x_j^2]
        block_dim: block size (must divide in_f)
        K:         codebook size
        n_iter:    Lloyd iterations

    Returns:
        _ReconLinear with quantized weights
    """
    out_f, in_f = W.shape
    assert in_f % block_dim == 0, f"in_f={in_f} must be divisible by block_dim={block_dim}"
    n_blocks_per_row = in_f // block_dim

    # Compute per-block scale: normalize H_diag within each block
    h_blocks = h_diag.reshape(n_blocks_per_row, block_dim).float()  # [n_blocks_per_row, block_dim]
    block_means = h_blocks.mean(dim=1, keepdim=True).clamp(min=1e-8)  # [n_blocks_per_row, 1]
    h_norm = h_blocks / block_means                                   # [n_blocks_per_row, block_dim], each row has mean=1.0
    s = h_norm.sqrt()                                                  # [n_blocks_per_row, block_dim]

    # Expand scale to all output rows: [out_f * n_blocks_per_row, block_dim]
    # s[b] applies to every output row for block-position b
    s_expanded = s.unsqueeze(0).expand(out_f, -1, -1).reshape(-1, block_dim)

    # Reshape W into blocks and scale
    W_blocks = W.float().reshape(-1, block_dim)  # [out_f * n_blocks_per_row, block_dim]
    W_scaled = W_blocks * s_expanded

    # k-means on scaled blocks — single global codebook across all scaled blocks
    centroids_scaled, labels = kmeans(W_scaled, K, n_iter=n_iter)

    # Reconstruct and unscale
    W_q_scaled = centroids_scaled[labels]        # [out_f * n_blocks_per_row, block_dim]
    W_q_blocks = W_q_scaled / s_expanded         # unscale back to original weight space
    W_q = W_q_blocks.reshape(out_f, in_f)

    return _ReconLinear(W_q, b)


def quantize_global_h(W, b, h_diag, block_dim, K, n_iter=50):
    """
    Global H-weighted k-means quantization (known failure, config E).

    Scales each input dimension j by sqrt(H_diag[j]) globally before k-means,
    then unscales at reconstruction. This is the approach from Exp 3-5b that
    produced PPL=300,000+.

    Included as a control to:
      1. Confirm this experiment reproduces the Exp 3-5b failure mode
      2. Provide a direct contrast to the block-local variant (configs C/D)

    Root cause of failure: H_diag range [0.005, 0.991] gives sqrt scale range
    ~[0.07, 0.99], a ~14× ratio. Hot dimensions compress into a narrow band in
    scaled space while cold dimensions spread over a wide range. The k-means
    codebook concentrates centroids on hot-scaled clusters; cold dimensions
    receive poor centroid coverage. At reconstruction (divide by sqrt(H_diag[j]))
    the small cold-dim errors in scaled space are amplified back to large errors
    in weight space, which devastates perplexity.

    Args:
        W:         [out_f, in_f] weight matrix
        b:         bias tensor or None
        h_diag:    [in_f] diagonal Hessian H_diag[j] = E[x_j^2]
        block_dim: block size (must divide in_f)
        K:         codebook size
        n_iter:    Lloyd iterations

    Returns:
        _ReconLinear with quantized weights
    """
    out_f, in_f = W.shape
    assert in_f % block_dim == 0, f"in_f={in_f} must be divisible by block_dim={block_dim}"
    n_blocks_per_row = in_f // block_dim

    # Global scale: sqrt(H_diag[j]) for each input dim j
    s = h_diag.float().clamp(min=1e-8).sqrt()  # [in_f]

    # Scale weight matrix column-wise
    W_scaled = W.float() * s.unsqueeze(0)  # [out_f, in_f]

    # Reshape into blocks
    W_blocks = W_scaled.reshape(-1, block_dim)  # [out_f * n_blocks_per_row, block_dim]

    # Build per-block scale tensor for reconstruction (matching block layout)
    # s: [in_f] → reshape to [n_blocks_per_row, block_dim] → expand for all output rows
    s_blocks = s.reshape(n_blocks_per_row, block_dim)                        # [n_blocks_per_row, block_dim]
    s_expanded = s_blocks.unsqueeze(0).expand(out_f, -1, -1).reshape(-1, block_dim)  # [out_f * n_blocks_per_row, block_dim]

    # k-means on globally-scaled blocks
    centroids_scaled, labels = kmeans(W_blocks, K, n_iter=n_iter)

    # Reconstruct and unscale
    W_q_scaled = centroids_scaled[labels]        # [out_f * n_blocks_per_row, block_dim]
    W_q_blocks = W_q_scaled / s_expanded         # unscale: divide by sqrt(H_diag[j])
    W_q = W_q_blocks.reshape(out_f, in_f)

    return _ReconLinear(W_q, b)


# ---------------------------------------------------------------------------
# H-RMSE diagnostic
# ---------------------------------------------------------------------------

def h_rmse(W: torch.Tensor, W_q: torch.Tensor, h_diag: torch.Tensor) -> float:
    """H-weighted RMSE: sqrt(mean((W - W_q)^2 * H_diag_j))."""
    err = (W.float() - W_q.float()) ** 2
    return (err * h_diag.float().unsqueeze(0)).mean().sqrt().item()


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

    # Target layer: h0.c_fc
    raw_layer     = model.transformer.h[0].mlp.c_fc
    target_linear = conv1d_to_linear(raw_layer)
    W = target_linear.weight.data.clone()   # [out_f, in_f] = [3072, 768]
    b = target_linear.bias.data.clone() if target_linear.bias is not None else None

    out_f, in_f = W.shape
    print(f"Target: h0.c_fc  {tuple(W.shape)}")

    # --- Baseline (full precision)
    print(f"\n[baseline] Full precision ...")
    ppl_base = eval_perplexity(model, tokenizer, eval_texts, MAX_LEN, DEVICE)
    print(f"  PPL = {ppl_base:.3f}")

    # --- H_diag calibration
    print(f"\nCalibrating H_diag ({N_CALIB} texts) ...")
    h_diag = estimate_h_diag(model, tokenizer, raw_layer, calib_texts, MAX_LEN, DEVICE)
    h_min  = h_diag.min().item()
    h_max  = h_diag.max().item()
    h_cv   = (h_diag.std() / h_diag.mean()).item()
    h_ratio = h_max / max(h_min, 1e-12)
    print(f"  H_diag: min={h_min:.4f}  max={h_max:.4f}  ratio={h_ratio:.1f}x  CV={h_cv:.3f}")

    # -----------------------------------------------------------------------
    # Configs to test
    # -----------------------------------------------------------------------
    configs = [
        # (label, method,           bd,  K,   description)
        ("A", "flat",          16, 256, "flat          bd=16 K=256  — reference (Exp 6)"),
        ("B", "flat",           8,  16, "flat          bd=8  K=16   — current best (Exp 8)"),
        ("C", "block_local_h", 16, 256, "block-local-H bd=16 K=256  — block-local fix of A"),
        ("D", "block_local_h",  8,  16, "block-local-H bd=8  K=16   — block-local fix of B"),
        ("E", "global_h",      16, 256, "global-H      bd=16 K=256  — known failure (Exp 3)"),
    ]

    results = []

    for cfg_label, method, bd, K, desc in configs:
        bpw = (K.bit_length() - 1) / bd
        print(f"\n[{cfg_label}] {desc}  bpw={bpw:.3f}")

        if method == "flat":
            q_layer = quantize_flat(W, b, bd, K)
        elif method == "block_local_h":
            q_layer = quantize_block_local_h(W, b, h_diag, bd, K)
        elif method == "global_h":
            q_layer = quantize_global_h(W, b, h_diag, bd, K)
        else:
            raise ValueError(f"Unknown method: {method}")

        # H-RMSE diagnostic
        W_q = q_layer.W_q
        hr  = h_rmse(W, W_q, h_diag)

        # PPL
        model_q = copy.deepcopy(model)
        model_q.transformer.h[0].mlp.c_fc = q_layer
        ppl = eval_perplexity(model_q, tokenizer, eval_texts, MAX_LEN, DEVICE)
        del model_q

        results.append((cfg_label, desc, bd, K, bpw, ppl, hr))
        print(f"  PPL = {ppl:.3f}  H-RMSE = {hr:.6f}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    # Reference for delta/improvement: config A (flat bd=16 K=256)
    ppl_ref_A = next(r[5] for r in results if r[0] == "A")

    print(f"\n{'='*85}")
    print(f"Experiment 15 Summary  —  Block-local H-weighted k-means  (h0.c_fc)")
    print(f"{'='*85}")
    print(f"  Baseline (FP32):  PPL = {ppl_base:.3f}")
    print()
    print(f"  {'Cfg':<4} {'bd':>4} {'K':>5} {'bpw':>6}  {'PPL':>10}  {'H-RMSE':>10}  "
          f"{'delta vs A':>12}  {'% vs A':>8}")
    print(f"  {'-'*4} {'-'*4} {'-'*5} {'-'*6}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*8}")

    for cfg_label, desc, bd, K, bpw, ppl, hr in results:
        delta = ppl - ppl_ref_A
        pct   = -delta / ppl_ref_A * 100  # positive = improvement over A
        print(f"  {cfg_label:<4} {bd:>4} {K:>5} {bpw:>6.3f}  {ppl:>10.3f}  {hr:>10.6f}  "
              f"{delta:>+12.3f}  {pct:>+7.1f}%")

    print()
    print(f"  Prior results (from experiment log):")
    print(f"    Exp 6  flat  bd=16 K=256 bpw=0.500: PPL=380.618")
    print(f"    Exp 8  flat  bd=8  K=16  bpw=0.500: PPL=154.340  (current best single-layer)")
    print(f"    Exp 10 SmoothQuant α=0.5 bd=16 K=256 bpw=0.500: PPL=169.704")
    print(f"    Exp 3-5b global H-weighted: PPL=300,000+  (catastrophic)")

    print()
    # Highlight whether block-local-H improved over flat at matching configs
    for flat_lbl, blh_lbl in [("A", "C"), ("B", "D")]:
        ppl_flat = next(r[5] for r in results if r[0] == flat_lbl)
        ppl_blh  = next(r[5] for r in results if r[0] == blh_lbl)
        delta    = ppl_blh - ppl_flat
        pct      = -delta / ppl_flat * 100
        sign     = "IMPROVEMENT" if delta < 0 else "REGRESSION"
        print(f"  Block-local-H ({blh_lbl}) vs flat ({flat_lbl}):  "
              f"delta={delta:+.3f}  ({pct:+.1f}%)  [{sign}]")

    # Confirm global-H failure
    ppl_global_h = next(r[5] for r in results if r[0] == "E")
    print(f"\n  Global-H (E): PPL={ppl_global_h:.3f}  "
          f"({'FAILURE confirmed' if ppl_global_h > 10_000 else 'unexpected — check implementation'})")


if __name__ == "__main__":
    main()
