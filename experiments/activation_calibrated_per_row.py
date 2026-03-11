"""
Activation-calibrated per-row k-means quantization.

Experiment 19B. Standard per-row k-means (Exp 14/17) treats all blocks in each
output row equally. But input activations are not uniform: some positions in the
hidden state consistently have large magnitude (important to reconstruct precisely)
while others are near-zero (tolerable to reconstruct loosely).

This experiment collects per-block input activation magnitudes from the FP32 model
and uses them to weight k-means: high-activation blocks attract tighter centroids.

Calibration: 20 WikiText-2 test texts, FP32 model only.
Activation weight for block j: mean(||x[:, j*d:(j+1)*d]||^2) across all tokens.

Weighted k-means:
  - Distance: w_j * ||block_j - centroid||^2
  - Centroid update: sum(w_j * block_j) / sum(w_j) for assigned blocks

Configs: K=16, 32, 64 × {unweighted, calibrated}, all bd=8, all-layer.
Reference PPLs (unweighted, from prior experiments):
  K=16 all-layer: 14353.6 (Exp 17A)
  K=64 all-layer:   421.9 (Exp 17C)

Usage:
    .venv/bin/python experiments/activation_calibrated_per_row.py
"""

import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from experiments.eval_perplexity import conv1d_to_linear, eval_perplexity


MODEL_NAME = "gpt2"
N_EVAL     = 50
N_CALIB    = 20
MAX_LEN    = 128
DEVICE     = "cpu"
BLOCK_DIM  = 8


class _ReconLinear(nn.Module):
    """Linear layer with precomputed reconstructed weight matrix."""
    def __init__(self, W_q, bias=None):
        super().__init__()
        self.register_buffer("W_q", W_q.float())
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None

    def forward(self, x):
        return F.linear(x, self.W_q, self.bias)


def collect_act_weights(model, tokenizer, calib_texts, bi, attr, block_dim, device):
    """
    Collect per-block input activation magnitudes for one MLP sublayer.

    Hooks the Conv1D at model.transformer.h[bi].mlp.{attr}, runs all calib_texts
    through the model, and computes per-block L2 magnitude averaged over all tokens.

    Args:
        model:       FP32 GPT-2 (not modified)
        tokenizer:   GPT-2 tokenizer
        calib_texts: list of strings for calibration
        bi:          transformer block index (0-11)
        attr:        "c_fc" or "c_proj"
        block_dim:   block size d
        device:      torch device string

    Returns:
        act_weights: [n_blocks] tensor of per-block importance weights,
                     where n_blocks = in_features // block_dim
    """
    buffers = []

    layer = getattr(model.transformer.h[bi].mlp, attr)

    def hook(m, inp, out):
        # inp[0]: [batch, seq, in_features] input to the Conv1D
        x = inp[0].detach().reshape(-1, inp[0].shape[-1]).cpu()
        buffers.append(x)

    handle = layer.register_forward_hook(hook)

    model.eval()
    with torch.no_grad():
        for text in calib_texts:
            enc = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=MAX_LEN
            )
            model(enc["input_ids"].to(device))

    handle.remove()

    x = torch.cat(buffers, dim=0)  # [N_tokens, in_features]
    in_f = x.shape[1]
    n_blocks = in_f // block_dim

    # Per-block mean squared L2 magnitude across all tokens
    act_weights = torch.zeros(n_blocks)
    for j in range(n_blocks):
        act_weights[j] = x[:, j * block_dim:(j + 1) * block_dim].pow(2).mean()

    return act_weights


def weighted_kmeans(blocks, weights, K, n_iter=50, seed=42):
    """
    Weighted k-means: importance-weighted centroid fitting.

    Assignment: block j → centroid with min weights[j] * ||block_j - centroid||^2
    Update: centroid_k = sum(weights[j] * block_j for j in cluster_k) /
                         sum(weights[j] for j in cluster_k)
    Init: weighted random sampling (prob ∝ weights) without replacement.

    Args:
        blocks:  [N, d] tensor of data points
        weights: [N] tensor of non-negative importance weights
        K:       number of centroids
        n_iter:  Lloyd iterations
        seed:    random seed

    Returns:
        centroids: [K, d]
        labels:    [N] long
    """
    torch.manual_seed(seed)
    N, d = blocks.shape
    assert K <= N, f"K={K} must be <= N={N}"

    # Weighted initialization: sample K distinct points with prob ∝ weights
    probs = weights / (weights.sum() + 1e-12)
    perm = torch.multinomial(probs, K, replacement=False)
    centroids = blocks[perm].clone()

    labels = torch.zeros(N, dtype=torch.long)

    for _ in range(n_iter):
        # [N, K] weighted squared distances
        dist = torch.cdist(blocks, centroids, p=2).pow(2)  # [N, K]
        weighted_dist = dist * weights.unsqueeze(1)         # [N, K]
        labels = weighted_dist.argmin(dim=1)                # [N]

        # Weighted centroid update
        new_centroids = torch.zeros_like(centroids)
        weight_sums   = torch.zeros(K)

        # Accumulate: new_centroids[k] += weights[j] * blocks[j] for all j in cluster k
        new_centroids.scatter_add_(
            0,
            labels.unsqueeze(1).expand(-1, d),
            blocks * weights.unsqueeze(1),
        )
        weight_sums.scatter_add_(0, labels, weights)

        nonempty = weight_sums > 0
        new_centroids[nonempty] /= weight_sums[nonempty].unsqueeze(1)
        new_centroids[~nonempty] = centroids[~nonempty]  # keep old centroid if empty
        centroids = new_centroids

    return centroids, labels


def quantize_per_row(W, b, block_dim, K, act_weights=None):
    """
    Per-row k-means quantization, optionally activation-weighted.

    Each output row W[i] is split into n_blocks = in_features // block_dim blocks.
    If act_weights is provided (shape [n_blocks]), runs weighted_kmeans using those
    weights. Otherwise runs standard unweighted k-means from src/codebook.

    Args:
        W:           [out_features, in_features] weight tensor
        b:           bias tensor or None
        block_dim:   block size d
        K:           codebook size per row
        act_weights: [n_blocks] tensor or None

    Returns:
        _ReconLinear with precomputed W_q
    """
    from src.codebook import kmeans as unweighted_kmeans

    W = W.float().cpu()
    out_f, in_f = W.shape
    assert in_f % block_dim == 0
    n_blocks = in_f // block_dim

    W_q = torch.zeros_like(W)

    for i in range(out_f):
        if i > 0 and i % 500 == 0:
            print(f"    row {i}/{out_f}...", flush=True)

        row_blocks = W[i].reshape(n_blocks, block_dim)  # [n_blocks, block_dim]

        if act_weights is not None:
            centroids_i, labels_i = weighted_kmeans(
                row_blocks, act_weights, K, n_iter=50, seed=42
            )
        else:
            centroids_i, labels_i = unweighted_kmeans(
                row_blocks, K, n_iter=50, seed=42
            )

        W_q[i] = centroids_i[labels_i].reshape(in_f)

    return _ReconLinear(W_q, b)


def build_all_layer_model(base_model, tokenizer, calib_texts, block_dim, K, calibrated):
    """
    Deepcopy base_model and quantize all 24 MLP sublayers.

    If calibrated=True: for each layer, first collect input activation weights
    from the FP32 base_model (not the working copy), then run weighted k-means.
    If calibrated=False: standard unweighted per-row k-means.

    Args:
        base_model:   original FP32 GPT-2 (never modified)
        tokenizer:    GPT-2 tokenizer
        calib_texts:  list of strings for activation collection
        block_dim:    block size d
        K:            codebook size per row
        calibrated:   bool — use activation-weighted k-means?

    Returns:
        Quantized deepcopy of base_model.
    """
    model = copy.deepcopy(base_model)

    for bi in range(12):
        for attr in ["c_fc", "c_proj"]:
            print(f"  Quantizing h{bi}.{attr} (K={K}, cal={calibrated})...", flush=True)

            raw = getattr(model.transformer.h[bi].mlp, attr)
            lin = conv1d_to_linear(raw)
            W = lin.weight.data.clone()
            b = lin.bias.data.clone() if lin.bias is not None else None

            act_weights = None
            if calibrated:
                # Collect activation weights from the FP32 base_model
                act_weights = collect_act_weights(
                    base_model, tokenizer, calib_texts, bi, attr, block_dim, DEVICE
                )

            layer = quantize_per_row(W, b, block_dim, K, act_weights=act_weights)
            setattr(model.transformer.h[bi].mlp, attr, layer)

    return model


def main():
    print(f"Loading {MODEL_NAME} ...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading WikiText-2 ...")
    test_data   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    all_texts   = [t for t in test_data["text"] if len(t.strip()) > 50]
    eval_texts  = all_texts[:N_EVAL]
    calib_texts = all_texts[N_EVAL:N_EVAL + N_CALIB]

    ppl_base = eval_perplexity(base_model, tokenizer, eval_texts, MAX_LEN, DEVICE)
    print(f"FP32 baseline PPL = {ppl_base:.3f}")

    # Configs: (K, calibrated)
    configs = [
        (16, False),
        (16, True),
        (32, False),
        (32, True),
        (64, False),
        (64, True),
    ]

    print(f"\n{'='*65}")
    print("Experiment 19B — Activation-Calibrated Per-Row K-Means")
    print(f"{'='*65}")
    print(f"{'Config':<22}  {'K':>4}  {'cal':>5}  {'PPL':>10}  {'vs uncal':>10}")
    print(f"{'-'*65}")

    ppl_by_k_uncal = {}
    results = []

    for (K, calibrated) in configs:
        label = f"per-row K={K} {'cal' if calibrated else 'uncal'}"
        print(f"\nBuilding: {label} ...", flush=True)

        q_model = build_all_layer_model(
            base_model, tokenizer, calib_texts, BLOCK_DIM, K, calibrated
        )
        ppl = eval_perplexity(q_model, tokenizer, eval_texts, MAX_LEN, DEVICE)
        del q_model

        if not calibrated:
            ppl_by_k_uncal[K] = ppl
            vs_uncal = "—"
        else:
            uncal_ppl = ppl_by_k_uncal.get(K)
            if uncal_ppl is not None:
                ratio = ppl / uncal_ppl
                vs_uncal = f"{ratio:.3f}×"
            else:
                vs_uncal = "?"

        print(
            f"{label:<22}  {K:>4}  {'yes' if calibrated else 'no':>5}  "
            f"{ppl:>10.3f}  {vs_uncal:>10}"
        )
        results.append((label, K, calibrated, ppl))

    print(f"\n{'='*65}")
    print("Summary — Experiment 19B")
    print(f"{'='*65}")
    print(f"  FP32 baseline: {ppl_base:.3f}")
    for label, K, calibrated, ppl in results:
        flag = "CAL" if calibrated else "   "
        print(f"  [{flag}] {label:<22} PPL = {ppl:.3f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
