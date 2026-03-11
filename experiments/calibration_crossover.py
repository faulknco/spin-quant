"""
Activation calibration crossover sweep.

Experiment 20A. Exp 19B showed activation-weighted k-means hurts at K=32
(2.6× worse) but helps at K=64 (27% better). Where exactly does the crossover
happen? This experiment sweeps K=40, 48, 56, 64, 80, 96, 128 paired with
uncalibrated and calibrated to find the crossover K.

Also extends the K=128 data point (19A only measured uncal K=128).

Usage:
    .venv/bin/python experiments/calibration_crossover.py
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
    def __init__(self, W_q, bias=None):
        super().__init__()
        self.register_buffer("W_q", W_q.float())
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None

    def forward(self, x):
        return F.linear(x, self.W_q, self.bias)


def collect_act_weights(model, tokenizer, calib_texts, bi, attr, block_dim, device):
    buffers = []
    layer = getattr(model.transformer.h[bi].mlp, attr)

    def hook(m, inp, out):
        x = inp[0].detach().reshape(-1, inp[0].shape[-1]).cpu()
        buffers.append(x)

    handle = layer.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for text in calib_texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
            model(enc["input_ids"].to(device))
    handle.remove()

    x = torch.cat(buffers, dim=0)
    in_f = x.shape[1]
    n_blocks = in_f // block_dim
    act_weights = torch.zeros(n_blocks)
    for j in range(n_blocks):
        act_weights[j] = x[:, j * block_dim:(j + 1) * block_dim].pow(2).mean()
    return act_weights


def weighted_kmeans(blocks, weights, K, n_iter=50, seed=42):
    torch.manual_seed(seed)
    N, d = blocks.shape
    K_eff = min(K, N)
    probs = weights / (weights.sum() + 1e-12)
    perm = torch.multinomial(probs, K_eff, replacement=False)
    centroids = blocks[perm].clone()
    labels = torch.zeros(N, dtype=torch.long)
    for _ in range(n_iter):
        dist = torch.cdist(blocks, centroids, p=2).pow(2)
        weighted_dist = dist * weights.unsqueeze(1)
        labels = weighted_dist.argmin(dim=1)
        new_centroids = torch.zeros_like(centroids)
        weight_sums = torch.zeros(K_eff)
        new_centroids.scatter_add_(0, labels.unsqueeze(1).expand(-1, d),
                                   blocks * weights.unsqueeze(1))
        weight_sums.scatter_add_(0, labels, weights)
        nonempty = weight_sums > 0
        new_centroids[nonempty] /= weight_sums[nonempty].unsqueeze(1)
        new_centroids[~nonempty] = centroids[~nonempty]
        centroids = new_centroids
    return centroids, labels


def quantize_per_row(W, b, block_dim, K, act_weights=None):
    from src.codebook import kmeans as unweighted_kmeans
    W = W.float().cpu()
    out_f, in_f = W.shape
    n_blocks = in_f // block_dim
    K_eff = min(K, n_blocks)
    W_q = torch.zeros_like(W)
    for i in range(out_f):
        row_blocks = W[i].reshape(n_blocks, block_dim)
        if act_weights is not None:
            centroids_i, labels_i = weighted_kmeans(row_blocks, act_weights, K_eff)
        else:
            centroids_i, labels_i = unweighted_kmeans(row_blocks, K_eff, n_iter=50, seed=42)
        W_q[i] = centroids_i[labels_i].reshape(in_f)
    return _ReconLinear(W_q, b)


def build_all_layer_model(base_model, tokenizer, calib_texts, block_dim, K, calibrated):
    model = copy.deepcopy(base_model)
    for bi in range(12):
        for attr in ["c_fc", "c_proj"]:
            raw = getattr(model.transformer.h[bi].mlp, attr)
            lin = conv1d_to_linear(raw)
            W = lin.weight.data.clone()
            b = lin.bias.data.clone() if lin.bias is not None else None
            act_weights = None
            if calibrated:
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

    # K values to sweep: bracketing the K=32→64 crossover plus high-K extension
    K_values = [40, 48, 56, 64, 80, 96, 128]

    print(f"\n{'='*70}")
    print("Experiment 20A — Calibration Crossover Sweep")
    print(f"{'='*70}")
    print(f"{'Config':<24}  {'K':>4}  {'cal':>5}  {'PPL':>10}  {'vs uncal':>10}")
    print(f"{'-'*70}")

    ppl_uncal = {}
    for K in K_values:
        for calibrated in [False, True]:
            label = f"K={K} {'cal' if calibrated else 'uncal'}"
            print(f"  Running {label}...", flush=True)
            q_model = build_all_layer_model(
                base_model, tokenizer, calib_texts, BLOCK_DIM, K, calibrated
            )
            ppl = eval_perplexity(q_model, tokenizer, eval_texts, MAX_LEN, DEVICE)
            del q_model

            if not calibrated:
                ppl_uncal[K] = ppl
                vs = "—"
            else:
                ratio = ppl / ppl_uncal[K] if K in ppl_uncal else float("nan")
                vs = f"{ratio:.3f}×"

            print(
                f"{'per-row ' + label:<24}  {K:>4}  {'yes' if calibrated else 'no':>5}  "
                f"{ppl:>10.3f}  {vs:>10}"
            )

    print(f"\n{'='*70}")
    print("Summary — crossover point (cal PPL < uncal PPL)")
    print(f"{'='*70}")
    for K in K_values:
        uncal = ppl_uncal.get(K, float("nan"))
        print(f"  K={K:>3}: uncal={uncal:>10.3f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
