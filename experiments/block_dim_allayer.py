"""
Block dimension sweep — all-layer, activation-calibrated.

Experiment 25. All post-Exp-16 calibrated work uses bd=8. This experiment
asks: does bd=4 (double the blocks per row, smaller block vectors) dominate
bd=8 on the PPL/bpw Pareto frontier when both use act_cal?

bpw formula: log2(K_eff) / block_dim, where K_eff = min(K, n_blocks_per_row)

Matched bpw pairs (bd=8 vs bd=4 at same bits):
  0.750 bpw:  bd=8 K=64   vs  bd=4 K=8
  1.000 bpw:  bd=4 K=16   (no clean bd=8 match)
  1.250 bpw:  bd=4 K=32
  1.500 bpw:  bd=4 K=64   (same K as bd=8 K=64, double bpw)

All MLP layers quantized. Attention stays FP32.

Usage:
    .venv/bin/python experiments/block_dim_allayer.py
"""

import sys, os, copy, math
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

MLP_LAYERS = [(bi, attr) for bi in range(12) for attr in ["c_fc", "c_proj"]]


class _ReconLinear(nn.Module):
    def __init__(self, W_q, bias=None):
        super().__init__()
        self.register_buffer("W_q", W_q.float())
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None

    def forward(self, x):
        return F.linear(x, self.W_q, self.bias)


def collect_act_weights(model, tokenizer, calib_texts, module, block_dim, device):
    buffers = []
    def hook(m, inp, out):
        buffers.append(inp[0].detach().reshape(-1, inp[0].shape[-1]).cpu())
    handle = module.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for text in calib_texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
            model(enc["input_ids"].to(device))
    handle.remove()
    x = torch.cat(buffers, dim=0)
    n_blocks = x.shape[1] // block_dim
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
        labels = (dist * weights.unsqueeze(1)).argmin(dim=1)
        new_c = torch.zeros_like(centroids)
        w_sum = torch.zeros(K_eff)
        new_c.scatter_add_(0, labels.unsqueeze(1).expand(-1, d), blocks * weights.unsqueeze(1))
        w_sum.scatter_add_(0, labels, weights)
        nonempty = w_sum > 0
        new_c[nonempty] /= w_sum[nonempty].unsqueeze(1)
        new_c[~nonempty] = centroids[~nonempty]
        centroids = new_c
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
            c, l = weighted_kmeans(row_blocks, act_weights, K_eff)
        else:
            c, l = unweighted_kmeans(row_blocks, K_eff, n_iter=50, seed=42)
        W_q[i] = c[l].reshape(in_f)
    return _ReconLinear(W_q, b)


def build_mlp_model(base_model, tokenizer, calib_texts, block_dim, K):
    model = copy.deepcopy(base_model)
    total_log2k = 0
    for bi, attr in MLP_LAYERS:
        raw      = getattr(model.transformer.h[bi].mlp, attr)
        raw_base = getattr(base_model.transformer.h[bi].mlp, attr)
        lin = conv1d_to_linear(raw)
        W = lin.weight.data.clone()
        b = lin.bias.data.clone() if lin.bias is not None else None
        in_f    = W.shape[1]
        n_blocks = in_f // block_dim
        K_eff   = min(K, n_blocks)
        total_log2k += math.log2(K_eff)
        act_w = collect_act_weights(
            base_model, tokenizer, calib_texts, raw_base, block_dim, DEVICE
        )
        layer = quantize_per_row(W, b, block_dim, K_eff, act_w)
        setattr(model.transformer.h[bi].mlp, attr, layer)
    bpw = total_log2k / (len(MLP_LAYERS) * block_dim)
    return model, bpw


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

    # (block_dim, K, label)
    configs = [
        (8,   64, "bd=8  K=64 "),    # 0.750 bpw — Exp 21 reference
        (8,  128, "bd=8  K=128"),    # 0.875 bpw
        (8,  256, "bd=8  K=256"),    # 1.000 bpw
        (4,    8, "bd=4  K=8  "),    # 0.750 bpw — matched to bd=8 K=64
        (4,   16, "bd=4  K=16 "),    # 1.000 bpw — matched to bd=8 K=256
        (4,   32, "bd=4  K=32 "),    # 1.250 bpw
        (4,   64, "bd=4  K=64 "),    # 1.500 bpw — same K as bd=8 K=64
        (4,  128, "bd=4  K=128"),    # 1.750 bpw — same K as bd=8 K=128
    ]

    print(f"\n{'='*66}")
    print("Experiment 25 -- Block Dimension Sweep (bd=4 vs bd=8, act_cal)")
    print(f"{'='*66}")
    print(f"{'Config':<16}  {'bd':>4}  {'K':>5}  {'bpw':>6}  {'PPL':>10}  {'vs FP32':>8}")
    print(f"{'-'*66}")

    results = []
    for (bd, K, label) in configs:
        print(f"  Running {label.strip()} ...", flush=True)
        q_model, bpw = build_mlp_model(base_model, tokenizer, calib_texts, bd, K)
        ppl = eval_perplexity(q_model, tokenizer, eval_texts, MAX_LEN, DEVICE)
        del q_model
        results.append((label, bd, K, bpw, ppl))
        print(f"{label:<16}  {bd:>4}  {K:>5}  {bpw:>6.3f}  {ppl:>10.3f}  {ppl/ppl_base:>7.3f}x")

    print(f"\n{'='*66}")
    print("Pareto -- sorted by bpw:")
    print(f"{'='*66}")
    print(f"{'Config':<16}  {'bd':>4}  {'K':>5}  {'bpw':>6}  {'PPL':>10}")
    print(f"{'-'*66}")
    for (label, bd, K, bpw, ppl) in sorted(results, key=lambda x: x[3]):
        print(f"{label:<16}  {bd:>4}  {K:>5}  {bpw:>6.3f}  {ppl:>10.3f}")
    print(f"\n  FP32 baseline: {ppl_base:.3f}")
    print(f"{'='*66}")


if __name__ == "__main__":
    main()
