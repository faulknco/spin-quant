"""
Full optimal model quantization.

Experiment 24. Combines all findings into one capstone experiment:
  - MLP layers: top-N K128 / rest K64, activation-calibrated  (Exp 21 best)
  - Attention layers: K=96 (exact reconstruction, lossless)    (Exp 22 finding)

K=96 is lossless for attention because n_blocks_per_row = 768//8 = 96 for both
c_attn [2304×768] and c_proj [768×768]. Every block gets its own centroid.

Sweep N=0,4,8,12,16,24 (same as Exp 21) so results are directly comparable.
Exp 21 calibrated PPLs are reused as MLP-only reference baselines.

Goal: confirm attention K=96 adds zero PPL cost at every N, and report the
definitive full-model Pareto frontier.

Usage:
    .venv/bin/python experiments/full_model_optimal.py
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
BLOCK_DIM  = 8
K_HIGH     = 128
K_LOW      = 64
K_ATTN     = 96   # lossless for attn (= n_blocks_per_row)

# Exp 18 ΔPPL ranking (most sensitive first)
RANKED = [
    "h6.c_proj", "h5.c_proj", "h7.c_proj", "h6.c_fc",
    "h10.c_proj", "h4.c_proj", "h8.c_proj", "h11.c_fc",
    "h7.c_fc", "h5.c_fc", "h11.c_proj", "h3.c_proj",
    "h4.c_fc", "h2.c_proj", "h9.c_proj", "h10.c_fc",
    "h3.c_fc", "h2.c_fc", "h0.c_proj", "h9.c_fc",
    "h1.c_fc", "h1.c_proj", "h0.c_fc", "h8.c_fc",
]

MLP_LAYERS  = [(bi, attr) for bi in range(12) for attr in ["c_fc", "c_proj"]]
ATTN_LAYERS = [(bi, attr) for bi in range(12) for attr in ["c_attn", "c_proj"]]

# Exp 21 calibrated MLP-only PPLs (reused as reference)
EXP21_CAL = {0: 321.676, 4: 217.041, 8: 180.037, 12: 146.579, 16: 120.085, 24: 84.192}


class _ReconLinear(nn.Module):
    def __init__(self, W_q, bias=None):
        super().__init__()
        self.register_buffer("W_q", W_q.float())
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None

    def forward(self, x):
        return F.linear(x, self.W_q, self.bias)


def collect_act_weights(model, tokenizer, calib_texts, module, block_dim, device):
    """Generic activation collector — works for any Conv1D/Linear sublayer."""
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


def build_full_model(base_model, tokenizer, calib_texts, N_sensitive):
    """
    Build quantized model with:
      - MLP: top-N K128, rest K64, activation-calibrated
      - Attention: K=96 (lossless), activation-calibrated
    Returns (model, mlp_bpw, attn_bpw, total_bpw).
    """
    sensitive = set(RANKED[:N_sensitive])
    model = copy.deepcopy(base_model)

    mlp_log2k_total = 0
    attn_log2k_total = 0

    # --- MLP layers ---
    for bi, attr in MLP_LAYERS:
        name = f"h{bi}.{attr}"
        K = K_HIGH if name in sensitive else K_LOW
        raw = getattr(model.transformer.h[bi].mlp, attr)
        raw_base = getattr(base_model.transformer.h[bi].mlp, attr)
        lin = conv1d_to_linear(raw)
        W = lin.weight.data.clone()
        b = lin.bias.data.clone() if lin.bias is not None else None
        n_blocks = W.shape[1] // BLOCK_DIM
        K_eff = min(K, n_blocks)
        mlp_log2k_total += math.log2(K_eff)
        act_weights = collect_act_weights(
            base_model, tokenizer, calib_texts, raw_base, BLOCK_DIM, DEVICE
        )
        layer = quantize_per_row(W, b, BLOCK_DIM, K_eff, act_weights)
        setattr(model.transformer.h[bi].mlp, attr, layer)

    # --- Attention layers ---
    for bi, attr in ATTN_LAYERS:
        raw = getattr(model.transformer.h[bi].attn, attr)
        raw_base = getattr(base_model.transformer.h[bi].attn, attr)
        lin = conv1d_to_linear(raw)
        W = lin.weight.data.clone()
        b = lin.bias.data.clone() if lin.bias is not None else None
        n_blocks = W.shape[1] // BLOCK_DIM
        K_eff = min(K_ATTN, n_blocks)
        attn_log2k_total += math.log2(K_eff)
        act_weights = collect_act_weights(
            base_model, tokenizer, calib_texts, raw_base, BLOCK_DIM, DEVICE
        )
        layer = quantize_per_row(W, b, BLOCK_DIM, K_eff, act_weights)
        setattr(model.transformer.h[bi].attn, attr, layer)

    n_mlp  = len(MLP_LAYERS)
    n_attn = len(ATTN_LAYERS)
    mlp_bpw  = mlp_log2k_total  / (n_mlp  * BLOCK_DIM)
    attn_bpw = attn_log2k_total / (n_attn * BLOCK_DIM)
    total_bpw = (mlp_log2k_total + attn_log2k_total) / ((n_mlp + n_attn) * BLOCK_DIM)

    return model, mlp_bpw, attn_bpw, total_bpw


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

    N_values = [0, 4, 8, 12, 16, 24]

    print(f"\n{'='*85}")
    print("Experiment 24 — Full Model Optimal (MLP top-N K128/K64 cal + Attn K96 cal)")
    print(f"{'='*85}")
    print(f"{'N':>3}  {'mlp_bpw':>8}  {'attn_bpw':>9}  {'total_bpw':>10}  "
          f"{'PPL':>10}  {'vs FP32':>8}  {'vs MLP-only':>12}  {'attn delta':>11}")
    print(f"{'-'*85}")

    results = {}
    for N in N_values:
        mlp_only_ppl = EXP21_CAL[N]
        print(f"  Building N={N}...", flush=True)
        q_model, mlp_bpw, attn_bpw, total_bpw = build_full_model(
            base_model, tokenizer, calib_texts, N
        )
        ppl = eval_perplexity(q_model, tokenizer, eval_texts, MAX_LEN, DEVICE)
        del q_model

        results[N] = ppl
        vs_fp32     = f"{ppl / ppl_base:.3f}×"
        vs_mlp      = f"{ppl / mlp_only_ppl:.3f}×"
        attn_delta  = f"{ppl - mlp_only_ppl:+.3f}"
        print(
            f"{N:>3}  {mlp_bpw:>8.3f}  {attn_bpw:>9.3f}  {total_bpw:>10.3f}  "
            f"{ppl:>10.3f}  {vs_fp32:>8}  {vs_mlp:>12}  {attn_delta:>11}"
        )

    print(f"\n{'='*85}")
    print("Full-model Pareto frontier (MLP + Attn, all calibrated):")
    print(f"{'='*85}")
    print(f"{'N':>3}  {'total_bpw':>10}  {'PPL':>10}  {'MLP-only PPL':>14}  {'attn cost':>10}")
    print(f"{'-'*85}")
    for N in N_values:
        mlp_only = EXP21_CAL[N]
        full = results[N]
        print(f"{N:>3}  {'—':>10}  {full:>10.3f}  {mlp_only:>14.3f}  {full - mlp_only:>+10.3f}")
    print(f"\n  FP32 baseline: {ppl_base:.3f}")
    print(f"{'='*85}")


if __name__ == "__main__":
    main()
