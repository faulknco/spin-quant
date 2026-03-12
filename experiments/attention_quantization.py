"""
Attention layer quantization.

Experiment 22. All prior experiments quantized only 24 MLP sublayers
(h0-h11 × c_fc, c_proj). GPT-2 also has 24 attention sublayers
(h0-h11 × attn.c_attn, attn.c_proj). This experiment extends per-row
k-means to the full model.

Attention layer shapes (after conv1d_to_linear):
  attn.c_attn:  [2304, 768]  out=2304 (Q+K+V combined), n_blocks_per_row=96
  attn.c_proj:  [768, 768]   out=768,                   n_blocks_per_row=96

Configs:
  A. Attn-only K=64 uncal        (baseline: how much does attn alone cost?)
  B. MLP K=64 cal + attn K=64 uncal
  C. MLP K=64 cal + attn K=64 cal
  D. MLP K=64 cal + attn K=96 uncal
  E. MLP K=64 cal + attn K=96 cal

Reference: MLP-only K=64 calibrated PPL=321.7 (Exp 19B/20A)

Usage:
    .venv/bin/python experiments/attention_quantization.py
"""

import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from experiments.eval_perplexity import conv1d_to_linear, eval_perplexity

MODEL_NAME      = "gpt2"
N_EVAL          = 50
N_CALIB         = 20
MAX_LEN         = 128
DEVICE          = "cpu"
BLOCK_DIM       = 8
MLP_CAL_K64_PPL = 321.676  # Exp 19B reference


class _ReconLinear(nn.Module):
    def __init__(self, W_q, bias=None):
        super().__init__()
        self.register_buffer("W_q", W_q.float())
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None

    def forward(self, x):
        return F.linear(x, self.W_q, self.bias)


def collect_act_weights_attn(model, tokenizer, calib_texts, bi, attr, block_dim, device):
    """Collect per-block input activation magnitudes for an attention sublayer."""
    buffers = []
    layer = getattr(model.transformer.h[bi].attn, attr)

    def hook(m, inp, out):
        buffers.append(inp[0].detach().reshape(-1, inp[0].shape[-1]).cpu())

    handle = layer.register_forward_hook(hook)
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
        act_weights[j] = x[:, j*block_dim:(j+1)*block_dim].pow(2).mean()
    return act_weights


def collect_act_weights_mlp(model, tokenizer, calib_texts, bi, attr, block_dim, device):
    """Collect per-block input activation magnitudes for an MLP sublayer."""
    buffers = []
    layer = getattr(model.transformer.h[bi].mlp, attr)

    def hook(m, inp, out):
        buffers.append(inp[0].detach().reshape(-1, inp[0].shape[-1]).cpu())

    handle = layer.register_forward_hook(hook)
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
        act_weights[j] = x[:, j*block_dim:(j+1)*block_dim].pow(2).mean()
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


def quantize_mlp_all(model, base_model, tokenizer, calib_texts, K, calibrated):
    """Replace all 24 MLP sublayers with per-row quantized versions."""
    for bi in range(12):
        for attr in ["c_fc", "c_proj"]:
            raw = getattr(model.transformer.h[bi].mlp, attr)
            lin = conv1d_to_linear(raw)
            W = lin.weight.data.clone()
            b = lin.bias.data.clone() if lin.bias is not None else None
            act_weights = None
            if calibrated:
                act_weights = collect_act_weights_mlp(
                    base_model, tokenizer, calib_texts, bi, attr, BLOCK_DIM, DEVICE
                )
            layer = quantize_per_row(W, b, BLOCK_DIM, K, act_weights)
            setattr(model.transformer.h[bi].mlp, attr, layer)


def quantize_attn_all(model, base_model, tokenizer, calib_texts, K, calibrated):
    """Replace all 24 attention sublayers with per-row quantized versions."""
    for bi in range(12):
        for attr in ["c_attn", "c_proj"]:
            raw = getattr(model.transformer.h[bi].attn, attr)
            lin = conv1d_to_linear(raw)
            W = lin.weight.data.clone()
            b = lin.bias.data.clone() if lin.bias is not None else None
            act_weights = None
            if calibrated:
                act_weights = collect_act_weights_attn(
                    base_model, tokenizer, calib_texts, bi, attr, BLOCK_DIM, DEVICE
                )
            layer = quantize_per_row(W, b, BLOCK_DIM, K, act_weights)
            setattr(model.transformer.h[bi].attn, attr, layer)


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
    print(f"MLP-only K=64 cal reference PPL = {MLP_CAL_K64_PPL:.3f}")

    configs = [
        # (label, mlp_K, mlp_cal, attn_K, attn_cal)
        ("attn-only K64 uncal",          None, False,  64, False),
        ("attn-only K64 cal",            None, False,  64, True),
        ("MLP K64cal + attn K64 uncal",    64, True,   64, False),
        ("MLP K64cal + attn K64 cal",      64, True,   64, True),
        ("MLP K64cal + attn K96 uncal",    64, True,   96, False),
        ("MLP K64cal + attn K96 cal",      64, True,   96, True),
    ]

    print(f"\n{'='*75}")
    print("Experiment 22 — Attention Layer Quantization")
    print(f"{'='*75}")
    print(f"{'Config':<35}  {'PPL':>10}  {'vs FP32':>9}  {'vs MLP-only':>12}")
    print(f"{'-'*75}")

    for (label, mlp_K, mlp_cal, attn_K, attn_cal) in configs:
        print(f"  Running: {label}...", flush=True)
        model = copy.deepcopy(base_model)

        if mlp_K is not None:
            quantize_mlp_all(model, base_model, tokenizer, calib_texts, mlp_K, mlp_cal)

        quantize_attn_all(model, base_model, tokenizer, calib_texts, attn_K, attn_cal)

        ppl = eval_perplexity(model, tokenizer, eval_texts, MAX_LEN, DEVICE)
        del model

        vs_fp32    = f"{ppl / ppl_base:.3f}×"
        vs_mlp     = f"{ppl / MLP_CAL_K64_PPL:.3f}×"
        print(f"{label:<35}  {ppl:>10.3f}  {vs_fp32:>9}  {vs_mlp:>12}")

    print(f"\n{'='*75}")
    print("Reference:")
    print(f"  FP32 baseline:            {ppl_base:.3f}")
    print(f"  MLP-only K=64 cal:        {MLP_CAL_K64_PPL:.3f}")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
