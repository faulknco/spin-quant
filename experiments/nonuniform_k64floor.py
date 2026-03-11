"""
Top-N K128/K64 sweep: non-uniform compression with K=64 floor.

Experiment 20B. Exp 19A showed that dropping background layers below K=64
is catastrophic. The one successful config was top-4 K128/K64 (PPL=289, 34%
better than uniform K=64). This experiment sweeps N (number of sensitive layers
getting K=128) while keeping background at K=64 minimum.

Sensitive layers selected by Exp 18 ΔPPL ranking (highest sensitivity first).
N=0 = uniform K=64 (baseline PPL=440). N=24 = uniform K=128 (PPL=88).

Usage:
    .venv/bin/python experiments/nonuniform_k64floor.py
"""

import sys, os, copy, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.codebook import kmeans
from experiments.eval_perplexity import conv1d_to_linear, eval_perplexity

MODEL_NAME = "gpt2"
N_EVAL     = 50
MAX_LEN    = 128
DEVICE     = "cpu"
BLOCK_DIM  = 8
K_HIGH     = 128
K_LOW      = 64

# Exp 18 ΔPPL ranking (most sensitive first)
RANKED = [
    "h6.c_proj", "h5.c_proj", "h7.c_proj", "h6.c_fc",
    "h10.c_proj", "h4.c_proj", "h8.c_proj", "h11.c_fc",
    "h7.c_fc", "h5.c_fc", "h11.c_proj", "h3.c_proj",
    "h4.c_fc", "h2.c_proj", "h9.c_proj", "h10.c_fc",
    "h3.c_fc", "h2.c_fc", "h0.c_proj", "h9.c_fc",
    "h1.c_fc", "h1.c_proj", "h0.c_fc", "h8.c_fc",
]

ALL_LAYERS = [(bi, attr) for bi in range(12) for attr in ["c_fc", "c_proj"]]


class _ReconLinear(nn.Module):
    def __init__(self, W_q, bias=None):
        super().__init__()
        self.register_buffer("W_q", W_q.float())
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None

    def forward(self, x):
        return F.linear(x, self.W_q, self.bias)


def quantize_per_row(W, b, block_dim, K):
    W = W.float().cpu()
    out_f, in_f = W.shape
    n_blocks = in_f // block_dim
    K_eff = min(K, n_blocks)
    W_q = torch.zeros_like(W)
    for i in range(out_f):
        row_blocks = W[i].reshape(n_blocks, block_dim)
        centroids_i, labels_i = kmeans(row_blocks, K_eff, n_iter=50, seed=42)
        W_q[i] = centroids_i[labels_i].reshape(in_f)
    return _ReconLinear(W_q, b)


def build_model(base_model, N_sensitive):
    sensitive = set(RANKED[:N_sensitive])
    model = copy.deepcopy(base_model)
    total_bits = 0
    for bi, attr in ALL_LAYERS:
        name = f"h{bi}.{attr}"
        K = K_HIGH if name in sensitive else K_LOW
        in_f = conv1d_to_linear(getattr(base_model.transformer.h[bi].mlp, attr)).weight.shape[1]
        n_blocks = in_f // BLOCK_DIM
        total_bits += math.log2(min(K, n_blocks))
        raw = getattr(model.transformer.h[bi].mlp, attr)
        lin = conv1d_to_linear(raw)
        W = lin.weight.data.clone()
        b = lin.bias.data.clone() if lin.bias is not None else None
        layer = quantize_per_row(W, b, BLOCK_DIM, K)
        setattr(model.transformer.h[bi].mlp, attr, layer)
    avg_bpw = total_bits / (len(ALL_LAYERS) * BLOCK_DIM)
    return model, avg_bpw


def main():
    print(f"Loading {MODEL_NAME} ...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading WikiText-2 ...")
    test_data  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_texts = [t for t in test_data["text"] if len(t.strip()) > 50][:N_EVAL]

    ppl_base = eval_perplexity(base_model, tokenizer, eval_texts, MAX_LEN, DEVICE)
    print(f"FP32 baseline PPL = {ppl_base:.3f}")

    N_values = [0, 2, 4, 6, 8, 10, 12, 16, 20, 24]

    print(f"\n{'='*70}")
    print(f"Experiment 20B — Top-N K128/K64 Sweep (K_low=64 floor)")
    print(f"{'='*70}")
    print(f"{'N':>3}  {'bpw':>6}  {'PPL':>10}  {'vs N=0':>10}  Sensitive layers")
    print(f"{'-'*70}")

    ppl_n0 = None
    for N in N_values:
        print(f"  Building N={N}...", flush=True)
        q_model, avg_bpw = build_model(base_model, N)
        ppl = eval_perplexity(q_model, tokenizer, eval_texts, MAX_LEN, DEVICE)
        del q_model

        if N == 0:
            ppl_n0 = ppl
        vs = f"{ppl / ppl_n0:.3f}×" if ppl_n0 else "—"
        top_layers = ", ".join(RANKED[:N]) if N <= 4 else f"{', '.join(RANKED[:4])}, ..."
        print(
            f"{N:>3}  {avg_bpw:>6.3f}  {ppl:>10.3f}  {vs:>10}  {top_layers}"
        )

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
