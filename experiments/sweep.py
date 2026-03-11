"""
Hyperparameter sweep: bits-per-weight vs perplexity delta.

Sweeps over (block_dim, K) for scalar codebook and plots the tradeoff curve.
This is the first experiment you'd run to understand the quantization landscape.

Usage:
    python experiments/sweep.py
"""

import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.layers import CodebookLinear
from experiments.eval_perplexity import eval_perplexity, get_mlp_layers, conv1d_to_linear


MODEL_NAME = "gpt2"
N_TEXTS = 100
MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


SWEEP = [
    # (block_dim, K)  =>  bits_per_weight = log2(K) / block_dim
    (16, 16),    # 0.25 bpw
    (16, 64),    # 0.375 bpw
    (16, 256),   # 0.5 bpw
    (8,  256),   # 1.0 bpw
    (4,  256),   # 2.0 bpw
    (4,  4096),  # 3.0 bpw
]


def quantize_all_mlp(model, block_dim, K):
    import copy
    m = copy.deepcopy(model)
    for _, parent, attr in get_mlp_layers(m):
        linear = conv1d_to_linear(getattr(parent, attr))
        setattr(parent, attr, CodebookLinear.from_linear(linear, block_dim=block_dim, K=K))
    return m


def main():
    print(f"Loading {MODEL_NAME} ...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading data ...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if len(t.strip()) > 50][:N_TEXTS]

    print("Baseline ...")
    ppl_base = eval_perplexity(model, tokenizer, texts, MAX_LENGTH, DEVICE)
    print(f"  Baseline PPL: {ppl_base:.2f}\n")

    results = []
    for block_dim, K in SWEEP:
        bpw = math.log2(K) / block_dim
        print(f"block_dim={block_dim}, K={K}, bpw={bpw:.3f}")
        qmodel = quantize_all_mlp(model, block_dim, K)
        ppl_q = eval_perplexity(qmodel, tokenizer, texts, MAX_LENGTH, DEVICE)
        delta = ppl_q - ppl_base
        results.append((bpw, ppl_q, delta))
        print(f"  PPL={ppl_q:.2f}  delta={delta:+.2f}\n")
        del qmodel

    print("\n=== Sweep Results ===")
    print(f"{'bpw':>6}  {'PPL':>8}  {'delta':>8}")
    for bpw, ppl, delta in results:
        print(f"{bpw:6.3f}  {ppl:8.2f}  {delta:+8.2f}")


if __name__ == "__main__":
    main()
