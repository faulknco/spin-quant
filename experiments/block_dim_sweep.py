"""
Experiment 8: Does block_dim (group size for k-means) affect the critical bpw?

Background:
- Experiment 6/7 found a sharp phase transition at bpw=0.5 for h0.c_fc with block_dim=16.
- bpw = log2(K) / block_dim, so the same bpw can be achieved with different (K, block_dim) pairs.
- Question: does smaller block_dim shift the critical bpw downward? Does it yield better PPL
  at the same nominal bpw?

Design:
- Target: h0.c_fc only (shape [3072, 768] after conv1d_to_linear)
- Test block_dims: [4, 8, 16]
- For each block_dim, test K values that achieve bpw = 0.25, 0.375, 0.5, 0.625, 0.75, 1.0
  (only K values that are powers of 2 between 2 and 4096 are tested)
- 75 eval texts, MODEL_NAME="gpt2", DEVICE="cpu", MAX_LEN=128

Usage:
    python experiments/block_dim_sweep.py
"""

import sys, os, copy, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.codebook import quantize_blocks
from experiments.eval_perplexity import conv1d_to_linear, eval_perplexity


MODEL_NAME = "gpt2"
N_EVAL     = 75
MAX_LEN    = 128
DEVICE     = "cpu"

# Target bpw values to test for each block_dim.
# K = 2^(bpw * block_dim); only include if K is an integer power of 2 in [2, 4096].
TARGET_BPWS = [0.25, 0.375, 0.5, 0.625, 0.75, 1.0]
BLOCK_DIMS   = [4, 8, 16]


def is_power_of_two(n: int) -> bool:
    return n >= 2 and (n & (n - 1)) == 0


def build_configs():
    """
    Return list of (block_dim, K, bpw) tuples.

    For each (block_dim, target_bpw), compute K = 2^(bpw * block_dim).
    Only keep configs where K is an exact power-of-two integer in [2, 4096].
    """
    configs = []
    for bd in BLOCK_DIMS:
        for bpw in TARGET_BPWS:
            exp = bpw * bd
            if abs(exp - round(exp)) > 1e-9:
                continue  # not an integer exponent
            K = int(round(2 ** exp))
            if not is_power_of_two(K):
                continue
            if K < 2 or K > 4096:
                continue
            actual_bpw = math.log2(K) / bd
            configs.append((bd, K, actual_bpw))
    return configs


class _FlatLinear(nn.Module):
    def __init__(self, centroids, labels, W_shape, bias=None):
        super().__init__()
        self.register_buffer("centroids", centroids.float())
        self.register_buffer("labels",    labels)
        self._W_shape = W_shape
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None

    def forward(self, x):
        W = self.centroids[self.labels].reshape(self._W_shape)
        return F.linear(x, W, self.bias)


def main():
    print(f"Experiment 8: block_dim sweep on h0.c_fc")
    print(f"  MODEL={MODEL_NAME}  N_EVAL={N_EVAL}  MAX_LEN={MAX_LEN}  DEVICE={DEVICE}")
    print()

    print(f"Loading {MODEL_NAME} ...")
    model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading WikiText-2 ...")
    test_data  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_texts = [t for t in test_data["text"] if len(t.strip()) > 50][:N_EVAL]

    # Target layer
    raw_layer     = model.transformer.h[0].mlp.c_fc
    target_linear = conv1d_to_linear(raw_layer)
    W = target_linear.weight.data.clone()    # [3072, 768]
    b = target_linear.bias.data.clone() if target_linear.bias is not None else None

    print(f"Target: h0.c_fc  shape={tuple(W.shape)}")

    # Baseline
    print("\nBaseline (full precision) ...")
    ppl_base = eval_perplexity(model, tokenizer, eval_texts, MAX_LEN, DEVICE)
    print(f"  PPL = {ppl_base:.3f}")

    configs = build_configs()

    # Print expected configs for reference
    print("\nConfigs to test:")
    print(f"  {'bd':>4}  {'K':>6}  {'bpw':>7}")
    print("  " + "-" * 22)
    for (bd, K, bpw) in configs:
        print(f"  {bd:>4}  {K:>6}  {bpw:>7.4f}")
    print()

    # Run sweep
    results = []  # (bd, K, bpw, ppl, delta)

    header = f"{'bd':>4}  {'K':>6}  {'bpw':>7}  {'PPL':>12}  {'delta':>12}  note"
    sep    = "-" * 65

    current_bd = None
    for (bd, K, bpw) in configs:
        if bd != current_bd:
            print(f"\n--- block_dim = {bd} ---")
            print(header)
            print(sep)
            current_bd = bd

        # Check divisibility
        if W.shape[0] % bd != 0 and W.shape[1] % bd != 0:
            print(f"  {bd:>4}  {K:>6}  {bpw:>7.4f}  SKIP (shape not divisible by block_dim)")
            continue

        try:
            centroids, labels, _ = quantize_blocks(W, bd, K, n_iter=50)
        except Exception as e:
            print(f"  {bd:>4}  {K:>6}  {bpw:>7.4f}  ERROR: {e}")
            continue

        layer = _FlatLinear(centroids, labels, W.shape, b)
        m = copy.deepcopy(model)
        m.transformer.h[0].mlp.c_fc = layer
        ppl = eval_perplexity(m, tokenizer, eval_texts, MAX_LEN, DEVICE)
        del m

        delta = ppl - ppl_base

        note = ""
        if ppl < 1000:
            note = "OK"
        if ppl > 10000:
            note = "diverged"
        elif ppl > 1000:
            note = "degraded"
        if ppl < ppl_base * 1.5:
            note = "good"

        print(f"  {bd:>4}  {K:>6}  {bpw:>7.4f}  {ppl:>12.3f}  {delta:>+12.3f}  {note}")
        results.append((bd, K, bpw, ppl, delta))

    # Summary
    print(f"\n{'='*65}")
    print("Experiment 8 Summary: Lowest critical bpw per block_dim")
    print(f"{'='*65}")
    print(f"  Baseline PPL:  {ppl_base:.3f}")
    print(f"  2x baseline:   {2*ppl_base:.3f}")
    print()

    for bd in BLOCK_DIMS:
        bd_results = [(K, bpw, ppl, delta) for (b, K, bpw, ppl, delta) in results if b == bd]
        if not bd_results:
            print(f"  block_dim={bd}: no results")
            continue

        ok_results = [(K, bpw, ppl) for (K, bpw, ppl, _) in bd_results if ppl < 1000]
        if ok_results:
            best = min(ok_results, key=lambda r: r[1])
            print(f"  block_dim={bd:>2}: lowest critical bpw = {best[1]:.4f}  (K={best[0]}, PPL={best[2]:.3f})")
        else:
            worst = min(bd_results, key=lambda r: r[2])
            print(f"  block_dim={bd:>2}: no config achieved PPL < 1000  (best PPL={worst[2]:.3f} at bpw={worst[1]:.4f})")

    print()
    print("Full table sorted by (block_dim, bpw):")
    print(f"  {'bd':>4}  {'K':>6}  {'bpw':>7}  {'PPL':>12}  {'delta':>12}  note")
    print("  " + "-" * 62)
    for (bd, K, bpw, ppl, delta) in sorted(results, key=lambda r: (r[0], r[2])):
        note = "PPL<1000" if ppl < 1000 else ""
        print(f"  {bd:>4}  {K:>6}  {bpw:>7.4f}  {ppl:>12.3f}  {delta:>+12.3f}  {note}")


if __name__ == "__main__":
    main()
