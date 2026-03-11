"""
Corrected bpw vs PPL sweep for flat k-means.

Experiment 6. Experiment 0 had the Conv1D silent bug (all deltas = 0). This is the
corrected single-layer sweep using the fixed conv1d_to_linear pipeline.

Sweeps: K in {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}
For each K, block_dim is chosen from {4, 8, 16} to span multiple bpw values.

bpw = log2(K) / block_dim

Configs that cover 0.25 to 8.0 bpw range.

Usage:
    python experiments/bpw_sweep_corrected.py
"""

import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.codebook import quantize_blocks
from experiments.eval_perplexity import conv1d_to_linear, eval_perplexity


MODEL_NAME = "gpt2"
N_EVAL     = 150
MAX_LEN    = 128
DEVICE     = "cpu"

# (block_dim, K) pairs → bpw = log2(K) / block_dim
CONFIGS = [
    # Very aggressive (< 1 bpw)
    (16,   2),   # bpw = 0.0625
    (16,   4),   # bpw = 0.125
    (16,   8),   # bpw = 0.1875
    (16,  16),   # bpw = 0.25
    (16,  32),   # bpw = 0.3125
    (16,  64),   # bpw = 0.375
    (16, 128),   # bpw = 0.4375
    (16, 256),   # bpw = 0.5   ← primary comparison point
    (16, 512),   # bpw = 0.5625
    (16,1024),   # bpw = 0.625
    (16,2048),   # bpw = 0.6875
    (16,4096),   # bpw = 0.75
    # Moderate (1–4 bpw)
    (4,  16),    # bpw = 1.0
    (4,  64),    # bpw = 1.5
    (4, 256),    # bpw = 2.0
    (4,1024),    # bpw = 2.5
    (4,4096),    # bpw = 3.0
    # Fine (4–8 bpw)
    (1,  16),    # bpw = 4.0
    (1, 256),    # bpw = 8.0
]


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

    print(f"Target: h0.c_fc  {tuple(W.shape)}")

    # Baseline
    print("\nBaseline (full precision) ...")
    ppl_base = eval_perplexity(model, tokenizer, eval_texts, MAX_LEN, DEVICE)
    print(f"  PPL = {ppl_base:.3f}")

    print(f"\n{'bd':>4}  {'K':>6}  {'bpw':>6}  {'PPL':>10}  {'delta':>10}  note")
    print("-" * 60)

    results = []
    for (bd, K) in CONFIGS:
        if W.shape[1] % bd != 0 and W.shape[0] % bd != 0:
            continue

        bpw = (K.bit_length() - 1) / bd
        try:
            centroids, labels, _ = quantize_blocks(W, bd, K, n_iter=50)
        except Exception as e:
            print(f"  {bd:>4}  {K:>6}  {bpw:>6.3f}  ERROR: {e}")
            continue

        layer = _FlatLinear(centroids, labels, W.shape, b)
        m = copy.deepcopy(model)
        m.transformer.h[0].mlp.c_fc = layer
        ppl = eval_perplexity(m, tokenizer, eval_texts, MAX_LEN, DEVICE)
        del m

        delta = ppl - ppl_base
        note = ""
        if ppl > 10000:
            note = "diverged"
        elif ppl > 1000:
            note = "degraded"
        elif ppl < ppl_base * 1.5:
            note = "← good"

        print(f"  {bd:>4}  {K:>6}  {bpw:>6.4f}  {ppl:>10.3f}  {delta:>+10.3f}  {note}")
        results.append((bpw, K, bd, ppl, delta))

    print(f"\n{'='*60}")
    print(f"Phase transition summary")
    print(f"{'='*60}")
    prev_ppl = None
    for (bpw, K, bd, ppl, delta) in sorted(results, key=lambda r: r[0]):
        if prev_ppl is not None and prev_ppl > 2 * ppl_base and ppl < prev_ppl * 0.5:
            print(f"  → Sharp improvement at bpw={bpw:.4f} (K={K}, bd={bd})")
        prev_ppl = ppl
    print(f"\n  Baseline PPL: {ppl_base:.3f}")
    print(f"  2× baseline:  {2*ppl_base:.3f}")
    good = [(bpw, ppl) for (bpw, K, bd, ppl, _) in results if ppl < 2*ppl_base]
    if good:
        print(f"  First config within 2× baseline: bpw={min(good, key=lambda r: r[0])[0]:.4f} PPL={min(good, key=lambda r: r[0])[1]:.3f}")


if __name__ == "__main__":
    main()
