"""
Experiment 9: Quantize ALL 24 GPT-2 MLP layers simultaneously and measure total PPL impact.

All previous experiments quantized only h0.c_fc (one of the 24 MLP projection layers in
GPT-2). Here we apply flat k-means quantization to every c_fc and c_proj in all 12
transformer blocks (24 Conv1D layers total) and measure the accumulated perplexity
degradation.

Hypothesis: errors accumulate across layers, producing much higher PPL deltas than the
single-layer results:
  bpw=0.500 (single-layer) → PPL ≈ 381  (delta ≈ +361)
  bpw=0.625 (single-layer) → PPL ≈ 100  (delta ≈ +80)
  bpw=1.000 (single-layer) → PPL ≈  71  (delta ≈ +51)

Configs tested (same three reference points):
  bpw=0.500  K=256,  block_dim=16
  bpw=0.625  K=1024, block_dim=16
  bpw=1.000  K=16,   block_dim=4

Usage:
    python experiments/all_layer_quant.py
"""

import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.codebook import quantize_blocks
from transformers.pytorch_utils import Conv1D
from experiments.eval_perplexity import conv1d_to_linear, eval_perplexity


MODEL_NAME = "gpt2"
N_EVAL     = 100
MAX_LEN    = 128
DEVICE     = "cpu"

# (label, K, block_dim, bpw) – three reference points from single-layer experiments
CONFIGS = [
    ("bpw=0.500", 256,  16, 0.500),
    ("bpw=0.625", 1024, 16, 0.625),
    ("bpw=1.000", 16,    4, 1.000),
]

# Single-layer reference deltas (from prior experiments)
SINGLE_LAYER_PPL = {
    "bpw=0.500": 381.0,
    "bpw=0.625": 100.0,
    "bpw=1.000":  71.0,
}


class _FlatLinear(nn.Module):
    """Reconstructs a weight matrix on-the-fly from a flat codebook."""

    def __init__(self, centroids, labels, W_shape, bias=None):
        super().__init__()
        self.register_buffer("centroids", centroids.float())
        self.register_buffer("labels",    labels)
        self._W_shape = W_shape
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None

    def forward(self, x):
        W = self.centroids[self.labels].reshape(self._W_shape)
        return F.linear(x, W, self.bias)


def quantize_all_mlp_layers(model, K: int, block_dim: int, label: str) -> nn.Module:
    """
    Deep-copy the model, then replace every c_fc and c_proj in all 12 transformer
    blocks with a _FlatLinear quantized layer.  Returns the modified copy.
    """
    print(f"\n[{label}]  K={K}, bd={block_dim} – deep-copying model ...")
    m = copy.deepcopy(model)

    for block_idx in range(12):
        block = m.transformer.h[block_idx]
        for attr in ("c_fc", "c_proj"):
            raw = getattr(block.mlp, attr)
            linear = conv1d_to_linear(raw)
            W = linear.weight.data.clone()   # [out, in] after conv1d_to_linear
            b = linear.bias.data.clone() if linear.bias is not None else None

            # quantize_blocks expects rows to be the vectors; W is [out, in] so each
            # row is one output neuron's weight vector – sensible grouping.
            centroids, labels_idx, _ = quantize_blocks(W, block_dim, K, n_iter=50)
            q_layer = _FlatLinear(centroids, labels_idx, W.shape, b)
            setattr(block.mlp, attr, q_layer)

            print(f"  h{block_idx}.{attr} done (PPL not yet eval'd)")

    return m


def main():
    print(f"Loading {MODEL_NAME} ...")
    model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading WikiText-2 ...")
    test_data  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_texts = [t for t in test_data["text"] if len(t.strip()) > 50][:N_EVAL]

    print(f"\nEvaluating baseline ({N_EVAL} texts) ...")
    ppl_base = eval_perplexity(model, tokenizer, eval_texts, MAX_LEN, DEVICE)
    print(f"  Baseline PPL = {ppl_base:.3f}")

    results = []
    for (label, K, block_dim, bpw) in CONFIGS:
        q_model = quantize_all_mlp_layers(model, K, block_dim, label)
        print(f"\n  Evaluating fully-quantized model [{label}] ...")
        ppl = eval_perplexity(q_model, tokenizer, eval_texts, MAX_LEN, DEVICE)
        delta = ppl - ppl_base
        results.append((label, K, block_dim, bpw, ppl, delta))
        print(f"  PPL = {ppl:.3f}  (delta = {delta:+.3f})")
        del q_model

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Experiment 9 – All-Layer Quantization Results")
    print("=" * 70)
    for (label, K, block_dim, bpw, ppl, delta) in results:
        print(f"  {label} (K={K}, bd={block_dim}): PPL = {ppl:>10.3f}  (delta = {delta:>+10.3f})")

    # ── Error-accumulation comparison ─────────────────────────────────────────
    print("\n" + "-" * 70)
    print("Error accumulation: single-layer vs all-layer")
    print(f"  {'Config':<20}  {'single-layer PPL':>18}  {'all-layer PPL':>14}  {'factor':>8}")
    print("-" * 70)
    for (label, K, block_dim, bpw, ppl_all, delta_all) in results:
        ppl_single = SINGLE_LAYER_PPL.get(label)
        if ppl_single is not None:
            delta_single = ppl_single - ppl_base
            factor = delta_all / delta_single if delta_single > 0 else float("nan")
            print(
                f"  {label:<20}  {ppl_single:>18.3f}  {ppl_all:>14.3f}  {factor:>7.2f}×"
            )

    print(f"\n  Baseline PPL (full precision): {ppl_base:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
