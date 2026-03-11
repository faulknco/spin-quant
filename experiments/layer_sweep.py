"""
Experiment 7: Layer-wise critical bpw sweep for GPT-2 MLP layers.

Tests whether the phase transition point (critical bpw) varies across MLP layers
in GPT-2. For each of 6 MLP layers (early, middle, late), we swap only that layer
with a quantized replacement at 5 different bpw values and measure perplexity.

Background:
- A sharp phase transition was found at bpw=0.5 (K=256, block_dim=16) for h0.c_fc
- Below bpw~0.5: PPL is catastrophically high (>9000)
- Above bpw~0.5: PPL is reasonable (~381)
- bpw = log2(K) / block_dim

This experiment checks if the critical bpw is the same for all layers or varies.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src.codebook import quantize_blocks
from experiments.eval_perplexity import conv1d_to_linear, eval_perplexity

MODEL_NAME = "gpt2"
DEVICE = "cpu"
MAX_LEN = 128
N_TEXTS = 75

BPW_CONFIGS = [
    (0.375,  64,   16),
    (0.4375, 128,  16),
    (0.500,  256,  16),
    (0.5625, 512,  16),
    (0.625,  1024, 16),
]

LAYERS = [
    ("h0.c_fc",   0, "c_fc"),
    ("h0.c_proj", 0, "c_proj"),
    ("h5.c_fc",   5, "c_fc"),
    ("h5.c_proj", 5, "c_proj"),
    ("h11.c_fc",  11, "c_fc"),
    ("h11.c_proj",11, "c_proj"),
]


class _FlatLinear(nn.Module):
    def __init__(self, centroids, labels, W_shape, bias=None):
        super().__init__()
        self.register_buffer("centroids", centroids.float())
        self.register_buffer("labels", labels)
        self._W_shape = W_shape
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None

    def forward(self, x):
        W = self.centroids[self.labels].reshape(self._W_shape)
        return F.linear(x, W, self.bias)


def get_layer_weight_and_bias(model, layer_idx, sublayer_name):
    mlp = model.transformer.h[layer_idx].mlp
    conv = getattr(mlp, sublayer_name)
    lin = conv1d_to_linear(conv)         # nn.Linear with weight [out, in]
    W = lin.weight.data.clone()
    bias = lin.bias.data.clone() if lin.bias is not None else None
    return W, bias


def make_quantized_model(base_model, layer_idx, sublayer_name, K, block_dim):
    model = copy.deepcopy(base_model)
    mlp = model.transformer.h[layer_idx].mlp
    conv = getattr(mlp, sublayer_name)

    lin = conv1d_to_linear(conv)         # nn.Linear with weight [out, in]
    W = lin.weight.data.clone()
    bias = lin.bias.data.clone() if lin.bias is not None else None

    centroids, labels, W_shape = quantize_blocks(W, K=K, block_dim=block_dim)

    replacement = _FlatLinear(centroids, labels, W_shape, bias=bias)
    setattr(mlp, sublayer_name, replacement)
    return model


def load_eval_texts(tokenizer, n_texts, max_len):
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if len(t.strip()) > 50][:n_texts]
    return texts


def main():
    print(f"Loading model {MODEL_NAME}...")
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    base_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    base_model.eval()

    print(f"Loading {N_TEXTS} eval texts...")
    texts = load_eval_texts(tokenizer, N_TEXTS, MAX_LEN)

    print("\nComputing baseline PPL (full precision)...")
    baseline_ppl = eval_perplexity(base_model, tokenizer, texts, max_length=MAX_LEN, device=DEVICE)
    print(f"Baseline PPL: {baseline_ppl:.2f}\n")

    # Print header
    bpw_labels = [f"bpw={bpw}" for bpw, _, _ in BPW_CONFIGS]
    header_layer  = f"{'Layer':<14}"
    header_shape  = f"{'shape':<14}"
    header_bpws   = "  ".join(f"{lbl:>10}" for lbl in bpw_labels)
    print(f"{header_layer} {header_shape} {header_bpws}")
    print("-" * (14 + 1 + 14 + 1 + len(header_bpws)))

    results = {}

    for layer_name, layer_idx, sublayer_name in LAYERS:
        W, _ = get_layer_weight_and_bias(base_model, layer_idx, sublayer_name)
        shape_str = str(tuple(W.shape))

        row_ppls = []
        for bpw, K, block_dim in BPW_CONFIGS:
            print(f"  Quantizing {layer_name} K={K} bd={block_dim} ...", flush=True)
            q_model = make_quantized_model(base_model, layer_idx, sublayer_name, K=K, block_dim=block_dim)
            q_model.eval()
            ppl = eval_perplexity(q_model, tokenizer, texts, max_length=MAX_LEN, device=DEVICE)
            row_ppls.append(ppl)
            del q_model

        results[layer_name] = row_ppls

        ppl_cols = "  ".join(f"{p:>10.0f}" for p in row_ppls)
        print(f"{layer_name:<14} {shape_str:<14} {ppl_cols}")

    # Summary: critical bpw = first bpw where PPL < 1000
    print("\n--- Summary ---")
    print(f"Baseline PPL: {baseline_ppl:.2f}")
    print(f"Critical bpw threshold: PPL < 1000\n")

    best_layer = None
    best_bpw = None

    for layer_name, _, _ in LAYERS:
        row_ppls = results[layer_name]
        critical = None
        for (bpw, K, block_dim), ppl in zip(BPW_CONFIGS, row_ppls):
            if ppl < 1000:
                critical = bpw
                break
        if critical is not None:
            print(f"  {layer_name:<14}: critical bpw = {critical}")
            if best_bpw is None or critical < best_bpw:
                best_bpw = critical
                best_layer = layer_name
        else:
            max_bpw = BPW_CONFIGS[-1][0]
            print(f"  {layer_name:<14}: critical bpw > {max_bpw} (never recovered in tested range)")

    if best_layer is not None:
        print(f"\nLowest critical bpw: {best_layer} at bpw={best_bpw}")
    else:
        print("\nNo layer recovered within the tested bpw range.")


if __name__ == "__main__":
    main()
