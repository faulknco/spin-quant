"""
SmoothQuant applied to all-layer quantization.

Experiment 12. Experiment 9 showed all-layer flat (K=256, bd=16) gives PPL=3042.
Experiment 10 showed single-layer SmoothQuant (α=0.5) gives PPL=170 vs 381 for flat.

Question: does the SmoothQuant improvement (55% single-layer) carry through to all-layer?
If proportional: all-layer smooth expected PPL ≈ 3042 × (170/381) ≈ 1357.

For each layer, SmoothQuant requires calibration activations (H_diag) specific to THAT
layer's inputs. Each MLP sublayer (c_fc, c_proj) sees different activations, so each
needs its own H_diag estimate.

Design:
  - Quantize all 24 MLP layers with SmoothQuant α=0.5, K=256, bd=16
  - For each layer: estimate H_diag from 50 calibration texts (same calib set)
  - Compare to: flat all-layer (PPL=3042 from Exp 9), baseline FP32

Also test the best single-layer config (bd=8, K=16, flat) applied to all layers
for comparison with Exp 9's all-layer results.

Usage:
    python experiments/smooth_all_layer.py
"""

import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.codebook import quantize_blocks
from src.hessian import estimate_h_diag, _ActivationCapture
from experiments.eval_perplexity import conv1d_to_linear, eval_perplexity


MODEL_NAME = "gpt2"
K_SMOOTH   = 256
BD_SMOOTH  = 16
K_FLAT8    = 16
BD_FLAT8   = 8
N_CALIB    = 50
N_EVAL     = 100
MAX_LEN    = 128
DEVICE     = "cpu"
ALPHA      = 0.5


class _FlatLinear(nn.Module):
    def __init__(self, centroids, labels, W_shape, bias=None):
        super().__init__()
        self.register_buffer("centroids", centroids.float())
        self.register_buffer("labels",    labels)
        self._W_shape = W_shape
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None
    def forward(self, x):
        return F.linear(x, self.centroids[self.labels].reshape(self._W_shape), self.bias)


class _SmoothLinear(nn.Module):
    def __init__(self, centroids, labels, W_shape, scale, bias=None):
        super().__init__()
        self.register_buffer("centroids", centroids.float())
        self.register_buffer("labels",    labels)
        self.register_buffer("scale",     scale.float())
        self._W_shape = W_shape
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None
    def forward(self, x):
        return F.linear(x * self.scale, self.centroids[self.labels].reshape(self._W_shape), self.bias)


def quantize_layer_flat(raw_layer, block_dim, K):
    lin  = conv1d_to_linear(raw_layer)
    W    = lin.weight.data.clone()
    b    = lin.bias.data.clone() if lin.bias is not None else None
    c, l, s = quantize_blocks(W, block_dim, K, n_iter=50)
    return _FlatLinear(c, l, s, b)


def quantize_layer_smooth(model, tokenizer, calib_texts, raw_layer, alpha, block_dim, K):
    lin = conv1d_to_linear(raw_layer)
    W   = lin.weight.data.clone()
    b   = lin.bias.data.clone() if lin.bias is not None else None

    h_diag    = estimate_h_diag(model, tokenizer, raw_layer, calib_texts, MAX_LEN, DEVICE)
    col_std   = W.float().std(dim=0).clamp(min=1e-8)
    h_sqrt    = h_diag.float().clamp(min=1e-8).sqrt()
    col_scale = (col_std ** alpha) * (h_sqrt ** (1 - alpha))
    col_scale = col_scale.clamp(min=1e-8)

    W_smooth  = W.float() / col_scale.unsqueeze(0)
    c, l, s   = quantize_blocks(W_smooth, block_dim, K, n_iter=50)
    return _SmoothLinear(c, l, s, col_scale, b)


def build_all_layer_model(base_model, tokenizer, calib_texts, mode, block_dim, K):
    """Deep-copy model and replace all 24 MLP layers."""
    model = copy.deepcopy(base_model)
    for bi in range(12):
        block = model.transformer.h[bi]
        for attr in ["c_fc", "c_proj"]:
            raw = getattr(block.mlp, attr)
            if mode == "flat":
                layer = quantize_layer_flat(raw, block_dim, K)
            elif mode == "smooth":
                layer = quantize_layer_smooth(
                    base_model, tokenizer, calib_texts, raw, ALPHA, block_dim, K
                )
            setattr(block.mlp, attr, layer)
            print(f"  h{bi}.{attr} done", flush=True)
    return model


def main():
    print(f"Loading {MODEL_NAME} ...")
    model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading WikiText-2 ...")
    train_data  = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test_data   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    calib_texts = [t for t in train_data["text"] if len(t.strip()) > 50][:N_CALIB]
    eval_texts  = [t for t in test_data["text"]  if len(t.strip()) > 50][:N_EVAL]

    print(f"\nBaseline (full precision) ...")
    ppl_base = eval_perplexity(model, tokenizer, eval_texts, MAX_LEN, DEVICE)
    print(f"  PPL = {ppl_base:.3f}")

    configs = [
        ("flat  bd=16 K=256", "flat",   BD_SMOOTH, K_SMOOTH),
        ("flat  bd= 8 K= 16", "flat",   BD_FLAT8,  K_FLAT8),
        ("smooth bd=16 K=256", "smooth", BD_SMOOTH, K_SMOOTH),
    ]

    results = {}
    for (label, mode, bd, K) in configs:
        bpw = (K.bit_length() - 1) / bd
        print(f"\n[{label}]  mode={mode}  bd={bd}  K={K}  bpw={bpw:.3f}")
        print("  Quantizing all 24 layers ...")
        q_model = build_all_layer_model(model, tokenizer, calib_texts, mode, bd, K)
        print(f"  Evaluating ...")
        ppl = eval_perplexity(q_model, tokenizer, eval_texts, MAX_LEN, DEVICE)
        del q_model
        print(f"  PPL = {ppl:.3f}  (delta = +{ppl - ppl_base:.3f})")
        results[label] = ppl

    # Reference from Exp 9
    EXP9_PPL = {"flat bd=16 K=256": 3042.435}

    print(f"\n{'='*65}")
    print(f"Experiment 12 Summary — All-layer quantization with SmoothQuant")
    print(f"{'='*65}")
    print(f"  Baseline (FP32):                    {ppl_base:>9.3f}")
    print(f"  [Exp 9 ref] flat bd=16 K=256:       {EXP9_PPL['flat bd=16 K=256']:>9.3f}  (24 layers, Exp 9)")
    for label, ppl in results.items():
        delta  = ppl - ppl_base
        factor = ppl / ppl_base
        print(f"  {label:<30}  {ppl:>9.3f}  (delta={delta:>+8.1f}, {factor:.1f}× baseline)")

    # Single-layer references
    SINGLE_LAYER = {
        "flat  bd=16 K=256": 380.618,
        "flat  bd= 8 K= 16": 154.340,
        "smooth bd=16 K=256": 169.704,
    }
    print(f"\n  Error accumulation (all-layer vs single-layer):")
    for label, ppl_all in results.items():
        ppl_single = SINGLE_LAYER.get(label, None)
        if ppl_single:
            factor = ppl_all / ppl_single
            print(f"  {label}: {ppl_single:.1f} → {ppl_all:.1f}  ({factor:.1f}× amplification)")


if __name__ == "__main__":
    main()
