"""
Collective behavior profiling: layer-by-layer error accumulation.

Experiment 18. Per-row all-layer (Exp 17) gives 7.5× error amplification
despite near-lossless single-layer performance. The mechanism is unknown:
is it early-layer dominated, uniform, or exponential?

This experiment quantizes layers one by one and records both PPL and
residual stream drift (vs FP32) after each addition. 13 checkpoints per step
(embedding output + after each of 12 transformer blocks).

Configs:
  - per-row bd=8 K=64 bpw=0.750 (primary — our best all-layer)
  - flat   bd=16 K=256 bpw=0.500 (reference — Exp 9)

Output: table of (layer, PPL, ΔPPL, worst_checkpoint, drift_mean, drift_max)
for each config. Marks tipping points (PPL > 2× previous checkpoint).

Usage:
    .venv/bin/python experiments/accumulation_profile.py
"""

import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.codebook import kmeans, quantize_blocks
from experiments.eval_perplexity import conv1d_to_linear, eval_perplexity


MODEL_NAME = "gpt2"
N_EVAL     = 50    # fewer texts for faster per-step eval
N_CALIB    = 10    # residual comparison texts
MAX_LEN    = 128
DEVICE     = "cpu"


class _ReconLinear(nn.Module):
    """Linear layer with precomputed reconstructed weight matrix."""
    def __init__(self, W_q, bias=None):
        super().__init__()
        self.register_buffer("W_q", W_q.float())
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None

    def forward(self, x):
        return F.linear(x, self.W_q, self.bias)


def quantize_per_row(W, b, block_dim, K):
    """
    Per-row k-means quantization: one codebook per output row.

    Each output row W[i] of shape [in_features] is split into
    n_blocks = in_features // block_dim blocks of shape [n_blocks, block_dim].
    K-means is run independently on each row's blocks. The reconstructed row
    W_q[i] = centroids_i[labels_i].reshape(1, in_features).

    Args:
        W:         [out_features, in_features] weight tensor
        b:         bias tensor or None
        block_dim: block size d (must divide in_features)
        K:         codebook size per row

    Returns:
        _ReconLinear with precomputed W_q of shape [out_features, in_features]
    """
    W = W.float().cpu()
    out_f, in_f = W.shape
    assert in_f % block_dim == 0, (
        f"in_features={in_f} must be divisible by block_dim={block_dim}"
    )
    n_blocks = in_f // block_dim

    W_q = torch.zeros_like(W)

    for i in range(out_f):
        if i > 0 and i % 500 == 0:
            print(f"    row {i}/{out_f}...", flush=True)

        # Shape: [n_blocks, block_dim]
        row_blocks = W[i].reshape(n_blocks, block_dim)

        # Run k-means on this row's blocks only
        centroids_i, labels_i = kmeans(row_blocks, K, n_iter=50, seed=42)

        # Reconstruct and store
        W_q[i] = centroids_i[labels_i].reshape(in_f)

    return _ReconLinear(W_q, b)


def quantize_flat(W, b, block_dim, K):
    """
    Flat k-means quantization: single shared codebook across all rows.

    All blocks from the entire weight matrix are pooled into one set and a
    single K-means codebook is trained on them. This matches the scheme used
    in Exp 9 (src/codebook.quantize_blocks).

    Args:
        W:         [out_features, in_features] weight tensor
        b:         bias tensor or None
        block_dim: block size d (must divide in_features)
        K:         codebook size

    Returns:
        _ReconLinear with precomputed W_q of shape [out_features, in_features]
    """
    c, l, s = quantize_blocks(W, block_dim, K, n_iter=50)
    W_q = c[l].reshape(s)
    return _ReconLinear(W_q, b)


def capture_residuals(model, tokenizer, texts, max_length, device):
    """
    Capture residual stream at each of 13 layer boundaries.

    GPT-2 residual stream checkpoints:
      - Checkpoint 0:  output of embedding dropout (before first transformer block)
      - Checkpoints 1-12: output of each transformer block h[0]..h[11]

    Args:
        model:      GPT-2 CausalLM (FP32 or quantized)
        tokenizer:  GPT-2 tokenizer
        texts:      list of strings to run through the model
        max_length: token truncation length
        device:     torch device string

    Returns:
        List of 13 tensors, each of shape [N_tokens, hidden_size].
        Index c corresponds to checkpoint c.
    """
    buffers = [[] for _ in range(13)]
    hooks = []

    # Checkpoint 0: output of embedding dropout (input to h[0])
    hooks.append(model.transformer.drop.register_forward_hook(
        lambda m, inp, out: buffers[0].append(
            out.detach().reshape(-1, out.shape[-1]).cpu()
        )
    ))

    # Checkpoints 1-12: output of each transformer block h[i]
    for i, block in enumerate(model.transformer.h):
        def make_hook(idx):
            def _hook(m, inp, out):
                # out is a tuple; out[0] is the hidden state tensor
                h = out[0].detach()
                buffers[idx].append(h.reshape(-1, h.shape[-1]).cpu())
            return _hook
        hooks.append(block.register_forward_hook(make_hook(i + 1)))

    model.eval()
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            )
            model(enc["input_ids"].to(device))

    for h in hooks:
        h.remove()

    return [
        torch.cat(b, dim=0) if b else None
        for b in buffers
    ]  # list of 13 tensors, each [N_tokens, hidden_size]


def residual_drift_stats(fp32_residuals, q_residuals):
    """
    Compare residual streams checkpoint by checkpoint.

    For each of the 13 checkpoints, computes element-wise absolute difference
    |q - fp32| and records mean, std, max. Returns the worst-checkpoint stats
    (highest mean drift) plus the full list of per-checkpoint means.

    Args:
        fp32_residuals: list of 13 tensors from the FP32 model
        q_residuals:    list of 13 tensors from the quantized model

    Returns:
        dict with keys:
          worst_ckpt  - checkpoint index with highest mean drift
          drift_mean  - mean absolute drift at worst checkpoint
          drift_std   - std of absolute drift at worst checkpoint
          drift_max   - max absolute drift at worst checkpoint
          all_means   - list of mean drifts for all 13 checkpoints
    """
    diffs = []
    for c in range(13):
        if fp32_residuals[c] is None or q_residuals[c] is None:
            continue
        diff = (q_residuals[c].float() - fp32_residuals[c].float()).abs()
        diffs.append({
            "checkpoint": c,
            "mean": diff.mean().item(),
            "std":  diff.std().item(),
            "max":  diff.max().item(),
        })

    # worst = checkpoint with highest mean drift
    worst = max(diffs, key=lambda d: d["mean"])
    return {
        "worst_ckpt": worst["checkpoint"],
        "drift_mean": worst["mean"],
        "drift_std":  worst["std"],
        "drift_max":  worst["max"],
        "all_means":  [d["mean"] for d in diffs],
    }


def run_accumulation(
    base_model, tokenizer, eval_texts, calib_texts, quantize_fn, block_dim, K, label
):
    """
    Quantize MLP sublayers one by one in forward order, recording PPL and
    residual stream drift after each addition.

    Layer order: h0.c_fc, h0.c_proj, h1.c_fc, h1.c_proj, ..., h11.c_fc, h11.c_proj
    (24 sublayers total — 2 per transformer block × 12 blocks).

    For each step:
      1. Replace the next sublayer with its quantized version (in-place on working copy).
      2. Evaluate PPL on eval_texts.
      3. Run calib_texts through both FP32 and current working model; capture residuals
         at all 13 checkpoints; compute worst-checkpoint drift stats.
      4. Record and print results.

    Tipping point: any step where ppl > 2 × prev_ppl is flagged with " <- TIPPING".

    Args:
        base_model:   original FP32 GPT-2 (never modified)
        tokenizer:    GPT-2 tokenizer
        eval_texts:   list of strings for PPL evaluation (N_EVAL texts)
        calib_texts:  list of strings for residual capture (N_CALIB texts)
        quantize_fn:  callable(W, b, block_dim, K) -> nn.Module
        block_dim:    block size for quantize_fn
        K:            codebook size for quantize_fn
        label:        human-readable config label for table header

    Returns:
        list of dicts, one per step (including FP32 baseline at index 0):
          layer, n, ppl, delta_ppl, worst_ckpt, drift_mean, drift_max
    """
    layer_order = [
        (bi, attr)
        for bi in range(12)
        for attr in ["c_fc", "c_proj"]
    ]

    model = copy.deepcopy(base_model)
    fp32_residuals = capture_residuals(
        base_model, tokenizer, calib_texts, MAX_LEN, DEVICE
    )

    print(f"\n{'='*75}")
    print(f"Config: {label}  (bd={block_dim}, K={K})")
    print(f"{'='*75}")
    print(
        f"{'Layer':<18} {'#':>2}  {'PPL':>10}  {'ΔPPL':>10}  "
        f"{'worst_ckpt':>10}  {'drift_mean':>10}  {'drift_max':>10}"
    )
    print(f"{'-'*75}")

    ppl_base = eval_perplexity(base_model, tokenizer, eval_texts, MAX_LEN, DEVICE)
    print(
        f"{'FP32 baseline':<18} {0:>2}  {ppl_base:>10.3f}  {'—':>10}  "
        f"{'—':>10}  {'—':>10}  {'—':>10}"
    )

    records = [{
        "layer":      "FP32",
        "n":          0,
        "ppl":        ppl_base,
        "delta_ppl":  0.0,
        "worst_ckpt": None,
        "drift_mean": 0.0,
        "drift_max":  0.0,
    }]
    prev_ppl = ppl_base

    for n, (bi, attr) in enumerate(layer_order, start=1):
        # Quantize and replace the next sublayer in the working model
        raw = getattr(model.transformer.h[bi].mlp, attr)
        lin = conv1d_to_linear(raw)
        W = lin.weight.data.clone()
        b = lin.bias.data.clone() if lin.bias is not None else None
        layer = quantize_fn(W, b, block_dim, K)
        setattr(model.transformer.h[bi].mlp, attr, layer)

        # Evaluate PPL
        ppl = eval_perplexity(model, tokenizer, eval_texts, MAX_LEN, DEVICE)

        # Capture residual stream drift vs FP32
        q_residuals = capture_residuals(model, tokenizer, calib_texts, MAX_LEN, DEVICE)
        stats = residual_drift_stats(fp32_residuals, q_residuals)

        delta_ppl  = ppl - prev_ppl
        tipping    = " <- TIPPING" if ppl > 2 * prev_ppl else ""
        layer_name = f"h{bi}.{attr}"

        print(
            f"{layer_name:<18} {n:>2}  {ppl:>10.3f}  {delta_ppl:>+10.3f}  "
            f"{stats['worst_ckpt']:>10}  {stats['drift_mean']:>10.5f}  "
            f"{stats['drift_max']:>10.4f}{tipping}"
        )

        records.append({
            "layer":      layer_name,
            "n":          n,
            "ppl":        ppl,
            "delta_ppl":  delta_ppl,
            "worst_ckpt": stats["worst_ckpt"],
            "drift_mean": stats["drift_mean"],
            "drift_max":  stats["drift_max"],
        })
        prev_ppl = ppl

    return records


def main():
    print(f"Loading {MODEL_NAME} ...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading WikiText-2 ...")
    test_data   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    all_texts   = [t for t in test_data["text"] if len(t.strip()) > 50]
    eval_texts  = all_texts[:N_EVAL]
    calib_texts = all_texts[N_EVAL:N_EVAL + N_CALIB]

    print(f"  {N_EVAL} eval texts, {N_CALIB} calib texts, max_len={MAX_LEN}")

    # Two configs: primary (per-row, our best) and reference (flat, Exp 9)
    configs = [
        ("per-row bd=8 K=64",  quantize_per_row, 8,  64),
        ("flat   bd=16 K=256", quantize_flat,    16, 256),
    ]

    all_records = {}
    for label, quantize_fn, block_dim, K in configs:
        records = run_accumulation(
            base_model, tokenizer, eval_texts, calib_texts,
            quantize_fn, block_dim, K, label
        )
        all_records[label] = records

    # Summary: tipping points and final PPL for both configs
    print(f"\n{'='*75}")
    print("Summary — Experiment 18: Collective Behavior Profiling")
    print(f"{'='*75}")
    for label, records in all_records.items():
        final   = records[-1]
        fp32    = records[0]["ppl"]
        tippers = [r for r in records[1:] if r["ppl"] > 2 * records[records.index(r) - 1]["ppl"]
                   ] if False else []  # computed inline below
        print(f"\nConfig: {label}")
        print(f"  FP32 baseline:  {fp32:.3f}")
        print(f"  Final PPL:      {final['ppl']:.3f}")
        print(f"  Amplification:  {final['ppl'] / fp32:.2f}×")

        # Find tipping points
        tipping_layers = []
        for i in range(1, len(records)):
            prev = records[i - 1]["ppl"]
            curr = records[i]["ppl"]
            if curr > 2 * prev:
                tipping_layers.append(
                    f"    {records[i]['layer']} (step {records[i]['n']}): "
                    f"{prev:.3f} -> {curr:.3f} ({curr/prev:.2f}×)"
                )
        if tipping_layers:
            print(f"  Tipping points ({len(tipping_layers)}):")
            for t in tipping_layers:
                print(t)
        else:
            print("  No single tipping point (gradual accumulation)")

    print(f"\n{'='*75}")


if __name__ == "__main__":
    main()
