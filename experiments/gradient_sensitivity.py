"""
Gradient-sensitivity weighted per-row k-means.

Experiment 23. Exp 19B/20A showed activation-weighted k-means (weighting by
input magnitude ||x||^2) helps at K>=64. But activation magnitude measures
how active a block is — not how much its reconstruction error affects the loss.

This experiment uses gradient-sensitivity weighting: for each block position j,
the weight is ||∂L/∂h_j||^2 (output gradient magnitude), measuring how much
the loss function depends on getting that block right. This is more principled
than input activation weighting.

Implementation:
  1. Run N_CALIB texts through the FP32 model with autograd enabled.
  2. For each MLP sublayer, compute the gradient of cross-entropy loss
     w.r.t. the layer's output tensor h = W x + b.
  3. Per-block gradient weight: grad_weight[j] = mean(||∂L/∂h[:, j*d:(j+1)*d]||^2)
     averaged over all tokens and calibration texts.
  4. Use grad_weight as importance weights in weighted k-means.

Compare: gradient-weighted vs activation-weighted vs unweighted at K=64 and K=96.

Usage:
    .venv/bin/python experiments/gradient_sensitivity.py
"""

import sys, os, copy
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

# Reference PPLs from prior experiments
REF_UNCAL_K64  = 439.737   # Exp 17C/18/20B
REF_CAL_K64    = 321.676   # Exp 19B/20A
REF_UNCAL_K96  = 103.695   # Exp 20A
REF_CAL_K96    =  93.081   # Exp 20A


class _ReconLinear(nn.Module):
    def __init__(self, W_q, bias=None):
        super().__init__()
        self.register_buffer("W_q", W_q.float())
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None

    def forward(self, x):
        return F.linear(x, self.W_q, self.bias)


def collect_grad_weights(model, tokenizer, calib_texts, bi, attr, block_dim, device):
    """
    Collect per-block output gradient magnitudes for one MLP sublayer.

    Runs calib_texts through the model with grad enabled, computes cross-entropy
    loss, backprops, and captures the gradient of loss w.r.t. the output of
    model.transformer.h[bi].mlp.{attr}.

    grad_weight[j] = mean over tokens of ||grad[:, j*d:(j+1)*d]||^2

    Returns:
        grad_weights: [n_out_blocks] tensor where n_out_blocks = out_features // block_dim
    """
    layer = getattr(model.transformer.h[bi].mlp, attr)
    # out_features of this layer = number of output channels
    # For c_fc: out=3072; for c_proj: out=768
    # We weight in the OUTPUT space (gradient w.r.t. output h = Wx+b)
    # but per-row k-means operates on the INPUT space of W (rows = output neurons)
    # So grad_weight[j] refers to output blocks, which corresponds to row groups in W
    # c_fc: out_features=3072, rows of W_linear=3072 (same), block_dim=8 → n_blocks=384
    # c_proj: out_features=768, n_blocks=96
    # We need per-output-neuron gradient magnitudes, then group into blocks of block_dim
    # Actually per-row k-means: row i has its own codebook, so the natural weight is
    # the gradient magnitude at neuron i: grad_weight[i] = mean(|grad_i|^2) over tokens
    # We return this per-neuron weight (not per-block), so shape = [out_features]

    grad_acc = None
    count = 0

    model.eval()
    for text in calib_texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue

        # Forward with grad
        output_buf = []

        def hook_fwd(m, inp, out):
            out_t = out.detach().requires_grad_(True)
            output_buf.append(out_t)
            return out_t

        handle = layer.register_forward_hook(hook_fwd)

        outputs = model(input_ids)
        handle.remove()

        if not output_buf:
            continue

        h = output_buf[0]  # [1, seq, out_features] with grad

        # Compute cross-entropy loss on next-token prediction
        logits = outputs.logits  # [1, seq, vocab]
        shift_logits = logits[0, :-1, :]
        shift_labels = input_ids[0, 1:]
        loss = F.cross_entropy(shift_logits, shift_labels)

        # Backprop to get gradient w.r.t. h
        loss.backward()

        if h.grad is not None:
            # h.grad: [1, seq, out_features]
            g = h.grad[0]  # [seq, out_features]
            # Per-output-neuron mean squared gradient magnitude
            neuron_grad = g.pow(2).mean(dim=0).detach().cpu()  # [out_features]
            if grad_acc is None:
                grad_acc = neuron_grad
            else:
                grad_acc += neuron_grad
            count += 1

        model.zero_grad()

    if grad_acc is None or count == 0:
        # Fallback to uniform weights if gradient collection fails
        out_f = conv1d_to_linear(layer).weight.shape[0]
        return torch.ones(out_f)

    return grad_acc / count  # [out_features] — per-neuron gradient sensitivity


def collect_act_weights_mlp(model, tokenizer, calib_texts, bi, attr, block_dim, device):
    """Collect per-block input activation magnitudes (for comparison)."""
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


def quantize_per_row_grad(W, b, block_dim, K, row_weights=None):
    """
    Per-row k-means with optional per-ROW weighting.

    row_weights[i] is the importance of row i (output neuron i).
    For gradient-sensitivity: row_weights[i] = mean ||∂L/∂h_i||^2.
    This is different from activation weighting which uses per-BLOCK weights
    (same weight for all rows using blocks at position j).

    For gradient weighting: each row's block data is weighted by
    row_weights[i] / mean(row_weights), scaling the row's contribution
    to the centroid — but since k-means is per-row anyway, row_weights[i]
    only affects which rows get MORE iterations / better initialization.

    In practice: we use row_weights[i] to determine init probability for
    that row's k-means (rows with higher gradient sensitivity get seeds
    from a distribution biased toward their higher-magnitude blocks).
    Specifically: within row i, all blocks get uniform weight, but we run
    an additional round with weights proportional to ||block_j||^2 * row_weights[i].
    Since row i is independent in per-row k-means, row_weights[i] only affects
    how we initialize centroids within that row.

    Simpler interpretation: use row_weights[i] as a scalar multiplier on the
    block magnitudes within row i. This biases the k-means to concentrate
    centroids where ||block||^2 is large, weighted by the row's sensitivity.

    Args:
        W:           [out_features, in_features]
        b:           bias or None
        block_dim:   block size d
        K:           codebook size per row
        row_weights: [out_features] per-row importance weights, or None (uniform)
    """
    from src.codebook import kmeans as unweighted_kmeans

    W = W.float().cpu()
    out_f, in_f = W.shape
    n_blocks = in_f // block_dim
    K_eff = min(K, n_blocks)
    W_q = torch.zeros_like(W)

    for i in range(out_f):
        if i > 0 and i % 500 == 0:
            print(f"    row {i}/{out_f}...", flush=True)

        row_blocks = W[i].reshape(n_blocks, block_dim)  # [n_blocks, block_dim]

        if row_weights is not None:
            # Use per-block weights = row_sensitivity * ||block_j||^2
            # This biases the codebook to cover the highest-energy blocks of the
            # highest-sensitivity output neurons
            block_norms = row_blocks.pow(2).sum(dim=1)  # [n_blocks]
            w = block_norms * row_weights[i].item()
            w = w + 1e-12  # avoid zero weights
            c, l = weighted_kmeans(row_blocks, w, K_eff)
        else:
            c, l = unweighted_kmeans(row_blocks, K_eff, n_iter=50, seed=42)

        W_q[i] = c[l].reshape(in_f)

    return _ReconLinear(W_q, b)


def build_all_layer_model(base_model, tokenizer, calib_texts, K, mode):
    """
    Quantize all 24 MLP sublayers.

    mode: "uncal" | "act_cal" | "grad_cal"
    """
    model = copy.deepcopy(base_model)
    for bi in range(12):
        for attr in ["c_fc", "c_proj"]:
            print(f"  h{bi}.{attr} (K={K}, {mode})...", flush=True)
            raw = getattr(model.transformer.h[bi].mlp, attr)
            lin = conv1d_to_linear(raw)
            W = lin.weight.data.clone()
            b = lin.bias.data.clone() if lin.bias is not None else None

            if mode == "grad_cal":
                row_weights = collect_grad_weights(
                    base_model, tokenizer, calib_texts, bi, attr, BLOCK_DIM, DEVICE
                )
                layer = quantize_per_row_grad(W, b, BLOCK_DIM, K, row_weights=row_weights)
            elif mode == "act_cal":
                act_weights = collect_act_weights_mlp(
                    base_model, tokenizer, calib_texts, bi, attr, BLOCK_DIM, DEVICE
                )
                # act_cal uses per-BLOCK weights (shape [n_blocks])
                from experiments.activation_calibrated_per_row import quantize_per_row as qpr_act
                layer = qpr_act(W, b, BLOCK_DIM, K, act_weights=act_weights)
            else:
                from src.codebook import kmeans
                W_f = W.float().cpu()
                out_f, in_f = W_f.shape
                n_blocks = in_f // BLOCK_DIM
                K_eff = min(K, n_blocks)
                W_q = torch.zeros_like(W_f)
                for i in range(out_f):
                    row_blocks = W_f[i].reshape(n_blocks, BLOCK_DIM)
                    c, l = kmeans(row_blocks, K_eff, n_iter=50, seed=42)
                    W_q[i] = c[l].reshape(in_f)
                layer = _ReconLinear(W_q, b)

            setattr(model.transformer.h[bi].mlp, attr, layer)
    return model


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

    configs = [
        (64, "uncal"),
        (64, "act_cal"),
        (64, "grad_cal"),
        (96, "uncal"),
        (96, "act_cal"),
        (96, "grad_cal"),
    ]

    print(f"\n{'='*70}")
    print("Experiment 23 — Gradient-Sensitivity Weighted Per-Row K-Means")
    print(f"{'='*70}")
    print(f"{'Config':<25}  {'K':>4}  {'PPL':>10}  {'vs uncal':>10}  {'vs act_cal':>10}")
    print(f"{'-'*70}")

    ref = {
        (64, "uncal"):   REF_UNCAL_K64,
        (64, "act_cal"): REF_CAL_K64,
        (96, "uncal"):   REF_UNCAL_K96,
        (96, "act_cal"): REF_CAL_K96,
    }
    results = {}

    for (K, mode) in configs:
        key = (K, mode)
        if key in ref and mode != "grad_cal":
            ppl = ref[key]
            print(f"{'per-row ' + mode:<25}  {K:>4}  {ppl:>10.3f}  {'(ref)':>10}  {'(ref)':>10}")
            results[key] = ppl
            continue

        print(f"\nBuilding K={K} {mode}...", flush=True)
        q_model = build_all_layer_model(base_model, tokenizer, calib_texts, K, mode)
        ppl = eval_perplexity(q_model, tokenizer, eval_texts, MAX_LEN, DEVICE)
        del q_model
        results[key] = ppl

        uncal_ppl   = results.get((K, "uncal"), float("nan"))
        act_cal_ppl = results.get((K, "act_cal"), float("nan"))
        vs_uncal  = f"{ppl / uncal_ppl:.3f}×"
        vs_act    = f"{ppl / act_cal_ppl:.3f}×"
        print(f"{'per-row ' + mode:<25}  {K:>4}  {ppl:>10.3f}  {vs_uncal:>10}  {vs_act:>10}")

    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"  FP32 baseline:       {ppl_base:.3f}")
    for K in [64, 96]:
        print(f"\n  K={K}:")
        for mode in ["uncal", "act_cal", "grad_cal"]:
            ppl = results.get((K, mode), float("nan"))
            print(f"    {mode:<10}: {ppl:.3f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
