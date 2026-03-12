"""
Combined activation calibration + non-uniform compression.

Experiment 21. Exp 20B showed top-N K128/K64 (uncalibrated) gives a smooth
Pareto frontier. Exp 20A showed calibration helps at K>=64. This experiment
stacks both: apply activation-weighted k-means to ALL layers while using
non-uniform K assignment (top-N sensitive layers get K=128, rest get K=64).

Hypothesis: calibration should improve every config since K_low=64 and K_high=128
are both above the K>=64 threshold where calibration consistently helps.

Sweep: N=0,4,8,12,16,24 x {uncalibrated, calibrated}
Compare to 20B uncalibrated baseline and 19B/20A calibrated baselines.

Usage:
    .venv/bin/python experiments/combined_cal_nonuniform.py
"""

import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math

from experiments.eval_perplexity import conv1d_to_linear, eval_perplexity

MODEL_NAME = "gpt2"
N_EVAL     = 50
N_CALIB    = 20
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


def collect_act_weights(model, tokenizer, calib_texts, bi, attr, block_dim, device):
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


def build_model(base_model, tokenizer, calib_texts, N_sensitive, calibrated):
    sensitive = set(RANKED[:N_sensitive])
    model = copy.deepcopy(base_model)
    total_log2k = 0
    for bi, attr in ALL_LAYERS:
        name = f"h{bi}.{attr}"
        K = K_HIGH if name in sensitive else K_LOW
        raw = getattr(model.transformer.h[bi].mlp, attr)
        lin = conv1d_to_linear(raw)
        W = lin.weight.data.clone()
        b = lin.bias.data.clone() if lin.bias is not None else None
        n_blocks = W.shape[1] // BLOCK_DIM
        total_log2k += math.log2(min(K, n_blocks))
        act_weights = None
        if calibrated:
            act_weights = collect_act_weights(
                base_model, tokenizer, calib_texts, bi, attr, BLOCK_DIM, DEVICE
            )
        layer = quantize_per_row(W, b, BLOCK_DIM, K, act_weights=act_weights)
        setattr(model.transformer.h[bi].mlp, attr, layer)
    avg_bpw = total_log2k / (len(ALL_LAYERS) * BLOCK_DIM)
    return model, avg_bpw


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

    N_values = [0, 4, 8, 12, 16, 24]

    # Reference PPLs from prior experiments (uncalibrated, for comparison)
    ref_uncal = {0: 439.737, 4: 288.956, 8: 234.765, 12: 173.153, 16: 133.427, 24: 87.626}

    print(f"\n{'='*75}")
    print("Experiment 21 — Combined Calibration + Non-Uniform Compression")
    print(f"{'='*75}")
    print(f"{'N':>3}  {'cal':>5}  {'bpw':>6}  {'PPL':>10}  {'vs uncal(N)':>12}  {'vs cal K64':>12}")
    print(f"{'-'*75}")

    cal_k64_ppl = 321.676  # Exp 19B/20A K=64 calibrated

    ppl_cal = {}
    for N in N_values:
        for calibrated in [False, True]:
            if calibrated is False and N in ref_uncal:
                # Use cached value from 20B — skip re-running to save time
                ppl = ref_uncal[N]
                bpw_str = "—"
                vs_uncal = "—"
                vs_cal64 = f"{ppl / cal_k64_ppl:.3f}×"
                print(
                    f"{N:>3}  {'no':>5}  {'—':>6}  {ppl:>10.3f}  {'(ref 20B)':>12}  {vs_cal64:>12}"
                )
                continue

            print(f"  Building N={N} cal={calibrated}...", flush=True)
            q_model, avg_bpw = build_model(
                base_model, tokenizer, calib_texts, N, calibrated
            )
            ppl = eval_perplexity(q_model, tokenizer, eval_texts, MAX_LEN, DEVICE)
            del q_model

            ppl_cal[N] = ppl
            uncal_ref = ref_uncal.get(N, float("nan"))
            vs_uncal = f"{ppl / uncal_ref:.3f}×"
            vs_cal64 = f"{ppl / cal_k64_ppl:.3f}×"
            print(
                f"{N:>3}  {'yes':>5}  {avg_bpw:>6.3f}  {ppl:>10.3f}  {vs_uncal:>12}  {vs_cal64:>12}"
            )

    print(f"\n{'='*75}")
    print("Summary — calibrated vs uncalibrated at each N")
    print(f"{'='*75}")
    for N in N_values:
        uncal = ref_uncal.get(N, float("nan"))
        cal   = ppl_cal.get(N, float("nan"))
        delta_pct = (cal - uncal) / uncal * 100
        sign = "better" if cal < uncal else "worse"
        print(f"  N={N:>2}: uncal={uncal:>8.1f}  cal={cal:>8.1f}  {delta_pct:>+6.1f}%  ({sign})")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
