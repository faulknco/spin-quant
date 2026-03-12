"""
Greedy bpw budget allocation.

Experiment 26. Prior work uses binary K in {64, 128} with a top-N cutoff
(Exp 20B/21). This experiment uses a greedy algorithm to allocate a fixed
bpw budget across layers, choosing from K in {64, 96, 128, 192, 256, 384}
based on Exp 18 sensitivity scores.

Algorithm:
  1. Start: all 24 MLP layers at K=64 (0.75 bpw baseline)
  2. For each layer i and each upgrade step (K_curr -> K_next):
       score = sensitivity[i] / delta_bpw(K_curr, K_next, bd, in_features)
  3. Greedily pick highest-score upgrade that fits remaining budget
  4. Repeat until budget exhausted or no upgrade fits

Sensitivity proxy: Exp 18 DELTA_PPL values (PPL added when quantizing layer i
on top of all prior layers). Negative values treated as 0.

Compare against Exp 21 binary top-N frontier at the same bpw points.

Usage:
    .venv/bin/python experiments/budget_allocation.py
"""

import sys, os, copy, math
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

# Exp 18 DELTA_PPL per layer (sequential quantization order h0.c_fc -> h11.c_proj)
# Negative value for h8.c_fc clamped to 0 (non-monotone artifact)
SENSITIVITY = {
    "h0.c_fc":   0.144,
    "h0.c_proj": 3.875,
    "h1.c_fc":   3.146,
    "h1.c_proj": 0.799,
    "h2.c_fc":   4.574,
    "h2.c_proj": 8.619,
    "h3.c_fc":   6.509,
    "h3.c_proj": 12.515,
    "h4.c_fc":   11.420,
    "h4.c_proj": 21.201,
    "h5.c_fc":   15.420,
    "h5.c_proj": 38.180,
    "h6.c_fc":   33.156,
    "h6.c_proj": 79.425,
    "h7.c_fc":   16.348,
    "h7.c_proj": 33.582,
    "h8.c_fc":   0.0,     # -6.740 clamped to 0
    "h8.c_proj": 20.214,
    "h9.c_fc":   3.363,
    "h9.c_proj": 6.803,
    "h10.c_fc":  6.667,
    "h10.c_proj": 22.258,
    "h11.c_fc":  19.765,
    "h11.c_proj": 15.218,
}

# K candidates (must be powers of 2 or sensible values; capped at n_blocks_per_row per layer)
K_STEPS = [64, 96, 128, 192, 256, 384]

# Layer in_features (determines n_blocks_per_row and bpw per K)
IN_FEATURES = {
    "c_fc":   768,   # n_blocks_per_row = 768/8 = 96  -> K_max = 96
    "c_proj": 3072,  # n_blocks_per_row = 3072/8 = 384 -> K_max = 384
}

MLP_LAYERS = [(bi, attr) for bi in range(12) for attr in ["c_fc", "c_proj"]]

# Exp 21 calibrated MLP-only reference PPLs (binary top-N K128/K64)
EXP21_CAL = {0: 321.676, 4: 217.041, 8: 180.037, 12: 146.579, 16: 120.085, 24: 84.192}
EXP21_BPW = {0: 0.750,   4: 0.769,   8: 0.787,  12: 0.804,   16: 0.820,   24: 0.849}


def layer_name(bi, attr):
    return f"h{bi}.{attr}"


def k_to_bpw(K, attr):
    in_f = IN_FEATURES[attr]
    n_blocks = in_f // BLOCK_DIM
    K_eff = min(K, n_blocks)
    return math.log2(K_eff) / BLOCK_DIM


def greedy_allocate(budget_bpw):
    """
    Greedily assign K per layer given a total MLP bpw budget.
    Returns dict: layer_name -> K_eff
    """
    # Start all at K=64
    k_assign = {layer_name(bi, attr): 64 for bi, attr in MLP_LAYERS}
    used_bpw = sum(k_to_bpw(64, attr) for _, attr in MLP_LAYERS) / len(MLP_LAYERS)

    while True:
        best_score = -1
        best_layer = None
        best_K_new = None

        for bi, attr in MLP_LAYERS:
            name = layer_name(bi, attr)
            K_curr = k_assign[name]
            sens = SENSITIVITY[name]
            if sens == 0:
                continue

            in_f = IN_FEATURES[attr]
            n_blocks = in_f // BLOCK_DIM

            for K_next in K_STEPS:
                if K_next <= K_curr:
                    continue
                K_next_eff = min(K_next, n_blocks)
                if K_next_eff <= K_curr:
                    continue

                delta_bpw_layer = (math.log2(K_next_eff) - math.log2(K_curr)) / BLOCK_DIM
                delta_bpw_total = delta_bpw_layer / len(MLP_LAYERS)

                if used_bpw + delta_bpw_total > budget_bpw + 1e-9:
                    continue

                score = sens / delta_bpw_layer
                if score > best_score:
                    best_score = score
                    best_layer = (bi, attr, name)
                    best_K_new = K_next_eff

        if best_layer is None:
            break  # no upgrade fits in budget

        bi, attr, name = best_layer
        delta_bpw_layer = (math.log2(best_K_new) - math.log2(k_assign[name])) / BLOCK_DIM
        used_bpw += delta_bpw_layer / len(MLP_LAYERS)
        k_assign[name] = best_K_new

    return k_assign, used_bpw


class _ReconLinear(nn.Module):
    def __init__(self, W_q, bias=None):
        super().__init__()
        self.register_buffer("W_q", W_q.float())
        self.bias = nn.Parameter(bias.clone().float()) if bias is not None else None

    def forward(self, x):
        return F.linear(x, self.W_q, self.bias)


def collect_act_weights(model, tokenizer, calib_texts, module, block_dim, device):
    buffers = []
    def hook(m, inp, out):
        buffers.append(inp[0].detach().reshape(-1, inp[0].shape[-1]).cpu())
    handle = module.register_forward_hook(hook)
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
        act_weights[j] = x[:, j * block_dim:(j + 1) * block_dim].pow(2).mean()
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


def build_model_from_kmap(base_model, tokenizer, calib_texts, k_assign):
    model = copy.deepcopy(base_model)
    total_log2k = 0
    for bi, attr in MLP_LAYERS:
        name = layer_name(bi, attr)
        K = k_assign[name]
        raw      = getattr(model.transformer.h[bi].mlp, attr)
        raw_base = getattr(base_model.transformer.h[bi].mlp, attr)
        lin = conv1d_to_linear(raw)
        W = lin.weight.data.clone()
        b = lin.bias.data.clone() if lin.bias is not None else None
        in_f = W.shape[1]
        n_blocks = in_f // BLOCK_DIM
        K_eff = min(K, n_blocks)
        total_log2k += math.log2(K_eff)
        act_w = collect_act_weights(
            base_model, tokenizer, calib_texts, raw_base, BLOCK_DIM, DEVICE
        )
        layer = quantize_per_row(W, b, BLOCK_DIM, K_eff, act_w)
        setattr(model.transformer.h[bi].mlp, attr, layer)
    bpw = total_log2k / (len(MLP_LAYERS) * BLOCK_DIM)
    return model, bpw


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

    # Budget points matching Exp 21 bpw values for direct comparison
    budgets = [0.750, 0.769, 0.787, 0.804, 0.820, 0.836, 0.849, 0.875]

    print(f"\n{'='*80}")
    print("Experiment 26 -- Greedy bpw Budget Allocation")
    print(f"{'='*80}")
    print(f"{'budget':>8}  {'actual_bpw':>11}  {'PPL':>10}  {'Exp21_bpw':>10}  "
          f"{'Exp21_PPL':>10}  {'delta_PPL':>10}")
    print(f"{'-'*80}")

    seen_kassigns = set()
    results = []

    for budget in budgets:
        k_assign, actual_bpw = greedy_allocate(budget)
        # Deduplicate: skip if same K assignment as previous budget
        key = tuple(sorted(k_assign.items()))
        if key in seen_kassigns:
            print(f"  budget={budget:.3f}: same allocation as previous, skipping")
            continue
        seen_kassigns.add(key)

        # Find closest Exp21 reference point
        closest_n = min(EXP21_BPW, key=lambda n: abs(EXP21_BPW[n] - actual_bpw))
        exp21_ppl = EXP21_CAL[closest_n]
        exp21_bpw = EXP21_BPW[closest_n]

        print(f"  Building budget={budget:.3f} (actual {actual_bpw:.3f} bpw) ...", flush=True)
        # Print K distribution
        k_counts = {}
        for v in k_assign.values():
            k_counts[v] = k_counts.get(v, 0) + 1
        k_str = ", ".join(f"K={k}:{n}" for k, n in sorted(k_counts.items()))
        print(f"    K distribution: {k_str}")

        q_model, bpw_actual = build_model_from_kmap(
            base_model, tokenizer, calib_texts, k_assign
        )
        ppl = eval_perplexity(q_model, tokenizer, eval_texts, MAX_LEN, DEVICE)
        del q_model

        delta = ppl - exp21_ppl
        results.append((budget, bpw_actual, ppl, exp21_bpw, exp21_ppl, delta))
        print(f"{budget:>8.3f}  {bpw_actual:>11.3f}  {ppl:>10.3f}  "
              f"{exp21_bpw:>10.3f}  {exp21_ppl:>10.3f}  {delta:>+10.3f}")

    print(f"\n{'='*80}")
    print("Summary -- greedy vs binary top-N at matched bpw:")
    print(f"{'='*80}")
    print(f"{'budget':>8}  {'actual_bpw':>11}  {'greedy_PPL':>11}  "
          f"{'topN_PPL':>10}  {'improvement':>12}")
    print(f"{'-'*80}")
    for (budget, bpw_actual, ppl, exp21_bpw, exp21_ppl, delta) in results:
        improvement = f"{-delta:+.3f}" if delta < 0 else f"{-delta:+.3f}"
        better = "better" if delta < 0 else "worse"
        print(f"{budget:>8.3f}  {bpw_actual:>11.3f}  {ppl:>11.3f}  "
              f"{exp21_ppl:>10.3f}  {improvement:>10} ({better})")
    print(f"\n  FP32 baseline: {ppl_base:.3f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
