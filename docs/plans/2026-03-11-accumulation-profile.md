# Experiment 18: Collective Behavior Profiling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Profile how quantization error accumulates layer-by-layer across all 24 GPT-2 MLP sublayers, tracking both PPL and residual stream drift at each step.

**Architecture:** Sequential layer-by-layer quantization with PPL measurement and residual stream comparison (FP32 vs quantized) after each layer addition. Two configs run in sequence: per-row bd=8 K=64 (primary) and flat bd=16 K=256 (reference).

**Tech Stack:** PyTorch, HuggingFace transformers, WikiText-2, src/codebook.py (kmeans), experiments/eval_perplexity.py (conv1d_to_linear, eval_perplexity)

---

## Task 1: Create `experiments/accumulation_profile.py`

### 1a. Write the file

Create `/Users/faulknco/projects/spin-quant/experiments/accumulation_profile.py` with the exact content below.

```python
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
```

### 1b. Verify syntax

```bash
cd /Users/faulknco/projects/spin-quant && .venv/bin/python -c "import experiments.accumulation_profile"
```

Expected output: no errors, no tracebacks. The import chain is:
- `sys.path.insert` adds repo root, so `from src.codebook import kmeans, quantize_blocks` and
  `from experiments.eval_perplexity import conv1d_to_linear, eval_perplexity` both resolve.
- `quantize_blocks` is imported directly from `src/codebook.py` (it exists there; see line 57 of that file).
- No `__init__` files are needed — this is the same pattern used by `per_row_all_layer.py`.

If an `ImportError` appears on `quantize_blocks`, verify it is exported from `src/codebook.py`
(it is — defined at line 57). If the error is on `experiments.accumulation_profile` itself, check
the `sys.path.insert` line resolves to the repo root from `experiments/`.

### 1c. Commit

```bash
cd /Users/faulknco/projects/spin-quant
git add experiments/accumulation_profile.py
git commit -m "Add Experiment 18: layer-by-layer accumulation profile script"
```

---

## Task 2: Run the experiment

### 2a. Launch in background

```bash
cd /Users/faulknco/projects/spin-quant
.venv/bin/python experiments/accumulation_profile.py
```

Run this in the background. The script prints one table row per sublayer as it goes, so you can confirm forward progress immediately.

**Expected runtime:** 45–60 minutes on CPU.
- Per-row config (bd=8 K=64): 24 sublayers × (per-row k-means for all rows + PPL eval on 50 texts + residual capture on 10 texts). Per-row k-means is the dominant cost: c_fc has 3,072 output rows × 50 k-means iterations; c_proj has 768 output rows.
- Flat config (bd=16 K=256): 24 sublayers × (flat k-means on all blocks + PPL eval + residual capture). Flat k-means is faster per-layer than per-row but uses K=256.
- Residual capture: 10 calib texts × 24 steps × 2 configs = 480 forward passes (light).

**Expected output shape:**

```
Loading gpt2 ...
Loading WikiText-2 ...
  50 eval texts, 10 calib texts, max_len=128

===========================================================================
Config: per-row bd=8 K=64  (bd=8, K=64)
===========================================================================
Layer              #         PPL        ΔPPL  worst_ckpt  drift_mean   drift_max
---------------------------------------------------------------------------
FP32 baseline      0      59.640           —           —           —           —
h0.c_fc            1      XX.XXX     +XX.XXX          XX     X.XXXXX      X.XXXX
h0.c_proj          2      XX.XXX     +XX.XXX          XX     X.XXXXX      X.XXXX
...
h11.c_proj        24     XXX.XXX    +XXX.XXX          XX     X.XXXXX      X.XXXX

===========================================================================
Config: flat   bd=16 K=256  (bd=16, K=256)
===========================================================================
...

===========================================================================
Summary — Experiment 18: Collective Behavior Profiling
===========================================================================

Config: per-row bd=8 K=64
  FP32 baseline:  59.640
  Final PPL:      XXX.XXX
  Amplification:  X.XX×
  Tipping points (?):
    ...

Config: flat   bd=16 K=256
  FP32 baseline:  59.640
  Final PPL:      XXXX.XXX
  Amplification:  XX.XX×
  Tipping points (?):
    ...
```

### 2b. Collect results

Once the run completes, read the full terminal output. Record:

1. Per-row config table: all 25 rows (FP32 baseline + 24 sublayer steps), including PPL, ΔPPL, worst_ckpt, drift_mean, drift_max.
2. Flat config table: same 25 rows.
3. Tipping points for each config (layers where PPL > 2× previous checkpoint).
4. Final amplification factors.
5. Worst-checkpoint column: note whether drift concentrates at later checkpoints (indicating downstream amplification) or is uniformly distributed.

---

## Task 3: Record findings in FINDINGS_LOG.md

### 3a. Insert Experiment 18 section

Open `/Users/faulknco/projects/spin-quant/FINDINGS_LOG.md` and insert the following section **before** `## Experiment 17`. Fill in all `???` / `RESULT_*` / `AMP_*` placeholders with actual numbers from the run output.

```markdown
## Experiment 18 — Collective behavior profiling (layer-by-layer accumulation)

**Question:** How and where does quantization error amplify across 24 GPT-2 MLP
sublayers? Is accumulation linear, exponential, or step-like? Are certain layers
disproportionately responsible? Do per-row and flat have different profiles?

**Design:** `experiments/accumulation_profile.py`. Quantizes sublayers one at a time
in forward order (h0.c_fc → h0.c_proj → ... → h11.c_proj). After each addition:
evaluates PPL on 50 WikiText-2 texts and captures the residual stream at 13 checkpoints
(embedding dropout output + after each of 12 transformer blocks) for 10 calibration
texts, comparing the working model vs the frozen FP32 baseline. Records worst-checkpoint
drift stats (mean, std, max of |x_q - x_fp32|).

Two configs:
- **Primary:** per-row bd=8 K=64 bpw=0.750 (best all-layer from Exp 17)
- **Reference:** flat bd=16 K=256 bpw=0.500 (Exp 9 baseline)

---

### Results: per-row bd=8 K=64

```
Layer              #         PPL        ΔPPL  worst_ckpt  drift_mean   drift_max
---------------------------------------------------------------------------
FP32 baseline      0      59.640           —           —        0.000      0.0000
h0.c_fc            1        ???          ???         ???          ???         ???
h0.c_proj          2        ???          ???         ???          ???         ???
h1.c_fc            3        ???          ???         ???          ???         ???
h1.c_proj          4        ???          ???         ???          ???         ???
h2.c_fc            5        ???          ???         ???          ???         ???
h2.c_proj          6        ???          ???         ???          ???         ???
h3.c_fc            7        ???          ???         ???          ???         ???
h3.c_proj          8        ???          ???         ???          ???         ???
h4.c_fc            9        ???          ???         ???          ???         ???
h4.c_proj         10        ???          ???         ???          ???         ???
h5.c_fc           11        ???          ???         ???          ???         ???
h5.c_proj         12        ???          ???         ???          ???         ???
h6.c_fc           13        ???          ???         ???          ???         ???
h6.c_proj         14        ???          ???         ???          ???         ???
h7.c_fc           15        ???          ???         ???          ???         ???
h7.c_proj         16        ???          ???         ???          ???         ???
h8.c_fc           17        ???          ???         ???          ???         ???
h8.c_proj         18        ???          ???         ???          ???         ???
h9.c_fc           19        ???          ???         ???          ???         ???
h9.c_proj         20        ???          ???         ???          ???         ???
h10.c_fc          21        ???          ???         ???          ???         ???
h10.c_proj        22        ???          ???         ???          ???         ???
h11.c_fc          23        ???          ???         ???          ???         ???
h11.c_proj        24        ???          ???         ???          ???         ???
```

Final PPL: ???  |  Amplification: ???×  |  Tipping points: ???

---

### Results: flat bd=16 K=256

```
Layer              #         PPL        ΔPPL  worst_ckpt  drift_mean   drift_max
---------------------------------------------------------------------------
FP32 baseline      0      59.640           —           —        0.000      0.0000
h0.c_fc            1        ???          ???         ???          ???         ???
h0.c_proj          2        ???          ???         ???          ???         ???
h1.c_fc            3        ???          ???         ???          ???         ???
h1.c_proj          4        ???          ???         ???          ???         ???
h2.c_fc            5        ???          ???         ???          ???         ???
h2.c_proj          6        ???          ???         ???          ???         ???
h3.c_fc            7        ???          ???         ???          ???         ???
h3.c_proj          8        ???          ???         ???          ???         ???
h4.c_fc            9        ???          ???         ???          ???         ???
h4.c_proj         10        ???          ???         ???          ???         ???
h5.c_fc           11        ???          ???         ???          ???         ???
h5.c_proj         12        ???          ???         ???          ???         ???
h6.c_fc           13        ???          ???         ???          ???         ???
h6.c_proj         14        ???          ???         ???          ???         ???
h7.c_fc           15        ???          ???         ???          ???         ???
h7.c_proj         16        ???          ???         ???          ???         ???
h8.c_fc           17        ???          ???         ???          ???         ???
h8.c_proj         18        ???          ???         ???          ???         ???
h9.c_fc           19        ???          ???         ???          ???         ???
h9.c_proj         20        ???          ???         ???          ???         ???
h10.c_fc          21        ???          ???         ???          ???         ???
h10.c_proj        22        ???          ???         ???          ???         ???
h11.c_fc          23        ???          ???         ???          ???         ???
h11.c_proj        24        ???          ???         ???          ???         ???
```

Final PPL: ???  |  Amplification: ???×  |  Tipping points: ???

---

### Key findings

**1. Accumulation profile shape.**

Fill in after run. Possible outcomes and their interpretation:

- **Step-like (dominated by one layer):** A single sublayer (e.g., h0.c_fc) accounts for most
  of the PPL jump. The mechanism is a particularly high-sensitivity layer, not distributed
  accumulation. Fixing just that layer could recover most of the lost quality. Check: does the
  worst_ckpt shift immediately after that layer's addition?

- **Approximately linear:** Each sublayer contributes roughly equal ΔPPL. Error compounds at a
  constant per-layer rate, like compound interest. The residual stream drift should grow
  monotonically across all 24 steps. Suggests no single "bottleneck layer" — the problem is
  architectural depth.

- **Exponential / accelerating:** Later layers cause disproportionately larger ΔPPL jumps.
  The residual stream has already drifted by the time later layers are added, so their inputs
  x_l are far from the distribution they were trained on. Per-layer error then multiplies against
  an already-corrupted activation, compounding super-linearly. Confirmed if drift_mean grows
  exponentially across checkpoints.

- **Front-loaded (early layers dominate):** Early blocks (h0, h1) are the most sensitive.
  In GPT-2, lower blocks process more fundamental token representations; errors there cascade
  through all subsequent blocks. The worst_ckpt column should move to higher checkpoint indices
  as early layers are quantized (their error has had more blocks to propagate through).

**2. Comparison: per-row vs flat accumulation profiles.**

Expected: per-row has smaller ΔPPL per step (tighter reconstruction per row), but both
configs may share the same qualitative profile shape (linear/exponential/step-like). The
shape tells us about GPT-2's inter-layer error propagation; the magnitude tells us about
per-scheme reconstruction quality.

If the profiles have different shapes: the quantization scheme itself changes which layers
are most sensitive. Per-row's row-wise codebooks give each output neuron its own discrete
alphabet — this may redistribute sensitivity differently than flat's shared codebook.

Reference amplification factors (from prior experiments):
- Per-row K=64 all-layer (Exp 17C):      PPL = 421.9   → 7.5×  amplification
- Flat K=256 all-layer (Exp 9):           PPL = 3,042.4 → 8.0×  amplification

**3. Residual stream drift analysis.**

The worst_ckpt column indicates where in the residual stream the damage is most severe.

- If worst_ckpt = 12 (final block output) consistently: damage accumulates to the end;
  no single block is an amplification hotspot; the token predictions at the LM head see
  the worst drift.

- If worst_ckpt = 0 or 1 (early checkpoints): early corruption is not recovered by later
  blocks — GPT-2's residual stream propagates errors faithfully forward without attenuation.
  This is the expected physics: the residual connection x_{l+1} = x_l + f_l(x_l) ensures
  early errors are added into every subsequent state.

- If worst_ckpt jumps to progressively later indices as more layers are quantized: the drift
  front moves downstream with each additional quantized layer. The per-step worst-checkpoint
  trace describes how the error "wave" propagates through the network depth.

**4. Physics interpretation.**

GPT-2's residual stream is a conserved "spin current" flowing from embedding to LM head.
Each MLP sublayer acts as a local perturbation: W_q ≠ W means each layer injects
δh = (W_q - W)x into the stream. Because the residual connection carries all previous errors
forward unchanged, the stream accumulates perturbations additively:

  x_L ≈ x*_L + Σ_{l=1}^{L} δh_l(x_l)

where x*_L is the FP32 final state. The amplification arises from two mechanisms:

1. **Direct additive accumulation:** Each δh_l adds directly to the stream. With 24 sublayers
   and mean per-layer error ε, the total perturbation grows as O(24ε) if errors are
   uncorrelated and the stream remains approximately linear.

2. **Nonlinear compounding:** The GeLU nonlinearity inside each MLP means δh_l depends on x_l,
   which is itself perturbed by all previous layers. If earlier errors push x_l into a
   different GeLU regime, subsequent layers amplify those errors nonlinearly. This explains
   why single-layer PPL is near-lossless (one small perturbation → GeLU is approximately
   linear in a small neighborhood) but all-layer PPL explodes (24 perturbations → cumulative
   GeLU regime shift).

The worst_ckpt data distinguishes between these two mechanisms: if drift_max is concentrated
at the final checkpoint regardless of which layer was just added, mechanism 1 dominates
(pure accumulation). If drift_max jumps to later checkpoints specifically when early layers
are quantized, mechanism 2 dominates (early corruption causes nonlinear compounding in later
blocks).

**5. Implications for next experiments.**

Fill in after run. Possible directions depending on findings:

- **If step-like with one dominant layer:** Target that layer for higher-quality quantization
  (more K, smaller bd, or no quantization). A "mixed precision" strategy — full precision for
  the 2–3 most sensitive layers, heavy quantization for the rest — may recover most of the PPL.

- **If linear accumulation:** Per-layer improvement directly translates to all-layer improvement.
  The correct next step is to push per-layer quality higher (larger K, distillation-based
  codebook, or activation-aware calibration). Exp 19 candidate: activation-calibrated per-row.

- **If exponential / accelerating:** The quantization order matters. Quantizing later layers first
  (reverse order) may reduce compounding, since later layers would be quantized while the
  residual stream is still close to FP32. Exp 19 candidate: reverse-order accumulation profile.

- **If early layers dominate:** Layerwise distillation starting from h0 is the priority.
  The first 2–4 sublayers need near-perfect reconstruction; the rest can be compressed heavily.
  Exp 19 candidate: non-uniform compression — low bpw on h4+ layers, high bpw on h0–h3.

---
```

### 3b. Commit

```bash
cd /Users/faulknco/projects/spin-quant
git add FINDINGS_LOG.md
git commit -m "Record Experiment 18 findings: collective behavior profiling results"
```

---

## Reference: complete file listing

- Script to create:  `/Users/faulknco/projects/spin-quant/experiments/accumulation_profile.py`
- Log to update:     `/Users/faulknco/projects/spin-quant/FINDINGS_LOG.md`
- This plan:         `/Users/faulknco/projects/spin-quant/docs/plans/2026-03-11-accumulation-profile.md`
- Design doc:        `/Users/faulknco/projects/spin-quant/docs/plans/2026-03-11-accumulation-profile-design.md`
- Reference scripts: `/Users/faulknco/projects/spin-quant/experiments/per_row_all_layer.py`
                     `/Users/faulknco/projects/spin-quant/experiments/eval_perplexity.py`
                     `/Users/faulknco/projects/spin-quant/src/codebook.py`

## Checklist

- [ ] Task 1: `experiments/accumulation_profile.py` created and syntax-verified
- [ ] Task 1: committed
- [ ] Task 2: experiment run to completion (~45-60 min)
- [ ] Task 2: per-row and flat tables recorded in full
- [ ] Task 2: tipping points identified for each config
- [ ] Task 3: `## Experiment 18` section inserted in FINDINGS_LOG.md before `## Experiment 17`
- [ ] Task 3: all ??? placeholders replaced with actual numbers
- [ ] Task 3: committed
