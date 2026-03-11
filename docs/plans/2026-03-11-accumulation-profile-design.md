# Experiment 18: Collective Behavior Profiling — Design

**Date:** 2026-03-11
**Status:** Approved

---

## Background

Per-row bd=8 K=64 all-layer gives PPL=422 (7.5× amplification from single-layer PPL=56).
Flat bd=16 K=256 all-layer gives PPL=3,042 (8× amplification from single-layer PPL=381).

Single-layer per-row is near-lossless (PPL=60), but stacking 24 layers produces 7.5× error
amplification. The mechanism is unknown: is it dominated by a few sensitive early layers,
does it compound uniformly across all 24 layers, or is there a tipping point?

## Question

How and where does quantization error amplify across 24 layers?
- Is accumulation linear, exponential, or step-like?
- Are certain layers (e.g. h0) disproportionately responsible?
- Does residual stream drift explain the amplification?
- Do per-row and flat have different accumulation profiles?

## Design

### Algorithm

For each config (per-row bd=8 K=64, flat bd=16 K=256):

1. Start with FP32 base model (deepcopy for in-place modification)
2. For each of 24 sublayers in forward order (h0.c_fc, h0.c_proj, ..., h11.c_proj):
   a. Quantize and replace that layer in the working model
   b. Eval PPL on 50 texts
   c. Run 20 calibration texts through both FP32 and working model simultaneously,
      recording residual stream tensors at all 13 checkpoints (before h0, after each block)
   d. Compute per-checkpoint: mean(|x_q - x_fp32|), std(|x_q - x_fp32|), max(|x_q - x_fp32|)
   e. Record: (layer_idx, PPL, Δmean_max, Δstd_max, Δmax_max) — worst checkpoint stats

### Residual stream capture

GPT-2 residual stream checkpoints: x_0 (input embeds), x_1..x_12 (after each transformer block).
Hook `model.transformer.h[i]` output to capture x_{i+1} after each block processes it.
Run same token batch through FP32 and quantized models, compare at each checkpoint.

### Configs

- **Primary:** per-row bd=8 K=64 bpw=0.750
- **Reference:** flat bd=16 K=256 bpw=0.500

### Output

Two side-by-side tables (one per config):

```
Layer added        PPL    ΔPPL    Δresid_mean  Δresid_std  Δresid_max
FP32 (baseline)   59.6     —           —            —           —
+h0.c_fc           ???    ???         ???          ???         ???
+h0.c_proj         ???    ???         ???          ???         ???
...
+h11.c_proj        ???    ???         ???          ???         ???
```

Mark "TIPPING POINT" at the first layer where PPL jumps > 2× previous checkpoint.

### File

`experiments/accumulation_profile.py`

### Reference numbers

- FP32 baseline: ~59.6
- Per-row K=64 final all-layer: 421.9 (Exp 17C)
- Flat K=256 final all-layer: 3,042.4 (Exp 9)
