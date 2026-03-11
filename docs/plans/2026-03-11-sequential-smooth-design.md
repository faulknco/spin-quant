# Experiment 13: Sequential SmoothQuant Calibration — Design

**Date:** 2026-03-11
**Status:** Approved

---

## Background

### What we know so far (Experiments 1–12)

**Phase transition (Exp 6):** GPT-2 h0.c_fc has a sharp PPL discontinuity at bpw≈0.5.
Below the transition: chaotic/broken phase. Above: ordered phase. Near-lossless at bpw≈1.5.

**Layer sensitivity (Exp 7):** h0.c_fc is uniquely hard (analogous to UV scale in RG).
h0.c_proj, h5, h11 are near-lossless at bpw=0.375. Critical bpw=0.5 is a property of h0.c_fc.

**Block_dim effect (Exp 8):** Critical bpw=0.5 is universal across bd=4, 8, 16.
Best at bpw=0.5: bd=8, K=16 (PPL=154) beats bd=16, K=256 (PPL=360).

**All-layer quantization (Exp 9):** All 24 MLP layers simultaneously.
- bpw=0.5, bd=16, K=256: PPL=3,042 (8.0× amplification)
- Non-monotone: more bits → worse all-layer PPL (bpw=1.0 gives PPL=19,571)
- Best: flat bd=16 K=256 at bpw=0.5

**SmoothQuant (Exp 10/10b):** Scale migration to activation path.
- Forward: `F.linear(x * s_j, W / s_j, bias)` where s_j = std_j^α × H_j^(0.5*(1-α))
- α=0.5 optimal: PPL=170 vs 381 flat — 55% gap recovery at single-layer
- H-RMSE is NOT a predictor of PPL quality

**SmoothQuant + bd sweep (Exp 11):** Improvements don't combine.
- bd=8 flat: PPL=154 (best); smooth bd=16: PPL=170; smooth bd=8: PPL=271
- Each technique addresses column scale heterogeneity from a different angle; combining over-corrects

**All-layer SmoothQuant (Exp 12) — CATASTROPHIC FAILURE:**
- Independent calibration (all layers calibrated against base model): PPL=36,946
- vs flat all-layer: PPL=3,042 — SmoothQuant made it 12× WORSE
- Root cause: each layer's s_j calibrated for base model activations; when all layers are
  modified, layer N receives activations from already-modified layers 0..N-1 that don't
  match what layer N was calibrated on. Mismatch compounds multiplicatively: 217.7×
  amplification vs 8.0× for flat.

---

## Problem Statement

Independent per-layer SmoothQuant calibration breaks all-layer quantization because each
layer's scale is calibrated against the wrong activation distribution.

**Hypothesis:** Sequential calibration — processing layers in forward order and calibrating
each layer using the model with all prior layers already quantized — gives each layer access
to its true inference-time activation distribution, fixing the mismatch.

---

## Design

### Algorithm

Process all 24 MLP sublayers in forward order: h0.c_fc, h0.c_proj, h1.c_fc, ..., h11.c_proj.

For each layer i:
1. Run 50 calibration texts through the **current model** (with layers 0..i-1 already quantized)
2. Hook captures activations at layer i (still FP32, but downstream of quantized layers)
3. Compute col_scale: `s_j = std(W[:,j])^0.5 × sqrt(H_diag[j])^0.5` (α=0.5)
4. Quantize layer i: `W_smooth = W / s_j`, flat k-means bd=16 K=256
5. Replace layer i in the model with `_SmoothLinear` (forward: `F.linear(x * s_j, W_q, bias)`)
6. Proceed to layer i+1

### Key implementation change from Exp 12

```python
# Exp 12 (broken):
layer = quantize_layer_smooth(base_model, tokenizer, calib_texts, raw_base, ...)
#                              ^^^^^^^^^^  always uses base model activations

# Exp 13 (fix):
layer = quantize_layer_smooth(model, tokenizer, calib_texts, raw, ...)
#                              ^^^^^  uses current model (with prior layers quantized)
#                                                            ^^^  hook on live layer in model
```

### Configuration

| Parameter | Value |
|-----------|-------|
| Mode | smooth (SmoothQuant α=0.5) |
| block_dim | 16 |
| K | 256 |
| bpw | 0.5 |
| Calibration texts | 50 per layer |
| Eval texts | 100 |
| α | 0.5 |

Identical to Exp 12 in all parameters except the calibration model reference.

### Expected outcomes

- **Best case (full fix):** Sequential PPL ≈ proportional prediction ≈ 1,357
  (3,042 × 170/381 — if the single-layer 55% gain survives all-layer)
- **Partial fix:** Sequential PPL between 3,042 (flat) and 36,946 (independent smooth)
- **No fix:** Sequential PPL ≈ 36,946 — distribution mismatch is not the cause

### Output

```
Baseline (FP32):                          59.640
[Exp 9 ref]  flat bd=16 K=256:          3,042.435
[Exp 12 ref] indep smooth bd=16 K=256: 36,945.867
[Exp 13]     seq   smooth bd=16 K=256:      ???
```

---

## Files

- **New:** `experiments/sequential_smooth.py`
- **Reference:** `experiments/smooth_all_layer.py` (Exp 12 — source to copy from)
- **Log:** `FINDINGS_LOG.md` (update after run)

---

## Findings Log Summary

All findings from Exp 1–12 are recorded in `/Users/faulknco/projects/spin-quant/FINDINGS_LOG.md`.

Key cumulative results table:

| Experiment | Target | Config | PPL | Notes |
|------------|--------|--------|-----|-------|
| Exp 6 | h0.c_fc | flat bd=16 K=256 bpw=0.5 | 381 | Phase transition at bpw=0.5 |
| Exp 8 | h0.c_fc | flat bd=8 K=16 bpw=0.5 | 154 | Best single-layer |
| Exp 10 | h0.c_fc | smooth α=0.5 bd=16 K=256 | 170 | SmoothQuant breakthrough |
| Exp 9 | all 24 layers | flat bd=16 K=256 | 3,042 | Best all-layer so far |
| Exp 12 | all 24 layers | smooth bd=16 K=256 (indep) | 36,946 | Catastrophic mismatch |
| **Exp 13** | **all 24 layers** | **smooth bd=16 K=256 (seq)** | **TBD** | **This experiment** |
