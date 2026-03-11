# Experiment 17: Per-Row All-Layer Quantization — Design

**Date:** 2026-03-11
**Status:** Approved

---

## Background

Per-row codebooks (Exp 14) dramatically improved single-layer PPL:
- Per-row bd=8 K=16 bpw=0.5: PPL=71 (vs flat 154)
- Per-row bd=16 K=32 bpw=0.313: PPL=58 (near-lossless)
- Per-row bd=8 K=64 bpw=0.75: PPL=56.4 (99.9% gap recovery)

Flat all-layer (Exp 9, all 24 MLP layers) gave PPL=3,042 at bpw=0.5 (8× amplification).
SmoothQuant all-layer (Exp 12/13) catastrophically failed regardless of calibration strategy.

Per-row codebooks don't alter the activation distribution (precomputed W_q forward pass),
so they should avoid the inter-layer mismatch that killed SmoothQuant all-layer. The question
is whether the error amplification factor is smaller (proportional to single-layer improvement)
or whether all-layer stacking reveals new failure modes.

## Question

Does the per-row single-layer improvement (2.2× at bpw=0.5) carry through to all 24 layers?
Expected if proportional: 3,042 × (71/154) ≈ 1,400 at bpw=0.5.

## Design

**Target:** All 24 MLP layers (h0-h11 × c_fc, c_proj), same as Exp 9.

**Configs:**
- (A) per-row bd=8  K=16  bpw=0.500 — direct comparison to flat all-layer Exp 9 (PPL=3,042)
- (B) per-row bd=16 K=32  bpw=0.313 — near-lossless single-layer, test all-layer
- (C) per-row bd=8  K=64  bpw=0.750 — highest quality single-layer (PPL=56.4), test all-layer

**Reference numbers:**
- Baseline FP32: PPL=59.640
- Flat all-layer bd=16 K=256 bpw=0.5 (Exp 9): PPL=3,042
- Per-row single-layer bd=8 K=16 bpw=0.5 (Exp 14): PPL=71
- Per-row single-layer bd=16 K=32 bpw=0.313 (Exp 14): PPL=58
- Per-row single-layer bd=8 K=64 bpw=0.75 (Exp 14): PPL=56

**Key implementation note:** Per-row quantization of each layer can use the base model
(no activation distribution change), so no sequential calibration needed — just deepcopy
and replace all layers with per-row quantized versions.

**File:** `experiments/per_row_all_layer.py`
**Calib texts:** 0 (per-row needs no H_diag)
**Eval texts:** 100, MAX_LEN=128
