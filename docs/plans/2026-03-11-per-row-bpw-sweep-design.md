# Experiment 16: Per-Row BPW Sweep — Design

**Date:** 2026-03-11
**Status:** Approved

---

## Background

Exp 14 showed per-row codebooks are transformative at bpw=0.5:
- Flat bd=16 K=256: PPL=381 (phase transition at bpw≈0.5)
- Per-row bd=8 K=16: PPL=71 (95% gap recovery at same bpw)
- Per-row bd=16 K=32 (bpw=0.313): PPL=58 — near-lossless BELOW flat's critical threshold
- Per-row bd=8 K=64 (bpw=0.75): PPL=56.4 — 99.9% gap recovery

The flat phase transition at bpw≈0.5 is a property of the shared codebook — below K=256, the
single codebook can't cover both high-norm and low-norm rows. Per-row eliminates this by giving
each row its own K centroids. The per-row critical bpw must be lower than 0.5, but we don't
know where it is or what shape the transition takes.

## Question

Where is the per-row phase transition? Is it sharp (first-order, like flat) or gradual?

## Design

**Target:** h0.c_fc only (hardest layer, reference for all prior experiments)

**Sweep:**
- bd=8: K = 2, 4, 8, 16, 32, 64, 96  → bpw = 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875
- bd=16: K = 2, 4, 8, 16, 32, 48     → bpw = 0.063, 0.125, 0.250, 0.313, 0.375, 0.396

Constraint: K ≤ n_blocks_per_row = in_features / block_dim (768/8=96 for bd=8; 768/16=48 for bd=16)

**Reference lines from prior experiments (flat):**
- bd=8  K=16  bpw=0.5: PPL=154 (Exp 8)
- bd=16 K=256 bpw=0.5: PPL=381 (Exp 6)
- bd=8  K=32  bpw=0.625: PPL=16,423 (non-monotone chaos, Exp 8)

**Output:** PPL table by bpw for each bd, clearly marking where per-row breaks down.

**File:** `experiments/per_row_bpw_sweep.py`
**Eval texts:** 100, MAX_LEN=128
