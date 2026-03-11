# Experiment 19B: Activation-Calibrated Per-Row K-Means — Design

**Date:** 2026-03-11
**Status:** Approved

---

## Background

Per-row k-means (Exp 14) treats all n_blocks_per_row blocks in each output row equally
during k-means. But not all blocks are equally important: if input activations at block
position j are near zero, the quantization error for that block barely affects output.
Weighting blocks by their activation magnitude during k-means should concentrate
reconstruction quality where it matters most.

## Question

Does weighting k-means by FP32 input activation magnitudes reduce PPL, especially at
lower K where the codebook is most budget-constrained?

## Design

### Algorithm

For each MLP sublayer (c_fc or c_proj) in GPT-2:

1. **Collect activation weights:**
   Run N_CALIB=20 calibration texts through the FP32 model with forward hooks.
   For each layer, capture input tensor x of shape [N_tokens, in_features].
   Compute per-block weight: `act_weight[j] = mean(||x[:, j*d:(j+1)*d]||²)`
   for j = 0..n_blocks-1. This gives a vector of shape [n_blocks] per layer.

2. **Weighted per-row k-means:**
   For row i, the block data is `row_blocks[i]` of shape [n_blocks, block_dim].
   In standard k-means, each block contributes equally to centroid updates.
   In weighted k-means, block j is scaled by `act_weight[j]` in both:
   - Distance computation: `d(block, centroid) = act_weight[j] * ||block - centroid||²`
   - Centroid update: `centroid_k = Σ_j (act_weight[j] * block_j) / Σ_j act_weight[j]`
     (only for blocks assigned to centroid k)

3. **Quantize all 24 MLP sublayers** with activation-calibrated per-row k-means.

4. **Evaluate PPL** on 50 WikiText-2 texts.

### Activation capture

Hook `model.transformer.h[bi].mlp.c_fc` input (= output of LayerNorm before MLP).
Hook `model.transformer.h[bi].mlp.c_proj` input (= output of GeLU after c_fc).
Use FP32 model only. Collect all N_CALIB texts, concatenate tokens, compute block-wise
L2 mean across all tokens.

### Weighted k-means implementation

Local function `weighted_kmeans(blocks, weights, K, n_iter, seed)` in the script.
`blocks`: [N, d] tensor of all blocks for one row.
`weights`: [N] tensor of per-block importance weights.
Initialize centroids by weighted random sampling (probability ∝ weight).
Each iteration: assign each block to nearest centroid (distance scaled by weight),
update centroids as weighted mean of assigned blocks.

No changes to `src/codebook.py` needed.

### Configs (all bd=8, all-layer quantization)

| Config | K   | calibrated | Notes                    |
|--------|-----|------------|--------------------------|
| A      | 16  | no         | Unweighted baseline      |
| B      | 16  | yes        | Calibrated — tight budget |
| C      | 32  | no         | Unweighted baseline      |
| D      | 32  | yes        | Calibrated               |
| E      | 64  | no         | Unweighted (Exp 17C ref) |
| F      | 64  | yes        | Calibrated sanity check  |

### Output

```
Config       K  calibrated      PPL      delta vs uncal
--------------------------------------------------------
(A) K=16     16     no       14353.6         —
(B) K=16     16    yes          ???       ???
(C) K=32     32     no         ???         —
(D) K=32     32    yes          ???       ???
(E) K=64     64     no         421.9         —
(F) K=64     64    yes          ???       ???
```

Note: K=16 uncalibrated PPL=14353.6 from Exp 17A.

### File

`experiments/activation_calibrated_per_row.py`

### Reference numbers

- Per-row K=16 all-layer (Exp 17A): PPL = 14,353.6
- Per-row K=64 all-layer (Exp 17C): PPL = 421.9
- FP32 baseline:                     PPL ≈ 59.6–63.3
