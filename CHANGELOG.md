# Changelog

All experiments and notable changes, in reverse chronological order.

---

## 2026-03-12

### Experiment 26 — Greedy bpw budget allocation (negative result)
- Greedy allocation with K∈{64,96,128,192,256,384} and Exp 18 sensitivity scores
- **Result**: worse than binary top-N at every bpw point (12–33 PPL)
- Root cause: sensitivity proxy overestimates marginal gains beyond K=128
- `experiments/budget_allocation.py`

### Experiment 25 — Block dimension sweep bd=4 vs bd=8 (negative result)
- Swept bd∈{4,8} × K at matched bpw and matched K, all act_cal
- **Result**: bd=4 catastrophic at matched bpw (PPL=10,008 vs 321 at 0.75 bpw)
- bd=4 phase transition requires K≥64 at 1.5 bpw; bd=8 strictly dominates
- `experiments/block_dim_allayer.py`

### Experiment 24 — Full optimal model (capstone)
- Combined MLP top-N K128/K64 act_cal + attention K=96 (lossless)
- **Result**: attention K=96 adds exactly zero PPL cost at all N values
- Best: PPL=84.2 at 0.836 bpw (1.33× FP32), all 48 sublayers quantized
- `experiments/full_model_optimal.py`

### Experiment 23 — Gradient-sensitivity weighted k-means (negative result)
- Per-row gradient norm × activation weight as k-means block weights
- **Result**: worse than activation calibration (K=64: 356.5 vs 321.7)
- Root cause: gradient norm is a per-row scalar, doesn't affect within-row centroid placement
- `experiments/gradient_sensitivity.py`

### Experiment 22 — Attention layer quantization
- Extended per-row k-means to attention sublayers (c_attn, c_proj)
- **Key finding**: K=96 attention is lossless (K=n_blocks_per_row=96 → exact reconstruction)
- Combined MLP K64cal + attn K64cal PPL=343.5 (calibration cannot fully rescue K=64 attn)
- `experiments/attention_quantization.py`

---

## 2026-03-11

### Experiment 21 — Combined calibration + non-uniform compression
- Stacked activation calibration on top of top-N K128/K64 assignment
- **Result**: calibration improves every N by 10–27%; techniques are additive
- N=24 calibrated PPL=84.2 at 0.849 bpw
- `experiments/combined_cal_nonuniform.py`

### Experiment 20B — Top-N K128/K64 sweep (K=64 floor)
- Swept N=0..24 sensitive layers at K=128, rest K=64, uncalibrated
- **Result**: smooth monotone Pareto frontier; N=24 PPL=87.6 at 0.849 bpw
- `experiments/nonuniform_k64floor.py`

### Experiment 20A — Calibration crossover sweep
- Swept K=40–128 × {uncalibrated, calibrated}
- **Result**: non-monotone crossover — K=48 helped, K=56 hurt, K≥64 consistently helped
- Phase transition boundary at K≈56–64 separates chaotic from ordered regime
- `experiments/calibration_crossover.py`

### Experiment 19B — Activation-calibrated per-row k-means
- Block weights = mean squared input activation magnitude
- **Result**: −27% PPL at K=64 (439→321); hurts at K≤32
- `experiments/activation_calibrated_per_row.py`

### Experiment 19A — Non-uniform compression (data-driven top-N)
- Top-N Exp 18 ΔPPL-ranked layers get K=128; rest K=32
- **Result**: K=32 background catastrophic; K=64 floor is essential
- `experiments/nonuniform_compression.py`

### Experiment 18 — Layer-by-layer accumulation profiling
- Sequential single-layer quantization; PPL + residual stream drift at each step
- **Result**: gradual accumulation (no tipping point); worst_ckpt always=12; h6.c_proj most sensitive (+79.4 ΔPPL)
- Established Exp 18 ΔPPL sensitivity ranking used in all subsequent experiments
- `experiments/accumulation_profile.py`

### Experiment 17 — Per-row all-layer quantization
- Applied per-row k-means to all 24 MLP layers, uniform K
- **Result**: PPL=422 at 0.75 bpw (K=64), 7× better than flat k-means at same bpw
- `experiments/per_row_all_layer.py`

### Experiment 16 — Per-row bpw sweep
- bpw sweep for per-row k-means on h0.c_fc
- **Result**: critical bpw=0.188 (vs 0.5 for flat); K=n_blocks_per_row → lossless
- `experiments/per_row_bpw_sweep.py`

### Experiment 14 — Per-row codebooks (major breakthrough)
- One k-means codebook per output row instead of one per layer
- **Result**: PPL=56.4 at 0.75 bpw (single layer) — essentially lossless
- `experiments/per_row_codebook.py`

### Earlier experiments (Exp 1–13)

| # | Description | Outcome |
|---|-------------|---------|
| 1–3 | Initial single-layer sweep, geometry | Established framework |
| 4–5 | Hessian-weighted k-means | Marginal gain; hard to scale |
| 6 | Corrected bpw sweep | Phase transition at bpw≈0.5, bd=16 |
| 7 | Layer-dependent critical bpw | Transition point varies by layer |
| 8 | block_dim effect | Larger bd → lower critical bpw |
| 9 | All-layer flat quantization | PPL=3000+ at 0.75 bpw — baseline |
| 10 | SmoothQuant scale migration | PPL breakthrough for single layer, doesn't scale |
| 11 | SmoothQuant + block_dim | No interaction benefit |
| 12 | All-layer SmoothQuant | PPL=3000+ — SmoothQuant doesn't help at full scale |
| 13 | Sequential SmoothQuant | PPL=31,662 — worse; calibration ordering matters |
| 15 | Block-local Hessian k-means | No improvement over standard k-means |
