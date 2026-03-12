# spin-quant

Physics-inspired k-means codebook quantization for LLM weights.

Research project exploring how far per-row k-means quantization can compress GPT-2 weights
while preserving language model quality, using techniques motivated by statistical physics.

---

## Key Results

Full model quantization of GPT-2 (117M parameters), all 48 sublayers (MLP + attention):

| Config | bpw | PPL | vs FP32 | compression |
|--------|-----|-----|---------|-------------|
| FP32 baseline | 32.0 | 63.3 | 1.0× | 1× |
| N=0 (uniform K=64, act_cal) | 0.787 | 321.7 | 5.08× | ~13× |
| N=8 (top-8 K=128, rest K=64, act_cal) | 0.805 | 180.0 | 2.85× | ~13× |
| N=16 (top-16 K=128, rest K=64, act_cal) | 0.822 | 120.1 | 1.90× | ~13× |
| N=24 (all K=128, act_cal) | 0.836 | 84.2 | 1.33× | ~13× |

All configs include attention layers quantized at K=96 (lossless — exact reconstruction).

**Best single result: PPL=84.2 at 0.836 bpw, 38× parameter compression vs FP32 bit count.**

---

## Method

### Core technique: per-row k-means with activation calibration

Each weight matrix row is independently quantized using k-means:
- `block_dim=8`: each row is split into blocks of 8 weights
- `K` centroids per row: each block replaced by its nearest centroid
- `bpw = log2(K) / block_dim`: bits per weight
- **Activation calibration**: blocks weighted by mean input activation magnitude
  `||x_j||²` during k-means, aligning the codebook with the actual inference distribution

### Non-uniform K allocation (Exp 18 + 20B/21)

Not all layers are equally sensitive to quantization. Layer sensitivity was measured by
sequential single-layer quantization (Exp 18 ΔPPL ranking). The top-N most sensitive
layers receive K=128; the rest receive K=64. This gives a smooth PPL/bpw Pareto frontier.

Top 8 most sensitive MLP layers (Exp 18 ΔPPL):

| Layer | ΔPPL |
|-------|------|
| h6.c_proj | +79.4 |
| h5.c_proj | +38.2 |
| h7.c_proj | +33.6 |
| h6.c_fc | +33.2 |
| h10.c_proj | +22.3 |
| h4.c_proj | +21.2 |
| h8.c_proj | +20.2 |
| h11.c_fc | +19.8 |

### Attention layers (Exp 22)

Attention sublayers (`c_attn` [2304×768], `c_proj` [768×768]) have `n_blocks_per_row = 96`
at `block_dim=8`. Setting K=96 gives **exact reconstruction** (lossless) at 0.823 bpw —
every block gets its own centroid. Full-model quantization adds zero PPL cost for attention.

---

## Phase transition

A sharp phase transition exists at K≈64 for `block_dim=8`. Below this threshold,
quantization error is catastrophic (PPL > 1000); above it, quality degrades gracefully.
This is analogous to a first-order spin ordering transition: K=64 is the critical point.

- K < 64: chaotic phase — k-means finds poor local minima, PPL non-monotone in K
- K ≥ 64: ordered phase — PPL decreases smoothly with increasing K
- K = n_blocks_per_row: exact reconstruction (lossless)

Activation calibration helps only in the ordered phase (K≥64); it hurts in the chaotic
phase (K≤32), consistent with the calibration signal being meaningful only when the
codebook is large enough to represent the distribution.

---

## Negative results

- **bd=4** (Exp 25): doubles blocks per row but phase transition moves to K≈64 at 1.5 bpw.
  bd=8 dominates the Pareto frontier — bd=4 is never worth the bpw cost.
- **Gradient calibration** (Exp 23): per-row gradient norms add no signal over activation
  calibration. Structural reason: gradient norm is a per-row scalar that doesn't change
  within-row centroid placement in per-row k-means.
- **Greedy bpw allocation** (Exp 26): sensitivity scores are not valid marginal benefit
  proxies for multi-step K upgrades. The binary K∈{64,128} top-N scheme beats greedy
  allocation by 12–33 PPL at every budget point.

---

## Experiments

| # | Name | Key finding |
|---|------|-------------|
| 6 | bpw sweep (corrected) | Phase transition at bpw≈0.5 for flat k-means, bd=16 |
| 8 | block_dim effect | Critical bpw decreases with larger block_dim |
| 10 | SmoothQuant | Scale migration helps but per-row codebooks dominate |
| 14 | Per-row codebooks | **Major breakthrough** — PPL=56.4 at 0.75 bpw (single layer) |
| 16 | Per-row bpw sweep | Critical bpw=0.188 for bd=8; K=n_blocks → exact reconstruction |
| 17 | Per-row all-layer | PPL=422 at 0.75 bpw (all MLP, flat K, uncalibrated) |
| 18 | Accumulation profile | Layer sensitivity ranking; residual stream integrates errors |
| 19A | Non-uniform compression | K=64 floor is essential; K=32 background is catastrophic |
| 19B | Activation calibration | −27% PPL at K=64; hurts at K≤32 |
| 20A | Calibration crossover | Non-monotone crossover at K=48–56; helps at K≥64 |
| 20B | Top-N K128/K64 sweep | Smooth Pareto frontier; N=24 PPL=87.6 uncalibrated |
| 21 | Combined cal + non-uniform | Techniques are additive; +10–27% improvement at every N |
| 22 | Attention quantization | K=96 attention is lossless (exact reconstruction) |
| 23 | Gradient calibration | Negative — same as act_cal structurally in per-row k-means |
| 24 | Full optimal model | Capstone — PPL=84.2 at 0.836 bpw, attention K=96 free |
| 25 | Block dim sweep | Negative — bd=4 catastrophic at matched bpw; bd=8 optimal |
| 26 | Greedy budget allocation | Negative — binary top-N beats greedy by 12–33 PPL |

Full results and analysis: [`FINDINGS_LOG.md`](FINDINGS_LOG.md)

---

## Running experiments

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run any experiment
.venv/bin/python experiments/full_model_optimal.py
.venv/bin/python experiments/block_dim_allayer.py
.venv/bin/python experiments/budget_allocation.py
```

All experiments use GPT-2 (gpt2) from HuggingFace and WikiText-2 for evaluation.
N_EVAL=50 texts, N_CALIB=20 texts, MAX_LEN=128 tokens.

---

## Structure

```
src/
  codebook.py          k-means implementation
  layers.py            quantized layer classes
  physics.py           physics-inspired variants (O(N), RG, etc.)
  frequency.py         DCT/WHT quantization
experiments/
  eval_perplexity.py   shared PPL evaluation utility
  full_model_optimal.py  Exp 24 — capstone full model
  combined_cal_nonuniform.py  Exp 21
  attention_quantization.py   Exp 22
  budget_allocation.py        Exp 26
  block_dim_allayer.py        Exp 25
  ... (see experiments/ for all)
docs/plans/            implementation plans for major experiments
FINDINGS_LOG.md        detailed results and analysis for all experiments
RESEARCH_IDEAS.md      physics-motivated design space
```

---

## Future work

- **FW-3**: Embedding + lm_head quantization (`wte`, `wpe`, `lm_head`)
- **FW-4**: Cross-row shared codebooks with gradient weighting
- **FW-5**: Residual (multi-stage / RVQ) quantization
- **FW-6**: Scale to GPT-2 medium/large
