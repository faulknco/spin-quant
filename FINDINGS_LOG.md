# spin-quant: Findings Log

Running record of all experiments, results, discussions and interpretations.
Updated after each session.

---

## Session 1 — Project genesis

### Source material
PDF: "What is the current research around creating infrastructure for AI in assembly code?"
(Perplexity research log, downloaded to ~/Downloads)

The document covered a chain of questions that converged on the core idea:

1. **AI infra in assembly** → not mainstream; real work is in kernel-level optimisation
   (SIMD/vector intrinsics, cuDNN-like stacks) and hardware co-design (custom ISAs for tensor ops).

2. **BitNet and 1-bit LLMs** → active research area. BitNet b1.58 (ternary weights, 8-bit activations)
   shows parity with FP16 models at 2B params while using far less compute.
   bitnet.cpp achieves up to 6.25× speedup over FP baseline on CPU via AVX2/NEON kernels.

3. **Ising model analogy** → user asked whether classical spin systems (Ising: {-1,+1}) could map
   onto LLM quantization. Answer: conceptually yes (discrete weight alphabet ≈ spin states),
   but pure 1-bit is too coarse because different channels need different dynamic ranges.
   BitNet uses ~1.58 bits = ternary code + shared block scale to buy flexibility.

4. **Heisenberg model** → more degrees of freedom (continuous spin on S², not ±1).
   Does NOT map more cleanly to LLM quantization — LLM weights are not unit vectors with O(3) symmetry.
   But the factorisation "norm × direction" (O(N)-style) is conceptually related.

5. **High-dimensional functions in physics** → quantum many-body wavefunctions, Green's functions,
   band structures: all are compressed with tensor networks (MPS/PEPS), basis truncations
   (Wannier functions, plane waves), and lattice RG coarse-graining.
   These are the same problem family as LLM codebook quantization.

6. **Practical codebooks for LLMs** → grouped quantization (per-channel scales) already does a
   primitive version. Tensor-network factorizations of weight matrices are being explored but niche.
   Physics-style hierarchical/learned codebooks are not yet mainstream.

7. **Concrete prototype** → user preferred PyTorch + physics background.
   PDF gave a full prototype: k-means codebook on weight blocks, O(N)-style norm×direction
   factorization, module wrapping as CodebookLinear.

### Project created
Path: `/Users/faulknco/projects/spin-quant`
Stack: Python 3.12, PyTorch, HuggingFace transformers, uv venv
Git: initialised, all work committed

---

## Experiment 0 — Initial sweep (FAILED: silent bug)

**What ran:** `experiments/sweep.py` — bits-per-weight vs PPL sweep over GPT-2 MLP layers.

**Result:**
```
bpw     PPL    delta
0.250   59.64  +0.00
0.375   59.64  +0.00
0.500   59.64  +0.00
1.000   59.64  +0.00
2.000   59.64  +0.00
3.000   59.64  +0.00
```

**Finding: SILENT BUG.** Delta = 0.00 across ALL settings because no layers were actually quantized.

**Root cause:** HuggingFace GPT-2 uses `transformers.pytorch_utils.Conv1D` (weight shape `[in, out]`)
rather than `torch.nn.Linear` (`[out, in]`). The `isinstance(layer, torch.nn.Linear)` check
silently returned False for every layer — the "quantized" model was always the full-precision model.

**Fix:** Added `conv1d_to_linear()` helper that transposes Conv1D weights to nn.Linear layout.
Detection now uses `isinstance(layer, (torch.nn.Linear, Conv1D))`.

**Lesson:** Always verify that quantization is actually happening — instrument with a layer count print.

---

## Brainstorm — Physics-inspired schemes beyond codebook

Discussed during session while sweep was (incorrectly) running.

### Implemented schemes

| Scheme | Physics origin | Core idea |
|--------|---------------|-----------|
| Scalar codebook | Ising spin alphabet | k-means on weight blocks, store centroid indices |
| O(N) factorization | Heisenberg spin (norm × direction) | Block = norm × unit-vector; separate codebooks |
| RG hierarchical | Wilson RG (coarse + residual) | Two-level codebook: global coarse + per-block correction |
| DCT quantization | Fourier/momentum-space truncation | DCT each block, keep top-k low-freq coefficients |
| WHT quantization | Spin-wave / Hadamard basis | Same but with ±1 Hadamard transform (hardware-efficient) |

### Proposed schemes (not yet implemented)

**Taylor/power series expansion**
Physics: perturbation theory (expand around free-field solution).
Idea: W ≈ Σ αₙ Bₙ where {Bₙ} is a fixed structured basis (Fourier matrices, Hadamard).
Only store scalar coefficients αₙ. Connection: LoRA uses rank-r additive decomposition;
this generalises with physics-motivated fixed bases.

**MPS tensor network decomposition**
Physics: Matrix Product States from quantum many-body physics.
Bond dimension χ controls entanglement/expressivity tradeoff.
Idea: reshape weight matrix into high-order tensor, decompose as MPS chain.
Different from SVD: captures higher-order correlations in weight structure.

**Spin-glass / Hessian-aware quantization** ← CURRENT TARGET
Physics: Sherrington-Kirkpatrick model. Optimal discrete assignment = ground state of
H({σ}) = Σᵢⱼ Jᵢⱼ σᵢ σⱼ where Jᵢⱼ = loss Hessian (sensitivity coupling matrix).
Idea: k-means in H^{1/2}-scaled weight space so we minimise loss-weighted error,
not flat RMSE. The Hessian diagonal Hⱼⱼ = E[xⱼ²] from calibration activations.

**RG flow across layers (layer-wise bit allocation)**
Physics: Wilson RG — theory parameters run with energy scale.
Observation: early layers may need more bits (fine-grained features), late layers fewer (semantic).
Idea: measure per-layer sensitivity, assign bit budgets along the RG flow direction.

---

## Experiment 1 — DCT power spectrum

**What ran:** `experiments/dct_spectrum.py`
**Status:** Spectrum completed; RMSE sweep killed by OOM (K=65536, ~90GB distance matrix).
**Fix applied:** Capped K at 4096 in the sweep.

**Results — DCT spectrum of h0.c_fc (block_dim=16):**
```
k=0   6.19%  ##
k=1   6.20%  ##
k=2   6.14%  ##
...
k=15  6.37%  ##
```
Every mode carries ~6.25% of energy — **completely flat, identical to white noise.**
Top 4 DCT modes = 24.8% energy (random baseline = exactly 25%).

**Layer-by-layer summary (top-4 DCT energy %):**
```
h0.c_fc:  24.8%    h0.c_proj: 25.8%
h1.c_fc:  25.0%    h1.c_proj: 25.8%
h2.c_fc:  25.3%    h2.c_proj: 26.1%
h3.c_fc:  25.1%    h3.c_proj: 26.2%
h4.c_fc:  24.7%    h4.c_proj: 25.7%
h5.c_fc:  24.9%    h5.c_proj: 25.3%
```
Uniform at ~25% across all layers. No layer has more Fourier structure than any other.

### Interpretation
**DCT quantization is not viable for LLM weights.**
GPT-2 weight blocks show no low-frequency spatial structure — they are incoherent in the
standard Fourier basis. This rules out JPEG-style compression for raw weight blocks.

**Connection to QuIP#:** QuIP# (Cornell/Meta, 2024) found the same thing and solved it by
applying a random Hadamard rotation first (incoherence processing), then quantizing.
The rotation spreads outliers uniformly, making scalar quantization work better.
Our WHT scheme is a first step in this direction.

---

## Experiment 2 — SVD singular value spectrum

**What ran:** `experiments/svd_spectrum.py`
**Status:** Fully completed.

**Results — all 24 MLP layers:**
```
Layer         shape        r90   r90%    α
h0.c_fc   (3072, 768)     501   65.2%  0.564
h0.c_proj  (768,3072)     513   66.8%  0.493
h1.c_fc   (3072, 768)     529   68.9%  0.495
h1.c_proj  (768,3072)     543   70.7%  0.401
... (all 24 layers) ...
h11.c_proj (768,3072)     530   69.0%  0.512
```
All layers: r90 ≈ 65-70% of full rank, α ≈ 0.40-0.57.

**Detailed spectrum for h0.c_fc:**
```
effective rank for 90% variance: 501  (65.2% of full rank)
effective rank for 99% variance: 704  (91.7% of full rank)
spectral decay exponent α:       0.564
σ[0..19]/σ[0]: 1.000  0.596  0.451  0.416  0.414  0.401  0.386  0.361  ...
```

**SVD vs DCT spectrum (first 16 modes):**
```
k=0   SVD: 14.9%   DCT: 6.2%   ← SVD has a genuine dominant mode
k=1   SVD:  8.9%   DCT: 6.2%   ← still some structure
k=2   SVD:  6.7%   DCT: 6.1%   ← rapidly approaching flat
k=3+  SVD: ≈ DCT   (both flat from here)
```

**RMSE sweep (h0.c_fc, bpw vs RMSE):**
```
rank    bpw     var%    RMSE_trunc   RMSE_svdq
1       0.013    3.8%    0.138484     0.138485
2       0.026    5.1%    0.137516     0.137517
4       0.052    6.5%    0.136483     0.136485
8       0.104    8.8%    0.134788     0.134797
16      0.208   12.2%    0.132255     0.132268
32      0.417   17.8%    0.127995     0.128014
64      0.834   27.2%    0.120447     0.120480
96      1.251   35.3%    0.113571     0.113616
128     1.668   42.4%    0.107129     0.107186
192     2.501   54.7%    0.095033     0.095116
256     3.335   64.8%    0.083719     0.083832
```

**Head-to-head at ~0.5 bpw:**
```
SVD-quant (r=38):             bpw=0.495  RMSE=0.126529  var=19.7%
Scalar codebook (K=256,d=16): bpw=0.500  RMSE=0.106799
Winner: Scalar codebook  (18% lower RMSE)
```
To match scalar codebook's RMSE=0.107, SVD needs r=128 at bpw=1.668 — **3.3× more bits.**

### Interpretation

**α ≈ 0.5 is the Marchenko-Pastur universality class.**
For a Gaussian random matrix with aspect ratio m/n=4 (GPT-2 c_fc: 3072/768), the Marchenko-Pastur
distribution predicts singular value decay consistent with α≈0.5. The fact that ALL 24 layers
show α≈0.5 is a strong signal: GPT-2 base weights sit in the bulk of random matrix theory.

**Implications:**
1. The weights are NOT low-rank in any practical sense. You need ~65% of modes for 90% variance.
2. SVD truncation (LoRA, etc.) works for *fine-tuning deltas ΔW* (which are low-rank)
   but NOT for compressing base weights (which are high-rank).
3. The consistent α across all layers suggests the spectral distribution is set by
   training dynamics (SGD noise pushes toward this universal distribution), not by task structure.

**Why scalar codebook wins:**
SVD finds the best global linear subspace. k-means finds the best local discrete alphabet
matching the actual weight block distribution. For high-rank matrices in the Marchenko-Pastur
bulk, there is no exploitable global subspace — but local distributions are still clusterable.

**The deeper question:**
The weights W are high-rank and random-looking. But the **Hessian H_W** (loss sensitivity
w.r.t. weights) may be low-rank — it encodes which directions in weight space actually
affect model outputs. This is the spin-glass coupling matrix Jᵢⱼ.
Quantizing in H^{1/2}-scaled space (Hessian-weighted codebook) should beat vanilla k-means
by protecting the sensitive directions while being aggressive on insensitive ones.

---

## Experiment 3 — Hessian-weighted codebook [IN PROGRESS]

**Hypothesis:** Minimising H-weighted error ||H^{1/2}(W - Wq)||² beats flat RMSE k-means
because not all weight directions equally affect model outputs.

**Method:**
- Calibration forward pass: 100 tokens through GPT-2, capture input activations to h0.c_fc
- H_diag[j] = E[x_j²] = mean squared activation of input dimension j (diagonal Hessian approx)
- Scale weights: W_scaled[i,j] = W[i,j] × sqrt(H_diag[j])
- Run k-means on scaled blocks
- Unscale centroids: C_unscaled = C_scaled / sqrt(H_diag[block_indices])
- Compare: flat k-means RMSE vs H-weighted k-means RMSE (both flat and H-weighted metrics)

**Status:** Completed.

**Results — H_diag statistics (h0.c_fc, 100 calibration texts):**
```
mean:             0.0222
std:              0.0519
min:              0.004670
max:              0.9910
dynamic range:    23.3 dB  (214× from min to max)
coeff variation:  2.336    (>> 1 = highly non-uniform)
'hot' dims (>10x mean): 0.7%   (~5-6 out of 768 dims)
```

**Results — comparison table (block_dim=16):**
```
K      bpw    flat_RMSE   h_RMSE_flat   h_RMSE_hq    gain%
16    0.250    0.125252     0.017313     0.016643      3.9%
64    0.375    0.115218     0.016134     0.015070      6.6%
256   0.500    0.106731     0.015016     0.013856      7.7%
1024  0.625    0.097229     0.013806     0.012570      9.0%
```

### Interpretation

**The Hessian is highly non-uniform (CV = 2.336, DR = 23.3 dB).**
The activation energy E[x_j²] varies by 214× across the 768 input dimensions.
Only 0.7% of dimensions (~5-6) are "hot" (>10× mean) — these are the LLM outlier dimensions
documented in LLM.int8() and SmoothQuant literature. Our H-diagonal weighting naturally
protects them without any special outlier handling.

**Hessian weighting consistently reduces the loss-weighted error (gain 3.9%→9.0%).**
The gain increases monotonically with K. This is physically meaningful:
- At small K: quantization is coarse everywhere; H-weighting helps little because the
  codebook is too small to allocate extra coverage to hot dimensions.
- At large K: finer codebook allows H-weighting to specifically cluster centroids near
  the values of hot-dimension weights; gain compounds with each additional centroid.
- Trend suggests gain would continue growing to ~12-15% at K=4096 before plateauing.

**Spin-glass framing:**
The coupling matrix J_ij is sparse: 99.3% of dimensions are weakly coupled (low E[x_j²]),
0.7% are strongly coupled ("frustrated spins"). Flat k-means treats all equally and
wastes codebook entries on insensitive dimensions. H-weighted k-means concentrates
entries near the sensitive directions — exactly what the spin-glass ground state search
should do.

**Practical conclusion:** Hessian weighting gives a real, consistent improvement in the
metric that matters (loss-weighted error), with gains increasing as codebook gets finer.
The absolute gain (~8% at K=256) is moderate, suggesting the H_diag approximation
captures the effect partially. Full off-diagonal Hessian (GPTQ approach) would likely
give larger gains at the cost of more computation.

---

## Open questions

1. Does Hessian-weighted codebook outperform flat k-means on PPL (not just H-weighted RMSE)?
   → Need PPL evaluation with Conv1D fix. Expected: yes, mirroring RMSE gain.
2. Does the gain increase further with more calibration data (>100 texts)?
   → H_diag estimate from 100 texts may be noisy; more data should sharpen it.
3. Is H_diag structured (low-rank)? What is its effective rank?
   → If H_diag itself has structure (e.g. follows a power law), this motivates a
      "spectral Hessian" approach: quantize in the eigenbasis of H, not just diagonal.
4. Does α≈0.5 hold for larger models (LLaMA, Mistral)?
   → If yes, Marchenko-Pastur bulk is a universal property of SGD-trained weights.
5. Can full off-diagonal Hessian (GPTQ-style) give substantially larger gains than diagonal?
   → Diagonal H_diag captures input sensitivity but not inter-weight correlations.
6. Can MPS bond dimension replace codebook size K as a compression knob?
7. Does per-layer bit allocation (RG flow) help when combined with Hessian weighting?
8. What is the "phase transition" bpw below which PPL diverges?
   (analogous to a phase transition in the spin model at critical temperature)

## Scheme performance summary (h0.c_fc, bpw≈0.5)

| Scheme | bpw | RMSE | H-RMSE | Notes |
|--------|-----|------|--------|-------|
| Flat k-means (scalar) | 0.500 | 0.1067 | 0.01502 | baseline |
| SVD-quant (r=38) | 0.495 | 0.1265 | — | 18% worse RMSE |
| H-weighted k-means | 0.500 | — | 0.01386 | 7.7% better H-RMSE |
| DCT (keep_k=8) | 0.500 | >> flat | — | not viable, flat spectrum |

**Ranking so far:** H-weighted k-means > Flat k-means >> DCT ≈ SVD-quant

The flat k-means scalar codebook is still the best RMSE baseline. Hessian weighting
improves the *right* metric (loss-weighted) by ~8% at no extra storage cost.

---

---

## Extended discussion — what the results mean together

### The null results are as informative as the positive ones

We tested three natural physics-inspired decompositions:
  - **DCT/Fourier** (momentum-space truncation): flat spectrum → weights have no spatial frequency structure
  - **SVD** (eigenmode truncation): α≈0.5 Marchenko-Pastur bulk → weights have no exploitable low-rank structure
  - **Hessian diagonal** (sensitivity weighting): CV=2.336, gain 3.9–9.0% → real but moderate effect

Together these paint a coherent picture: **GPT-2 base weights are high-rank, spatially incoherent,
but non-uniformly sensitive to perturbation.** The structure is not in W itself but in how W
interacts with the data distribution — i.e., in the loss landscape geometry.

### Why Marchenko-Pastur α≈0.5 matters

The Marchenko-Pastur distribution is the limiting spectral distribution of large random matrices
with iid entries. Finding α≈0.5 uniformly across all 24 layers is not a coincidence — it is a
signature of how SGD training works:

- SGD adds gradient noise proportional to the batch variance
- Over many steps this noise acts like random perturbations to the weights
- The weights converge to a distribution where the task-relevant signal sits above a
  "noise floor" consistent with the Marchenko-Pastur bulk

This is analogous to **thermal equilibration** in statistical mechanics: a system in contact with
a heat bath (the stochastic gradient noise) equilibrates to the Boltzmann distribution, which for
a quadratic Hamiltonian is Gaussian — and Gaussian random matrices have Marchenko-Pastur spectra.

**Implication:** The "task information" in GPT-2 is NOT stored in a small number of dominant
singular modes. It is distributed across essentially all modes — a fundamentally delocalized
representation. This is why compression is hard and why you need more bits than you might expect.

### The Hessian gain mechanism

The 7.7% gain at K=256 comes from a simple geometric argument:

Flat k-means places 256 centroids to minimise the average squared distance from any weight block
to its nearest centroid. In 16-dimensional space this is an optimal packing problem — the 256
centroids tile the space uniformly.

But the *loss function* doesn't care uniformly about all dimensions. Dimension j costs
H_diag[j] × ε² to misquantize by ε. So the loss-optimal placement of centroids should be
denser near the high-H_diag dimensions (the ~5-6 outlier dimensions with max activation
energy 0.99 vs mean 0.022).

H-weighted k-means achieves this by scaling the space before running k-means: in the scaled
space, centroids pack uniformly, but when unscaled this corresponds to denser packing in
the sensitive directions. This is geometrically identical to placing more centroids per unit
volume in the high-curvature directions of the loss landscape.

The gain grows with K because:
- At K=16: only 16 centroids in 16D space → only 1 centroid per dimension on average,
  so reallocation gains nothing; the error is dominated by coarse discretization everywhere
- At K=256: finer coverage → reallocation of even 1-2 centroids toward hot dimensions matters
- At K=1024: finer still → more opportunity for selective concentration

Extrapolating: at K=4096 we'd expect gain ≈ 12-14%, plateauing because H_diag is diagonal
(we're not accounting for inter-weight correlations). Full GPTQ with off-diagonal Hessian
likely achieves 20-30% gain by also rotating the quantization axes to align with the
loss Hessian eigenvectors — not just scaling.

### Connection to GPTQ and the full spin-glass treatment

Our diagonal H_diag approach is a first-order approximation to what GPTQ does:

| Step | Our approach | GPTQ |
|------|-------------|------|
| Hessian estimation | H_diag[j] = E[x_j²] (diagonal only) | Full H = E[xx^T] (n×n matrix) |
| Quantization | k-means in H^{1/2}-scaled space | Greedy column-by-column with error feedback |
| Error propagation | None — blocks quantized independently | Quantization error in column j is propagated to remaining columns |
| Gain vs flat | ~8% H-RMSE reduction | ~20-30% PPL reduction (literature) |

In the spin-glass framing, our approach finds the ground state of a *diagonal* spin glass
(no inter-site coupling). GPTQ finds the approximate ground state of the full coupled system
using a Cholesky-based greedy solver — equivalent to one step of belief propagation on the
full coupling graph.

The next natural experiment is to ask: **is the off-diagonal H actually low-rank?**
If E[xx^T] ≈ U Λ U^T with small rank, we can rotate into the Hessian eigenbasis cheaply,
apply diagonal scaling in that basis, and approximate the full GPTQ gain at much lower cost.
This is the "spectral Hessian" idea — the real synthesis of SVD and Hessian approaches.

### Roadmap going forward

**Tier 1 — immediate experiments (single layer, fast):**
1. PPL eval: flat k-means vs H-weighted k-means (single layer swap) — IN PROGRESS
2. H_diag rank analysis: is E[xx^T] actually low-rank? What is its effective rank?
3. Calibration data scaling: does more calibration data sharpen H_diag and increase the gain?

**Tier 2 — new schemes (medium effort):**
4. Spectral Hessian: quantize in eigenbasis of E[xx^T], apply diagonal scaling there
5. Per-layer bit allocation (RG flow): assign K based on per-layer H-RMSE sensitivity
6. H-weighted RG: combine hierarchical codebook with Hessian scaling

**Tier 3 — architecture experiments (high effort):**
7. Replicate on LLaMA/Mistral: test α≈0.5 universality on a proper instruction-tuned model
8. MPS decomposition: implement and test bond-dimension vs K tradeoff
9. Full GPTQ-style error propagation on top of H-weighted codebook

---

## Experiment 4 — Hessian PPL evaluation [IN PROGRESS]

**Hypothesis:** The 7.7% H-RMSE gain from Hessian weighting translates to measurable
perplexity improvement when a single layer is quantized.

**Design:**
- Single layer: h0.c_fc only (to avoid memory pressure)
- 3 conditions: baseline (FP32), flat k-means K=256, H-weighted k-means K=256
- Calibration: 100 texts from WikiText-2 train split
- Evaluation: 100 texts from WikiText-2 test split, max_length=128
- Metric: perplexity (exp of mean NLL)

**Status:** Running.

---

## Key references from source PDF

- BitNet b1.58 2B4T: arxiv.org/html/2504.12285v1
- bitnet.cpp: arxiv.org/html/2502.11880v1
- QuIP# (Hadamard incoherence): implied by QuIP# paper, Cornell/Meta 2024
- Residual quantization with implicit neural codebooks: ai.meta.com/research/publications/...
- Awesome-LLM-Quantization: github.com/pprp/Awesome-LLM-Quantization
