# Physics-Inspired LLM Quantization — Research Ideas

Captured from initial brainstorm session. Each scheme has a physics origin, a core idea,
and a difficulty/novelty estimate.

---

## Implemented

### 1. Scalar Codebook (Ising-style)
**Physics origin:** Discrete spin alphabet {s₁, s₂, ..., sK} — finite local state space.
**Idea:** Reshape weight matrix into blocks of size d. Run k-means to learn K centroids.
Replace each block with its nearest centroid index. Store indices + codebook.
**Effective bits:** log₂(K) / d
**Files:** `src/codebook.py`, `src/layers.py:CodebookLinear`

### 2. O(N)-style: Direction + Norm Factorization (Heisenberg-inspired)
**Physics origin:** Heisenberg spin = unit vector on S^{d-1} × scalar magnitude.
**Idea:** Factor each weight block as norm × unit-direction. Quantize direction with a
spherical codebook (k-means on normalized vectors), norm with a discrete ladder.
**Effective bits:** (log₂(K_dir) + log₂(n_levels)) / d
**Files:** `src/physics.py:on_quantize`, `src/layers.py:ONLinear`

### 3. RG Hierarchical Codebook (Renormalization Group)
**Physics origin:** Wilson's RG — integrate out short-wavelength modes, keep long-wavelength.
**Idea:** Two-level codebook: coarse centroids capture global structure (UV-integrated modes),
residual centroids correct per-block deviations (short-wavelength corrections).
Quantize as coarse_centroid + residual_centroid.
**Files:** `src/physics.py:rg_quantize`, `src/layers.py:RGLinear`

### 4. DCT Quantization (Frequency Domain / Momentum-Space Truncation)
**Physics origin:** Lattice field theory — fields expanded in Fourier/momentum modes.
Low-k modes carry most energy; high-k (UV) modes are small and can be truncated.
Energy compaction property: most variance lives in a few low-frequency coefficients.
**Idea:** Apply DCT-II to each weight block. Keep only the top-keep_k low-frequency
coefficients. Scalar-quantize retained coefficients to n_levels bins. Discard the rest.
**Effective bits:** (keep_k × log₂(n_levels)) / d
**Key diagnostic:** Plot DCT power spectrum of weight blocks — if it decays fast, this wins.
**Files:** `src/frequency.py:dct_quantize`, `src/layers.py:FreqLinear(mode="dct")`

### 5. Walsh-Hadamard (WHT) Quantization
**Physics origin:** Spin-wave basis on a discrete lattice. Hadamard matrix entries are ±1/√d,
analogous to a discrete Fourier basis but hardware-efficient (no float multiplies).
**Idea:** Same pipeline as DCT but using Hadamard basis. WHT is its own inverse (H @ H = I),
making reconstruction trivial. Rows ordered by "sequency" (sign changes ≈ frequency).
**Connection to current work:** QuIP# (Cornell/Meta, 2024) uses Hadamard incoherence processing.
**Files:** `src/frequency.py:wht_quantize`, `src/layers.py:FreqLinear(mode="wht")`

---

## Proposed (not yet implemented)

### 6. Taylor/Power Series in Weight Space (Perturbation Theory)
**Physics origin:** Perturbation theory in QFT — expand interacting theory around free-field solution.
**Idea:** Represent W ≈ α₀·B₀ + α₁·B₁ + ... + αₙ·Bₙ where {Bₙ} is a fixed structured basis
(Fourier matrices, Hadamard, or learned via SVD). Store only the scalar coefficients αₙ.
**Variants:**
- Structured basis (Fourier/Hadamard): basis is fixed, only αₙ stored → ultra low bit budget
- Learned basis (SVD/PCA): basis learned from data, αₙ quantized → similar to LoRA but quantized
**Connection:** LoRA uses rank-r additive decomposition; this generalises with physics-motivated bases.
**Difficulty:** Low-Medium | **Novelty:** Medium

### 7. Tensor Network / MPS Decomposition
**Physics origin:** Matrix Product States (MPS) from quantum many-body physics.
MPS represents exponentially large wavefunctions as chains of small tensors.
Bond dimension χ controls the entanglement/expressivity tradeoff.
**Idea:** Reshape each weight matrix into a high-order tensor, then decompose as MPS:
  W[i,j] → reshape → T[i₁,...,iₙ,j₁,...,jₘ] → A₁[i₁,χ] · A₂[χ,i₂,χ] · ... · Aₙ[χ,iₙ]
**Why different from SVD/LoRA:** SVD is rank-2 (U·V^T). MPS captures higher-order correlations.
**Connection:** ITensor library, recent papers on tensor-network transformers.
**Difficulty:** Medium | **Novelty:** High

### 8. Spin Glass / Hessian-Aware Quantization
**Physics origin:** Sherrington-Kirkpatrick spin glass model.
Finding the optimal discrete assignment is equivalent to minimising a spin glass Hamiltonian
where coupling Jᵢⱼ encodes the second-order loss sensitivity (weight Hessian).
**Idea:** Frame quantization as ground-state search:
  H({σ}) = Σᵢⱼ Jᵢⱼ σᵢ σⱼ,  σᵢ ∈ {-1, 0, +1}
Jᵢⱼ = loss Hessian w.r.t. weight pair (i,j). Solve via belief propagation or simulated annealing.
**Connection to existing work:** GPTQ uses second-order information but not the spin-glass framing.
The spin-glass framing opens up different algorithms (TAP equations, cavity method).
**Difficulty:** High | **Novelty:** High

### 9. RG Flow Across Transformer Layers (Layer-wise Bit Allocation)
**Physics origin:** Wilson's RG flow — theory parameters run with energy scale.
**Idea:** Not all transformer layers are equally sensitive to quantization.
Early layers ≈ UV (fine-grained features) → need more bits.
Late layers ≈ IR (coarse semantic features) → tolerate fewer bits.
Measure per-layer quantization sensitivity (e.g. output activation RMSE or PPL delta per layer).
Use this as a "beta function" to assign bit budgets that decrease (or increase) monotonically.
**Connection:** Layer-wise quantization sensitivity papers (SmoothQuant, LLM.int8()).
The RG framing gives a principled reason and a systematic bit-allocation algorithm.
**Difficulty:** Low | **Novelty:** Medium-High

---

## Implementation Priority

| Scheme | Novelty | Difficulty | Status |
|--------|---------|------------|--------|
| Scalar codebook (Ising) | Low | Done | ✅ |
| O(N)-style (Heisenberg) | Medium | Done | ✅ |
| RG hierarchical | Medium | Done | ✅ |
| **DCT (frequency domain)** | **Medium** | **Done** | **✅** |
| **WHT (Hadamard)** | **Medium** | **Done** | **✅** |
| **Per-row codebooks + act_cal** | **High** | **Done** | **✅ (Exp 14–26)** |
| **RG flow across layers (non-uniform K)** | **Medium-High** | **Done** | **✅ (Exp 18–21)** |
| Spin glass / Hessian-aware | High | High | Partially explored (Exp 4–5) |
| Taylor/power series | Medium | Low | Proposed |
| MPS decomposition | High | Medium | Proposed |

---

## Key Experimental Questions

1. **Does DCT power spectrum decay fast for LLM weight blocks?**
   → Run `experiments/dct_spectrum.py` and plot. If yes, DCT quantization is strongly motivated.

2. **Does low-frequency structure align with layer depth?**
   → Later layers may have smoother (lower-frequency) weights and tolerate more truncation.

3. **Does O(N) factorization (norm × direction) lose less than scalar codebook at the same bpw?**
   → Compare sweep curves. The physics predicts it should capture rotational structure better.

4. **Does the RG residual step give diminishing returns?**
   → Plot coarse-only RMSE vs coarse+residual RMSE as K_coarse varies. Expect RG-like behaviour.

5. **Can MPS bond dimension be a better knob than codebook size K?**
   → At same parameter count, MPS may exploit weight correlations that flat codebooks miss.
