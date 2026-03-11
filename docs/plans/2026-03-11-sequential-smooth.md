# Experiment 13: Sequential SmoothQuant Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement sequential SmoothQuant calibration for all-layer GPT-2 quantization, where each layer is calibrated using the model with all prior layers already quantized.

**Architecture:** Copy `experiments/smooth_all_layer.py`, change `build_all_layer_model` to pass the in-progress model (not `base_model`) to `estimate_h_diag`. This gives each layer's calibration access to its true inference-time activation distribution.

**Tech Stack:** PyTorch, HuggingFace transformers, WikiText-2, existing `src/hessian.py` and `src/codebook.py`

---

### Task 1: Create `experiments/sequential_smooth.py`

**Files:**
- Create: `experiments/sequential_smooth.py`
- Reference: `experiments/smooth_all_layer.py` (source to copy from)

**Step 1: Copy smooth_all_layer.py to sequential_smooth.py**

```bash
cp experiments/smooth_all_layer.py experiments/sequential_smooth.py
```

**Step 2: Update the module docstring**

Replace the docstring at the top of `sequential_smooth.py`:

```python
"""
Sequential SmoothQuant calibration for all-layer quantization.

Experiment 13. Experiment 12 showed that independent SmoothQuant calibration
(all layers calibrated against base model activations) gives PPL=36,946 —
12× WORSE than flat all-layer (PPL=3,042, Exp 9).

Root cause: each layer's col_scale is calibrated against the base model's
activation distribution. When all layers are modified, layer N receives
activations from already-modified layers 0..N-1, not matching its calibration.

Fix: sequential calibration. Process layers in forward order. For each layer i:
  - Calibrate using the CURRENT model (with layers 0..i-1 already quantized)
  - The hook fires on the live layer in the model (still FP32, but downstream
    of quantized layers) — capturing the true inference-time activation distribution
  - Quantize and replace layer i, then move to i+1

One-line change from Exp 12: pass `model` (not `base_model`) to estimate_h_diag,
and use `raw` (live layer in model) not `raw_base` (base model layer) for the hook.

Design:
  - Quantize all 24 MLP layers with SmoothQuant α=0.5, K=256, bd=16
  - Sequential calibration: each layer sees actual downstream activations
  - Compare to: flat all-layer (PPL=3,042), indep smooth all-layer (PPL=36,946)

Expected if sequential fixes the mismatch: PPL ≈ 3,042 × (170/381) ≈ 1,357

Usage:
    python experiments/sequential_smooth.py
"""
```

**Step 3: Fix `build_all_layer_model` — the core change**

In the `smooth` branch of `build_all_layer_model`, change two things:
1. Pass `model` instead of `base_model` to `quantize_layer_smooth`
2. Pass `raw` instead of `raw_base` as the layer module (hook target)

Old code (Exp 12):
```python
raw      = getattr(block.mlp, attr)
raw_base = getattr(base_model.transformer.h[bi].mlp, attr)
if mode == "flat":
    layer = quantize_layer_flat(raw, block_dim, K)
elif mode == "smooth":
    layer = quantize_layer_smooth(
        base_model, tokenizer, calib_texts, raw_base, ALPHA, block_dim, K
    )
```

New code (Exp 13):
```python
raw = getattr(block.mlp, attr)
if mode == "flat":
    layer = quantize_layer_flat(raw, block_dim, K)
elif mode == "smooth":
    # Sequential: use model (with already-quantized prior layers), not base_model.
    # Use raw (live layer in model) for the hook — captures true inference activations.
    layer = quantize_layer_smooth(
        model, tokenizer, calib_texts, raw, ALPHA, block_dim, K
    )
```

Remove the `raw_base` variable entirely.

**Step 4: Update the configs list**

Replace the configs list in `main()` to only run smooth (flat configs have Exp 9/12 results):

```python
configs = [
    ("smooth bd=16 K=256 (sequential)", "smooth", BD_SMOOTH, K_SMOOTH),
]
```

**Step 5: Update the summary section**

Replace the summary print block at the end of `main()`:

```python
EXP9_PPL  = {"flat bd=16 K=256": 3042.435}
EXP12_PPL = {"smooth bd=16 K=256 (indep)": 36945.867}

print(f"\n{'='*70}")
print(f"Experiment 13 Summary — Sequential SmoothQuant calibration")
print(f"{'='*70}")
print(f"  Baseline (FP32):                          {ppl_base:>9.3f}")
print(f"  [Exp 9 ref]  flat   bd=16 K=256:          {EXP9_PPL['flat bd=16 K=256']:>9.3f}  (all-layer, indep flat)")
print(f"  [Exp 12 ref] smooth bd=16 K=256 (indep):  {EXP12_PPL['smooth bd=16 K=256 (indep)']:>9.3f}  (all-layer, indep smooth)")
for label, ppl in results.items():
    delta  = ppl - ppl_base
    factor = ppl / ppl_base
    print(f"  {label:<40}  {ppl:>9.3f}  (delta={delta:>+8.1f}, {factor:.1f}× baseline)")

SINGLE_LAYER = {"smooth bd=16 K=256 (sequential)": 169.704}
print(f"\n  Error accumulation (all-layer vs single-layer):")
for label, ppl_all in results.items():
    ppl_single = SINGLE_LAYER.get(label, None)
    if ppl_single:
        factor = ppl_all / ppl_single
        exp12_factor = EXP12_PPL.get("smooth bd=16 K=256 (indep)", None)
        print(f"  single-layer: {ppl_single:.1f}  →  all-layer: {ppl_all:.1f}  ({factor:.1f}× amplification)")
        print(f"  vs Exp 12 (indep):  {exp12_factor:.1f}  →  {ppl_all:.1f}  ({'BETTER' if ppl_all < exp12_factor else 'WORSE'})")
        print(f"  vs Exp 9  (flat):   3042.4  →  {ppl_all:.1f}  ({'BETTER' if ppl_all < 3042.435 else 'WORSE'} than flat)")
```

**Step 6: Verify the script is syntactically correct**

```bash
cd /Users/faulknco/projects/spin-quant && python -c "import experiments.sequential_smooth"
```

Expected: no output (clean import). If syntax error, fix it.

**Step 7: Commit**

```bash
cd /Users/faulknco/projects/spin-quant
git add experiments/sequential_smooth.py docs/plans/2026-03-11-sequential-smooth-design.md docs/plans/2026-03-11-sequential-smooth.md
git commit -m "exp13: add sequential SmoothQuant calibration script"
```

---

### Task 2: Run Experiment 13

**Files:**
- Read output: task output file (background task)

**Step 1: Launch experiment as background task**

```bash
cd /Users/faulknco/projects/spin-quant && python experiments/sequential_smooth.py
```

Run in background using TaskCreate. Expected runtime: ~20-30 minutes (48 calibration passes + eval).

**Step 2: Wait for completion notification**

Do not poll. Wait for background task notification. Then read the output file.

**Step 3: Verify output structure**

Expected output should contain:
- `Baseline (FP32): 59.640`
- 48 lines of `h{N}.{attr} done`
- `PPL = ???` for smooth sequential config
- Summary table

---

### Task 3: Record findings in FINDINGS_LOG.md

**Files:**
- Modify: `FINDINGS_LOG.md`

**Step 1: Read the experiment output**

Read the task output file.

**Step 2: Append Exp 13 section to FINDINGS_LOG.md**

Insert a new `## Experiment 13` section immediately before `## Experiment 12` in FINDINGS_LOG.md.

Include:
- Results table comparing Exp 9 (flat), Exp 12 (indep smooth), Exp 13 (seq smooth)
- Amplification factors (single-layer 170 → all-layer result)
- Root cause analysis: did sequential calibration fix the distribution mismatch?
- Physics interpretation based on actual results
- Practical implications

**Step 3: Commit**

```bash
cd /Users/faulknco/projects/spin-quant
git add FINDINGS_LOG.md
git commit -m "exp13: record sequential SmoothQuant findings"
```

---

## Reference Numbers

| Experiment | Config | PPL |
|------------|--------|-----|
| Baseline | FP32 | 59.640 |
| Exp 6 (single-layer) | flat bd=16 K=256 | 380.618 |
| Exp 10 (single-layer) | smooth α=0.5 bd=16 K=256 | 169.704 |
| Exp 9 (all-layer) | flat bd=16 K=256 | 3,042.435 |
| Exp 12 (all-layer) | smooth indep bd=16 K=256 | 36,945.867 |
| **Exp 13 (all-layer)** | **smooth seq bd=16 K=256** | **TBD** |

Prediction if fully fixed: ~1,357. Prediction if partially fixed: 3,042–36,946. Prediction if not fixed: ~36,946.
