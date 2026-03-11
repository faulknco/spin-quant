# Experiment 19A: Data-Driven Non-Uniform Compression — Design

**Date:** 2026-03-11
**Status:** Approved

---

## Background

Exp 18 measured layer-by-layer ΔPPL for per-row K=64 across all 24 GPT-2 MLP sublayers.
The accumulation is gradual (no single tipping point) but uneven — h6.c_proj alone
contributed +79.4 ΔPPL, while h0.c_fc contributed only +0.14. This 550× range in
per-layer sensitivity suggests that a budget-neutral reallocation (more K for sensitive
layers, less for background) could reduce all-layer PPL without increasing average bpw.

## Question

Can we improve all-layer PPL by using the Exp 18 ΔPPL rankings to assign K non-uniformly,
holding average bpw approximately fixed?

## Design

### ΔPPL rankings from Exp 18 (per-row K=64, sorted descending)

| Rank | Layer      | ΔPPL   |
|------|------------|--------|
| 1    | h6.c_proj  | +79.4  |
| 2    | h5.c_proj  | +38.2  |
| 3    | h7.c_proj  | +33.6  |
| 4    | h6.c_fc    | +33.2  |
| 5    | h10.c_proj | +22.3  |
| 6    | h4.c_proj  | +21.2  |
| 7    | h8.c_proj  | +20.2  |
| 8    | h11.c_fc   | +19.8  |
| 9    | h11.c_proj | +15.2  |
| 10   | h5.c_fc    | +15.4  |
| 11   | h3.c_proj  | +12.5  |
| 12   | h4.c_fc    | +11.4  |
| ...  | ...        | ...    |
| 24   | h0.c_fc    | +0.14  |

### Algorithm

1. Load Exp 18 ΔPPL ranking (hardcoded from results).
2. For each config (N, K_high, K_low):
   a. Select top-N sublayers by ΔPPL as "sensitive"; remaining 24-N as "background".
   b. Quantize all 24 sublayers with per-row k-means: sensitive get K_high, background get K_low.
   c. Evaluate all-layer PPL on 50 WikiText-2 texts.
3. Print results table comparing to uniform K=64 (PPL=421.9) baseline.

### Configs

| Config | N  | K_high | K_low | Notes                        |
|--------|----|--------|-------|------------------------------|
| A      | 4  | 256    | 32    | Only top-4; big quality gap  |
| B      | 8  | 256    | 32    | Top-8; budget-neutral (0.75 bpw avg) |
| C      | 12 | 128    | 32    | Top-12; moderate split       |
| D      | 8  | 128    | 32    | Top-8; conservative K_high   |
| E      | 4  | 128    | 64    | Minimal reallocation         |

Also include: uniform K=64 (baseline), uniform K=128 (upper bound).

### Output

```
Config         N   K_high  K_low     PPL      delta vs K=64  avg_bpw
----------------------------------------------------------------------
Uniform K=64   —      64     64    421.9          —           0.750
Uniform K=128  —     128    128    ???        ???             0.875
(A) top-4     4     256     32    ???        ???             ~0.729
(B) top-8     8     256     32    ???        ???             0.750
(C) top-12   12     128     32    ???        ???             ~0.729
(D) top-8     8     128     32    ???        ???             ~0.708
(E) top-4     4     128     64    ???        ???             ~0.760
```

### File

`experiments/nonuniform_compression.py`

### Reference numbers

- Uniform K=64 all-layer (Exp 17C):  PPL = 421.9
- FP32 baseline:                      PPL ≈ 59.6–63.3
