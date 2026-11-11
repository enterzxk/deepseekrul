# Supported improvements (factorial, replacement-based)

Scope: 2x2x2 factorial + two controls, seed 101, four C-MAPSS subsets.
Selection is validation-only; test numbers are descriptive (user-requested).
Design: `FACTORIAL_DESIGN.md`; full tables: `DSPT_ABLATION_ANALYSIS.md` §4b.

## 1. Accepted

### A1. Single-scale conv + cross-sensor attention + HI supervision

- **Validation (4-subset mean):** 12.779 vs FULL 13.075 (-2.3%); vs
  single-scale + mean + no HI 13.632 (-6.3%).
- **Official test (descriptive, 4-subset mean):** 10.983 vs FULL 11.311
  (-2.9%); better on all four subsets (10.196/10.258/11.652/11.827 vs
  10.288/10.523/11.803/12.628).
- **Noise:** paired engine bootstrap vs FULL gives CIs that all contain 0
  (FD001 [-2.41,+0.47], FD002 [-0.99,+0.64], FD003 [-1.61,+0.69],
  FD004 [-1.00,+1.40]). The honest label is **comparable**, with the simpler
  model preferred by the validation rule.
- **Mechanism:** attention and HI carry the main effects; multi-scale does not.

## 2. Rejected

### R1. Multi-scale convolution as a contribution

- Validation main effect -0.14 (small), test -0.06 (negligible).
- MSxSA interaction +0.73 mean (+1.33 FD001, +1.00 FD003): multi-scale and
  attention are redundant; adding multi-scale on top of attention costs more
  than its standalone benefit.
- The selected model uses the **single-scale** replacement (k=3 + the same
  projection). Multi-scale is not claimed.

### R2. FULL as the best model

FULL is 13.075 validation, worse than the selected cell (12.779) and worse than
several other cells. It is not selected.

### R3. HI as a standalone module

HI main effect is -0.05 validation and -0.18 test, but subset-dependent: FD003
is +0.10 validation and +0.26 test. HI is retained only as part of the selected
cell; it is not claimed as a general improvement.

## 3. Controls

| control | validation mean | test mean | reading |
|---|---:|---:|---|
| linear-only (no temporal conv) | 12.957 | 11.618 | the conv still matters on test |
| static sensor weights (no attention) | 13.163 | 11.417 | attention is better on validation; test comparable |

## 4. What is not claimed

- No significance claim over FULL: all bootstrap CIs contain 0.
- No cross-seed claim: one seed (101), as requested.
- No universal module claim: effects are subset-dependent and reported per
  subset.
- No test-based selection: the test table is descriptive.
