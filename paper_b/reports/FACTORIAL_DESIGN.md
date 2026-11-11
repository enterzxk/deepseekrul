# Factorial ablation design (replacement-based)

Implemented 2026-11-11 after the user's methodology note. This supersedes the
delete-from-FULL ablation for the paper's module claims.

## 1. Replacement, not deletion

Each module is replaced by the simplest interface-compatible substitute, so the
rest of the network, the tensor shapes and the training budget stay the same.

| module | "on" | "off" replacement | paper wording |
|---|---|---|---|
| multi-scale conv | k=3/5/7 + projection | single k=3 conv + the **same** projection | w/ single-scale conv |
| | | projection only, no conv | w/ linear projection only |
| cross-sensor attention | one attention layer + learned pooling | mean pooling over sensors | w/ mean pooling |
| | | learned static sensor weights, no attention | w/ static sensor weights |
| HI supervision | head present, lambda=0.5 | head present, lambda=0 | w/o HI supervision |

The HI head is **always built**; only the loss weight changes. The comparison
is therefore "is there HI supervision", not "is there a head".

## 2. 2x2x2 factorial

Factors and levels:

- **MS**: `single_proj` (single k=3 + projection) vs `full` (k=3/5/7 + projection).
- **SA**: `mean` (mean pooling) vs `basic` (cross-sensor attention + learned pooling).
- **HI**: `hi_lambda=0` vs `0.5`, with `hi_source='hf'` always.

Eight cells, seed 101, all four subsets. Two extra controls at the baseline
level: `linear_only` (no temporal conv) and `static` sensor weights.

Tags:

```
F_single_proj_mean_hi0/1     F_single_proj_basic_hi0/1
F_full_mean_hi0/1            F_full_basic_hi0/1
C_linear_mean_hi0            C_single_static_hi0
```

## 3. Analysis

For a factor F, the **main effect** is the mean over all cells with F on minus
the mean over all cells with F off, averaged over the other factors and the four
subsets. Two-way and three-way interactions are the usual factorial contrasts.
Effects are reported separately for validation RMSE, validation NASA score and
(descriptively) test RMSE.

Noise: comparisons are paired by seed and, for the engine-level metric, by
engine. A bootstrap over engines (1000 resamples) gives a 95% CI for the RMSE
difference between the selected cell and FULL. A difference whose CI contains 0
is reported as **comparable**, not as a win.

## 4. Selection rule (validation only)

1. Compute each cell's mean validation RMSE over the four subsets.
2. Keep all cells within 1% of the best.
3. Among them, prefer the lower mean validation NASA score.
4. If still tied within 1%, prefer fewer parameters.
5. The test set is not used for selection; the test ablation is descriptive.

## 5. What is not done

- No per-module hyperparameter tuning for FULL (HI weight, dropout, width).
  This is a stated limitation: the comparison is at a fixed training budget.
- One seed first (101), as requested. Seed variance is reported from the
  already-computed three-seed FD001/FD002 runs where available.
