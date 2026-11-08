# PAPER — RESULTS SECTION (draft with final seed-101 numbers, 2026-11-05 21:45)

All numbers from `RESULTS.csv`, `ABLATIONS.csv`, `MEASUREMENT_CONDITIONS.csv`,
`observability.json`, `PER_ENGINE.csv`, `paper_tables/per_fold.csv`.
Protocol: full budget (80 epochs, all training windows, fixed final epoch), five
engine-disjoint outer folds, RUL capped at 125, engine-equal-weight pooled metrics,
engine-clustered paired bootstrap (5000).

## A. Main comparison (seed 101)

**FD002** (260 engines, 41,019 windows). Reference arm (lowest outer-validation RMSE):
FULL, 13.054.

| Arm | RMSE | MAE | NASA (per-window mean) | Bias | Delta vs reference [95% CI] |
|---|---:|---:|---:|---:|---|
| M0 | 13.505 | 9.329 | 4.012 | +1.208 | +0.451 [+0.071, +0.823] |
| OPS | 13.376 | 9.291 | 3.904 | +0.776 | +0.322 [+0.060, +0.583] |
| **FULL** | **13.054** | **9.098** | **3.492** | **+0.288** | — |

FULL is better than M0 on 5/5 folds (per-fold deltas -0.03 to -0.98).

**FD004** (249 engines, 49,048 windows). Reference arm: FULL, 13.253.

| Arm | RMSE | MAE | NASA | Bias | Delta vs reference [95% CI] |
|---|---:|---:|---:|---:|---|
| M0 | 13.398 | 8.810 | **8.980** | -0.171 | +0.145 [-0.240, +0.518] |
| OPS | 13.676 | 8.721 | 30.644 | -0.108 | +0.422 [+0.123, +0.756] |
| **FULL** | **13.253** | **8.662** | 11.251 | -1.038 | — |

FULL is better than M0 on 3/5 folds only, the pooled interval crosses zero, and FULL's
NASA score is worse than M0's. The additive prior is therefore not uniformly better on
FD004.

## B. Ablations (each vs M0 on the folds where both exist)

| Variant | Subset | Folds | Delta RMSE [95% CI] | Verdict |
|---|---|---:|---|---|
| NORM (cycle coordinate replaced by c/mu_T) | FD002 | 4 | +1.034 [+0.595, +1.468] | significantly worse |
| NORM | FD004 | 4 | +1.320 [+0.783, +1.879] | significantly worse |
| PHASE (RUL = phi * mu_T) | FD002 | 2 | +3.909 [+2.383, +5.421] | significantly worse |
| DIFF (window minus early-life reference) | FD002 | 2 | -0.038 [-0.534, +0.472] | neutral |
| EQBAL (engine-balanced sample weights) | FD002 | 1 | -0.092 [-0.873, +0.704] | neutral |
| GRAPH (sensor-wise attention, +40% params) | FD002 | 1 | -0.019 | neutral |

## C. Observability of the lifetime scale

| Subset | 50 | 100 | 150 | 190 |
|---|---:|---:|---:|---:|
| FD002 corr(mu_T, T) | 0.26 | 0.75 | 0.92 | 0.94 |
| FD002 MAE (cycles) | 35.5 | 22.4 | 11.1 | 9.9 |
| FD004 corr | 0.53 | 0.74 | 0.88 | 0.87 |
| FD004 MAE (cycles) | 49.1 | 36.2 | 21.1 | 19.9 |

## D. Identifiability of adaptive correction (FD002, 4 folds)

Six observable families are essentially uncorrelated with the individual-versus-population
advantage (|rho| < 0.08). Quantile and age gates give 13.03-13.62 against a fixed 0.5
fusion at 12.89; the per-window oracle is 10.97. The sensitivity coefficient correlates
0.37-0.39 with the absolute error but 0.02 with the relative advantage.

## E. Measurement conditions (FD002, folds 0 and 2, weights unchanged)

| Condition | M0 | FULL |
|---|---:|---:|
| clean | 13.372 | 13.221 |
| MC1 (20% missing, forward fill) | 15.289 | 14.862 |
| MC2 (1 sigma offset on three channels) | 36.937 | 39.692 |

MC3 (terminal-condition support tertiles) is flat (13.0-13.6 for all arms).

## F. Endpoint and budget dependence

On FD002 the three endpoints select three different winners: all windows -> FULL 13.054;
one truncation per engine -> M0 11.662; official test -> OPS 11.842. A leave-one-fold-out
inverse-variance fusion reaches 12.657 (-3.0% vs FULL) on all windows but 11.967 on the
official test, worse than OPS. NORM is better than M0 at 24 epochs (-4.7%, three seeds)
and worse at 80 epochs (+7.5%).

## G. Metric behaviour

NASA score on FD004 is dominated by 1-5 engines (OPS: 90% of the total from five engines,
one engine contributing 5256), so the NASA ranking is unstable at this fleet size. The
95% interval half-width of the RMSE difference is 0.376 cycles on FD002, i.e. about 2.9%
relative, so a 3% effect sits at the edge of detectability.
