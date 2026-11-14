# Task C: what the four conditional-error tables can and cannot say

Generated numbers: `TASK_C_CONDITIONAL_ERRORS.csv` (156 rows).
Script: `task_c_conditional_errors.py` (read-only; no training).

## Schema facts that had to be established first

| fact | check | result |
|---|---|---|
| `val_pred.csv` has no cycle / regime / sensor columns | header | `unit, cut, y_true, y_pred` |
| `cut` is the CMAPSS cycle index | join `(unit, cut)` to `train_FD00X.txt` on `(unit, cycle)` | **19170 points, 0 unmatched** |
| operating conditions | `KMeans(k=6, random_state=7)` on the three op-setting columns, the same construction as `data.py::OpScaler` | reproduced per subset |
| convention | primary: label capped at 125, prediction not clipped | all numbers below use it |

## Three structural findings, before any number

1. **`FD001` and `FD003` are single-condition subsets.** Forcing `k=6` splits
   sensor noise rather than operating conditions, so a regime label there is an
   artefact and consecutive cycles flip between clusters. Regime-based results
   are only interpretable on `FD002` and `FD004`.
2. **On `FD002`/`FD004` the operating condition changes almost every cycle**
   (post-switch windows are 81% of all windows: 5697/7020 and 5535/6750). There
   is no persistent regime to transition *between*, so "the first N windows
   after a switch" is not a well-defined stratum on this data. What the table
   actually compares is "regime differs from the previous cycle" vs "regime
   repeats", and the two groups differ in regime composition as well, so the
   difference is not a clean transition effect.
3. **Same cycle does not mean same age.** Engines have different lifetimes, so
   at a fixed cycle their true RULs differ widely; that is exactly what the
   cross-engine dispersion in A1 measures.

## A1: at the same cycle, how far apart are engines

Cross-engine sd of the signed error, one number per (run, cycle bin), CI over 27
runs. `FD001` falls monotonically with cycle; `FD004` does not, because its
engine lifetimes are far more heterogeneous.

| subset | cycle 50-74 | 125-149 | 200-224 | worst bin |
|---|---|---|---|---|
| FD001 | 17.540 [16.72, 18.36] | 6.893 [6.50, 7.29] | 4.897 [4.57, 5.23] | 17.540 |
| FD004 | 9.898 [8.88, 10.92] | 19.414 [19.11, 19.71] | 10.367 [9.62, 11.12] | 36.199 at 225-249 |

Reading: at the same age the error of two engines can differ by 5 to 36 RUL
units. "Cycle explains everything" is not supported by these numbers.

## A2: post-switch vs steady

| subset | paired abs-error difference | 95% CI over engines | n engines |
|---|---|---|---|
| FD001 | -1.960 | [-3.366, -0.554] | 324 |
| FD002 | +0.236 | [-0.268, +0.740] | 972 |
| FD003 | +1.402 | [+0.736, +2.068] | 432 |
| FD004 | -4.011 | [-4.960, -3.063] | 783 |

The signs disagree across subsets and the two subsets with real operating
conditions give opposite signs. Given finding 2, this table does not establish a
transition effect; it establishes that the stratum as constructed is not
interpretable. It is reported as a negative result about the analysis, not as a
result about the model.

## A3: capped region vs degradation region

| subset | y_true == 125 | y_true < 125 | signed error in capped region |
|---|---|---|---|
| FD001 | 12.360 (n=189) | 7.514 (n=2511) | -12.280 |
| FD002 | 11.510 (n=594) | 7.941 (n=6426) | -10.320 |
| FD003 | 8.335 (n=297) | 8.859 (n=2403) | -7.820 |
| FD004 | 13.313 (n=1377) | 8.119 (n=5373) | -11.944 |

The capped region is not just noisier, it is **biased**: the signed error is
negative everywhere, so predictions in the early-life region sit well below the
cap. Three of four subsets have a larger absolute error there.

## A4: sensor information inside (cycle bin x regime) cells

Spearman correlation of absolute error against the first principal component of
the regime-wise z-scored sensors, computed inside cells with at least 50 points
and 5 engines: 58 cells.

```
rho range   -0.713 .. +0.828
median      +0.099
|rho| > 0.1 47/58 cells
|rho| > 0.2 33/58 cells
negative    25/58 cells
```

Sensor variation does carry information about the error magnitude after holding
age and regime fixed, but **the sign is not consistent across cells**: 25 cells
negative, 33 positive. So the defensible statement is "the model's error
depends on sensor state within a fixed age and regime", not "more sensor signal
means lower error".

## What is not in these tables

* No test-set numbers: everything here is validation.
* No cross-model comparison: the predictions are pooled over one parent arm and
  the eight A8 arms, and the tables describe the model family's error
  structure, not a difference between arms.
* No transition effect: see finding 2.
