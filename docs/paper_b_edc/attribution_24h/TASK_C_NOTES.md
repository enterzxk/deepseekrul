# Task C: what the four conditional-error tables can and cannot say

Numbers: `TASK_C_CONDITIONAL_ERRORS.csv` (162 rows).
Script: `task_c_conditional_errors.py` (read-only; no training).

## Schema facts that had to be established first

| fact | check | result |
|---|---|---|
| `val_pred.csv` has no cycle / regime / sensor columns | header | `unit, cut, y_true, y_pred` |
| `cut` is the CMAPSS absolute cycle index | join `(unit, cut)` to `train_FD00X.txt` on `(unit, cycle)` | **19170 points, 0 unmatched** |
| operating conditions | `KMeans(k=6, random_state=7, n_init=20)` on the three op-setting columns, the same construction as `data.py::OpScaler` | reproduced per subset |
| lifetime per engine | max cycle of that unit in the train file | `cut / lifetime` lands on the five truncation points |
| convention | primary: label capped at 125, prediction not clipped | all numbers below use it |

## Two axes, never substituted for each other

The question "at the same running age, how far apart are engines" is about the
**absolute cycle**. A second, different question is "at the same fraction of its
own life". A machine at cycle 100 may be halfway through its life or 80% of the
way through it, so the two answers differ and both are reported:

* `A1a_same_age` — bins of 50 absolute cycles.
* `A1b_same_life_fraction` — bins of `cut / lifetime`, i.e. the five truncation
  points 0.40 / 0.55 / 0.70 / 0.85 / 0.95.

An earlier pass used 25-cycle bins and produced 74 bins with a median of one
engine each; the cross-engine spread is undefined in such a bin. That was a
resolution problem, not a reason to abandon the absolute axis.

## A1a: same running age

Cross-engine spread of the error, one mean per engine, median and IQR across
engines, CI over 27 runs. Bins with fewer than 10 engines carry a
`engines<10: raw value only` note.

| subset | bin | engines | median | IQR |
|---|---|---|---|---|
| FD001 | 50-99 | 18 | -5.301 | 21.944 |
| FD001 | 100-149 | 20 | -0.899 | 6.360 |
| FD001 | 150-199 | 15 | -1.667 | 5.697 |
| FD001 | 200-249 | 6 | -0.405 | 6.115 |
| FD002 | 50-99 | 46 | -2.407 | 18.653 |
| FD002 | 100-149 | 52 | +0.505 | 7.634 |
| FD002 | 250-299 | 5 | +4.972 | 4.171 |
| FD004 | 50-99 | 28 | +0.111 | 14.518 |
| FD004 | 150-199 | 47 | +0.611 | 7.665 |
| FD004 | 250-299 | 17 | +1.050 | 6.852 |
| FD004 | 300-349 | 9 | -1.343 | 9.669 (raw only) |

Reading: at the same running age the engines are not exchangeable. In the early
bins the inter-quartile range of the error is 18-29 RUL units, i.e. the middle
half of engines already spans that much error at one age. The spread shrinks
with age. "Cycle explains the remaining life" is not supported.

## A1b: same fraction of life

| subset | 0.40 | 0.55 | 0.70 | 0.85 | 0.95 |
|---|---|---|---|---|---|
| FD001 median | -7.046 | +3.977 | -1.986 | -2.481 | -0.757 |
| FD001 IQR | 24.130 | 11.106 | 12.400 | 4.530 | 4.576 |
| FD004 median | -0.074 | -0.204 | -0.389 | +0.819 | +2.071 |
| FD004 IQR | 8.532 | 19.949 | 13.086 | 5.472 | 4.159 |

The two axes disagree in sign at several points (FD001 at life 0.55 is +3.98
while the same subset at cycle 100-149 is -0.90). That is the expected
consequence of engines having different lifetimes, and it is why the two are
reported separately.

## A2: post-switch vs steady — not interpretable on this data

| subset | paired abs-error difference | 95% CI over engines | n engines |
|---|---|---|---|
| FD001 | -1.960 | [-3.366, -0.554] | 324 |
| FD002 | +0.236 | [-0.268, +0.740] | 972 |
| FD003 | +1.402 | [+0.736, +2.068] | 432 |
| FD004 | -4.011 | [-4.960, -3.063] | 783 |

Two structural reasons, both in the data:

1. **`FD001` and `FD003` are single-condition subsets.** Forcing `k=6` splits
   noise, so consecutive cycles flip between clusters and "post-switch" is an
   artefact.
2. **On `FD002`/`FD004` the condition changes almost every cycle** — post-switch
   windows are 81% of all windows (5697/7020 and 5535/6750). There is no
   persistent regime to transition between, so "the first N windows after a
   switch" is not a defined stratum.

The signs disagree across subsets, which is what a non-interpretable stratum
looks like. This is a negative result about the analysis, not about the model.

## A3: capped region vs degradation region

| subset | y_true == 125 | y_true < 125 | signed error in capped region |
|---|---|---|---|
| FD001 | 12.360 (n=189) | 7.514 (n=2511) | -12.280 |
| FD002 | 11.510 (n=594) | 7.941 (n=6426) | -10.320 |
| FD003 | 8.335 (n=297) | 8.859 (n=2403) | -7.820 |
| FD004 | 13.313 (n=1377) | 8.119 (n=5373) | -11.944 |

The capped region is biased, not merely noisier: the signed error is negative in
all four subsets, so predictions there sit 8-12 RUL below the cap, and the
absolute error is larger in three of four.

## A4: sensor information inside (50-cycle bin x regime) cells

Spearman correlation of absolute error against the first principal component of
regime-wise z-scored sensors, inside cells with at least 50 points and 5
engines: 58 cells.

```
rho range   -0.713 .. +0.828
median      +0.099
|rho| > 0.1 47/58 cells
|rho| > 0.2 33/58 cells
negative    25/58 cells
```

Sensor state carries information about the error after holding age and regime
fixed, but the sign is not consistent: 25 cells negative, 33 positive. The
defensible statement is "the error depends on sensor state within a fixed age
and regime", not "more sensor signal means lower error".

## What is not in these tables

* No test-set numbers: everything is validation.
* No cross-arm comparison: predictions are pooled over one parent arm and the
  eight A8 arms; these tables describe the family's error structure.
* No transition effect: see A2.
