# Task C: what the four conditional-error tables can and cannot say

Numbers: `TASK_C_CONDITIONAL_ERRORS.csv` (226 rows).
Script: `task_c_conditional_errors.py` (read-only; no training).

**Pool:** 108 runs = the eight A8 arms (96 cells) + the strong parent
`F_full_basic_hi1` (12 cells), 19170 validation points. The snapshot is a point
in time: the raw tree held 277 runs when this was written, so the run count is
not a constant and should never be quoted without a date.

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

Two structural reasons, both measured:

1. **`FD001` and `FD003` have no operating conditions to find.** `op3` is
   constant at 100.0000 and `op1` moves only within `[-0.0087, +0.0087]`, so
   `k=6` splits that noise. By contrast `FD002`/`FD004` have `op1` up to 42.008
   and `op3` spanning 60-100, i.e. six real conditions.
2. **The condition changes almost every cycle in all four subsets** — post-switch
   windows are 84.0% / 81.2% / 80.0% / 82.0% of all windows. This is not a
   property of two subsets; at the `k=6` granularity the dataset has no
   persistent regime anywhere.

So "there is no transition to speak of" and "the error does not rise after a
transition" are two different statements, and only the first is supported here.
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
engines.

**Pooled over the nine arms (58 cells):**

```
rho range   -0.7544 .. +0.7247
median      +0.1819
negative    20/58
|rho| > 0.2 37/58
|rho| > 0.3 26/58
```

**The same test run inside each arm (45 cells, one arm at a time):**

```
rho range   -0.1612 .. +0.5710
median      +0.2103
negative     7/45
```

Per arm:

| arm | cells | positive | negative | median rho |
|---|---|---|---|---|
| A8_base_L70 | 5 | 4 | 1 | +0.286 |
| A8_full_L70 | 5 | 4 | 1 | +0.103 |
| A8_hi_L70 | 5 | 4 | 1 | +0.235 |
| A8_ms_L70 | 5 | 5 | 0 | +0.196 |
| A8_mshi_L70 | 5 | 5 | 0 | +0.280 |
| A8_mssa_L70 | 5 | 4 | 1 | +0.111 |
| A8_sa_L70 | 5 | 4 | 1 | +0.270 |
| A8_sahi_L70 | 5 | 4 | 1 | +0.160 |
| F_full_basic_hi1 | 5 | 4 | 1 | +0.103 |

What the per-arm check changes: **no arm is internally all-negative, and every
arm is predominantly positive (at least 4 of 5 cells)**. So the direction does
not depend on the module configuration. But the negative share roughly doubles
when arms are pooled (7/45 = 16% inside arms, 20/58 = 34% pooled), so the pooled
table overstates how unstable the direction is: mixing arms adds sign flips that
are not present within any single configuration.

The defensible statement is therefore: **after holding age and regime fixed,
error still depends on sensor state, and the association is predominantly
positive (larger PC1, larger error) in every arm; the pooled table's larger
share of negative cells is a mixing artefact, not evidence that the direction is
arbitrary.**

**But `|error| ~ PC1` measures heteroscedasticity, not predictive power.** It
says the error is larger in some sensor regions; it does not say the sensor
predicts the remaining life. The task asks the second question, so it is
answered separately below.

## A5: does PC1 explain the label after age and condition?

Within-cell regression of `y_true` on PC1 (cells are 50-cycle bin x regime, so
age and condition are held fixed by construction), cluster-robust SE over
engines. `y_true` and PC1 do not depend on the model, so this is computed on
unique `(subset, unit, cut)` rows rather than once per arm.

| subset | slope | 95% CI | R2 increment | engines |
|---|---|---|---|---|
| FD001 | +9.008 | [6.504, 11.511] | 0.473 | 20 |
| FD002 | +11.188 | [9.194, 13.182] | 0.476 | 52 |
| FD003 | +9.271 | [4.529, 14.013] | 0.160 | 20 |
| FD004 | -7.632 | [-9.683, -5.581] | 0.227 | 50 |

**The sensor carries a large incremental signal**: within a fixed age bin and
regime, PC1 explains 16-48% of the remaining label variance, and one SD of PC1
moves the label by 7.6-11.2 RUL.

The sign is **not** comparable across subsets: PC1 comes from an SVD whose
orientation is arbitrary per subset, so a negative slope in FD004 means the
label runs opposite to that subset's PC1 axis, not that FD004 has opposite
physics. Only the magnitude and the R2 increment are interpretable across
subsets.

## A6: does the model's signed error track that signal?

Same within-cell construction, response `err` (signed), one arm at a time:
36 regressions (9 arms x 4 subsets).

```
CI excludes zero   0 / 36
R2 increment       median 0.0044, max 0.0189
```

**The signed error has no detectable dependence on PC1 within cells.** So the
model does not convert the available sensor signal into a directional bias; the
error is heteroscedastic along the sensor axis (A4) but not shifted along it
(A6).

The 4 statement that survives all three checks:

> After age and condition are held fixed, the sensor state carries a large
> incremental signal about the label (R2 increment 0.16-0.48), the residual
> magnitude depends on it (predominantly positive inside every arm), but the
> signed error does not: no arm shows a detectable slope (0/36). The model's
> error is spread by sensor state, not biased by it.

## What is not in these tables

* No test-set numbers: everything is validation.
* No cross-arm comparison: predictions are pooled over one parent arm and the
  eight A8 arms; these tables describe the family's error structure.
* No transition effect: see A2.
