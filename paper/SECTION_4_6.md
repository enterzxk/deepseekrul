# 4.6 Measurement uncertainty budget

Sections 4.1–4.3 report intervals one contrast at a time. This section assembles them
into a single budget, so that a reader can see where the uncertainty in a C-MAPSS
comparison actually comes from and which knob would reduce it. The treatment follows
the GUM convention: each source is evaluated separately, combined in quadrature, and
expanded by a coverage factor.

## 4.6.1 The measurement model

The measurand is the difference in RMSE between two trained models, evaluated on a
fixed pool of engines:

```
Δ = RMSE(model 1) − RMSE(model 2)
```

Δ is not a property of the two models alone. Holding the models fixed, Δ still varies
with (i) which engines are held out, (ii) which initialisation is used, (iii) which fold
partition is drawn, and (iv) which architecture pair is being compared. The budget
treats those four as separate input quantities plus a determinism term:

| symbol | source | type | estimate | how it was measured |
|---|---|---|---|---|
| `u_seed` | initialisation | A | SD across 3 seeds / √3 | E10, E10b (same fold, same architecture) |
| `u_engine` | engine sampling within folds | A | per-engine SD / √(n_fold · k) | bootstrap SE, E9 |
| `u_fold` | fold partition | A | SD of the 5 fold estimates / √5 | E10, E10b |
| `u_arch` | architecture choice | A | SD across 5 architectures | E11 |
| `u_repro` | re-run determinism | B | 0.0000 RMSE | E4 from-scratch rerun |

Combination: `u_c = sqrt(Σ u_i²)`; expanded uncertainty `U = 2 u_c` (k = 2, ≈95%).

## 4.6.2 The budget

| estimand | u_seed | u_engine | u_fold | u_c | U (k=2) | dominant term |
|---|---:|---:|---:|---:|---:|---|
| PE − A (bounded encoding) | 0.158 | 0.328 | 0.493 | **0.613** | **1.227** | u_fold (65%) |
| BE − A (time removed) | 0.203 | 0.328 | 0.498 | **0.630** | **1.260** | u_fold (63%) |

`results/dsh/e12/uncertainty_budget.json`, `scripts/dsh_e12_uncertainty.py`.

Read against the observed effects, the budget explains both verdicts in §4.3 without
any further argument:

* BE − A = **+1.133** against U = 1.26 → the interval clears zero, but only just.
* PE − A = **−0.126** against U = 1.23 → indistinguishable from zero.

A single number — **U ≈ 1.2 RMSE at 260 engines** — summarises the resolution of every
comparison in this paper, and it is consistent with the independent MDE calculation of
§4.2.1 (1.16–1.46 at 260 engines) computed from a completely different route.

## 4.6.3 Why more engines do not help

The budget answers the obvious follow-up question directly. Holding the fold count at
five and increasing the engines per fold:

| engines per fold | total engines | u_engine | u_fold | u_c | engine share |
|---:|---:|---:|---:|---:|---:|
| 52 | 260 | 0.328 | 0.493 | **0.613** | 28.6% |
| 104 | 520 | 0.232 | 0.493 | 0.568 | 16.7% |
| 260 | 1300 | 0.147 | 0.493 | **0.539** | 7.4% |

**Quintupling the engine pool from 260 to 1300 moves u_c by 12%**, because the fold term
does not shrink: it is set by how much the effect itself varies from one partition to
another, not by how many engines each partition contains. Data collection on this
benchmark has hit a floor that more data does not move.

That floor is a property of the *effect*, not of the estimator. It is also the reason
§4.2.5 recommends reporting fold-level estimates: their spread is the thing that limits
the claim, so it should be visible rather than absorbed into a single interval.

## 4.6.4 A caveat about independence

The GUM combination assumes the components are independent. They are not, exactly:
`s_u` and `u_engine` both shrink as engines are pooled, and the fold term partly
refolds engine heterogeneity. The practical consequence is that `u_c` from this budget
(0.613) is **larger** than the directly bootstrapped standard error for the same
contrast (0.475, §4.3.2). The budget is therefore conservative, which is the right
direction for a stated uncertainty, but a reader should not treat the two numbers as
independent confirmations of each other. The paper reports both and says why they
differ.

## 4.6.5 What the budget implies for reported results

Three consequences, all of which the manuscript applies:

1. **Quote U, not just a CI.** A CI computed at one fold count is conditional; U states
   the resolution of the measurement itself, which is what a reader needs in order to
   judge someone else's claim.
2. **An effect below 1.2 RMSE is not measurable at this scale.** That is ≈9% of the
   pooled baseline of 13.98, which is why the project's original 5% acceptance
   criterion could not have been met by any method, however good.
3. **The dominant term names the next experiment.** Today it is `u_fold`, so the
   useful next step is more independent partitions, not more engines and not more
   seeds. Once `u_fold` falls below `u_engine` (at roughly 30 folds for these data),
   the ordering flips and engines become the lever again.
