# 4.6 Measurement uncertainty budget

Sections 4.1–4.3 report intervals one contrast at a time. This section assembles them
into a single budget, so that a reader can see where the uncertainty in a C-MAPSS
comparison actually comes from and which knob would reduce it. The treatment follows
the GUM convention: each source is evaluated separately, combined in quadrature, and
expanded by a coverage factor.

## 4.6.1 The measurement model

The measurand is the difference in RMSE between two trained models, evaluated on a fixed
pool of engines:

```
Δ = RMSE(model 1) − RMSE(model 2)
```

Δ is not a property of the two models alone. Holding the models fixed, Δ still varies with
(i) which engines are held out, (ii) which initialisation is used, and (iii) which fold
partition is drawn. The budget evaluates those three as input quantities, plus a
determinism term:

| symbol | source | type | estimate | how it was measured |
|---|---|---|---|---|
| `u_seed` | initialisation | A | SD across 3 seeds / √3 | E10, E10b (same fold, same architecture) |
| `u_fold` | engine sampling (partition level) | A | SD of the 5 fold estimates / √5 | E10, E10b |
| `u_repro` | re-run determinism | B | 0.0000 RMSE | E4 from-scratch rerun |
| — | engine sampling (within fold) | A | 0.328, **not combined** — see below | bootstrap SE, E9 |

**Why `u_engine` is not in the quadrature.** An earlier version of this section added a
within-fold bootstrap SE (0.328) to the fold term (0.493). That double counts: both
quantities estimate the same thing — how much an RMSE moves when the engine sample changes
— and the fold term is the larger, more conservative measurement of it, because each fold
estimate is itself a 52-engine draw. Adding them as if independent inflated `u_c` from 0.52
to 0.61 and, more importantly, made the fold term look like a component that could not
shrink. It can; §4.6.3 shows how.

`u_arch` is deliberately absent: the spread across five different architectures is a
between-model variance, not a contribution to the uncertainty of one pairwise contrast.

Combination: `u_c = sqrt(u_seed² + u_fold²)`; expanded uncertainty `U = 2 u_c` (k = 2).

## 4.6.2 The budget

| estimand | u_seed | u_fold | u_c | U (k=2) | cross-check: clustered bootstrap |
|---|---:|---:|---:|---:|---:|
| PE − A (bounded encoding) | 0.158 | 0.493 | **0.518** | **1.036** | 0.475 → U 0.95 |
| BE − A (time removed) | 0.203 | 0.498 | **0.538** | **1.076** | 0.475 → U 0.95 |

`results/dsh/e12/uncertainty_budget.json` (components), `results/dsh/e16/mde_corrected.json`
(scaling). The two routes agree to within 9%, and the GUM route is the conservative one.

Read against the observed effects, the budget explains both verdicts in §4.3:

* BE − A = **+1.133** against U = 1.08 → clears zero, by a margin of one twentieth of the
  baseline.
* PE − A = **−0.126** against U = 1.04 → indistinguishable from zero.

A single number — **U ≈ 1.0 RMSE at 260 engines, or 7% of the baseline** — summarises the
resolution of every comparison in this paper. It is consistent with the independent MDE
calculation of §4.2.1 (1.31–1.54 at 260 engines), which arrives at the same scale from a
completely different route.

## 4.6.3 The sample size a 5% claim would need

Because `u_fold` is an engine-sampling quantity measured at 52 engines per fold, it shrinks
as more engines are held out, like every other sampling term:

```
u_c(n) ≈ u_c(260) · sqrt(260 / n)
```

| total engines | u_c | U (k=2) | U as % of a 14-RMSE baseline |
|---:|---:|---:|---:|
| 260 (FD002) | 0.518 | 1.04 | 7.4% |
| 360 (FD001+FD002) | 0.440 | 0.88 | 6.3% |
| 708 (all four subsets) | 0.314 | 0.63 | 4.5% |
| 1300 (projected) | 0.232 | 0.46 | 3.3% |

**To resolve a 5% difference (0.70 RMSE) at 95% with 80% power this design needs about 575
held-out engines.** FD002 alone provides 260, which is why no 5% claim in this literature
can be settled on one subset; the four subsets together (708) are just sufficient.

This replaces an earlier conclusion of this section, which held that the fold term sets a
floor that more data cannot move and that quintupling the engine pool would change `u_c` by
only 12%. That was a consequence of the double counting described in §4.6.1; the correct
scaling is the 1/√n above, and the "data collection has hit a floor" claim is withdrawn.

What does *not* scale is the effect itself: it varies across folds by more than its own mean
(§4.2.2), which is why §4.2.4 recommends reporting fold-level estimates beside any pooled
one. More engines narrow the interval around the pooled value; they do not make the
fold-to-fold spread of the effect disappear, and that spread is what a reader needs in order
to judge whether a reported improvement would survive a different split.
