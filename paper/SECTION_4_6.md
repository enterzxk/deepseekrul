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

## 4.6.3 Two different sample-size questions

"How many engines do I need?" has two answers, and they must not be mixed.

| target | definition | formula for a 0.70-RMSE (5%) difference |
|---|---|---|
| **T1 interval precision** | expanded uncertainty U = 2u_c does not exceed 0.70 | `n = (2σ / 0.70)²` |
| **T2 detection power** | 80% power to reject zero at two-sided α = 0.05 when the true difference is 0.70 | `n = ((1.96 + 0.84)σ / 0.70)²` |

T2 is strictly larger because it also pays the power term. An earlier draft of this section
reported only T1 (≈575 engines) and let it read as a power requirement; that was wrong.

Because σ varies across folds by about 30% (§4.2.1), both answers are ranges:

| target | σ = 6.26 (fold 0) | σ = 8.88 (3-seed pooled) |
|---|---:|---:|
| **T1** | **320 engines** | **644 engines** |
| **T2** | **627 engines** | **1262 engines** |

A non-parametric paired power simulation (20 000 resamples per grid point, drawing per-engine
influence values) agrees with the closed form: at σ = 6.26 the simulated power reaches 0.85 at
n = 708, slightly conservative against the formula's 627
(`results/dsh/e19/power_and_folds.json`).

**Consequence.** FD002 alone (260 engines) meets neither target. The four subsets together
(708) meet T1 and meet T2 only at the lower σ estimate. The correct statement for the
guidelines is therefore:

> **A 5% RMSE difference is visible at k = 2 from roughly 320–640 held-out engines, and
> detectable at 80% power from roughly 630–1260.** On a single C-MAPSS subset neither is
> available, so a 5% improvement claim is not decidable there — independently of method
> quality.

What does *not* scale is the effect itself: it varies across folds by more than its own mean
(§4.2.2), so §4.2.4 recommends reporting fold-level estimates beside any pooled one.
