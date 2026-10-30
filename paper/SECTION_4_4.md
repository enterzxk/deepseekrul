# 4.4 The lifespan-scale headroom is real, and the estimator that would capture it is not

Sections 4.1–4.3 treat the time index as a feature to be encoded or removed. This
section asks a different question: if a model were told the *remaining life* of
the engine it is looking at — one number per engine, the scale `T` — how much
would that be worth? And can that number be estimated from the observed prefix?

The answers are clean and opposite: the upper bound is large, and the deployable
version of it is indistinguishable from doing nothing.

## 4.4.1 The upper bound: +8.2 RMSE

An oracle arm replaces the cycle coordinate `min(c/250, 1)` with `c / T_true`,
where `T_true` is the engine's full lifespan. This uses information that no
deployment has, so it is not a method — it is a measurement of how much the
lifespan scale is worth.

| arm | RMSE | MAE | NASA mean | late | severe e>10 |
|---|---:|---:|---:|---:|---:|
| cycle (`min(c/250,1)`) | 21.32 ± 1.04 | 15.38 | 17.13 | 0.436 | 0.276 |
| **oracle (`c/T_true`)** | **13.09 ± 2.08** | 9.16 | **2.71** | 0.429 | 0.212 |

Engine-clustered paired bootstrap (4 000 resamples at the time, sign convention
Δ = RMSE(first) − RMSE(second)):

> **oracle − cycle = +8.23 [+5.30, +11.02]**, the only contrast in the whole study
> whose interval clears zero by a wide margin.

The effect is not an artefact of one metric: NASA mean falls from 17.13 to 2.71
and severe late reporting from 0.276 to 0.212, while the late-reporting rate is
essentially unchanged (0.436 → 0.429). Linearly rescaling the time axis makes the
model's errors symmetric in *relative* rather than absolute life, which is the
behaviour the NASA score rewards.

This is the strongest quantitative signal the project produced, and it is the
reason the lifespan-scale direction was pursued for three further rounds rather
than closed.

## 4.4.2 The deployable version: +0.08, interval crossing zero

Replacing `T_true` with an estimate `T̂` obtained from the observed prefix only —
a per-engine linear extrapolation of a health index, with a fallback to the
training-set median lifespan when the slope is non-positive — gives:

| arm | RMSE | MAE | NASA mean | late | severe e>10 |
|---|---:|---:|---:|---:|---:|
| cycle | 21.32 ± 1.04 | 15.38 | 17.13 | 0.436 | 0.276 |
| constant scale (`c/median(T_train)`) | 20.15 ± 1.54 | 15.55 | 11.20 | 0.423 | 0.276 |
| **per-engine estimate `c/T̂`** | **21.24 ± 1.71** | 15.99 | 10.95 | 0.429 | 0.263 |
| rate-stratified variant | 20.07 ± 1.22 | 15.56 | 8.44 | 0.468 | 0.301 |

| contrast | ΔRMSE | 95% CI | excludes 0 |
|---|---:|---|---|
| estimate − cycle | **+0.08** | [−2.10, +2.27] | no |
| estimate − constant scale | −1.09 | [−3.25, +1.06] | no |
| rate-stratified − cycle | +1.25 | [−1.31, +3.63] | no |

Three statements follow, and all three matter for the paper's thesis:

1. **The per-engine estimator is not better than a constant.** A single number —
   the median training lifespan — captures everything the estimator contributes.
2. **The constant is not better than the plain cycle coordinate.** `pooled` beats
   `cycle` by 1.17 RMSE with an interval that crosses zero.
3. **So the deployable gain is zero**, against an oracle gain of 8.23. The
   headroom exists; nothing in this pipeline reaches it.

## 4.4.3 Why it fails: monotonicity is not the bottleneck

The obvious diagnosis is that the health index is not monotone, so a linear
extrapolation of it is meaningless. That diagnosis was tested directly
(`paper/E2_PREREGISTRATION.md`, criteria frozen before the run;
`paper/E2_RESULT.md`).

Three health-index variants were built on FD002, in increasing order of
constraint:

| variant | monotonicity score | median Spearman vs time | corr(T̂, T) all | corr(T̂, T) excluding fallbacks | median T̂/T | fallback rate |
|---|---:|---:|---:|---:|---:|---:|
| H0 cluster-z → PCA1 | 0.505 | 0.377 | 0.084 | 0.319 | 1.92 | 3.7% |
| H1 + EMA(0.3) | 0.518 | 0.674 | 0.157 | 0.346 | 1.92 | 3.9% |
| H2 + isotonic projection | **1.000** | **0.976** | 0.280 | **0.347** | 1.92 | 0.9% |

The isotonic projection drives the index from 0.505 to a **perfect 1.000**
monotonicity and its rank correlation with time from 0.377 to **0.976**. The
downstream quantity — the correlation between estimated and true lifespan — moves
from 0.319 to **0.347**. The pre-registered gate was 0.5: **FAIL**.

Two further observations from the same run:

* **The bias is structural, not stochastic.** The median `T̂/T` is **1.92 in every
  variant**, and 1.90 in the independently written four-arm implementation. A
  1.9× multiplicative bias that survives three different health indices, two
  code paths and 1 383 observations is a property of the extrapolation, not of
  the index.
* **A monotonicity criterion alone would have passed.** H0's median Spearman of
  0.377 already exceeds the "≥0.3" threshold that a naive reading of the task
  specification would use, while the estimator it feeds is unusable. This is the
  concrete reason section 5's G1 insists that a resolution criterion must name the
  quantity it applies to: the index, or the estimator downstream of it.

## 4.4.4 A withdrawn number, and what it cost

An earlier version of this section reported that the deployable estimator
"recovered 38% of the oracle gain" (18.18 vs 21.32). That number is withdrawn.
Two independent defects produced it:

1. **The slope denominator was wrong** in the training-side estimator
   (`k·Σt − (Σt)²` instead of `k·Σt² − (Σt)²`), which is negative for `k > 3`,
   so the slope was never positive and **32 498 of 32 498 windows (100%)** fell
   back to the pooled constant. The "per-engine" arm was a constant arm.
2. **The validation-side estimator was given the true lifespan as its cutoff**
   (`that(frame, int(life[u]))`), so it saw the entire degradation trajectory and
   was then clipped at `1.2 · T_true`. In 52 held-out engines, **67.3%** had
   `T̂` exactly equal to `1.2 · T_true`, with `corr(T̂, T) = 0.936`. The validation
   feature was a rescaled oracle.

After fixing both (prefix-only estimator, cutoff replicated and asserted against
the emitted feature), the correlation with true lifespan is **0.196** and the
median ratio **1.90**, and the interval on the contrast crosses zero.

The episode is reported here rather than deleted because it is the paper's second
worked example of the same failure mode as section 4.3: a number that looked like
a positive result was produced by an evaluation defect, and only a converged or
corrected estimate revealed it. `paper/E3_AMENDMENT_1.md` records the parallel
case for the time-encoding margin.

## What this section establishes, and what it does not

* **Supported.** Lifespan-scale normalisation is worth **+8.23 RMSE
  [+5.30, +11.02]** when the lifespan is known.
* **Supported.** Every deployable estimator tested recovers none of it: the best
  is +0.08 with an interval crossing zero, and it is not better than a constant
  median scale.
* **Supported.** The failure is not caused by a non-monotone health index: forcing
  monotonicity to 1.000 moves the estimator correlation only to 0.347.
* **Not supported.** That no estimator could work. A different observable — one
  that actually grows with damage and is calibrated in absolute units — was not
  found here; the paper reports the search, its gate, and its failure.
* **Not supported.** Any statement about FD003/FD004 or the official test set.
  Every number in this section comes from FD002 training engines.
