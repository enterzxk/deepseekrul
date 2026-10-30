# E7 — Multi-cutoff resolution: post-hoc re-analysis (not pre-registered)

**Disclosure first.** Unlike E2/E3/E3b/E3c/E4, this analysis was **not frozen in a
separate pre-registration**. The design grid (`single` = {100}, `pair` = {100,140},
`all` = {60,100,140,180}) was written into `scripts/eval_resolution.py` before the run,
but no sha256 was recorded beforehand, and the script was executed within minutes of
being written. It is therefore reported as **exploratory**, with the claims marked
accordingly in `paper/CLAIMS.md`.

Why the risk is limited: the direction is mechanically expected — more observations per
engine at fixed engine count reduces the standard error — so this is not a search over
many designs hoping for a favourable draw. What is empirical is the *magnitude*, and
that is what the numbers below report.

## Design

Re-analysis of `protocol_c_predictions.csv`, which every run already stores: one
prediction per engine per absolute cutoff (60/100/140/180). Engines are restricted to
those that contribute every cutoff in the design, so the only quantity that changes is
observations per engine. Contrast: engine-clustered bootstrap, 100k resamples, three
RNG streams, seeds averaged — the E6 recipe, unchanged.

## Result

| contrast | design | engines | obs/engine | ΔRMSE | 95% CI | width | ratio |
|---|---|---:|---:|---:|---|---:|---:|
| FD002 A vs BE | single | 156 | 1 | +0.002 | [−1.672, +1.690] | 3.362 | 1.00 |
| | pair | 156 | 2 | −0.625 | [−1.891, +0.638] | 2.528 | **0.75** |
| | all | 123 | 4 | −0.672 | [−1.854, +0.474] | 2.328 | 0.69 |
| FD002 A vs PE | single | 156 | 1 | +1.179 | [−0.567, +2.941] | 3.508 | 1.00 |
| | pair | 156 | 2 | +1.360 | [+0.207, +2.510] | 2.303 | **0.66** |
| | all | 123 | 4 | +0.943 | [−0.025, +1.898] | 1.923 | 0.55 |
| FD001 A vs BE | single | 60 | 1 | −0.962 | [−3.333, +1.302] | 4.635 | 1.00 |
| | pair | 51 | 2 | −1.387 | [−3.014, +0.268] | 3.282 | **0.71** |
| | all | 42 | 4 | −2.079 | [−3.820, −0.316] | 3.503 | 0.76 |

Two readings:

1. **Two cutoffs on the same engines cut the interval to 66–75%.** No retraining, no
   new engines; the engine-level variance is shared across cutoffs while the residual
   noise is not.
2. **Four cutoffs can backfire.** On FD001 the 180-cycle cutoff removes too many short
   engines (60 → 42), and the width rises relative to the two-cutoff design. The design
   has to be checked per subset, not assumed.

## The incidental finding, which matters more

At fixed absolute cutoffs the time contrast collapses: **+0.002** at cutoff 100 and
−0.625 at 100+140, against **+2.693** under the deployment-style matched cutoff
(§4.5.2). Same models, same engines, same statistic. The cut placement, not the
method, decides whether the time channel looks load-bearing.

## Boundaries

* Absolute cutoffs and the matched cutoff do not have the same target distribution, so
  the numbers in the last paragraph are not estimates of one quantity under two
  estimators; they are two protocols that answer different questions. That is the
  point, and §4.5 states it that way.
* FD001's `all` row is confounded by engine attrition (42 of 60), as noted.
* Not pre-registered (see the disclosure above).

Artefacts: `results/dsh/e7/RESOLUTION.json`, `scripts/eval_resolution.py`.
