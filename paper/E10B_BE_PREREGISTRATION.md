# E10b — Pre-registration: does a large effect survive the same 5-fold design?

Frozen: 2026-10-30T23:05+08:00, before any BE fold-1..4 run was launched.

## 1. Why

E10 (`paper/E10_CV_PREREGISTRATION.md`) held out every FD002 engine once and
found that the PE-versus-A difference, which looked like −0.90 (−7.7%) on the
single archived split, is **−0.126 (−0.9%) with a 95% CI of [−1.07, +0.80]**
once all 260 engines are held out. Per-fold estimates range from **−1.02 to
+1.70**, and PE wins on 51% of individual engines — indistinguishable from a coin
flip.

That result raises an obvious question the paper must answer rather than dodge:
**is the whole design unable to detect anything, or is 0.9 simply below its
resolution?** E3 also reported a much larger contrast on the same pipeline:
removing the time index altogether (arm BE) costs **+2.69 RMSE [+1.35, +4.10]**
on the single split. If that effect also collapses under 5-fold, then the design
is uninformative and the paper's thesis weakens. If it survives, the paper can
quantify the boundary: **which effect sizes are reproducible at 260 engines and
which are not.**

## 2. Design

* Arms: **BE** (no time channel) and **A** (linear cycle) — A already exists for
  all five folds from E10, so only BE must be trained.
* Folds 1–4 × seeds {2027, 2028, 2029} = **12 new runs**. Fold 0 reuses
  `experiments/e3/BE_s{2027,2028,2029}` verbatim.
* Identical configuration to E3/E10: `--subset FD002 --split-seed 7 --fold k
  --ref-mode global --epochs 80 --patience 20 --d-model 64 --num-layers 1`.
* Estimand and interval: **exactly the E10 recipe** — pooled MSE over 260 engines
  × 3 seeds, sqrt, fold-stratified engine bootstrap, B = 100 000, RNG 9000–9004,
  MC error reported. Sign convention Δ = RMSE(first) − RMSE(second), first = BE,
  so a **positive** value means removing time hurts.

## 3. Decision rule (fixed now)

| outcome | condition | interpretation |
|---|---|---|
| **REPRODUCED** | 95% CI lower bound **> 0** | the time channel is load-bearing at 260 engines; the paper reports the effect size with its interval |
| **ATTENUATED** | point estimate > 0 but CI crosses 0 | the single-split +2.69 was partly split-specific; the paper reports both numbers and drops "load-bearing" |
| **COLLAPSED** | point estimate ≤ 0 | the 5-fold design cannot resolve even a 2.7 effect; the paper must report that its own design is uninformative and the thesis weakens accordingly |

No margin, no alternative metric, no seed search. If the outcome is ATTENUATED or
COLLAPSED, that is what gets written.

## 4. What it buys the paper

Combined with E10, this gives a **calibration of the design** rather than a
single verdict:

| effect size (RMSE) | single split (52 engines) | five folds (260 engines) |
|---|---|---|
| 2.7 (remove time) | significant | **this experiment** |
| 0.9 (bounded encoding) | looked like −7.7% | −0.9%, CI crosses 0 |
| ~0.1 (lifespan estimator) | +0.08, CI crosses 0 | not tested |

If BE reproduces, the paper can state a usable rule: at 260 engines this design
resolves effects of order 2 RMSE but not of order 1, so **any published claim of a
5% improvement (≈0.7 RMSE at this baseline) is below the resolution of the
benchmark** — which is the paper's central claim, now supported by two contrasts
that differ only in effect size.

## 5. Cost and artefacts

12 runs × ~20 min at 4 concurrent ≈ 1 GPU-hour.

```
experiments/e10b/BE_s{2027,2028,2029}_f{1,2,3,4}/FD002/*/
results/dsh/e10/E10b_be_cv.json
paper/E10B_RESULT.md
```

## 6. Boundaries

FD002 only. FD003/FD004 remain sealed; no official test file is read. The A arm
is not retrained — it is taken from E10, which is the same code, same seeds, same
folds, so the pairing is exact.

## 7. Amendments

(none yet)
