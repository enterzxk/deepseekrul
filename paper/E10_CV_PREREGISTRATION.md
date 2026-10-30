# E10 — Pre-registration: 5-fold cross-validated PE vs A on FD002

Frozen: 2026-10-30, before any fold-1..4 run was launched.
Author of the design: this round; no result of this experiment existed at freeze time.

## 1. Why this experiment exists

The bounded sinusoidal encoding (PE) beats the linear cycle term on the point
estimate in both subsets, with every seed agreeing:

| subset | ΔRMSE (PE − A) | per-seed | engine-clustered 95% CI |
|---|---:|---|---|
| FD002 (52 held-out) | **−0.908** (−7.7%) | −0.73 / −0.87 / −1.13 | [−2.03, +0.29] |
| FD001 (20 held-out) | **−0.888** (−7.8%) | −1.74 / −0.17 / −0.75 | [−2.23, +0.32] |

Neither interval excludes zero, so the paper can currently only say "tends to PE".
A variance decomposition (`scripts/dsh_e9_variance_decomp.py`,
`results/dsh/e9/variance_decomp.json`) shows why:

| subset | engine share of variance | seed share | engine SE | seed SE (3 seeds) |
|---|---:|---:|---:|---:|
| FD002 | **97.6%** | 2.4% | 0.733 | 0.116 |
| FD001 | 79.2% | 20.8% | 0.894 | 0.458 |

**More seeds cannot fix this** (10 seeds would move the SE from 0.116 to 0.063
while the engine term stays at 0.733). The only lever is more independently
held-out engines. FD002 has 260 engines; the single 80/20 split exposes 52.

Therefore: hold out every engine exactly once, via 5-fold cross-validation.

## 2. Design

* **Partition.** `prepare()` draws `rng = default_rng(split_seed + 20261020)`,
  then `shuffled = rng.permutation(units)`, and holds out the first
  `n_val = round(0.2·N)` entries. Fold *k* holds out the k-th contiguous block
  of the **same** permutation. Hence fold 0 is bit-identical to the archived
  split, and folds 0..4 partition all 260 engines exactly once.
* **Verified before freezing** (`scripts/.foldcheck.py`, run 2026-10-30):

  | subset | fold0 == archived val_units | blocks | union | duplicates |
  |---|---|---|---|---|
  | FD002 | **True** (52 engines) | 5 × 52 | **260 / 260** | 0 |
  | FD001 | **True** (20 engines) | 5 × 20 | **100 / 100** | 0 |

* **Arms.** `A` (`min(c/250,1)` linear cycle) and `PE` (bounded 4-period
  sin/cos). No BE, no PS: those contrasts already have resolved intervals and
  are not the subject of this experiment.
* **Seeds.** 2027, 2028, 2029 (unchanged from E3, for comparability).
* **New runs.** fold ∈ {1,2,3,4} × {A, PE} × 3 seeds = **24 runs**, FD002 only.
  fold 0 reuses `experiments/e3/{A,PE}_s{2027,2028,2029}` verbatim — no rerun.
* **Budget.** `--epochs 80 --patience 20 --d-model 64 --num-layers 1
  --ref-mode global --split-seed 7 --fold k`, identical to E3 otherwise.
* **Prediction of record.** `matched_cutoff_predictions.csv` (one deployment-style
  cut per engine), the same artefact the E3 numbers came from.

## 3. Estimand and interval

* **Pooled MSE, then square root.** For each (fold, seed, engine) we have the
  signed errors `e_A`, `e_PE`. Define
  `MSE_a = mean over all 260×3 observations of e_a²`, and
  `ΔRMSE = sqrt(MSE_A) − sqrt(MSE_PE)`.
  Pooling at the MSE level (rather than averaging 15 per-fold RMSEs) keeps the
  estimand identical to the reported RMSE and gives every engine equal weight.
* **Interval.** Stratified engine-clustered bootstrap: within each fold,
  resample its 52 engines with replacement, keep all 3 seeds of a resampled
  engine together, rebuild the 260-engine pool, recompute `ΔRMSE`.
  **B = 100 000, RNG seeds 9000–9004 reported separately, MC error reported.**
  This is the same recipe as `paper/E6_STATS_AUDIT.md`, so the number is
  comparable with every other interval in the paper.
* **Sign convention.** `Δ = RMSE(first) − RMSE(second)`, negative favours the
  second arm. Here: PE − A, negative favours PE.

## 4. Decision rule (fixed now, not after seeing the numbers)

| outcome | condition | what the paper may say |
|---|---|---|
| **ESTABLISHED** | 95% CI upper bound **< 0** and \|Δ\|/RMSE_A **≥ 5%** | "PE improves accuracy by X% (95% CI …, 260 engines held out once each)" |
| **DIRECTIONAL** | point estimate ≤ −5% but CI crosses 0 | unchanged wording: "tends to PE, not established" |
| **NOT REPRODUCED** | point estimate > −5% | PE discussed only as a robustness note; Track M closes |
| **REVERSED** | point estimate > 0 | report as a failed replication of the E3 point estimate |

No margin is used in this experiment. The E3 non-inferiority margin (+0.30)
was already shown to be inside the resolution of 52 engines
(`paper/E3_AMENDMENT_1.md`); this experiment asks the simpler question: **is the
improvement distinguishable from zero once every engine has been held out?**

## 5. Power expectation (stated before running)

From the variance decomposition, if the per-fold effect equals the E3 effect
(−0.908) and the engine SE scales as 1/√(number of distinct held-out engines):

| engines held out | projected SE | projected 95% CI | excludes 0 |
|---:|---:|---|---|
| 52 (single split, observed) | 0.733 | [−2.35, +0.53] | no |
| 104 (2 folds) | 0.519 | [−1.92, +0.11] | no |
| 156 (3 folds) | 0.423 | [−1.74, −0.08] | marginal |
| **260 (5 folds)** | **0.328** | **[−1.55, −0.27]** | **yes, with margin** |

Two honest caveats, recorded now:

1. The projection assumes the effect is homogeneous across folds. If the effect
   is driven by a subset of engines, pooling will dilute it and the interval
   will still cross zero. **That outcome is informative and will be reported
   as such** — it would mean the E3 point estimate was fold-specific.
2. Fold-level models differ (different 208-engine training sets), which adds a
   between-fold component the single-split SE does not contain. The projection
   is therefore optimistic; a realised SE of 0.4–0.45 is entirely possible.

## 6. Failure handling

* If a run crashes: fix only the crash, keep the config, rerun that cell; record
  the incident in `paper/RUNLOG.md`. No config search.
* If the CI crosses zero: **stop, do not add seeds, folds or arms to chase
  significance.** Write it up as DIRECTIONAL and close Track M.
* If the effect reverses: report it, and treat the E3 point estimate as
  fold-specific in §4.3.

## 7. Artefacts

```
experiments/e10/{A,PE}_s{2027,2028,2029}_f{1,2,3,4}/FD002/*/
    metrics.json                        (with fold, prereg_sha, started_at, finished_at)
    matched_cutoff_predictions.csv
results/dsh/e10/E10_cv.json             pooled estimand, per-fold, per-seed, bootstrap
paper/E10_RESULT.md
```

## 8. Relation to the manuscript

* **If ESTABLISHED:** §4.3.2 upgrades from "tends to PE" to a measured
  improvement, and the paper gains the Track M ingredient it currently lacks
  (relative gain ≥5%, CI excluding 0, NASA/late/severe not degraded, mechanism
  identified by E3c). §4.2 also gains a worked example of "resolution bought by
  evaluation design rather than by a new module".
* **If DIRECTIONAL:** §4.3 keeps today's wording and the paper stays Track E.
  The negative result joins the two existing self-corrections (E3 margin, the
  21-pair table) as evidence for the thesis.
* Either way the 5-fold out-of-fold predictions become the paper's primary
  artefact for §4.3, replacing the single-split numbers.

## 9. Boundaries

* FD002 only. FD001 has 100 engines total (20 per fold); five folds give 100
  distinct engines, whose projected SE (0.894/√5 = 0.40) still leaves the upper
  bound near zero. FD001 is therefore reported as a replication check, not as a
  second test.
* FD003/FD004 remain sealed. No official test file is read at any point.
* `fold = -1` is the default and reproduces the archived single split exactly;
  the equality check (`scripts/dsh_e3_equality.py --mode check`) is rerun after
  this patch and its output pasted into `paper/RUNLOG.md`.

## 10. Amendments

(none yet)
