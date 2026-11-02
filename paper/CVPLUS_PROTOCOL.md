# Cross-fitted conformal RUL intervals — frozen protocol

Status: **FROZEN 2026-11-02T14:30+08:00**, before any per-fold test prediction exists on disk.
Supersedes nothing; runs beside `paper/RECALIBRATION_LIMITS.md` and
`paper/COVERAGE_AUDIT_RESULTS.md`, neither of which changes.

## 1. Question

Every post-hoc recalibration family failed on this benchmark (0/8 beat "do nothing",
`RECALIBRATION_LIMITS.md`). The diagnosis there was *non-exchangeability*: the calibration
rows come from fold models, the deployed rows come from a different object, so a width
tuned on dev does not transfer.

This protocol tests the one repair that follows from that diagnosis: **make the deployed
predictor and the calibration rows come from the same model-quality distribution**, then
build the interval by conformal quantiles instead of by a fitted parametric width.

Two things are being separated for the first time on this benchmark:

* *validity* — does a 90% interval contain 90% of engines, measured without the integer
  endpoint convention that inflates PICP by 0.014–0.024?
* *price* — how much wider must the interval be than the widths currently reported?

## 2. Data and predictions (fixed before running)

* Arm A only. Subsets FD001–FD004. Seeds 2027–2029. Folds 0–4. 60 checkpoints, none retrained.
* Test prediction of fold model $(s,k)$ on engine $i$: `experiments/calx/fold_test/`,
  produced by `scripts/calx/dump_fold_test.py` with that fold's own training scaler,
  exactly as `evaluate_subset` builds it. This artefact does not exist upstream — E21
  averages the 15 fold models before writing, so the members are unrecoverable from
  `results/dsh/e21/preds/`.
* Calibration rows: `experiments/cal/dev_preds/`, one row per (engine, fold, seed); an
  engine appears only under the fold that held it out. $n$ = 100 / 260 / 100 / 249.
* Target $Y=\min(\text{RUL},125)$, integer. Test $n$ = 93 / 235 / 97 / 228.
* Levels: 0.90 (primary), 0.50 (secondary). Both reported, no level is dropped for being
  unflattering.

## 3. Coverage convention (this is the part earlier work got wrong)

Coverage of a discrete predictive distribution is measured by **randomized PIT**: for a pmf
$p$ on $\{0,\dots,125\}$ and observation $y$, draw $U\sim\text{Unif}(0,1)$ and count
$F(y-1)+U\cdot p(y) \le \tau$. An interval $[L,U]$ is scored by the same rule: the
indicator contains $y$, with the boundary case resolved by $U$. The integer-endpoint
convention (`L ≤ y ≤ U` on a grid) is reported only as a sensitivity row, never as the
headline — it was measured to inflate 90% coverage by 0.014–0.024 and 50% by 0.071–0.119.

## 4. Four methods, one per row of the result table

Let $\hat\mu_k(x_i)$ be fold model $k$'s prediction, $r_j = y_j - \hat\mu_{k(j)}(x_j)$ the
signed out-of-fold residual of calibration engine $j$, $n$ the calibration size.

**M0 `base`** — the frozen 66-parameter censored layer (`experiments/cal/audit/*.npz`,
`p_cens`). Reference, not a candidate. Its width was tuned so cross-fitted dev PICP90 = 0.90.

**M1 `cv+`** — constant-width cross-conformal. Pool the $n\cdot K$ values
$\{\hat\mu_k(x_i) \pm |r_j|\}$ and take the $\alpha$ and $1-\alpha$ empirical quantiles
(Barber et al. 2021). Distribution-free; worst case $1-2\alpha$, expected $\approx 1-\alpha$
because the fold models are highly correlated.

**M2 `ncv+`** — normalised cross-conformal. Score $s_j = |r_j| / \hat\sigma(\hat\mu_{k(j)})$,
interval $\hat\mu_k(x_i) \pm q\,\hat\sigma(\hat\mu_k(x_i))$, $q$ the conformal quantile of
$s$. $\hat\sigma$ is the frozen layer's own scale head — used as a *shape*, never as a
calibrated width, so this row tests whether the layer's variance profile is right even
though its level is wrong.

**M3 `cw-cp`** — covariate-shift weighted conformal. Weights
$w(\hat\mu) = \hat f_{\text{test}}(\hat\mu) / \hat f_{\text{dev}}(\hat\mu)$, both Gaussian
KDE with Silverman bandwidth on the respective $\hat\mu$ samples (test side uses the
15-model mean, dev side the out-of-fold mean). Weights clipped to their own 1st/99th
percentile; effective sample size reported. Weighted quantile per Tibshirani et al. (2019).
This row is the one the diagnosis predicts should work: KS on $|\text{error}|$ was
non-significant in 3/4 subsets, so the conditional error law is stable and only the
$\hat\mu$ marginal moved.

**Reference ceiling `oracle90`** — for each subset, the single multiplicative factor on
`base`'s width that makes *test* randomized coverage exactly 0.90. Not deployable; it is the
width any correct method must pay.

## 5. Decision rules (fixed now, evaluated once)

* **R1 (replication).** `base` test coverage at 0.90 is below nominal by more than
  1.96 SE in ≥3 of 4 subsets. Expected from `RECALIBRATION_LIMITS.md`; if it fails, the
  audit numbers and this run disagree and the run is reported as inconsistent, not merged.
* **R2 (validity).** `ncv+` and `cw-cp` reach ≥0.90 − 1.96 SE on randomized coverage in ≥3
  of 4 subsets. `cv+` is additionally held to its own $1-2\alpha$ floor of 0.80.
* **R3 (price).** The valid method's mean width at 0.90, divided by `base`'s, is the price.
  It is reported per subset with a bootstrap interval over engines. No threshold is set in
  advance: the number is the result, not a pass/fail.
* **R4 (sharpness claim).** `cw-cp` is called *better than* `cv+` only if it is narrower by
  ≥10% in ≥3 of 4 subsets while R2 still holds for it. Below that, the weighting is called
  uninformative and the row is reported as such.

Negative outcomes are reportable outcomes. If all four fail R2, the conclusion is that the
non-exchangeability is not an artefact of model averaging and conformal machinery does not
repair it either; if they pass but R3 is large, the conclusion is that the literature's
reported widths are unattainable at nominal coverage and the correct number is the one
measured here.

## Amendment 1 (2026-11-02T14:40) — what R4 compares

R4 as written compares `cw-cp` with `cv+`. That comparison is confounded: `cv+` is a
constant-width interval over a population whose error scale varies several-fold with RUL,
so any normalised method is narrower for reasons that have nothing to do with the weights.
R4 therefore compares **`cw-cp` against `ncv+`** — same score, same centres, weights on
versus off — which isolates the covariate-shift correction. `cv+` stays in the table as the
distribution-free reference, and the `cv+`→`ncv+` width drop is reported separately as the
value of normalising. No test number had been seen when this was amended.

## 6. Prohibitions

No subset, seed, fold, level, bandwidth or clipping quantile may be chosen after seeing a
test coverage number. No method may be added after the first run and reported as
pre-registered; later additions are labelled exploratory. The 653-engine pooled row is
computed but is not used to rescue a per-subset failure.

## 7. Artifacts

`scripts/calx/dump_fold_test.py` → `experiments/calx/fold_test/`;
`scripts/calx/conformal.py` → `experiments/calx/conformal.json`;
`paper/CONFORMAL_RESULTS.md`; figures under `figures/calx/`.

## 8. Outcome (appended 2026-11-02T15:20, after the one frozen run)

Executed once, as written. Verdicts against section 5:

* **R1 replication — passes, with a correction.** The deployed layer lands at
  0.924 / 0.869 / 0.932 / 0.842 randomized at nominal 0.90. Only FD004 (−0.058) is near its
  own detection floor (0.063 at n=228); the other three are *over*-covered, not under.
  The 0.79–0.86 quoted in `RECALIBRATION_LIMITS.md` came from a second implementation of the
  same 66-parameter architecture (`calibrate.CalibNet`, t5) that also lives in the repo. Both
  reproduce; the write-ups never named which. `CONFORMAL_RESULTS.md` §5 carries the table.
* **R2 validity — passes formally, fails on price.** The three symmetric rows reach
  0.949–0.983 at 1.35–1.95× the base width. Valid and useless.
* **R3 price — 1.35–1.95×.** Reported per subset with bootstrap intervals.
* **R4 weighting — fails, recorded as uninformative.** `cw-cp` is 1–2% narrower than `ncv+`
  with effective sample sizes of 92–235; the covariate-shift weighting has nothing to act on.

Post-hoc and labelled exploratory (section 6): dropping the symmetry assumption. The
signed-residual construction reaches 0.900–0.925 at the base width in all four fleets
(§2–§3 of `CONFORMAL_RESULTS.md`). It was not pre-registered and is not presented as such.

One artefact was missing upstream and had to be produced: per-fold test predictions
(`dump_fold_test.py`). Its 15-model average reproduces the published ensemble to 2.1e-07.
