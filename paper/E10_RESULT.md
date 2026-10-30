# E10 — Result: the bounded-encoding advantage does not survive cross-validation

Run: 2026-10-30 21:10–22:50 local, A40, 24 runs (A/PE × 3 seeds × folds 1–4) plus
the 6 archived fold-0 runs reused verbatim.
Pre-registration: `paper/E10_CV_PREREGISTRATION.md`, sha `aa7072f7…`, frozen before
any fold-1 run existed. Analysis: `scripts/dsh_e10_cv_analyze.py`,
`results/dsh/e10/E10_cv.json`.

## Verdict: **NOT_REPRODUCED**

The single-split estimate of −0.908 (−7.7%) collapses to **−0.126 (−0.9%)** when
every FD002 engine is held out exactly once. The 95% interval crosses zero.

| | pooled, 260 engines × 3 seeds |
|---|---:|
| RMSE, arm A (`min(c/250,1)`) | 13.983 |
| RMSE, arm PE (bounded sin/cos) | 13.857 |
| **ΔRMSE (PE − A)** | **−0.126 (−0.9%)** |
| 95% CI (fold-stratified engine bootstrap, 100 000 × 5 RNG) | **[−1.073, +0.795]** |
| Monte-Carlo error on the upper bound | 0.0034 |

Pre-registered rule (`E10_CV_PREREGISTRATION.md` §4): ESTABLISHED required the
upper bound **< 0** and |Δ|/RMSE_A ≥ 5%. Neither holds. The outcome is
**NOT_REPRODUCED**, not DIRECTIONAL: the point estimate is also below the 5%
materiality threshold.

## The per-fold spread is the actual finding

| fold | held-out engines | RMSE A | RMSE PE | Δ (PE − A) | engines where PE wins |
|---|---:|---:|---:|---:|---:|
| 0 (the archived split) | 52 | 12.679 | 11.775 | **−0.904** | 67% |
| 1 | 52 | 13.494 | 15.192 | **+1.699** | 38% |
| 2 | 52 | 15.097 | 14.074 | −1.023 | 48% |
| 3 | 52 | 14.351 | 13.874 | −0.477 | 48% |
| 4 | 52 | 14.176 | 14.145 | −0.031 | 56% |
| **pooled** | **260** | 13.983 | 13.857 | **−0.126** | **51%** |

Three facts, each verifiable from the table:

1. **The sign is not stable.** Fold 0 says PE is 0.90 better; fold 1 says it is
   1.70 worse. The spread between fold estimates is **2.62 RMSE**, which is 21% of
   the baseline and roughly 1.8× the single-split MDE (1.46). Every fold is a
   valid 80/20 split of the same dataset; nothing distinguishes them except which
   engines happened to land in the holdout.
2. **The seed agreement inside a fold is misleading.** Within fold 1, all three
   seeds agree (+1.65, +2.02, +1.42); within fold 0, all three agree
   (−0.73, −0.87, −1.13). Seed replication reproduces the fold's answer
   faithfully and says nothing about whether the answer generalises. This is the
   concrete instance of the variance decomposition in §4.1: seeds are 2.4% of the
   variance, engines are 97.6%.
3. **Per-engine, PE wins half the time.** Across all 260 engines, PE is better on
   **51%** of them. The pooled improvement of 0.9% is not a small consistent gain;
   it is the residue of a symmetric distribution of wins and losses.

## The archived E3 result was fold 0

The E3 report (`paper/E3_RESULT.md`) quotes A = 12.676, PE = 11.768, Δ = −0.908.
Those are the **fold-0** numbers. Fold 0 is also the easiest fold for arm A
(12.679 against 13.5–15.1 elsewhere), so the absolute levels in E3 are not
representative either. The relative claim (−7.7%) and the absolute levels were
both properties of one holdout draw.

This does not retroactively invalidate the E3 *machinery*: the runs, the equality
checks, the seed pairing and the bootstrap are all correct, and the E3 amendment
already withdrew the non-inferiority verdict on separate grounds. What E10 adds is
that the point estimate itself was not a property of the design change.

## Neither lifespan nor degradation rate explains the sign

Exploratory stratification of the per-engine difference
(`scripts/dsh_e10_strata.py`, `results/dsh/e10/E10_strata.json`). **Not
pre-registered; it cannot upgrade the verdict and is reported only as a mechanism
probe.**

| stratum | n | mean Δ per engine | 95% CI | PE wins |
|---|---:|---:|---|---:|
| short lifespan (tertile 1) | 92 | −0.223 | [−1.48, +1.03] | 55% |
| mid lifespan | 81 | −0.576 | [−1.80, +0.65] | 53% |
| long lifespan (tertile 3) | 87 | +0.418 | [−0.70, +1.53] | 46% |

`corr(Δ, lifespan) = +0.015`, Spearman +0.054. There is no conditional story
here: the encoding does not help long-lived engines and hurt short-lived ones, or
vice versa. The difference is noise around zero at every stratum.

## What this changes in the manuscript

| claim | before E10 | after E10 |
|---|---|---|
| C14: bounded encoding is non-inferior, directionally better | partial (margin not met) | **withdrawn**: the direction itself does not replicate |
| §4.3.2 "tends to PE" | allowed | **not allowed**; the pooled effect is 0.9% with an interval crossing zero |
| §4.3.3 "same effect, two verdicts across subsets" | FD001 vs FD002 | **replaced** by the stronger within-subset result: five folds of the *same* subset give −1.02 to +1.70 |
| §4.2 resolution | MDE from engine count | **plus an empirical calibration**: at 260 engines, a 0.9 effect is not resolvable and a 2.7 effect is (E10b) |

The result strengthens rather than weakens the paper's thesis. A design change
that a single 80/20 split reports as a 7.7% improvement is, at 260 engines, a 0.9%
non-result — and the same 52-engine split that produced it also produced three
seeds that all agreed with it.

## Boundaries

* FD002 only; FD003/FD004 remain sealed; no official test file read.
* Pooled MSE over 260 engines × 3 seeds. Pooling assumes the effect is not
  concentrated in a few engines; the per-fold table and the 51% win rate show this
  assumption is not what fails here.
* The bootstrap resamples engines within folds, so it is conditional on these five
  folds. Treated as a random effect instead, the fold-level interval is wider
  (−0.15 ± 2.78 × 1.10/√5 ≈ [−1.52, +1.22]). Both cross zero; the conclusion does
  not depend on the choice, and both are reported.
