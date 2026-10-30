# E10b — Result: the time-channel cost survives cross-validation, at less than half its single-split size

Run: 2026-10-30 23:00–23:34 local, A40, 12 runs (BE × 3 seeds × folds 1–4) plus the
3 archived fold-0 runs; the A arm is taken from E10 (same code, seeds, folds).
Pre-registration: `paper/E10B_BE_PREREGISTRATION.md`, sha `532a1d5d…`.
Analysis: `scripts/dsh_e10b_analyze.py`, `results/dsh/e10/E10b_be_cv.json`.

## Verdict: **REPRODUCED** — and the single split overstated it 2.4×

| | pooled, 260 engines × 3 seeds |
|---|---:|
| RMSE, arm A (`min(c/250,1)`) | 13.983 |
| RMSE, arm BE (no time channel) | 15.116 |
| **ΔRMSE (BE − A), positive = removing time hurts** | **+1.133 (+8.1%)** |
| 95% CI (fold-stratified engine bootstrap, 100 000 × 5 RNG) | **[+0.245, +2.023]** |
| Monte-Carlo error on the lower bound | 0.004 |

The interval clears zero. **The time channel is load-bearing at 260 engines**: removing
it costs 8.1% RMSE.

## But the single split reported nearly three times as much

| fold | 0 (archived) | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| Δ (BE − A) | **+2.677** | **−0.306** | +1.088 | +1.571 | +0.566 |

* Pooled: **+1.133**, between-fold SD **1.11**.
* The archived E3 single-split value was **+2.693** — 2.4× the pooled estimate, and
  larger than every one of the other four folds.
* Fold 1 has the **opposite sign**: on those 52 engines, removing the time channel
  *improves* RMSE by 0.31.

Three seeds per fold agree with their own fold (fold 0: +2.71/+2.59/+2.73;
fold 3: +1.46/+1.84/+1.65), so this is not seed noise. It is the same phenomenon
documented in E10: within-fold agreement is not evidence of between-fold stability.

## The two contrasts together calibrate the design

This is the pair that makes the paper's argument quantitative. Same dataset, same
protocol, same 260 engines, same estimator, same interval recipe — the only thing
that differs is the size of the effect.

| contrast | effect (single split) | effect (5 folds) | 95% CI (5 folds) | verdict |
|---|---:|---:|---|---|
| **BE − A** (remove the time channel) | +2.693 | **+1.133 (+8.1%)** | [+0.245, +2.023] | **reproduced** |
| **PE − A** (bounded vs linear time) | −0.908 | **−0.126 (−0.9%)** | [−1.073, +0.795] | **not reproduced** |

Reading the two rows together:

1. **An 8% effect is resolvable at 260 engines; a 0.9% effect is not.** The
   boundary sits between them, and both sides are now measured rather than
   assumed. Expressed against the pooled baseline of 13.98, the smallest effect
   this design resolves is roughly 1.1–1.3 RMSE (**8–9%**), consistent with the
   independent MDE calculation in §4.2 (1.16–1.46 at 260 engines).
2. **A single 80/20 split is not a noisy estimate of the truth — it is a biased
   one.** Both contrasts were overstated on fold 0: 2.4× for BE, 7× for PE. The
   direction of the bias is the same in both cases, which suggests fold 0 is
   systematically "easy" for the time channel (arm A scores 12.68 there against
   13.5–15.1 elsewhere). A split that is easy for one arm inflates any contrast
   involving it.
3. **Report the fold-level estimates, not only the pooled one.** The pooled
   number +1.13 with CI [+0.25, +2.02] is honest but incomplete; the per-fold
   spread (−0.31 to +2.68) tells the reader that a single-split replication could
   plausibly land anywhere in that range.

## What the manuscript may now claim

* **Supported (new, C25).** Removing the time index from the frozen pipeline
  degrades RMSE by **8.1% [+1.8%, +14.5%]** measured as a ratio, over 260 held-out
  FD002 engines. The time index is not decorative: it is a load-bearing input.
* **Supported.** The same design cannot resolve an effect of 0.9%: the bounded
  encoding's advantage disappears (C14 withdrawn).
* **Supported.** A single 80/20 split overstated both effects, by 2.4× and 7×
  respectively, with three mutually agreeing seeds in each case.
* **Not supported.** That the single-split numbers are wrong in an unpredictable
  way. In both cases they were too large, and the fold that produced them is the
  easiest fold for the baseline arm — but two contrasts is not enough to establish
  a directional rule, and the paper says so.

## Boundaries

* FD002 only. FD003/FD004 remain sealed; no official test file is read.
* The A arm is the E10 arm, not a retrain, so the pairing is exact by construction.
* Pooled MSE over 260 engines × 3 seeds; bootstrap resamples engines within folds
  and is therefore conditional on these five folds. Treated as a random effect,
  the fold-level interval is wider (1.12 ± 2.78 × 1.11/√5 ≈ [−0.26, +2.50]) and
  would *not* clear zero. **Both are reported.** The pre-registered rule used the
  bootstrap, so the verdict stands as REPRODUCED, but the random-effect reading is
  disclosed in §4.3 and §6 as a boundary: five folds is a small number of folds,
  and a design with five folds cannot distinguish an 8% effect from a 4% one at
  the fold level.
