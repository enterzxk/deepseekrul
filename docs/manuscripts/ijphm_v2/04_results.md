# 4. Results

<!-- draft note: every number is bound to CLAIM_LEDGER.md (C-ids are stripped at typesetting)
     and asserted by scripts/paper_evidence.py (17/17) or freeze_audit.py (50/50).
     Data source: docs/tables/mcd80_cells_seeded.csv (seeded, authoritative). -->

## 4.1 The efficiency gain is reproducible, and it is not bought with coverage

At the nominal level ($\delta=0.1$), CQR lowers the Winkler score relative to the constant-width
conformal baseline in **75 of 80 cells** and in **20 of 20 folds**; the cell-level mean change is
**−10.22%** (median −11.30%). <!-- C1 -->

The two arms do not trade coverage for sharpness. Averaged over the 80 cells, realised coverage is
**0.8952 for CQR and 0.8994 for the constant-width arm**, a difference of **−0.42 pp**; per subset the
differences are −0.08, −0.43, −1.36 and +0.19 pp. <!-- C3 --> The match is a property of the subset
means, not of every cell: the largest per-cell coverage difference is **8.84 pp**. We state this
explicitly because a reader who checks individual cells will find it.

We also verified that the comparison is not an artefact of a single training run: the four seeds share
the engine partition, and 16 of 20 folds have all four seeds agreeing in sign, with a within-fold seed
spread whose median is 6.13 pp and whose maximum is 36.63 pp (46.82% of the descriptive variance is
within-fold). <!-- C13 -->

![Per-cell relative Winkler change at the nominal level: CQR versus constant width (left) and the MC-Dropout scale versus the same-centre constant-width control (right), over the 80 cells. Negative is better.](docs/figures/ijphm_v2/fig1_effect_distribution.pdf){#fig:effect width=100%}

## 4.2 Marginal coverage does not imply engine-level reliability

The per-engine coverage distribution is heavy-tailed downwards. Across the 80 cells, the **lowest
single-engine coverage is 0.4214 (FD001), 0.2375 (FD002), 0.4933 (FD003) and 0.1217 (FD004)**.
<!-- C4 --> **73 of the 80 cells** contain at least one engine below 0.80, and in **18 of 20 folds** the
seed-averaged per-engine coverage falls below 0.80 for at least one engine. <!-- C4 --> <!-- C16 -->

The extremes are not isolated failures of one cell. Taking, for each cell, the minimum over its test
engines and then the median of those minima across the 20 cells of a subset gives **0.6950 (FD001),
0.5438 (FD002), 0.7110 (FD003) and 0.5181 (FD004)**; the typical cell's worst engine is itself well
below nominal. All per-engine numbers in this section come from the 80-cell matrix and are recomputed
under the same seed-42 calibration/test split for every cell, so they are directly comparable across
seeds. The 20-fold count in the previous paragraph is the seed-averaged version of the same 80-cell
computation, not a separate single-seed run. <!-- C17 -->

![Per-engine coverage of the frozen CQR across the 80 cells, by subset. Dashed line: nominal 0.90.](docs/figures/ijphm_v2/fig3_engine_coverage.pdf){#fig:engine width=70%}

This is not training noise. Treating the four seeds as replicates of the same engine under the same
partition, the intraclass correlation of per-engine coverage is **ICC(1,1) = 0.780**, the mean pairwise
Spearman correlation across seeds is **+0.762** (99% of 120 seed pairs positive), the bottom-quartile
engine sets overlap with a mean Jaccard of **0.599**, and the within-engine standard deviation across
seeds is **0.032** (population SD; the sample SD with 4 seeds is 0.0372). <!-- C5 --> An engine that is
under-covered by one training run tends to be under-covered by the others.

We stress what this does and does not show. It shows that, under this frozen partition and training
configuration, the failure is a repeatable property of the engine. It does not show that the failure
is intrinsic to the engine, that it transfers across partitions, or that it is identifiable at
deployment time.

## 4.3 The epistemic scale helps or hurts depending on operating-condition structure

Replacing the learned half-width with a per-sample MC-Dropout scale changes the Winkler score by
**+9.49% on average** over the 80 cells, with a **median of −0.97%**. <!-- C6 --> The mean and the
median point in opposite directions, so neither "better" nor "worse" is an accurate one-line summary.
Two facts constrain the reading: the median is the same order as the sampling noise (median absolute
change 0.63 pp), and the mean is dominated by a small number of catastrophic cells.

The split is by subset (same-centre control, so only the scale differs):

| Subset | Cells better | Mean change | Median change |
|---|---|---|---|
| FD001 (single OC) | 8/20 | +18.77% | +10.65% |
| FD002 (multi OC) | **19/20** | **−7.09%** | −7.29% |
| FD003 (single OC) | **1/20** | **+28.06%** | +24.99% |
| FD004 (multi OC) | 15/20 | −1.78% | −2.58% |

<!-- C7 --> <!-- C22 --> FD002 and FD003 are both far outside the noise floor (p90 = 3.20 pp); FD004's −1.78% is
not, and we do not treat it as evidence of an improvement.

Aggregating to the fold level makes the structure sharper. Of the ten folds from multi-operating-
condition subsets, **9 are improved** (mean −4.44%); of the ten folds from single-condition subsets,
**2 are improved** (mean **+23.41%**); Fisher's exact two-sided test on this 2×2 table gives
**p = 0.0055**. <!-- C8 --> The test is descriptive: the 20 folds share training data and are not
independent draws, so the p-value quantifies the strength of the separation in this design, not a
population-level effect.

![Fold-level relative Winkler change of the dropout scale against the same-centre constant-width control, grouped by operating-condition structure. Horizontal bars are means; each dot is one fold.](docs/figures/ijphm_v2/fig2_fold_separation.pdf){#fig:fold width=70%}

## 4.4 A natural calibration-side diagnostic does not survive its own pre-registered test

Before looking at any of the above as a mechanism, we registered a hypothesis: that the scale is
useful exactly when it ranks calibration engines by difficulty, measured by
$\rho_{cal}=\mathrm{Spearman}\{(\bar s_i,\ \bar a_i)\}_{i\in\text{calib}}$, the rank correlation between
an engine's mean dropout scale and its mean absolute centre error, computed on calibration engines
only. Four gates were fixed in advance; three fail. <!-- C9 -->

| Gate | Statistic | Threshold | Result |
|---|---|---|---|
| G-A | $\mathrm{Spearman}(\rho_{cal},\ r_{scale})$ over 20 folds | ≤ −0.35, CI upper < 0 | **−0.039** [−0.431, +0.317] — fail |
| G-B | $\rho_{cal}$ multi-OC minus single-OC | ≥ 0.20, ≥4/5 folds | +0.216, 4/5 — pass |
| G-C | $\mathrm{Spearman}(\text{scale shift},\ \text{coverage gap})$ | ≤ −0.30, CI upper < 0 | −0.221 [−0.712, +0.300] — fail |
| G-D | go/no-go rule at $\tau=0.5$ (fixed) | mean ≤ −2%, FD003 ≤ +2% | +2.79%, FD003 +13.77% — fail |

<!-- C19 -->

The failure has a measurable cause. Bootstrapping engines *within* each cell (post-hoc; not part of the
pre-registered gates), the sampling standard deviation of $\rho_{cal}$ is **0.490 (FD001) and 0.414
(FD003)**, where only 6 engines are available for calibration, against 0.249 (FD002) and 0.171 (FD004)
with 15–16. The corresponding per-cell 95% half-widths are **0.898, 0.479, 0.747 and 0.324**.
<!-- C10 --> <!-- C20 --> With six calibration engines the diagnostic's own uncertainty spans most of its range, so
a rule built on it cannot be reliably calibrated at this sample size — which is what G-A and G-D show.

We read this as an estimation result, not an impossibility result. It does not show that no
calibration-side diagnostic can work; it shows that this one, at this calibration size, cannot.
The distinction matters for practice: the usual way to justify a conditional scale is to check that it
correlates with residuals on the calibration set, and that check is uninformative here.

## 4.5 The other candidate scales are not better

Two further comparisons place the dropout scale in context.

**Against the learned conditional width.** On the same 80 cells, the MC-Dropout arm is worse than CQR in
**72 of 80 cells and in 20 of 20 folds**, with a mean change of **+21.42%** and a median of +15.31%.
<!-- C2 --> On this comparison the learned half-width is better; we do not claim that it captures
"more" of any particular quantity.

**Against the prior-uncertainty scale.** Permuting the prior standard deviation $\sigma_T$ across windows
— which destroys any association between the scale and the sample while keeping its marginal
distribution — changes the Winkler score by **+0.246% on average**, with **11 of 20 folds** becoming
worse and FD004 improving by 3.02%. <!-- C12 --> The permutation is within noise, so we do not treat
$\sigma_T$ as a useful conditional scale.

**Cross-domain.** On six run-to-full-failure FEMTO bearings [@nectoux2012], with the conformal
quantile calibrated on 300 held-out windows per bearing and evaluated on 300 disjoint test windows,
CQR improves on the constant-width baseline in **6 of 6** bearings with a mean change of **−16.30%**.
<!-- C11 --> For the record, an earlier **internal** evaluation of the same dumps calibrated and
evaluated on the same windows and scored each frontier point at its own $\alpha$; it gave 5/6 and
−12.21%. It is noted only to explain the difference from earlier project records and is not used as
evidence here. The split protocol above supports the efficiency result of Section 4.1 only; it is too
small a sample to say anything about the scale or the per-engine findings.
