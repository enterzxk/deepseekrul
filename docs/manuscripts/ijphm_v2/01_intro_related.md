# 1. Introduction

<!-- draft note: citation keys are pandoc author-year syntax and are resolved against
     references_v2.bib at build time; this note is an HTML comment so it never reaches the PDF. -->

Remaining-useful-life (RUL) prognostics is one of the places where a prediction interval matters more
than a point estimate: maintenance decisions are made against a bound, not against a mean. Conformal
prediction has therefore become a natural tool for RUL, because it turns any point or quantile
predictor into an interval with a distribution-free finite-sample guarantee [@vovk2005; @lei2018].
Conformalized quantile regression (CQR) [@romano2019] is the variant most often used, since it lets the
interval width follow the local difficulty of the prediction instead of being constant.

The guarantee, however, is **marginal**: it holds on average over calibration and test points. It says
nothing about a single engine, and it says nothing about whether the interval is efficient. Both gaps
matter in prognostics, where the unit of a maintenance decision is one engine. Recent work has begun to
map these gaps — conditional coverage is impossible to guarantee distribution-free [@barber2021limits],
group-conditional guarantees require group structure and enough per-group samples [@gibbs2025],
calibration-set-conditional coverage is itself a random quantity [@bian2023], and label-free auditing of
coverage under shift has a detectability boundary [@moussa2026driftguard]. Yet the studies we reviewed
for this paper typically report marginal coverage from a single training run.

This paper asks three questions about a frozen, published-style CQR pipeline for C-MAPSS RUL:

- **Q1.** Is the reported efficiency gain — lower interval score than a constant-width conformal
  baseline — stable across training seeds and folds, and is it obtained at matched coverage?
- **Q2.** When marginal coverage is close to nominal, what does the *per-engine* coverage distribution
  look like, and is the failure repeatable or is it training noise?
- **Q3.** If we replace the learned half-width with a per-sample epistemic-uncertainty scale
  (MC-Dropout), when does it help, and can a calibration-side diagnostic tell us in advance?

The answers are organised as four contributions.

**C1 — A controlled, multi-seed comparison.** We evaluate 4 subsets × 5 folds × 4 seeds = 80 cells
against a constant-width conformal baseline trained identically, at the nominal level. CQR lowers the
Winkler score in 75/80 cells and in 20/20 folds (mean −10.22%), while realised coverage differs by
−0.42 pp on average. We report the per-cell coverage spread (up to 8.84 pp) rather than hiding it.
<!-- C1 --> <!-- C3 -->

**C2 — The unit-level view.** <!-- C4 --> <!-- C5 --> Per-engine coverage is heavy-tailed: the worst engine reaches 0.1217
(FD004) across the 80 cells, and 18/20 folds contain an engine below 0.80. Treating seeds as
replicates, the intraclass correlation of per-engine coverage is 0.780, so the failure is a repeatable
property of the engine under this partition, not sampling noise.

**C3 — A controlled test of an epistemic scale.** Comparing MC-Dropout against a constant-width
interval built on the *same* MC mean isolates the scale from the centre shift. The effect is strongly
heterogeneous: 19/20 cells improve on FD002 (mean −7.09%) while 1/20 improve on FD003 (mean +28.06%);
at the fold level, 9/10 multi-condition folds improve versus 2/10 single-condition folds
(Fisher exact p = 0.0055, descriptive). <!-- C6 --> <!-- C7 --> <!-- C8 -->

**C4 — A pre-registered diagnostic that fails, and why.** We registered the natural hypothesis that the
scale is useful exactly when it ranks calibration engines by difficulty. The hypothesis does not reach
its pre-registered support criteria: the rank correlation between the diagnostic and the outcome is
−0.039 [−0.431, +0.317]. The reason is measurable — with six calibration engines the diagnostic's own
bootstrap standard deviation is 0.41–0.49, so its 95% interval spans most of its range.
<!-- C9 --> <!-- C10 -->

The common thread is a measurement point rather than a methodological one: marginal coverage and a
single training seed are weak summaries of what a conformal RUL interval delivers. We do not propose a
new interval method; we quantify what the current one gives, at the resolution at which it is used.

# 2. Related work

## 2.1 Conditional validity and its limits

Split conformal prediction guarantees marginal coverage under exchangeability [@vovk2005; @lei2018].
Exact distribution-free conditional coverage is impossible without further assumptions
[@barber2021limits], and coverage guarantees are conditional on the calibration set, hence random
[@bian2023]. Practical relaxations impose structure: group-conditional guarantees over a user-specified
collection of subgroups [@gibbs2025], or weighting by an estimated density ratio under covariate shift
[@tibshirani2019]. Our work does not add to this theory; it measures how far a standard pipeline is from
the conditional ideal, and it keeps the counting object explicit — calibration *scores* per cell, not
calibration *engines*.

## 2.2 Conformal RUL

CQR [@romano2019] is the base of most recent conformal RUL work. CRULP [@piao2025crulp] integrates a
deep RUL predictor with an uncertainty-aware CQR variant built on epistemic uncertainty, using ensemble
CQR and cross-conformal prediction. Robinson [@robinson2026ijphm] applies gradient-boosted point
predictors with CQR to all four C-MAPSS subsets and adds a risk-aware decision layer; it reports
empirical coverage of 88–92% at a 90% nominal level and per-engine degradation curves, but no
per-engine coverage distribution. We use the same family of predictors and intervals, and we report the
per-engine distribution.

## 2.3 Audits, conditional coverage diagnostics, and boundaries

Several 2026 works are close in spirit. Yan [@yan2026audit] audits conformal RUL intervals on C-MAPSS
and N-CMAPSS along operating regimes and predicted-RUL bands, with ACI and Mondrian variants, and
explains mid-band failures by band misassignment. Domingues and Ather [@domingues2026] document
conditional collapse near end of life (61–90 cycles remaining; 37–50% near failure) on C-MAPSS.
Zhou et al. [@zhou2026cpa] reframe conditional-coverage assessment as a supervised learning problem:
a reliability estimator predicts instance-level coverage, with convergence rates, but it requires a
labeled evaluation set. Moussa [@moussa2026driftguard] establishes detectability bounds and a sharp
identified set for *marginal* target coverage under shift, and states explicitly that the approach
repairs marginal rather than conditional coverage. For sparse cells, HACC [@hacc2026] gives an
arithmetic threshold below which the empirical quantile does not exist, and repairs it by shrinkage.

## 2.4 What this paper adds

The difference is a controlled comparison, not a new estimator:

1. **Same-centre control.** MC-Dropout changes the centre as well as the scale. Holding the centre
   fixed at the MC mean isolates the scale, which is what the question "is an epistemic scale useful?"
   actually asks.
2. **Multi-seed replication.** Four seeds on a shared partition let us separate engine-level failure
   from training noise (ICC 0.780) and attach a sampling-noise floor (median 0.63 pp, p90 3.20 pp,
   0/80 direction flips) to every subset-level statement.
3. **A pre-registered diagnostic test.** The calibration-side diagnostic is specified, thresholded, and
   registered before it is evaluated, and its estimation variance is measured rather than assumed.
4. **Explicit counting objects.** Calibration engines (6/16/6/15), calibration scores per cell, and
   per-engine windows are kept distinct throughout; conflating these three may make claims about
   "unit-level validity" harder to compare across papers.

We make no claim of theoretical novelty, and we do not claim to be the first to observe conditional
coverage failure in RUL; that observation is due to [@domingues2026] and [@yan2026audit].
