# 6 Limitations

## 6.1 One dataset family, two subsets

Every number in this paper comes from C-MAPSS training trajectories, with one exception
that is stated here and in the supplement.

**Training engines.** FD001 and FD002 only. **FD003 and FD004 remain sealed** and were
never read by any script.

**Official test files.** `test_FD002` and `RUL_FD002` were read **once**, on 2026-10-31,
after every model in §4.2.4 had been trained, frozen and archived
(`paper/TEST_SET_PROTOCOL_CHANGE.md`, authorised before the read; `paper/E15_TEST_RESULT.md`
for the protocol and results). They were not used for model selection, tuning, calibration
or any decision, and the pipeline was verified against the training path before the read
(feature equality 0.0, prediction replay 0.084 RMSE). `test_FD001`, `test_FD003` and
`test_FD004` were never used for any reported metric; an earlier exploratory script read
`test_FD001` and `test_FD002` once to count engines and produced no prediction, which is
disclosed in the supplement. FD003 and FD004 test data remain unread.

Consequences: nothing here generalises to other turbofan datasets, to bearings,
or to systems where the failure criterion is physical rather than simulated. The
C-MAPSS "failure" is a defined point on a simulated degradation trajectory
(Saxena et al., 2008); the lifespan scale `T` used in section 4.4 inherits that
convention.

## 6.2 Resolution is bounded by engine count, and FD001 is small

The minimum detectable effect is a property of the held-out engine count, not of
the method. FD002 has 260 engines and reaches MDE ≈ 1.54 RMSE at 260 held-out
engines (section 4.2). FD001 has 100 engines; a 5-fold split exposes 20 per fold,
and the same contrast that is significant on FD002 (BE − A = +2.69) is
directionally identical but **below the FD001 MDE of 2.38** on FD001
(+1.67, interval crossing zero). Section 4.3.3 reports this as a boundary of the
measurement, not as a contradiction between subsets.

The paper therefore cannot claim that any effect smaller than roughly one RMSE is
absent. It can only claim that such an effect is not resolvable here.

## 6.3 Probe and neural settings are not on the same scale

The linear-probe experiments (section 4.3.1) report absolute RMSE values near 21,
while the frozen neural pipeline reports values near 12.7. The two use different
preprocessing (the probe does not apply `OperatingConditionScaler`), different
model classes and different training budgets. **Only the within-setting contrasts
are interpretable**; the absolute levels must not be compared across the two.
Every table in this paper keeps the settings separated and labels them.

## 6.4 Capacity is not exactly matched in the design-change comparison

The bounded sinusoidal encoding adds a `Linear(8, 64)` projection in place of
`Linear(1, 64)`: **+448 parameters (+1.07%)**. The paper's claim about it is
*non-inferiority*, not superiority, and the claim was in any case not established
once the bootstrap was converged (`paper/E3_AMENDMENT_1.md`). A capacity control
was nevertheless run in a stronger form than parameter matching: the scrambled-age
variant (section 4.3.2) keeps the same 8→64 projection and the same parameter
count while destroying the alignment between the encoding and the engine's own
age. It is **+3.51 RMSE [+2.21, +4.86] worse** than the aligned encoding and
statistically indistinguishable from removing time altogether (−0.08, interval
crossing zero). This rules out "the gain is the extra 448 parameters", but it does
not rule out that a *different* 448-parameter head could also fail — the control
tests this head, not all heads.

## 6.5 Pooling folds assumes a homogeneous effect

The 5-fold estimator in section 4.3.3 pools the out-of-fold predictions of all
260 engines and reports one interval. This is valid if the effect is roughly
homogeneous across folds; if the improvement were concentrated in the engines of
one or two folds, pooling would dilute it and the interval would remain wide.
Per-fold estimates are reported alongside the pooled one precisely so that a
reader can see whether this happened. The pre-registration
(`paper/E10_CV_PREREGISTRATION.md` §5) states this limitation before the numbers
were seen.

A second, smaller issue: each fold trains a different model on a different
208-engine subset, so the pooled estimate carries a between-fold component that a
single-split standard error does not contain. The projected interval in the
pre-registration is therefore optimistic by construction.

## 6.6 The oracle bound is an upper bound, not a target

The +8.23 RMSE attributable to lifespan-scale normalisation (section 4.4.1)
requires the true lifespan, which no deployment has. It quantifies the value of
the information, not the value of any method. After three attempts — a
prefix-extrapolation estimator, a rate-stratified variant, and a monotone
health-index rebuild — the best deployable estimator recovers +0.08 with an
interval crossing zero. The paper reports the bound and the failure together;
quoting the bound alone would be misleading.

## 6.7 Two self-corrections are part of the record

The paper contains three numbers that were reported and later withdrawn:

| number | why it was wrong | where the correction is recorded |
|---|---|---|
| "the estimator recovers 38% of the oracle gain" | 100% fallback to a constant slope **and** a validation-side estimator fed the true lifespan | section 4.4.4; `docs/director/REVIEW_20261022_R3.md` §J |
| "PE is non-inferior to the cycle term" | a 4 000-resample bound (+0.291) that a converged 100 000-resample bound (+0.317) does not reproduce | `paper/E3_AMENDMENT_1.md` |
| "21 paired comparisons, none survives Bonferroni" | no per-engine artefact was ever stored, so the table cannot be re-derived | `paper/E6_STATS_AUDIT.md`; `paper/CLAIMS.md` C3 |

They are retained rather than deleted because each one is an instance of the
paper's own thesis: a comparison whose resolution was never checked. A reader who
wants to reproduce a *failure* has the same artefacts as a reader who wants to
reproduce a success.

## 6.8 Scope of the reproducibility claim

Section 7 shows a from-scratch rerun that reproduces the archived metrics to
**0.0000 RMSE** and a weight replay that matches to **7.2e-6** in prediction
space. That is determinism on one machine, one CUDA version and one PyTorch
version (A40, CUDA 12.1, torch 2.5.1+cu121, Python 3.9.25, host YNDXTEA0034). It
is **not** a claim of cross-platform portability: different BLAS libraries, GPU
architectures or PyTorch builds can change the last digits, and the paper does not
test that.

## 6.9 What the paper deliberately does not do

* It does not claim state of the art, a new architecture, or a "first" of any kind.
  The literature check (`paper/E1_LITERATURE.md`) found prior work on leakage,
  seed variance, benchmarking variance and C-MAPSS reporting inconsistency; the
  contribution is positioned against them, not above them.
* It does not report FD003/FD004 or official-test performance.
* It does not convert a null result into a positive one. Where an interval crosses
  zero, the text says so, including in the abstract.

## 6.10 Exclusion of short test trajectories

Of the 259 engines in `test_FD002`, **24 were excluded** by `scripts/dsh_test_verify.py:155`,
which skips any trajectory shorter than the 50-cycle window (their observed lengths are
21–49 cycles). This is a property of the protocol, not an implementation choice: a 50-cycle
window cannot be formed from 21 cycles.

Those 24 are not a random subset. Their true RUL averages **138.4** against **75.3** for the
235 retained engines, and their total life averages 175.5 against 216.2. The evaluation
therefore loses exactly the cases with the longest remaining life, which are the hardest.

The bias is bounded rather than re-measured: a constant-125 predictor scores RMSE 28.72 on the
excluded set, and pooling that with the retained 23.96 gives 24.44, **+0.48 (+2.0%)**. That is
an illustrative bound, not a second evaluation; the one-shot test result stays frozen at 23.96
on 235 engines, and it is optimistic by roughly this order.

Any comparison of test-set RMSE across papers must state how short trajectories were handled.

