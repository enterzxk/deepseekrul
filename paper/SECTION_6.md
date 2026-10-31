# 6 Limitations

## 6.1 One dataset family, four subsets of one simulator

Every number in this paper comes from C-MAPSS. Both the development analyses and the
test analyses use only this family, and all four subsets were used in both roles.

**Training engines.** FD001–FD004 training trajectories, through the identical
pre-registered 5-fold design (§4.3.3b; pre-registrations `59bb5d9b`, `3f0142f9` and
Amendment 2). FD003/FD004 train files were unsealed for this purpose under
`paper/FD0034_PROTOCOL_CHANGE.md`.

**Official test files.** Read **once per subset, after freezing**, never for selection,
tuning, calibration or any decision. `test_FD002`/`RUL_FD002` were read on 2026-10-31
under the E15 one-shot protocol after every §4.2.4 model was frozen
(`paper/TEST_SET_PROTOCOL_CHANGE.md`, authorised before the read; feature equality 0.0,
prediction replay 0.084 RMSE). The remaining three subsets were read under the unified
E21 protocol (`paper/E21_TEST_ALL_PREREGISTRATION.md`, sha `1579e0e0`) after all
training was complete: feature verification passed on all four subsets (max
difference 0.0) before any test file was read, the FD003/FD004 test and RUL files were
unsealed at that point with sha256 recorded and mtimes untouched
(`results/dsh/e21/unseal_hashes.sha256`), and exclusions matched the registered counts
(§6.10). The E21 FD002 evaluation supersedes the E15 numbers with a strictly larger
model set (15 models including fold 0 vs 12); both are archived.

Consequences: nothing here generalises to other turbofan datasets, to bearings,
or to systems where the failure criterion is physical rather than simulated. The
C-MAPSS "failure" is a defined point on a simulated degradation trajectory
(Saxena et al., 2008); the lifespan scale `T` used in section 4.4 inherits that
convention. The four subsets are four operating regimes of one simulator, so
"across C-MAPSS" means "across regimes of this simulator", nothing more.

## 6.2 Resolution is bounded by engine count, and FD001 is small

The minimum detectable effect is a property of the held-out engine count, not of
the method. FD002 has 260 engines and reaches MDE ≈ 1.54 RMSE at 260 held-out
engines (section 4.2). FD001 has 100 engines; a 5-fold split exposes 20 per fold,
and the same contrast that is significant on FD002 (BE − A = +2.69) is
directionally identical but **below the FD001 MDE of 4.84** on FD001
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
* It does not convert a null result into a positive one. Where an interval crosses
  zero, the text says so, including in the abstract. FD003's reversal (§4.3.3b) and
  FD002's test-side equivalence (§4.3.6) are reported in the same breath as the
  significant pooled effects.

## 6.10 Exclusion of short test trajectories

The unified E21 protocol (`paper/E21_TEST_ALL_PREREGISTRATION.md`) skips any test
trajectory shorter than the 50-cycle window — a property of the protocol, not an
implementation choice: a 50-cycle window cannot be formed from 21 cycles. The counts
registered before unsealing are the counts observed:

| subset | total | excluded | excluded length | excluded true-RUL mean | retained true-RUL mean |
|---|---:|---:|---|---:|---:|
| FD001 | 100 | 7 | 31–49 | 119.0 | 72.2 |
| FD002 | 259 | 24 | 21–49 | 138.4 | 75.3 |
| FD003 | 100 | 3 | 38–47 | 136.3 | 73.4 |
| FD004 | 248 | 20 | 19–48 | 157.2 | 80.4 |

(`results/dsh/e21/test_all.json` `exclusions`; FD002's total-life comparison, 175.5 vs
216.2, is in `results/dsh/e19/test_exclusion.json`.)

In every subset the excluded engines are exactly the longest-remaining-life cases — the
hardest ones — so every test RMSE in §4.3.6 is optimistic by an order that is bounded,
not re-measured. For FD002 the bound is concrete: a constant-125 predictor scores 28.72
on the excluded 24, and pooling that with the E21 FD002 ensemble 23.386 gives 23.93,
**+0.54 (+2.3%)**. The same construction on the other subsets moves the score by a
similar small order. These are illustrative bounds, not second evaluations; the reported
test results stay frozen at the retained-engine counts, and any comparison of test-set
RMSE across papers must state how short trajectories were handled.

