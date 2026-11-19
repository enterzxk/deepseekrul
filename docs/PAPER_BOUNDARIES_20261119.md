# Paper boundaries — C-MAPSS RUL (Paper B)

**Date:** 2026-11-19 · **Scope:** LTAG + RHSE-TSH · **Status:** claim freeze before drafting

> **2026-11-20 correction — read before using any R5/R6 number.**
> The pre-fix `tsh_rul_eval.py` built the last-value arm from the **clean** counterfactual
> `x[fb-1]` instead of the observed `x_f[fb-1]`. Every R5/R6 number in this file is therefore
> **oracle-leaked and invalid**; the sentence “hold wins late drift” is **withdrawn**.
> Post-fix FD001 s101: bias R3 0.862 / R5 0.059; drift R3 0.861 / R5 0.313; stuck R3 0.895 /
> R5 −0.012. The Phase 1c queue (60 RUL + 2 EOL, 3 seeds) is producing the corrected R5 table;
> until it lands, cite **R3 only** and mark R5 as pending. R3 numbers are unaffected by the leak.
> The Gate-1 “drift R4 ≥ R5 + 0.10” condition must be re-evaluated with corrected R5, not with
> the 0.984 figure in §1.4/§5.

> This file records what the current evidence can and cannot support. It runs no experiments and
> introduces no measurements: every number is quoted from `PHASE1_FINAL_20261119.md`,
> `PHASE1B_HC_NEGATIVE_20261119.md`, `PHASE1_TSH_PILOT_20261119.md`, `LTAG_TEST_20261118.md` or
> `PHASE0_BUGFIX_20261119.md`, with the section named. Where two sources disagree, §5 records the
> conflict instead of choosing silently.
>
> **How to use it:** §1 is claim text that can be lifted into the manuscript; §2 is the set of
> claims the paper must not make (rebuttal shield); §3 is the Limitations subsection; §4 is the
> narrative.

## 0. Metric definitions — read before quoting any number

### 0.1 RMSE is not one number

| Estimator | Definition | Used by |
|---|---|---|
| Engine-mean RMSE | mean over engines of per-engine RMSE, each engine weighted 1/E | LTAG validation (`learned_gate_global.py::engine_mean_rmse`), 5 cuts/engine |
| Pooled RMSE | `sqrt(mean squared error)` over all rows | RHSE-TSH R0–R7 and clean cost (`tsh_rul_eval.py`), 5 cuts/engine pooled over (engine, cut) rows |
| Pooled over test engines | `sqrt(mean squared error)` over engines | LTAG official test (`cata_test_eval.py`); one window per engine, so it equals engine-mean there |

Consequences:

- Pooled ≥ engine-mean when engines have equal window counts (Jensen). They are different
  estimators: never subtract one from the other, and never put them in one column.
- **RHSE-TSH numbers are pooled over (engine, cut) rows**, not engine-mean. R0 = 12.647 (FD001)
  is a pooled number over five cuts per engine.
- **LTAG validation numbers are engine-mean** (11.006 for FD001). **LTAG test numbers are pooled
  over engines** (12.995); with one test window per engine the two coincide numerically.
- RHSE-TSH R0 and LTAG K=1 are different pipelines and must not be assumed to share a baseline:
  FD001 is 12.647 (pooled, RHSE) vs 11.696 (engine-mean, LTAG). The Jensen term here is
  second-order (`pooled² − engine-mean² = Var_e(RMSE_e)`, zero when per-engine RMSEs are equal),
  so it cannot explain a 0.95-cycle gap. Treat the two as different baselines until reconciled.

### 0.2 Recovery

`recovery = (RMSE_R1 − RMSE_arm) / (RMSE_R1 − RMSE_R0)`, computed on pooled RMSE, denominator
floored at 1e-9 (`tsh_rul_eval.py`).

- Arms: R0 clean; R1 faulted, no repair; R2 oracle Hθ; R3 TSH+Hθ; R4 oracle last-value;
  R5 TSH+last-value; R6 onset-aware hybrid; R7 5-cycle causal median filter.
- Recovery can be negative (arm worse than faulted) or >1 (arm better than clean).
- It is undefined/unstable when the fault barely moves RMSE: on FD001 noise, R1 = 12.526 <
  R0 = 12.647, so the denominator is negative and the ratio is meaningless. Report absolute RMSE
  in that cell, not recovery.
- "Recovery" is therefore a **normalised pooled-RMSE gap**, not an average over engines and not a
  probability of recovery.

### 0.3 EOL false-alarm rate

From `tsh_eol_fpr.py`: EOL = last 20% of the valid span
(`e0 = first + int(0.8*(last−first+1))`), clean validation, s101, five cuts, gamma = 1.30. Both
rates are computed per window and then averaged over windows.

| Estimand | Definition | FD001–FD004 |
|---|---|---|
| per-sensor-cycle | per window, fraction of (cycle, sensor) entries flagged in EOL | 0.32 / 1.04 / 1.31 / 1.17 % |
| any-sensor per cycle (source label "any-window") | per window, fraction of EOL cycles with ≥1 sensor flagged | 4.43 / 13.02 / 9.71 / 12.39 % |

- The pre-registered 2% gate is met **only** under per-sensor-cycle.
- The evaluator docstring states the gate as "2% (per window, any sensor)", i.e. the second row,
  which does **not** pass. The paper must declare the estimand in Methods and report both; it
  must not present "≤2%" as estimand-free.
- The first TSH version had 3.6–5.3% per-sensor-cycle and 40–50% any-sensor; the 15-cycle
  run-length rule produced the table above at a recovery cost (§1.5).

### 0.4 Conventions

- Δ (LTAG) = K=1 − LTAG; positive = improvement.
- Clean cost = R3c − R0 (same pooled estimator).
- RHSE-TSH conditions: validation split s101, five cuts, mag=3 in z-score space, sensors 0/5/10,
  gamma=1.30, 15-cycle run-length, 2-of-3 latch, kmax=4, onset 0.2/0.5.
- "Test" means the official C-MAPSS test set: sealed for RHSE-TSH, opened once for LTAG.

## 1. What can be claimed

### 1.1 LTAG — main result (test-level)

**Claim.** With the B0 backbone and the LTAG configuration frozen before test, LTAG reduced RMSE
against the single-window baseline in **15/15 (subset, seed) cells** of the official C-MAPSS test
set, mean **ΔRMSE +0.398 cycles**: FD001 +0.426 (6/6, 95% CI [0.231, 0.620]), FD002 +0.162
(3/3, [−0.008, 0.333]), FD003 +0.656 (3/3, [0.264, 1.048]), FD004 +0.321 (3/3, [0.033, 0.609]).

| Subset | Seeds | K=1 | Fixed median | LTAG | Δ | Sign | 95% CI |
|---|---:|---:|---:|---:|---:|---|---|
| FD001 | 6 | 13.421 | 13.364 | 12.995 | +0.426 | 6/6 | [+0.231, +0.620] |
| FD002 | 3 | 11.514 | 11.803 | 11.352 | +0.162 | 3/3 | [−0.008, +0.333] |
| FD003 | 3 | 12.979 | 13.002 | 12.323 | +0.656 | 3/3 | [+0.264, +1.048] |
| FD004 | 3 | 13.031 | 13.600 | 12.710 | +0.321 | 3/3 | [+0.033, +0.609] |
| **All** | **15** | — | — | — | **+0.398** | **15/15** | — |

**Conditions.**

- Configuration frozen in `FREEZE_LTAG.json` (model sha256 `11bd3a61…`); test labels used only for
  scoring; no test-based selection of gate, features, K or Δ.
- Gate trained only on training engines (5-fold GroupKFold by `(subset, engine)`); validation
  engines never used for fitting.
- Features observable at test; no engine total life `n` and no `c/n`.
- K=3 causal windows ending at `c, c−5, c−10`; age alignment `a_t = p_t − (c−(t+1))`;
  `pred = g·p_med + (1−g)·p_k1`.
- Δ = K=1 − LTAG; intervals as reported (their construction must be stated in Methods).

**Mechanism support (validation, engine-mean; Δ vs K=1).**

| Arm | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| Fixed median | +0.533 | −0.208 | −0.219 | −0.864 |
| RGTA (τ=0.05) | +0.615 | +0.041 | −0.185 | −0.322 |
| Same-rate random | +0.496 | −0.193 | +0.068 | −0.365 |
| **LTAG** | **+0.690** | **+0.307** | **+0.375** | **+0.047** |
| LTAG (test) | +0.426 | +0.162 | +0.656 | +0.321 |

- Fixed median: aggregation without selection hurts the multi-condition subsets.
- Hand-written RGTA: a hard threshold is not a selector (FD004 −0.322).
- Same-rate random: the gain is not "more aggregation" (FD002 −0.193, FD004 −0.365).
- LTAG: 14/15 validation cells positive (FD004 s202 −0.038).
- Un-aligned K=3 on FD001 is 12.17 vs K=1 11.56 → age alignment is necessary.
- Engine-disjoint OOF is positive on all four subsets: +0.665 / +0.742 / +0.710 / +0.577.
- Test random-gate control: LTAG better in 14/15 cells; FD001 s606 −0.026 against a random
  sd of 0.305.

**Boundary.** FD002's interval crosses zero: three seeds positive, but not significant. The test
is a **confirmatory one-shot**, not a pristine blind test. The gate is inference-time only on a
frozen backbone.

### 1.2 RHSE-TSH — single-condition front-end (FD001/FD003)

**Claim.** Under single-condition sensor faults, the TSH front-end plus reconstruction recovered
**0.765–0.895** of the fault-induced pooled-RMSE increase in five of six cells (FD001/FD003 bias
and drift; FD001 stuck), at a clean cost of +0.024 (FD001) and +0.149 (FD003).

| Subset | Fault | R3 (TSH+Hθ) | R5 (TSH+hold) | R3−R5 | Clean cost | Claimable? |
|---|---|---:|---:|---:|---:|---|
| FD001 | bias | 0.862 | 0.782 | +0.080 | +0.024 | yes (Hθ edge small) |
| FD001 | drift | 0.855 | 0.984 | −0.129 | +0.024 | yes (hold wins) |
| FD001 | stuck | 0.895 | 0.294 | +0.601 | +0.024 | yes (Hθ edge large) |
| FD003 | bias | 0.808 | 0.791 | +0.017 | +0.149 | yes (Hθ edge negligible) |
| FD003 | drift | 0.765 | 0.772 | −0.007 | +0.149 | yes (Hθ ≈ hold) |
| FD003 | stuck | 0.440 | 0.101 | +0.339 | +0.149 | **no** (<0.65; §2.2) |

The claim is about **detection + repair**. The incremental value of spatial reconstruction over
temporal hold is fault-type dependent and must be stated as such:

- **stuck:** Hθ clearly beats hold (FD001 +0.601, FD003 +0.339);
- **early drift (onset 0.20):** Hθ beats hold by +0.057 (0.890 vs 0.833), and the hybrid reaches
  0.912;
- **late drift (onset 0.50):** hold wins (0.984 vs 0.855);
- **bias:** Hθ's edge is small (FD003 +0.017, FD001 +0.080). The simple-RHSE Phase 0 run found
  +0.022 on FD001 bias (q=0.5), so this is not an artefact of TSH.

The onset-aware hybrid is the honest policy: FD001 drift 0.972 late / 0.912 early; FD003 drift
0.793. It is a **selection rule**, not evidence that Hθ dominates everywhere.

**Conditions.** Validation only; s101; five cuts; mag=3 z-score; sensors 0/5/10; gamma=1.30;
post-15-cycle-run-length. Test sealed.

### 1.3 Clean cost

| Subset | R0 | R3c | Cost | Gate ≤0.15 |
|---|---:|---:|---:|---|
| FD001 | 12.647 | 12.671 | +0.024 | pass |
| FD002 | 12.732 | 12.864 | +0.132 | pass |
| FD003 | 14.299 | 14.448 | +0.149 | pass (0.001 margin) |
| FD004 | 18.442 | 18.406 | −0.036 | pass (not a gain) |

**Claim.** All four subsets stay within the pre-registered 0.15 pooled-RMSE budget.

**Cautions.**

- FD003 passes by 0.001 and FD002 by 0.018, on a single split seed. Do not call the margin robust.
- FD004's −0.036 is estimator noise, not an improvement; do not claim a gain.
- The cost is measured on clean windows and excludes the recovery lost to the 15-cycle delay; that
  loss is in §1.5. Report both.

### 1.4 EOL false-alarm rate

| Subset | per-sensor-cycle | any-sensor per cycle | 2% gate (per-sensor-cycle) |
|---|---:|---:|---|
| FD001 | 0.32% | 4.43% | pass |
| FD002 | 1.04% | 13.02% | pass |
| FD003 | 1.31% | 9.71% | pass |
| FD004 | 1.17% | 12.39% | pass |

**Claim.** On clean validation, the per-sensor-cycle EOL false-alarm rate is **0.32–1.31%**,
within the 2% gate.

**Conditions.**

- Last 20% of the valid span, clean data, gamma=1.30, post-run-length, s101, five cuts.
- The any-sensor-per-cycle rate is 4.43–13.02% and does **not** meet 2%. Both must be reported.
- The estimand must be declared in Methods; the gate is not estimand-free (§0.3).
- The first version's 3.6–5.3% / 40–50% is the counterfactual that motivates the run-length rule;
  cite it only as the before/after trade-off.

### 1.5 Stuck: R3 vs R5

| Subset | R3 | R5 | R3−R5 | Claimable? |
|---|---:|---:|---:|---|
| FD001 | 0.895 | 0.294 | +0.601 | yes |
| FD004 | 0.702 | −0.758 | +1.460 | yes |
| FD002 | 0.167 | −0.776 | +0.943 | no |
| FD003 | 0.440 | 0.101 | +0.339 | no |

**Claim.** On FD001 and FD004, R3 (TSH+Hθ) recovers 0.895 and 0.702, while R5 (TSH+hold)
recovers 0.294 and −0.758; the R3−R5 gaps are +0.601 and +1.460. Both R3 values are ≥0.65.

**Cautions.**

- Post-run-length numbers. The run-length rule is what buys the EOL FPR; its price is visible
  here: FD001 0.927→0.895 and FD004 0.964→0.702 (pre→post).
- R5 is detected-then-hold, not an oracle; the comparison isolates the reconstruction choice
  given the same detector.
- R4 (oracle last-value) per-cell values are not in the final table. `PHASE1_FINAL` §5 labels the
  stuck-lead row "R4", but its numbers equal R3−R5 (+0.60/+0.94/+0.34/+1.46); cite them as
  R3−R5 and confirm the arm label before submission.
- FD002 (0.167/−0.776) and FD003 (0.440/0.101) are **not** claimable: FD002 is below the gate;
  FD003 is below the gate and its lead is over a collapsed baseline (§2.2).

## 2. What cannot be claimed

### 2.1 FD002/FD004 multi-condition drift and noise

| Subset | Fault | R3 | R5 | Hybrid | Gate ≥0.65 |
|---|---|---:|---:|---:|---|
| FD002 | drift | 0.603 | 0.525 | 0.536 | fail |
| FD004 | drift | 0.518 | 0.322 | 0.307 | fail |
| FD002 | noise | −2.106 | −6.854 | — | fail |
| FD004 | noise | 0.338 | −1.173 | — | fail |
| FD002 | stuck | 0.167 | −0.776 | — | fail |

- **Noise:** FD002's R2 (oracle Hθ) is 13.050 vs R1 12.966 → the reconstruction is worse than no
  repair, so the inductive bias is wrong for noise, not just the detector.
- **Stuck:** FD002's R2 oracle recovers only 0.573 → even knowing the faulty sensor, Hθ cannot
  reach 0.65.
- **H_c does not fix it:** per-condition H gives reconstruction MSE 0.6106→0.6166 (FD002, ratio
  1.010) and 0.4686→0.4776 (FD004, 1.019), with 5,600–12,040 rows per condition. The failure is
  not a single-H mismatch.
- **Filtering does not unify noise either:** R7 is best on FD001 (12.424 vs clean 12.647) but worse
  than faulted on FD002 (13.078 vs 12.966) and FD004 (18.948 vs 18.911).

**Therefore:** no multi-condition drift/noise claim, no "H_c fixes multi-condition" claim, no
"noise is handled" claim. This is boundary evidence, not a result.

### 2.2 FD003 stuck 0.440

- R3 0.440 < 0.65; R5 0.101. The +0.339 lead over R5 is real but is a lead over a collapsed
  baseline; it does not satisfy the absolute recovery gate.
- The 15-cycle run-length rule cost this cell 0.148 (0.588→0.440) while buying the EOL FPR. Report
  it as the latency/false-alarm trade-off, not as a recovery success.
- FD002 stuck (0.167) is excluded for the same reason.

### 2.3 No universal module claim

| Gate 1 condition (frozen) | Verdict | Numbers |
|---|---|---|
| clean cost ≤0.15 | pass | +0.024 / +0.132 / +0.149 / −0.036 |
| EOL FPR ≤2% (per-sensor-cycle) | pass | 0.32–1.31% |
| mean recovery ≥0.65, ≥3/4 subsets | **fail** | FD002/FD004 below; FD003 borderline |
| R4 stuck lead R5 ≥0.10 | pass (label caveat) | +0.60 / +0.94 / +0.34 / +1.46 |
| R4 drift lead R5 ≥0.10 | **fail** | FD001 late drift R5 0.984; only FD004 +0.20 |
| onset=0.20 lead R5 ≥0.10 | partial | FD001 drift +0.079; stuck passes |

- The pre-registered average-recovery condition failed: FD002 and FD004 are clearly below 0.65,
  FD003 is borderline, FD001 is the only clean pass. Gate 1 formally failed; Phase 2 was not
  started.
- The drift condition "R4 ≥ R5 + 0.10" was **infeasible** on FD001 late drift because R5 = 0.984
  leaves only 0.016 of headroom. That is a flaw in the gate, not a pass. A non-inferiority
  reformulation (e.g. R6 ≥ R5 − 0.05) is a **new, post-hoc criterion**; if used, it must be
  labelled as such and the frozen gate must still be reported as failed.
- Therefore RHSE-TSH is a **single-condition sensor-fault front-end with an interpretable
  operating point**, not a general robustness module. No "sensor-fault-robust RUL", no "works
  across C-MAPSS", no "universal".

### 2.4 Also forbidden

- "LTAG improves the backbone / is a new RUL predictor" → inference-time gate on frozen B0.
- "Blind test" → confirmatory one-shot on a historically observed test set.
- "Significant on FD002" → CI [−0.008, +0.333].
- "RHSE-TSH test result" → validation only; test sealed.
- "Clean cost is negligible/robust" → FD003 margin 0.001, single split seed.
- "R4 beats R5 on stuck" → the final table reports R3, not R4.
- Subtracting LTAG and RHSE numbers, or putting them in one table → different estimands and
  pipelines (§0.1).

## 3. Five limitations for IEEE TIM

1. **Scope of the fault-tolerance claim.** The front-end is validated only on the single-condition
   subsets FD001 and FD003. On the multi-condition subsets FD002 and FD004, drift recovery falls
   to 0.518–0.603 and noise recovery to 0.338 and −2.106, and a condition-specific measurement
   matrix does not help (reconstruction-MSE ratio 1.010 and 1.019). We therefore claim
   single-condition, fault-type-specific behaviour only; multi-condition sensor-fault robustness
   is open.

2. **Detection latency versus false-alarm trade-off.** The 15-cycle run-length rule was required
   to reduce the clean end-of-life false-alarm rate from 3.6–5.3% to 0.32–1.31% (per sensor-cycle),
   but it delays stuck detection and lowers recovery (FD001 0.927→0.895, FD004 0.964→0.702,
   FD003 0.588→0.440). The operating point is a design choice with a measurable cost, and FD003
   stuck falls below the 0.65 recovery gate.

3. **Estimand sensitivity.** Fault results are pooled over (engine, cut) rows, whereas the LTAG
   validation results are engine-mean; pooled RMSE is never below engine-mean and the two are not
   interchangeable. The end-of-life false-alarm rate has two defensible estimands (0.32–1.31% per
   sensor-cycle versus 4.43–13.02% any-sensor per cycle), and only the former meets the 2% gate.
   The recovery ratio is undefined when the fault barely changes RMSE (FD001 noise: R1 12.526 <
   R0 12.647), so that cell is reported as absolute RMSE.

4. **Evidence breadth.** The fault study uses one split seed (s101), five cut points, three of
   fourteen sensors, one fault magnitude (mag=3, z-score space) and one backbone. No multi-seed,
   multi-magnitude or all-sensor sweep was run; stuck injection in physical units is still
   pending; no end-to-end joint training was attempted. The Gate 1 failure stopped Phase 2, so the
   end-to-end variant is untested rather than refuted.

5. **Test-set status and statistical power.** The LTAG test evaluation is a confirmatory one-shot
   on a test set that has been observed repeatedly in this project, not a pristine blind test. The
   FD002 interval crosses zero ([−0.008, +0.333], three seeds), so its improvement is directional
   but not significant. LTAG is an inference-time gate on a frozen backbone; it is not a new RUL
   predictor and does not improve the backbone.

## 4. Recommended narrative

### 4.1 Main contribution: LTAG

One method, one test table, one ablation set.

- **One-sentence claim:** "A learned, inference-time gate over causally age-aligned multi-window
  predictions improves a frozen RUL backbone on all 15 official-test cells (mean +0.398 cycles),
  while fixed median aggregation, a hand-written dispersion gate, and a same-rate random gate do
  not."
- **Why it is a contribution:** the ablation set removes the three obvious alternative
  explanations (aggregation itself, a threshold rule, more weight on the median). Age alignment
  is shown to be necessary (un-aligned K=3 is worse than K=1 on FD001). The gate uses only
  observable features.
- **Evidence hierarchy:** official test 15/15 (+0.398) → validation 14/15 → engine-disjoint OOF
  positive on all four subsets → random-gate control 14/15.
- **Keep it inference-time:** no backbone retraining, no new predictor.

### 4.2 Conditional front-end: RHSE-TSH

Put it in an appendix or a bounded subsection, not in the abstract's main claim.

- **One-sentence claim:** "On the single-condition subsets FD001 and FD003, an SPC front-end plus
  measurement-model reconstruction recovers 0.765–0.895 of the fault-induced RMSE increase at a
  clean cost of +0.024/+0.149 and a per-sensor-cycle EOL false-alarm rate of 0.32–1.31%; the
  advantage over last-value holds for stuck and early drift, not for late drift or bias."
- **What it contributes:** an interpretable detector with a measurable operating point (EOL FPR,
  latency, clean cost) and a falsifiable mechanism (fault type determines whether spatial or
  temporal redundancy is the right repair). The hybrid is the operational form of that mechanism.
- **What it is not:** a universal module, a test-level result, or an end-to-end method.

### 4.3 Negative results as boundary

Give them a short, explicit section.

- Multi-condition failure (FD002/FD004), the H_c null, FD003 stuck, and the Gate 1 failure explain
  why the paper stops at a front-end and does not claim end-to-end robustness.
- Present them as the validity domain, not as a second contribution and not as a confession. A
  reviewer will ask "does it work on FD002/FD004?"; the paper should answer first, with numbers,
  and then scope the claim.
- Do not let the negative results dominate the abstract or the introduction.

### 4.4 Section mapping

| Section | Content |
|---|---|
| Abstract | LTAG (+0.398, 15/15) + one clause on the scoped front-end |
| §1 Introduction | Problem: when to aggregate multi-window predictions under regime shift |
| §3 Method | LTAG |
| §4 Results | Test table, per-seed, ablations, random control |
| §5 / Appendix A | RHSE-TSH single-condition, EOL FPR, latency trade-off |
| §6 Limitations | The five items in §3 |
| §7 Conclusion | LTAG as the transferable result; front-end and negative results as scope |

### 4.5 Sentence pairs

| Do not write | Write instead |
|---|---|
| "RHSE-TSH is robust to sensor faults across C-MAPSS." | "On the single-condition subsets FD001 and FD003, RHSE-TSH recovers 0.765–0.895 of the fault-induced RMSE increase at a clean cost of +0.024/+0.149." |
| "LTAG significantly improves FD002." | "LTAG improved all three FD002 seeds by a mean 0.162 cycles (95% CI [−0.008, 0.333])." |
| "The gate improves the backbone." | "The gate is applied at inference time to a frozen backbone." |
| "The EOL false-alarm rate is below 2%." | "The per-sensor-cycle EOL false-alarm rate is 0.32–1.31%; the any-sensor-per-cycle rate is 4.43–13.02%." |
| "FD003 stuck is recovered." | "FD003 stuck recovery is 0.440, below the 0.65 gate, reflecting the 15-cycle detection latency." |

## 5. Provenance and unresolved inconsistencies

| # | Item | What the sources say | Action |
|---|---|---|---|
| 1 | FD002 clean cost | Phase 1b: +0.227 ("fail"); FINAL: +0.132 ("pass") | Use FINAL; reconcile from the same eval log before submission |
| 2 | FD002/FD004 bias cells | Pilot: 0.909/0.849 and 0.824/0.436; FINAL: 0.913/0.844 and 0.801/0.502 | Cite FINAL only; small unexplained drift |
| 3 | Stuck cells | Pilot pre-run-length vs FINAL post: FD001 0.927→0.895, FD002 −0.047→0.167, FD003 0.588→0.440, FD004 0.964→0.702 | Always label pre/post; the 15-cycle rule explains the change |
| 4 | FINAL §5 "R4" row | Numbers equal R3−R5 (+0.60/+0.94/+0.34/+1.46) | Cite as R3−R5; confirm the arm label |
| 5 | EOL FPR gate | Evaluator docstring: 2% per window, any sensor; report: 2% per-sensor-cycle | Declare the estimand; report both; no estimand-free pass |
| 6 | LTAG vs RHSE baselines | Different estimands/pipelines; FD001 12.647 vs 11.696 | No joint table until a shared R0 is established |

**Bottom line.** The paper has one test-level method claim (LTAG, 15/15, +0.398) and one
conditional, validation-only front-end claim (RHSE-TSH, single-condition FD001/FD003, 0.765–0.895,
clean cost +0.024/+0.149, EOL FPR 0.32–1.31% per sensor-cycle). Everything multi-condition,
everything test-level for RHSE-TSH, and every "universal module" statement is outside the
evidence. The negative results are the boundary that makes the two positive claims credible.
