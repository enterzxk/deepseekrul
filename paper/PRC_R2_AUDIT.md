# PRC-R2 audit: is there a methodological contribution? (NO-GO on the current design)

Date: 2026-11-01. Evidence: `results/prc/coverage_r2_conditional_200.json`,
`results/prc/consistency_conditional.json`. Code: `scripts/prc/`. This report is
written to be read against the numbers, not against the intention.

## 1. What was tested

Four known-truth scenarios built on a shared latent-prediction structure
(`simulator_v2.py`), all protocols derived from the **same two prediction vectors**
(cap transform, coverage of short vs mid life segments), so protocol dependence is real
and an independence assumption is false:

| scenario | true deltas (cap_all, uncap_all, short, mid) | intended state |
|---|---|---|
| A_dominant | +4.38, +0.79, +7.12, +4.92 | A robustly better |
| B_dominant | −4.35, −0.78, −6.94, −5.10 | B robustly better |
| sign_flip | −1.31, −0.65, **+18.63**, **−18.33** | protocol dependence confirmed |
| near_zero | +0.17, +0.03, +0.34, +0.09 | inconclusive |

200 Monte Carlo replicates per cell, n=200 engines, 499 bootstrap replicates,
conditional inference (fold and seed constants fixed). Coverage MC CIs are in the JSON.

## 2. Result

| scenario | PRC band coverage [MC 95% CI] | PRC decision correct | IUT decision correct | Bonferroni |
|---|---|---:|---:|---:|
| A_dominant | 0.86 [0.82, 0.91] | **0.23** | **0.89** | 0.22 |
| B_dominant | 0.84 [0.80, 0.90] | **0.31** | **0.93** | 0.30 |
| sign_flip | 0.95 [0.92, 0.98] | 1.00 | 1.00 | 0.62 |
| near_zero | 0.92 [0.88, 0.96] | 1.00 | 0.77 | 0.98 |

Two findings, both against the design:

1. **The simultaneous band does not clear its own gate.** Coverage 0.84–0.86 in the two
   dominant scenarios against a 0.93 threshold, even with a Bonferroni quantile that
   should be conservative. The residual failure is the finite-sample bias of an RMSE
   difference measured in §3 of `consistency_conditional.json`: E[delta_hat] is 3–5%
   below the population value at delta ≈ 7, and a basic bootstrap cannot correct a bias
   that scales with the estimand.
2. **Splitting alpha across protocols is what kills the four-state decision.** With
   alpha/k = 0.0125 the band half-width is ~3.3 RMSE; A_dominant's smallest true delta
   is 0.79, so min's LCB can never exceed zero and "A robust" is essentially
   unreachable (23% correct). The IUT needs no band: each component test runs at alpha,
   its size is controlled by the IUT property, and the four states are mutually
   exclusive by construction, so no multiplicity correction is required. IUT reaches
   89–93% correct on the same data.

**The IUT is not just competitive; it is the correct tool for this decision.** The
four-state decision is a union of three IUTs whose rejection regions are disjoint
(all-positive vs all-negative vs mixed-sign), so at most one can be false and the
family-wise error rate is bounded by alpha without correction. The simultaneous band is
a diagnostic ("which protocol binds"), not an inference device.

## 3. Verdict

**NO-GO** on PRC as an IEEE TIM *methodological* contribution, under the standard the
brief set: it does not beat the honest baseline on decision accuracy, it fails its own
coverage gate, and its central construct (the simultaneous band as the basis of a
decision) is dominated by an existing method (IUT) that is simpler and has a cleaner
theory.

## 4. What survives, and how much it is worth

| component | status after this round |
|---|---|
| Four-state IUT framework (A / B / dependence / inconclusive) with disjoint rejection regions and no multiplicity correction | **defensible**: a correct, size-controlled decision rule; the novelty is compositional, not algorithmic |
| Conditional vs unconditional inference distinction (engines only vs nested over training randomness) | **necessary and non-obvious**; a methods note, not a new estimator |
| Simultaneous band | **diagnostic only**; must not be the basis of a claim until coverage clears 0.93 |
| Budget allocator | **unvalidated**; no evidence yet that it beats "add engines" |
| Protocol-transform simulator with shared latent predictions | **useful infrastructure** for any future claim; not a contribution by itself |

An honest reading: the E-series empirical work (resolution, protocol sensitivity,
label-definition effects) is stronger than the methodological work. The methodological
story that can be defended today is "a correctly specified four-state decision rule for
protocol-dependent comparisons, with a calibrated audit trail" — a methods contribution
of modest scope, which TIM may or may not consider sufficient on its own.

## 5. If the work continues (conditions, not plans)

1. The IUT framework must be shown to change a **real** conclusion on C-MAPSS or an
   independent dataset — a pair that a naive per-protocol test calls significant and
   the four-state rule calls protocol-dependent, or vice versa. Without such a case it
   is a formalism.
2. The allocator must be validated against simulated power (±5 points) before it is
   mentioned in a paper.
3. The band, if used at all, must be reported as a diagnostic with its measured
   coverage, including when that coverage is below nominal.
4. A defensible paper can be written **without** a new estimator: the empirical
   protocol-sensitivity results plus the correctly specified decision rule. Whether that
   clears TIM's bar is a judgement for the authors, and this report does not assume it
   does.

## 6. Reproduction

```
python scripts/prc/exp_consistency.py 200 200 conditional   # bias / target match
python scripts/prc/exp_coverage_r2.py 200 200 conditional   # the table above
```
Full grid (2000 replicates, both inference modes, both resampling schemes) is a
background run; the conclusion above does not depend on it, because the failure is in
the band's construction, not in sampling noise.
