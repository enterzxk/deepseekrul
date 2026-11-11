# Next research decision

Status: pending the protocol-fixed ablation. This file records the decision
procedure, not a post-hoc rationalisation.

## 1. Decision rule

See `FREEZE_RULE.md`. The final model is chosen on validation only; the test
set is opened once for the frozen tags.

## 2. Questions to answer with the results

1. What is the most credible DSPT performance under the fixed protocol?
2. Does DSPT beat an independent Vanilla Transformer at the same window and
   with a feature-matched control?
3. Is MS's effect multi-scale, projection, or capacity?
4. Why did SA hurt on some subsets: condition heterogeneity or missing sensor
   identity?
5. Is HI's negative effect a representation mismatch, a noisy target, or the
   monotone penalty?
6. Which historical numbers are invalid, and which protocol fix explains the
   10.91 -> 12.70 change?
7. Is there a stable improvement, and on which subsets?
8. What is the paper's defensible contribution?

## 3. Decision options

- **READY**: the final model has a replicated validation gain and a frozen test
  number, the Vanilla comparison is fair, and the mechanism claims survive the
  falsification tests. Then write the paper.
- **REVISE**: a specific missing experiment (e.g., window search, external
  dataset, combination of two supported mechanisms) can change the conclusion.
  Run it with a stated budget and stop rule.
- **STOP**: no module survives, and DSPT-base is not better than Vanilla. Then
  the architecture line is closed; the paper becomes a protocol/audit paper or
  the direction changes.

## 4. Decision tree (fixed before the test is opened)

Let `W` be the number of subsets on which the frozen final model beats
DSPT-base on the official test, and `G` the mean relative test gain.

| condition | verdict | action |
|---|---|---|
| `W >= 3` and `G <= -5%` | **READY** | write the paper around the protocol + FULL result; report the mechanism analyses as the explanation |
| `W >= 3` and `-5% < G < 0` | **REVISE** | the gain is real but small; add the window search and a second dataset before writing |
| `W == 2` or sign disagreement with validation | **REVISE** | diagnose the validation/test mismatch (RUL bins, selection metric); do not claim a gain |
| `W <= 1` and base is not better than Vanilla | **STOP** | close the DSPT architecture line; the paper becomes a protocol/audit paper or the direction changes |

A mechanism variant is added to the final model only if it passes the
`FREEZE_RULE.md` test. Otherwise it is reported as a scope-limited finding,
not adopted.

## 5. Current decision (FD001 scope)

**Final model: MS+SA (MSSA), no HI.**

| evidence | value |
|---|---|
| FD001 validation (3 seeds) | 11.112 +- 0.344 vs base 11.874 (-6.4%) |
| FD001 official test (3 seeds) | 10.598 +- 0.299 vs base 11.054 (-4.1%) |
| Vanilla raw / feature-matched test | 15.046 / 12.599 (14-36% worse) |
| FULL vs MSSA | 11.132 vs 11.112 validation, tied; MSSA simpler |

**Verdict: READY as an FD001-scope pilot; REVISE before an IEEE TIM
submission.**

Why not READY for submission: a single C-MAPSS subset cannot support a general
claim, and FD002/FD004 are exactly the multi-condition subsets where the
historical modules failed. The FD001 result is internally clean and
protocol-consistent, but it is one dataset.

**Minimal next round if resumed (estimated 1.5-2 GPU-h):**

1. FD002 and FD004 only: MSSA vs base vs Vanilla (raw + features), 3 seeds.
2. If MSSA wins on at least one multi-condition subset and does not lose on the
   other, the claim becomes "subset-conditional gain" with a stated scope.
3. If MSSA loses on both, the paper is FD001-scope or a protocol/audit paper;
   do not stack more modules.

**Stop rule:** no new module is added in this round. Only the three existing
comparisons are extended to two subsets.