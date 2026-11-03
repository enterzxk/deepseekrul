# Paper Structure Map (writing gate compliant)

> Frozen 2026-11-03. Sections marked [GATED] may only be drafted after the
> corresponding gate passes (per the gated execution spec, section 8).
> All numbers are script-generated from metrics.json / predictions.csv —
> never hand-copied.

## Outcome branches (decided by gates, not by preference)

- **Branch A** (G0+G1+G2 pass): Individualized Degradation Clock +
  degradation-conditioned temporal/sensor modeling + uncertainty-aware gate.
- **Branch B** (G0 pass, G1 modules fail): Contribution 1 = causal nested
  lifetime prior + individualized clock (analysis of *why* joint learning
  fails: LSC detach/oracle evidence); no B/C/D module claims.
- **Branch C** (G0 or G1 fails): analysis / negative-results report; no TIM
  method submission.

## Section map

| # | Section | Gate dependency | Status |
|---|---------|-----------------|--------|
| 1 | Introduction | [GATED] after G1 verdict sets the branch | empty |
| 2 | Related Work | none (writing gate allows) | 01_related_work.md |
| 3 | Dataset & Protocol | none (frozen facts) | 02_dataset_protocol.md |
| 4 | Experimental Setup | none (arms + frozen criteria) | 03_experimental_setup.md |
| 5 | Method | [GATED] branch-dependent | empty |
| 6 | Results (main) | [GATED] G3 | empty |
| 7 | Ablation & Mechanism | [GATED] G1 modules | empty |
| 8 | Uncertainty analysis | [GATED] G2.1 | empty |
| 9 | Complexity | after architecture freeze | empty |
| 10 | Discussion & Limitations | [GATED] | empty |

## Table plan (all script-generated)

- T1 datasets & protocol (facts, no results)
- T2 published results of compared 2026 papers (Table A, "as reported",
  protocol caveats annotated)
- T3 unified-protocol main results (Table C): PE / IDA-PE / CET-MSA /
  (reproducible baselines) — 5-fold CV × 3 seeds
- T4 ablation: M0/M1/M1p/shuffled/constant/oracle
- T5 complexity (params, FLOPs, latency)
- T6 lifetime stratification (short/mid/long)
- T7 uncertainty-gate behavior by u-tercile [GATED G2.1]

## Figure plan

- F1 method overview [GATED]
- F2 individualized clock construction (staircase prior + tau) — can draft
  mechanics now, results panels gated
- F3 attention/sensor conditioning [GATED]
- F4 four-subset results [GATED G3]
- F5 shuffled/constant/oracle mechanism bars [GATED]
- F6 uncertainty gate [GATED G2.1]
- F7 cross-prefix trajectories [GATED module E]
