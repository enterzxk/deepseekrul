# Paper B current status

Updated 2026-11-11 after the scope reduction to **FD001 only**. All numbers
below come from `paper_b/results/fd001_final_table.csv` and the per-run
`metrics.json` / `test_metrics.json` files.

## 1. Where the project stands

| item | state |
|---|---|
| branch | `punet-interval`; local commits on top of `95e729e` |
| Paper A | untouched |
| active code | `paper_b/models`, `paper_b/training`, `paper_b/evaluation` |
| scope this round | **FD001 only** (multi-subset grid stopped at the user's request) |
| running jobs | none |
| final model | **MS+SA (MSSA), no HI** |
| official test | opened once, for the frozen FD001 tags only |

## 2. FD001 final results (3 seeds, 101/202/303)

| model | validation RMSE | official test RMSE | test vs base |
|---|---:|---:|---:|
| DSPT base | 11.874 +- 0.258 | 11.054 +- 0.350 | - |
| **DSPT MS+SA (final)** | **11.112 +- 0.344** | **10.598 +- 0.299** | **-4.1%** |
| DSPT FULL | 11.132 +- 0.234 | not evaluated | - |
| Vanilla (raw) | 13.366 +- 0.184 | 15.046 +- 1.111 | +36.1% |
| Vanilla (features+cycle) | 12.503 +- 0.666 | 12.599 +- 0.415 | +14.0% |

- MS+SA is -6.4% on validation and -4.1% on test relative to base; the two
  directions agree.
- FULL is tied with MS+SA on validation (0.2%, far below the seed std); the
  simpler MS+SA is selected by the pre-registered tie-break.
- HI alone is +0.5% (worse); adding HI to MS+SA gives no gain. HI is dropped.
- DSPT beats the feature-matched Vanilla by 16% on test, so the architecture
  adds value beyond the three-channel input and cycle encoding.

## 3. Historical evidence status

| item | status |
|---|---|
| Archived 32-cell table | checkpoint-consistent (corr 0.9935-0.9995); protocol-confounded, not fabricated |
| P0-fixed recomputation | moves absolute values, does not flip the archived ranking |
| OOF 60-cell | **INVALID** (truncation points sampled from official test RUL) |
| Historical negative ablation | **reversed** under the fixed FD001 protocol |

Attribution (FD001/seed 101, diagnostic): global HI leakage buys 3.0% test;
the EMA fix buys 5.8% test but looks worse on validation; removing the padding
mask buys 5.4% test. See `ATTRIBUTION.md`.

## 4. What was not run

- FD002/FD003/FD004 completion (FD002 has 24 cells, FD003 18, FD004 0).
- MS/SA/HI mechanism grids, window search, padding-mask control, cut
  sensitivity, determinism check. These remain planned, not claimed.

## 5. Decision

**READY as an FD001-scope pilot; REVISE before an IEEE TIM submission.**

The FD001 result is internally clean and protocol-consistent. A single C-MAPSS
subset cannot support a general claim; the minimal next round is FD002/FD004
for MSSA/base/Vanilla (3 seeds, ~1.5-2 GPU-h). If MSSA does not hold on the
multi-condition subsets, the paper stays FD001-scope or becomes a protocol/audit
paper.

## 6. Caveats

- FD001 only; no cross-subset claim.
- n=3 seeds; the MSSA-vs-base gap is larger than the seed std, but a paired
  test with n=3 has low power.
- The official FD001 test set has been observed before; the freeze manifest
  limits further selection but cannot make it pristine.
- The mechanism explanations (SA condition heterogeneity, HI target noise, MS
  scale redundancy) are probe-level, not ablation-confirmed in this round.
