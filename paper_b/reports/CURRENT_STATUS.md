# Paper B current status

Updated 2026-11-11 after the replacement-based 2x2x2 factorial (seed 101, four
subsets). Authoritative numbers: `results/factorial_val.csv`,
`results/factorial_test.csv`, `results/factorial_effects_*.csv`.

## 1. Scope and final model

| item | state |
|---|---|
| branch | `punet-interval`; local commits on top of `95e729e` |
| Paper A | untouched |
| ablation | 2x2x2 replacement design + 2 controls (10 cells x 4 subsets x seed 101 = 40 runs) |
| selection | validation only: 4-subset mean RMSE -> 1% band -> NASA -> parameters |
| **final model** | **single-scale conv (k=3) + cross-sensor attention + HI supervision** |
| test | opened for all 40 cells at the user's request; descriptive, not selective |
| running jobs | none |

## 2. Factorial headline (seed 101)

| cell | validation mean | test mean (descriptive) |
|---|---:|---:|
| single + attention + HI (selected) | **12.779** | **10.983** |
| full + attention + HI (FULL) | 13.075 | 11.311 |
| single + attention, no HI | 12.882 | 11.671 |
| single + mean pooling, no HI | 13.632 | 11.406 |
| full + mean pooling, no HI | 13.110 | 11.207 |
| linear-only control | 12.957 | 11.618 |
| static sensor weights control | 13.163 | 11.417 |

Selected vs FULL: validation -2.3%, test -2.9% (better on all four subsets).
Paired engine bootstrap vs FULL: all 95% CIs contain 0, so the correct label is
**comparable**, with the simpler model preferred by the validation rule.

## 3. Main effects (validation RMSE; negative = helps)

| subset | MS | SA | HI | MSxSA |
|---|---:|---:|---:|---:|
| FD001 | +0.04 | -0.52 | -0.15 | +1.33 |
| FD002 | -0.23 | -0.07 | -0.01 | +0.41 |
| FD003 | -0.03 | +0.01 | +0.10 | +1.00 |
| FD004 | -0.35 | -1.00 | -0.14 | +0.20 |
| mean | -0.14 | -0.39 | -0.05 | +0.73 |

1. **Multi-scale has no robust effect**; the large positive MSxSA interaction
   shows it is redundant with attention. The selected model uses the
   single-scale replacement.
2. **Attention is the largest validation effect**, concentrated on FD001 and
   FD004; on test it is subset-dependent (helps FD001/FD002, hurts FD003).
3. **HI is weakly positive and subset-dependent** (FD003 is worse with HI).
4. **FULL is not the best cell**; the redundancy between multi-scale and
   attention explains why adding modules did not help.

## 4. Historical evidence status

| item | status |
|---|---|
| Archived 32-cell table | checkpoint-consistent (corr 0.9935-0.9995); protocol-confounded, not fabricated |
| OOF 60-cell | **INVALID** (truncation points sampled from official test RUL) |
| Historical negative ablation | explained by protocol + functional redundancy, not by a single bug |

Attribution (FD001/seed 101, diagnostic): global HI leakage buys 3.0% test;
the EMA fix buys 5.8% test but looks worse on validation; removing the padding
mask buys 5.4% test (`ATTRIBUTION.md`).

## 5. Decision

**READY as a single-seed, four-subset pilot; REVISE before an IEEE TIM
submission.**

The factorial is internally consistent and the selected model is simpler than
FULL, but the gain is within noise, the test table is descriptive, and only one
seed was run. The minimal next round is seeds 202/303 for the selected cell and
FULL on all four subsets (16 runs), then a paired bootstrap with three seeds.
No new module is added.

## 6. Caveats

- One seed (101), as requested; seed variance is not estimated in this round.
- Test numbers are descriptive and were opened at the user's request.
- The single-scale replacement uses k=3; "best scale" was not separately tuned.
- The HI head is present in both HI levels; only the loss weight changes.
- No claim that the selected cell is significantly better than FULL.
