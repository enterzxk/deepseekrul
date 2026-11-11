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

**READY as a four-subset pilot; REVISE before an IEEE TIM submission.**

The factorial is internally consistent and the selected model is simpler than
FULL, but the gain is within noise and the test table is descriptive. The
three-seed check in section 7 settles the seed question: the seed-101 margin
does not replicate, and the winner flips on seed 202. No claim of a performance
advantage over FULL survives three seeds. No new module is added.

## 6. Caveats

- Three seeds (101/202/303) for the three headline cells; the other seven
  factorial cells are still single-seed.
- Test numbers are descriptive and were opened at the user's request.
- The single-scale replacement uses k=3; "best scale" was not separately tuned.
- The HI head is present in both HI levels; only the loss weight changes.
- No claim that the selected cell is significantly better than FULL.

## 7. Three-seed check (seeds 202/303)

24 runs added 2026-11-11: the selected cell, the factorial base and FULL on all
four subsets at seeds 202 and 303, with hyperparameters identical to seed 101.
Grid `experiments/grids/factorial_s23.json`, manifest
`results/FREEZE_FACTORIAL_S23.json`, tables `results/seed_robustness_*.csv`,
generated report `SEED_ROBUSTNESS.md`.

Validation RMSE (mean +- sd over seeds 101/202/303):

| cell | FD001 | FD002 | FD003 | FD004 | mean |
|---|---:|---:|---:|---:|---:|
| single + mean, no HI (base) | 12.093 +- 0.311 | 11.884 +- 0.311 | 12.669 +- 0.214 | 17.026 +- 1.071 | 13.418 |
| **single + attention + HI (selected)** | 10.760 +- 0.178 | 12.055 +- 0.414 | 11.992 +- 0.306 | 16.910 +- 0.134 | **12.929** |
| full + attention + HI (FULL) | 11.132 +- 0.286 | 11.941 +- 0.300 | 12.342 +- 0.854 | 16.514 +- 0.184 | 12.982 |

Official test RMSE (descriptive, mean over seeds): selected 11.134, FULL 11.512,
base 11.655.

What three seeds change:

1. **The seed-101 margin does not replicate.** The validation gap between the
   selected cell and FULL shrinks from -2.3% (seed 101) to -0.4% (three-seed
   mean, 12.929 vs 12.982), and the winner **flips on seed 202**, where FULL
   wins (12.628 vs 13.035). Per-seed winner: cand / full / cand.
2. **Paired test deltas are not sign-stable.** selected - FULL: mean -0.378
   (sd 1.057), better in 8/12 (subset, seed) pairs; only FD003 keeps the sign
   across all three seeds; engine-bootstrap CIs exclude 0 in 3/12 pairs.
   selected - base: mean -0.521 (sd 0.814), better in 9/12 pairs; FD004 is the
   only subset whose t interval excludes 0.
3. **The correct claim is "comparable", not "better".** The selected cell is
   still what the validation rule picks on the seed-averaged metric and it is
   the simplest of the three, but no performance advantage over FULL survives
   three seeds.
