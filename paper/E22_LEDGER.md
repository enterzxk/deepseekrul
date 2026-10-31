# E22 ledger

| item | file | sha256 (first 16) | frozen at |
|---|---|---|---|
| E22a prereg | paper/E22A_FULL_TEST_PREREGISTRATION.md | 8f18b75c999be9ca | 2026-10-31T23:12+08:00 |
| E22b prereg (incl. Amendment 1) | paper/E22B_DUALMIXER_PREREGISTRATION.md | 29113e037b441127 | 2026-11-01T00:22+08:00 |

## E22a outcome (2026-11-01T00:10+08:00)

All 707 official test engines, frozen first-frame padding, 15-model ensembles.
Capped truth (headline) / uncapped truth (real-life error analysis):

| subset | arm | capped RMSE (95% CI) | uncapped RMSE | NASA mean cap | NASA mean raw |
|---|---|---:|---:|---:|---:|
| FD001 (100) | A | 10.382 [8.54, 12.16] | 11.662 | 1.60 | 1.85 |
| FD001 | BE | 12.170 [9.94, 14.32] | 13.209 | 2.23 | 2.46 |
| FD001 | PE | 9.816 [8.13, 11.54] | 10.940 | 1.48 | 1.69 |
| FD002 (259) | A | 16.659 [13.66, 19.61] | 28.403 | 9.45 | 53.41 |
| FD002 | BE | 14.098 [12.24, 15.93] | 26.232 | 3.56 | 21.60 |
| FD002 | PE | 11.509 [9.94, 13.11] | 24.012 | 2.69 | 18.31 |
| FD003 (100) | A | 9.947 [7.90, 11.88] | 11.449 | 1.77 | 2.06 |
| FD003 | BE | 11.524 [9.37, 13.59] | 13.077 | 2.47 | 2.84 |
| FD003 | PE | 10.063 [7.89, 12.22] | 11.809 | 1.98 | 2.31 |
| FD004 (248) | A | 13.640 [11.67, 15.68] | 26.900 | 3.59 | 24.52 |
| FD004 | BE | 13.340 [11.49, 15.21] | 25.919 | 3.58 | 18.08 |
| FD004 | PE | 11.794 [9.95, 13.83] | 24.725 | 2.84 | 16.04 |

**Pre-registered prediction P1 FAILED.** P1 expected the full set to raise FD002/FD004
RMSE by about the §6.10 bound (≤ ±5%). Observed: FD002 arm A +39% (11.96 → 16.66),
while FD002 PE moved +0.4% (11.46 → 11.51). The failure is informative: the 54
short-trajectory engines do not perturb every arm equally — they are a stress test that
the linear time channel fails and the bounded encoding passes.

**New finding (not pre-registered).** On FD002 the arm ranking flips between coverage
regimes: with the 235 retained engines A is best (11.96 vs PE 11.46 — A better by 0.50);
with all 259 engines PE is best by 5.15. Coverage definition is therefore a ranking
variable, which is RQ1 evidence for E22c.

## E22b status (2026-11-01T00:30+08:00)

FD001 queue running: 15 A-zscore runs + 15 Dual-Mixer runs (5 folds × 3 seeds).
Smoke tests: A-zscore fold0 seed2027 val 9.397 (arms match, 41,858 params);
Dual-Mixer fold0 seed2027 val 15.1 on z-score input after the Amendment-1 fix (val 43
on the project's min-max input — the failed fit is documented in the amendment).
