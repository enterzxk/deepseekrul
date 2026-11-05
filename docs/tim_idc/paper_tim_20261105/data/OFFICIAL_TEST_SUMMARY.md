# Official test — one-shot results (opened 2026-11-05 13:55 CST)

- Sealed predictions: **96** (48 lightweight 24ep/stride5 + 48 full-budget 80ep/stride1).
- Labels opened exactly once by `score_both.py`; no model, recipe, arm, or budget change after opening.
- Results files: `official_results.json` (lightweight), `official_fullbudget_results.json` (full-budget).
- The historical E22a official-test numbers (9.82/11.51/10.06/11.79) are a different protocol and are not comparable.

## 1. Full-budget (80 epochs, all windows) — main performance table

| subset | M0 | OPS | FULL | CET | FULL-M0 [95% CI] | FULL-OPS [95% CI] |
|---|---:|---:|---:|---:|---|---|
| FD001 | 9.852 | 9.968 | 9.443 | 18.668 | -0.409 [-0.850, +0.010] ns | -0.526 [-0.913, -0.146] SIG |
| FD002 | 12.084 | 11.842 | 12.432 | 17.831 | **+0.348 [-0.259, +0.948] ns (worse)** | **+0.590 [+0.036, +1.148] SIG (worse)** |
| FD003 | 10.823 | 10.952 | 10.386 | 16.112 | -0.437 [-0.799, -0.060] SIG | -0.566 [-0.952, -0.196] SIG |
| FD004 | 13.010 | 13.196 | 12.486 | 18.002 | -0.524 [-1.114, +0.080] ns | -0.710 [-1.377, -0.053] SIG |

Per-seed FULL-M0 (full-budget):
- FD001: -0.241 / -0.356 / -0.637 (all negative; ensemble CI upper just crosses 0)
- FD002: +0.146 / -0.122 / +0.819 (mixed; ensemble worse)
- FD003: -0.254 / -0.401 / -0.726 (all negative)
- FD004: -0.086 / -1.022 / -0.964 (all negative; ensemble CI crosses 0)

## 2. Lightweight (24 epochs, stride 5) — controlled contrast

| subset | M0 | OPS | FULL | CET | FULL-M0 [95% CI] | FULL-OPS [95% CI] |
|---|---:|---:|---:|---:|---|---|
| FD001 | 22.211 | 17.843 | 17.194 | 16.655 | -5.017 [-6.434, -3.539] SIG | -0.649 [-0.849, -0.432] SIG |
| FD002 | 12.997 | 13.593 | 12.864 | 21.646 | -0.134 [-0.454, +0.181] ns | -0.729 [-1.119, -0.349] SIG |
| FD003 | 17.038 | 16.563 | 15.140 | 16.386 | -1.898 [-2.671, -1.186] SIG | -1.423 [-1.833, -1.010] SIG |
| FD004 | 13.063 | 12.994 | 12.636 | 22.283 | -0.427 [-0.777, -0.070] SIG | -0.358 [-0.634, -0.075] SIG |

## 3. Frozen conformal intervals on the official test (90%, outer-val calibration)

Ensemble FULL, coverage / mean width (cycles):

| subset | lightweight | full-budget |
|---|---|---|
| FD001 | 1.000 / 83.2 | 1.000 / 79.1 |
| FD002 | 0.942 / 37.1 | 0.927 / 36.8 |
| FD003 | 0.940 / 46.6 | 0.970 / 45.2 |
| FD004 | 0.927 / 37.3 | 0.895 / 36.0 |

The intervals are over-conservative (FD001 width ~80 cycles) because the calibration is from the lightweight outer-val, whose FD001 model is much weaker than the full-budget model. The reliability layer is not a strong contribution as implemented.

## 4. Reading

1. **The full-budget official test does not support a robust FULL-M0 gain**: significant only on FD003; FD001 ns (upper +0.010), FD002 ns and worse, FD004 ns.
2. **On the primary FD002, the sensor prior hurts relative to the age/condition prior**: FULL-OPS = +0.590 SIG worse. This is the opposite of the lightweight result (-0.729 SIG) and the replication (-0.616).
3. **The benefit is budget-dependent**: the lightweight protocol shows FULL-M0 gains on FD001/FD003/FD004 and FULL-OPS on all four; the full-budget protocol removes most of them and flips FD002.
4. **CET is not a competitive baseline in the full-budget protocol** (17.8-18.7), and the adapted CET is not the official CET-MSA; the baseline table must be labeled carefully.
5. **The conformal layer is wide and not compelling**; it should be a secondary/caveated analysis, not the main innovation.

## 5. Bottom line

The one-shot test answers the question the development experiments could not: under the full training budget, the FULL sensor prior does **not** reliably beat the PE baseline, and on FD002 it is worse than the age/condition prior. The method claim "FULL improves RUL over PE" is not supported by the official test. The honest paper is a measurement/empirical study of when the causal lifetime prior helps and when it does not, not a "new SOTA method" paper.
