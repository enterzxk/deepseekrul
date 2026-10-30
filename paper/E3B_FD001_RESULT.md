# E3b — Result: FD001 replication (cross-subset)

Run 2026-10-30 00:28–00:52 UTC+8, 9 runs × FD001 (80/20, `--split-seed 7`,
`--epochs 80 --patience 20`), prereg `paper/E3B_FD001_PREREGISTRATION.md`
(sha `3c2b8ea0…`). Statistics: E6 recipe (engine-clustered, 100k resamples,
RNG 7000–7002).

| arm | RMSE (matched, 20 engines) | MAE | NASA | late | severe | params |
|---|---:|---:|---:|---:|---:|---:|
| A (cycle) | 12.224 ± 0.835 | 9.12 | 3.00 | 0.417 | 0.183 | 41,858 |
| BE (no time) | 13.847 ± 0.766 | 10.42 | 3.55 | 0.367 | 0.150 | 41,730 |
| **PE (bounded)** | **11.336 ± 0.105** | **8.15** | **2.67** | 0.433 | 0.200 | 42,306 |

Per-seed: A 13.074 / 11.406 / 12.193; BE 14.488 / 12.998 / 14.055;
PE 11.333 / 11.233 / 11.442.

| contrast | ΔRMSE | 95% CI | verdict |
|---|---:|---|---|
| PE − A | **−0.928** | [−2.234, +0.319] | marg. upper **+0.319 > +0.30** → **FAIL** (same as FD002) |
| BE − A | **+1.674** | [−0.250, +3.699] | **CI includes 0** → time effect *not resolvable* on FD001 |

## The interesting part: the verdict is predicted by the MDE

Same effect, two subsets, two verdicts — and the split is exactly where the
resolution calculation says it should be:

| subset | val engines | BE − A (point) | paired MDE | detectable? |
|---|---:|---:|---:|---|
| FD002 | 52 | +2.69 | **1.89** | yes (CI excludes 0) |
| FD001 | 20 | +1.67 | **2.38** | no (CI includes 0) |

The effect is of comparable size on both subsets (+1.67 vs +2.69, CIs overlap); what
differs is the number of held-out engines, and the MDE tracks the verdict. This is
the paper's thesis demonstrated on its own experiment rather than asserted.

MDE numbers come from `scripts/mde.py` on the same per-engine residuals:
`MDE(n) = 2.80 · sd_i(ΔRMSE_i) / √n`, sd 8.42 (FD002 A/BE, n=156 engine-seed obs),
6.58 (FD001, n=60).

## What this changes in the paper

* §4.3's time claim becomes: **load-bearing and resolvable on FD002; directionally
  consistent but not resolvable on FD001 (20 engines)**. Not "holds on C-MAPSS".
* The bounded-encoding claim is unchanged and now replicated in the negative sense:
  −0.90 (FD002) and −0.93 (FD001), both with upper bounds +0.317 / +0.319, both
  above the +0.30 margin. Two independent subsets failing the same margin by the same
  amount is evidence that the margin, not the encoding, is the binding constraint.
* PE's seed-to-seed SD on FD001 is 0.105 vs A's 0.835 — worth one sentence, not a
  claim (three seeds).

## Boundaries

* 20 held-out engines on FD001: the paired MDE is 2.38, so anything below ~2.4 RMSE
  is invisible here. That is a property of the split, not of the method.
* FD003/FD004 remain sealed; no claim of cross-family generality.
* Same probe-vs-pipeline caveat as E3: the FD001 pipeline runs with
  `condition_aware=False`, so its absolute level is not comparable to FD002's.

Artefacts: `experiments/e3_fd001/*/FD001/*/{metrics.json,matched_cutoff_predictions.csv}`,
`results/dsh/e3/FD001_contrasts.json`, `results/dsh/e6/{E6_stats,MDE}.json`.
