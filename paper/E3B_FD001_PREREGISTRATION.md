# E3b — Pre-registration: FD001 replication of the time-encoding contrast

Frozen 2026-10-30 00:25 UTC+8, before any FD001 run. sha256 recorded in
`paper/STATUS.json` and passed to every run via `--prereg-sha`.

## Why

Every neural result in the paper so far is FD002 only. FD001 is a different regime
(single operating condition, one fault mode, 100 engines), and the time-shortcut
claim is exactly the kind of finding that may be regime-specific. §4.3 currently
rests on one subset; a reviewer will ask for the other. FD003/FD004 remain sealed and
are **not** part of this registration.

## Design (identical to E3 except the subset)

| | |
|---|---|
| data | FD001, frozen `prepare()` (`condition_aware=False`), `--split-seed 7`, `--ref-mode global` |
| size | 100 engines, 15,731 training windows, 80/20 engine split |
| arms | A (cycle), BE (no time), PE (bounded sin/cos, same 8 channels, +448 params) |
| seeds | 2027, 2028, 2029 |
| budget | `--epochs 80 --patience 20 --d-model 64 --num-layers 1` |

## Hypotheses and rules (fixed now)

| id | hypothesis | rule |
|---|---|---|
| H1 | time remains load-bearing | BE − A > 0 with the engine-clustered 95% CI excluding 0; report the point estimate regardless |
| H2 | the bounded encoding is at least compatible | PE − A one-sided 97.5% upper bound < +0.30, NASA ≤ 1.05×A, late ≤ A + 0.02 |
| H3 | the effect is regime-dependent | compare the FD001 and FD002 point estimates; report both, claim a regime difference only if the CIs are disjoint |

Δ = RMSE(first) − RMSE(second), 100k engine-clustered resamples, RNG 7000–7002 —
the E6 recipe, unchanged.

## What a failure means

* H1 fails (CI includes 0) → §4.3 must say "the time channel is load-bearing on
  FD002 but not resolvable on FD001", and the cross-subset claim is dropped.
* H2 fails → the bounded encoding is not adoptable on either subset; §4.3 keeps
  "promising but unresolved" and adds that it does not replicate.
* Neither failure is a reason to re-run with another seed or to move the margin.

## Costs and artefacts

9 runs × ~8 min ≈ 25 min wall at 3-way concurrency, ≈ 0.6 GPU-hours.
Artefacts: `experiments/e3_fd001/<arm>_s<seed>/FD001/<arm>_s<seed>/{metrics.json,
matched_cutoff_predictions.csv}`, plus `results/dsh/e3/FD001_contrasts.json` from
`scripts/dsh_e3_contrasts.py --root experiments/e3_fd001 --dataset FD001`.
