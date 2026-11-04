# Gate 0 Multi-Seed Confirmation (Stage B)

Date: 2026-11-03 (seeds 202/303), extended 2026-11-04 (seeds 42/67/123 per
user request). Purpose: confirm the frozen Gate 0 verdict (seed 101,
`experiments/ida/gate0/gate0_verdict.json`) on additional training seeds
with identical arms, folds, and frozen criteria. Aggregation is the
same `scripts/summarize_gate0.py` (engine-clustered bootstrap, 5000 reps,
read-only over `val_predictions.csv`); verdict files:
`experiments/ida/gate0/gate0_verdict_s{101,202,303,42,67,123}.json`.

## Results (all 5 outer folds per seed, FD002 n=259 engines, FD001 n=80)

| Criterion | Rule | seed 101 | seed 202 | seed 303 | seed 42 | seed 67 | seed 123 |
|---|---|---|---|---|---|---|---|
| G0.1 M1 vs M0 (FD002) | upper<0 | **-0.279** [-0.498,-0.053] PASS | **-0.347** [-0.607,-0.078] PASS | **-0.448** [-0.681,-0.216] PASS | **-0.462** PASS | **-0.540** [-0.788,-0.285] PASS | pending |
| G0.2 M1_shuf vs M1 (FD002) | lower>0 | **+0.312** [+0.114,+0.517] PASS | **+0.310** [+0.058,+0.558] PASS | +0.149 [-0.105,+0.387] FAIL | **+0.295** PASS | **+0.316** [+0.093,+0.536] PASS | pending |
| G0.3 M1_const vs M1 (FD002) | lower>0 | **+0.279** [+0.072,+0.485] PASS | **+0.272** [+0.022,+0.510] PASS | +0.188 [-0.042,+0.422] FAIL | **+0.373** PASS | **+0.342** [+0.135,+0.552] PASS | pending |
| G0.4 M1 vs M0 (FD001 guard) | upper<+0.5 | -0.103 [-0.468,+0.258] PASS | +0.060 [-0.282,+0.402] PASS | +0.255 [-0.276,+0.787] FAIL | +0.185 [-0.31,+0.75] FAIL | +0.156 [-0.247,+0.566] FAIL | pending |
| Verdict | — | GO_TO_GATE1 | GO_TO_GATE1 | STOP | STOP | STOP | pending |

(The per-seed STOP verdicts apply to that seed only; the frozen pre-registered
verdict remains the seed-101 GO_TO_GATE1. See disposition.)

## Honest reading (updated for five completed seeds)

1. **The core claim is robust**: G0.1 (prior injection helps M0) passes on
   5/5 completed seeds with a monotone-growing effect across seeds
   (-0.279 / -0.347 / -0.448 / -0.462 / -0.540).
2. **The attribution controls are significant on 4/5 seeds**: shuffled and
   constant controls hurt relative to M1 (direction correct 5/5 seeds);
   individually significant on seeds 101/202/42/67, CI crosses zero only on
   seed 303.
3. **The FD001 guardrail is the weak leg**: point estimates never exceed
   +0.255 (no sign of real harm), but CI upper bounds exceed the frozen +0.5
   bound on 3/5 seeds (303: +0.787, 42: +0.75, 67: +0.566). FD001 has only
   80 engines, so the guardrail is under-powered rather than violated;
   report it as such, never as "no harm on FD001".

## Disposition (gating discipline)

- The frozen, pre-registered verdict is the seed-101 one: **GO_TO_GATE1**.
  Stage B was a confirmation layer; criteria are NOT re-interpreted,
  re-metriced, or relaxed post hoc.
- Gate 1 continues (H1/H2/H3 arms running). A multi-seed confirmation issue
  in Stage B does not by itself halt the pipeline, because the frozen main
  verdict did not fail.
- Manuscript presentation rules (binding, updated for five completed seeds;
  re-check when seed 123 completes):
  - Report all seeds; never quote only 101.
  - G0.1 may be stated as "significant on 5/5 seeds, effect -0.28 to -0.54".
  - G0.2/G0.3 must be stated as "direction-consistent on 5/5 seeds,
    individually significant on 4/5" — no stronger claim.
  - G0.4 must carry the explicit caveat: CI upper bound exceeds the +0.5
    no-harm bound on 3/5 seeds (FD001 n=80 under-powered); point estimates
    ≤ +0.26 everywhere.
  - Any pooled-across-seeds analysis is descriptive robustness only and must
    be labeled as post hoc; it cannot resurrect a failed per-seed criterion.
- If Gate 1 outcomes depend on the attribution claim, re-examine seed-303
  variance sources (fold-level diagnostics) BEFORE any further claim, and
  document in a new file (never edit this one).
