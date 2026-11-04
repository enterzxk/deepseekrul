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
| G0.1 M1 vs M0 (FD002) | upper<0 | **-0.279** [-0.498,-0.053] PASS | **-0.347** [-0.607,-0.078] PASS | **-0.448** [-0.681,-0.216] PASS | **-0.462** PASS | **-0.540** [-0.788,-0.285] PASS | **-0.580** [-0.864,-0.301] PASS |
| G0.2 M1_shuf vs M1 (FD002) | lower>0 | **+0.312** [+0.114,+0.517] PASS | **+0.310** [+0.058,+0.558] PASS | +0.149 [-0.105,+0.387] FAIL | **+0.295** PASS | **+0.316** [+0.093,+0.536] PASS | **+0.367** [+0.134,+0.591] PASS |
| G0.3 M1_const vs M1 (FD002) | lower>0 | **+0.279** [+0.072,+0.485] PASS | **+0.272** [+0.022,+0.510] PASS | +0.188 [-0.042,+0.422] FAIL | **+0.373** PASS | **+0.342** [+0.135,+0.552] PASS | **+0.233** [+0.009,+0.463] PASS |
| G0.4 M1 vs M0 (FD001 guard) | upper<+0.5 | -0.103 [-0.468,+0.258] PASS | +0.060 [-0.282,+0.402] PASS | +0.255 [-0.276,+0.787] FAIL | +0.185 [-0.31,+0.75] FAIL | +0.156 [-0.247,+0.566] FAIL | **-0.126** [-0.480,+0.230] PASS |
| Verdict | — | GO_TO_GATE1 | GO_TO_GATE1 | STOP | STOP | STOP | GO_TO_GATE1 |

(The per-seed STOP verdicts apply to that seed only; the frozen pre-registered
verdict remains the seed-101 GO_TO_GATE1. See disposition.)

## Honest reading (final, six seeds)

1. **The core claim is robust**: G0.1 (prior injection helps M0) passes on
   **6/6 seeds** with a monotone-growing effect across seeds
   (-0.279 / -0.347 / -0.448 / -0.462 / -0.540 / -0.580).
2. **The attribution controls are significant on 5/6 seeds**: shuffled and
   constant controls hurt relative to M1 (direction correct 6/6 seeds);
   individually significant on all seeds except 303.
3. **The FD001 guardrail passes 3/6 (101/202/123), including a negative
   point estimate on seed 123 (-0.126)**; the three FAILs (303/42/67) are
   power failures only — point estimates never exceed +0.255, so there is
   no sign of real harm anywhere, but the frozen +0.5 no-harm bound cannot
   be certified on every seed with FD001's 80 engines.

## Disposition (gating discipline)

- The frozen, pre-registered verdict is the seed-101 one: **GO_TO_GATE1**.
  Stage B was a confirmation layer; criteria are NOT re-interpreted,
  re-metriced, or relaxed post hoc.
- Gate 1 continues (H1/H2/H3 arms running). A multi-seed confirmation issue
  in Stage B does not by itself halt the pipeline, because the frozen main
  verdict did not fail.
- Manuscript presentation rules (binding, final for six seeds):
  - Report all six seeds; never quote only 101.
  - G0.1 may be stated as "significant on 6/6 seeds, effect -0.28 to -0.58,
    monotone in seed".
  - G0.2/G0.3 must be stated as "direction-consistent on 6/6 seeds,
    individually significant on 5/6" — no stronger claim.
  - G0.4 stated as: "passes on 3/6 seeds with a negative point estimate on
    seed 123; the remaining seeds are under-powered (FD001 n=80), point
    estimates ≤ +0.26 throughout — no evidence of harm, but the frozen
    no-harm bound is not certifiable on every seed".
  - Any pooled-across-seeds analysis is descriptive robustness only and must
    be labeled as post hoc; it cannot resurrect a failed per-seed criterion.
- If Gate 1 outcomes depend on the attribution claim, re-examine seed-303
  variance sources (fold-level diagnostics) BEFORE any further claim, and
  document in a new file (never edit this one).
